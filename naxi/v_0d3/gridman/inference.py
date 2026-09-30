"""会话级推理调度：缓存持有、每 patch 一次 prepare、逐 token 解码、bulk prefill、
可选 CUDA Graph 重放与 bf16 权重驻留（默认 CUDA 全开，CPU 自动降级为 eager）。

纪律：
- commit（lock_mem=False）仅允许满 patch_size token，走训练前向本身；
- lock_mem=True 的部分 patch 只许扩展、不许改写（extend-only）；
- 图输出一律 clone 后再暴露给会话状态，公共状态永不别名图内部存储。
"""

import torch

from naxi.v_0d3.ouro.core import OuroState
from naxi.v_0d3.ouro.decode import OuroDecoder, PreparedOuroState
from naxi.v_0d3.gridman.cuda_graph import StaticCudaGraph
from naxi.v_0d3.gridman.inference_runtime import commit_view, decode_view


def _clone_state(state: OuroState) -> OuroState:
    return OuroState(
        state.memory.clone(),
        tuple(r.clone() for r in state.recurrent),
        state.queue.clone(),
        state.mtm.clone(),
    )


class GridmanInference:
    @torch.inference_mode()
    def __init__(
        self,
        model,
        device: torch.device | str,
        init_state: OuroState | None = None,
        cuda_graph: bool = True,
        bf16_weights: bool = True,
        prefill_min: int = 3,
        allow_eager_prefill: bool = False,
    ):
        if model.training:
            raise ValueError("Cached inference requires model.eval()")
        self.model = model
        self.device = torch.device(device)
        # CUDA Graph 仅可捕获 CUDA kernel 流；CPU 自动降级为 eager
        self.cuda_graph_enabled = bool(cuda_graph) and self.device.type == "cuda"
        self.prefill_min = prefill_min
        self.use_prefill = self.cuda_graph_enabled or allow_eager_prefill
        self._decode_memory = None
        self._source_signature = tuple(
            (name, id(p), p.dtype, p.device, tuple(p.shape)) for name, p in model.named_parameters()
        )
        self._weight_memo = {}
        # bf16 权重副本服务于 graph 静态地址链路；eager 链路不建副本
        weight_dtype = torch.bfloat16 if (self.cuda_graph_enabled and bf16_weights) else None
        self.runtime = decode_view(model, weight_dtype, self._weight_memo)
        self.decoder = OuroDecoder(self.runtime.core_ouro)
        self.commit_runtime = commit_view(model, weight_dtype, self._weight_memo) if weight_dtype else model
        self._prepare_graph = self._decode_graph = self._commit_graph = self._prefill_graph = None
        self._graph_state = None
        self.patch_size = self.decoder.layout.patch_size
        # 会话初始状态快照：默认由 load_model_checkpoint 提供完整热启动
        # （训练稳态记忆现场——delta-rule 记忆依赖训练稳态，空白起步为分布外）；
        # reset 回到该快照
        self._session_init_state = (
            init_state if init_state is not None else model.init_inference_state(1, self.device)
        )
        self.state = _clone_state(self._session_init_state)
        self.cache = self.decoder.layout.allocate(self.runtime.core_ouro, 1, self.device)
        self.host_tokens = torch.empty(
            1, self.patch_size, dtype=torch.long, pin_memory=self.device.type == "cuda"
        )
        self.buffer_length = 0
        self.last_logits = None
        self._prefix = []

    # ---- 内部路径 ----

    def _decode_token(
        self, memory, prepared, delta, attention, emergent, scan, position, positions, token_buffer
    ):
        ids = token_buffer.index_select(1, position)
        hidden = self.decoder.decode(
            self.runtime.byte_emb(ids),
            memory,
            prepared,
            delta,
            attention,
            emergent,
            scan,
            position,
            positions,
        )
        return self.runtime.out_proj(self.runtime.out_norm(hidden))[:, 0]

    def _copy_graph_state(self) -> OuroState:
        if self._graph_state is None:
            self._graph_state = _clone_state(self.state)
        else:
            self._graph_state.memory.copy_(self.state.memory)
            for target, value in zip(self._graph_state.recurrent, self.state.recurrent):
                target.copy_(value)
            self._graph_state.queue.copy_(self.state.queue)
            self._graph_state.mtm.copy_(self.state.mtm)
        return self._graph_state

    def _prepare_decode_state(self, memory, recurrent, queue, mtm) -> PreparedOuroState:
        prepared = self.decoder.prepare(memory, recurrent, queue, mtm)
        if self.cuda_graph_enabled:
            prepared = PreparedOuroState(
                prepared.gate_k.to(torch.bfloat16), prepared.gate_v, prepared.depth_q
            )
        return prepared

    def _prepare(self):
        if not self.cuda_graph_enabled:
            return self._prepare_decode_state(*self.state)
        if self._decode_memory is None:
            self._decode_memory = self.state.memory.to(torch.bfloat16)
        else:
            self._decode_memory.copy_(self.state.memory)
        args = self._copy_graph_state()
        if self._prepare_graph is None:
            self._prepare_graph = StaticCudaGraph(self._prepare_decode_state, args)
        return self._prepare_graph.replay()

    def _commit_tokens(self, token_buffer, memory, recurrent, queue, mtm):
        # commit 复用训练前向（lock_mem=False，静态 logical_seq_len=patch_size）
        logits, next_state = self.commit_runtime(
            token_buffer, OuroState(memory, recurrent, queue, mtm), False, logical_seq_len=self.patch_size
        )
        return (next_state.memory, *next_state.recurrent, next_state.queue, next_state.mtm, logits[:, -1])

    @staticmethod
    def _unpack_commit(flat):
        n_rec = len(flat) - 4  # memory, *recurrent, queue, mtm, logits
        state = OuroState(flat[0], tuple(flat[1 : 1 + n_rec]), flat[1 + n_rec], flat[2 + n_rec])
        return state, flat[-1]

    def _commit(self):
        if not self.cuda_graph_enabled:
            return self._unpack_commit(self._commit_tokens(self.cache.token_buffer, *self.state))
        args = (self.cache.token_buffer, *self._copy_graph_state())
        if self._commit_graph is None:
            self._commit_graph = StaticCudaGraph(self._commit_tokens, args)
        # 公共状态与 logits 永不别名可重用的图输出存储
        return self._unpack_commit(tuple(t.clone() for t in self._commit_graph.replay()))

    def _decode(self):
        cache = self.cache
        if not self.cuda_graph_enabled:
            return self._decode_token(
                self.state.memory,
                cache.prepared,
                cache.delta,
                cache.attention,
                cache.emergent,
                cache.scan,
                cache.position,
                cache.positions,
                cache.token_buffer,
            )
        if self._decode_graph is None:
            # prepared 输出与图状态跨 patch 保持地址稳定
            args = (
                self._decode_memory,
                cache.prepared,
                cache.delta,
                cache.attention,
                cache.emergent,
                cache.scan,
                cache.position,
                cache.positions,
                cache.token_buffer,
            )
            self._decode_graph = StaticCudaGraph(
                self._decode_token, args, (*cache.delta, *cache.attention, *cache.emergent, cache.scan)
            )
        return self._decode_graph.replay()

    def _prefill_tokens(
        self, memory, recurrent, prepared, delta, attention, emergent, scan, position, positions, token_buffer
    ):
        scan.copy_(torch.stack(tuple(recurrent)))
        hidden = self.decoder.decode(
            self.runtime.byte_emb(token_buffer),
            memory,
            prepared,
            delta,
            attention,
            emergent,
            scan,
            position,
            positions,
            prefill=True,
        )
        hidden = hidden.index_select(1, position)
        return self.runtime.out_proj(self.runtime.out_norm(hidden))[:, 0]

    def _prefill(self):
        cache = self.cache
        memory = self._decode_memory if self.cuda_graph_enabled else self.state.memory
        recurrent = self._graph_state.recurrent if self.cuda_graph_enabled else self.state.recurrent
        args = (
            memory,
            recurrent,
            cache.prepared,
            cache.delta,
            cache.attention,
            cache.emergent,
            cache.scan,
            cache.position,
            cache.positions,
            cache.token_buffer,
        )
        if not self.cuda_graph_enabled:
            return self._prefill_tokens(*args)
        if self._prefill_graph is None:
            self._prefill_graph = StaticCudaGraph(
                self._prefill_tokens, args, (*cache.delta, *cache.attention, *cache.emergent, cache.scan)
            )
        return self._prefill_graph.replay()

    # ---- 会话管理 ----

    @torch.inference_mode()
    def warmup(self):
        """在首次提示词前捕获固定路径，保持状态、历史与 RNG 不受污染。CPU 无图可捕获。"""
        if not self.cuda_graph_enabled:
            return
        if self.buffer_length:
            raise ValueError("Warmup requires an empty pending patch")
        state, generation = self.state, self.cache.generation
        try:
            self.forward_prefix([0] * 16, True)
            self.forward_prefix([0] * 17, True)
            self.forward_prefix([0] * self.patch_size, False)
            torch.cuda.synchronize(self.device)
        finally:
            self.reset()
            self.state, self.cache.generation = state, generation

    @torch.inference_mode()
    def refresh_weights(self):
        """换权重后弃图并重置会话；device/dtype/结构变更需重建会话。"""
        if self.model.training:
            raise ValueError("Cached inference requires model.eval()")
        signature = tuple(
            (name, id(p), p.dtype, p.device, tuple(p.shape)) for name, p in self.model.named_parameters()
        )
        if signature != self._source_signature:
            raise ValueError("Model tensors/device/dtype changed; construct a new inference session")
        self._prepare_graph = self._decode_graph = self._commit_graph = self._prefill_graph = None
        self._graph_state = None
        if self.cuda_graph_enabled:
            for parameter in self.model.parameters():
                copied = self._weight_memo.get(id(parameter))
                if copied is not None:
                    copied.copy_(parameter)
        self.reset()

    @torch.inference_mode()
    def reset(self):
        """回到会话初始状态（热启动快照），不重新分配缓存竞技场。"""
        self.state = _clone_state(self._session_init_state)
        self.cache.generation += 1
        self.cache.length = 0
        self.cache.position.zero_()
        self.cache.prepared = None
        self.buffer_length = 0
        self._prefix = []
        self.last_logits = None

    def _store_prefix(self, tokens: list[int]):
        start, end = self.buffer_length, len(tokens)
        if not tokens or end < start or end > self.patch_size:
            raise ValueError(f"A pending patch must contain 1..{self.patch_size} tokens and only be extended")
        if tokens[:start] != self._prefix:
            raise ValueError("Cached prefix tokens cannot be rewritten; reset the session first")
        if start == end:
            return
        # 复用宿主/设备两侧 token 存储；同步的微小 H2D 保证 pinned 缓冲不可变至传输完成
        self.host_tokens[0, start:end] = torch.as_tensor(tokens[start:end], dtype=torch.long)
        self.cache.token_buffer[:, start:end].copy_(self.host_tokens[:, start:end])
        self.buffer_length = end
        self._prefix = list(tokens)

    @torch.inference_mode()
    def forward_prefix(self, tokens: list[int], lock_mem: bool):
        if not lock_mem and len(tokens) != self.patch_size:
            raise ValueError(f"Persistent state may only commit a complete {self.patch_size}-token patch")
        if lock_mem and len(tokens) == self.patch_size:
            raise ValueError(f"Token {self.patch_size} must use the unique bulk commit entry point")
        self._store_prefix(tokens)
        cache = self.cache
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            if not lock_mem:
                next_state, logits = self._commit()
                self.state = next_state
                self.last_logits = logits
                cache.generation += 1
                cache.length = 0
                cache.prepared = None
                self.buffer_length = 0
                self._prefix = []
                return logits
            if cache.prepared is None:
                cache.prepared = self._prepare()
                cache.scan.copy_(torch.stack(tuple(self.state.recurrent)))
            if self.use_prefill and len(tokens) - cache.length >= self.prefill_min:
                # 补零的未来 token 仅做因果内工作，永不提交
                cache.token_buffer[:, len(tokens) :].zero_()
                cache.position.fill_(len(tokens) - 1)
                self.last_logits = self._prefill()
            else:
                for position in range(cache.length, len(tokens)):
                    cache.position.fill_(position)
                    self.last_logits = self._decode()
            if self.cuda_graph_enabled and cache.length < len(tokens):
                self.last_logits = self.last_logits.clone()
            cache.length = len(tokens)
        return self.last_logits
