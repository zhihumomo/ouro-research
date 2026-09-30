"""推理侧增量解码：沿固定 64-token patch 图逐位置求值（纯 torch，无自定义算子）。

训练前向 (core.py) 保持唯一权威实现；本模块是推理专用的并行实现：
- 每 token 只计算当前位置，K/V/delta 历史驻留预分配缓存（index_copy_ 写入）；
- 每 patch 边界一次 prepare 完成全部状态条件化
  （MTM read -> temporal_attn -> state_ffn -> 各 state_attn 门控 K/V -> depth Q）；
- prefill 模式整 patch 并行（补零尾部仅做因果内工作，永不提交）；
- 满 patch 的记忆固化 (commit) 仍由训练前向完成，本模块不做记忆提交。

与 v_0d3 内核的对齐点：
- OuroState 四元组 (memory, recurrent tuple, queue, mtm)；
- STM 扫描为与训练逐字一致的顺序循环；prefill 后 scan 进位恢复到最后一个真实 token；
- 无效缓存槽位一律经 valid 掩码清洗后再参与乘算（防旧会话残留/NaN 污染）。
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

from naxi.v_0d3.ouro.core import Ouro


def sequential_scan(f: torch.Tensor, v: torch.Tensor, initial: torch.Tensor):
    """与训练 OuroSTM 逐字一致的顺序扫描（prefill 整 patch 模式使用）。"""
    states = []
    current = initial
    for position in range(f.shape[1]):
        current = f[:, position] * current + v[:, position]
        states.append(current)
    return torch.stack(states, dim=1)


class DeltaCacheSpec(NamedTuple):
    path: str
    width: int
    offset: int
    numel_per_batch: int


class PreparedOuroState(NamedTuple):
    gate_k: torch.Tensor
    gate_v: torch.Tensor
    depth_q: torch.Tensor


@dataclass
class OuroDecodeState:
    # 宿主计数器驱动会话状态机；张量全部为定地址的驻留缓存
    generation: int
    length: int
    position: torch.Tensor
    token_buffer: torch.Tensor
    delta: tuple[torch.Tensor, torch.Tensor]
    attention: tuple[torch.Tensor, torch.Tensor]
    emergent: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    scan: torch.Tensor
    positions: torch.Tensor
    prepared: PreparedOuroState | None = None


class DecodeLayout:
    """由模块拓扑静态推导的缓存布局（构建一次，运行期不变）。

    状态侧模块（temporal_attn/state_ffn/mtm、各处 state_proj、gate_attn.kv_proj、
    depth 注意力的 q/kv 投影）处理的是状态轴/深度轴，不占逐 token 时间缓存。
    """

    def __init__(self, core: Ouro):
        self.patch_size = core.temporal_queue_len
        patch = self.patch_size
        delta, attention, state_attention = [], [], []
        offset = 0
        for path, module in core.named_modules():
            kind = type(module).__name__
            if kind == "Attention":
                attention.append(path)
            elif kind == "OuroStateAttention":
                state_attention.append(path)
            if kind == "OuroCell":
                if (
                    path.startswith(("temporal_attn.", "state_ffn.", "mtm."))
                    or path.endswith(".state_proj")
                    or path.endswith(".gate_attn.kv_proj")
                    or path in ("final_depth_attn.q_proj", "final_depth_attn.kv_proj")
                ):
                    continue
                width = module.rank
            elif kind == "OuroLayer" and module.need_mem:
                width = core.embed_dim
            else:
                continue
            numel = patch * width
            delta.append(DeltaCacheSpec(path, width, offset, numel))
            offset += numel
        self.delta_specs = tuple(delta)
        self.delta_numel_per_batch = offset
        self.delta_indices = {spec.path: i for i, spec in enumerate(delta)}
        self.attention_paths = tuple(attention)
        self.attention_indices = {path: i for i, path in enumerate(attention)}
        self.state_attention_paths = tuple(state_attention)
        self.state_attention_indices = {path: i for i, path in enumerate(state_attention)}
        self.memory_indices = {name: i for i, (name, _) in enumerate(core.state_layout.specs)}

    def allocate(
        self,
        core: Ouro,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        state_dtype: torch.dtype = torch.float32,
    ) -> OuroDecodeState:
        patch = self.patch_size
        heads, head_dim = core.embed_dim // 64, 64
        device = torch.device(device)
        # CUDA autocast 将 normalize/LayerNorm 提升为 FP32；CPU autocast 策略不同。
        # 以探针推导缓存原生 dtype，而非对 K 做事后 cast。
        with torch.autocast(device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
            probe = torch.zeros(1, 1, head_dim, device=device, dtype=dtype)
            delta_dtype = F.normalize(probe, p=2, dim=-1, eps=1e-5).dtype
            norm_dtype = F.layer_norm(probe, (head_dim,)).dtype
        if device.type == "cuda":
            delta_dtype = norm_dtype = dtype
        return OuroDecodeState(
            generation=0,
            length=0,
            position=torch.zeros(1, dtype=torch.long, device=device),
            token_buffer=torch.empty(batch_size, patch, dtype=torch.long, device=device),
            delta=tuple(
                torch.empty(batch_size * self.delta_numel_per_batch, dtype=kind, device=device)
                for kind in (delta_dtype, dtype)
            ),
            attention=tuple(
                torch.empty(
                    len(self.attention_paths),
                    batch_size,
                    heads,
                    patch,
                    head_dim,
                    dtype=kind,
                    device=device,
                )
                for kind in (norm_dtype, dtype)
            ),
            emergent=tuple(
                torch.empty(core.blocks, batch_size, patch, core.embed_dim, dtype=kind, device=device)
                for kind in (dtype, norm_dtype, dtype)
            ),
            scan=torch.empty(core.blocks + 1, batch_size, core.embed_dim, dtype=state_dtype, device=device),
            positions=torch.arange(patch, dtype=torch.long, device=device),
        )

    def delta_views(self, arenas: tuple[torch.Tensor, torch.Tensor], batch_size: int):
        return tuple(
            tuple(
                arena.narrow(0, spec.offset * batch_size, spec.numel_per_batch * batch_size).view(
                    batch_size, self.patch_size, spec.width
                )
                for arena in arenas
            )
            for spec in self.delta_specs
        )


class _DecodeContext(NamedTuple):
    memory: tuple[torch.Tensor, ...]
    delta: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    attention: tuple[torch.Tensor, torch.Tensor]
    emergent: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    scan: torch.Tensor
    prepared: PreparedOuroState
    position: torch.Tensor
    valid: torch.Tensor
    prefill: bool = False


class OuroDecoder:
    """只读的模型/布局持有者；每个会话自带缓存张量。"""

    def __init__(self, core: Ouro):
        self.core = core
        self.patch_size = core.temporal_queue_len
        self.layout = DecodeLayout(core)
        self._masks: dict[torch.device, torch.Tensor] = {}
        # 避免在任何被捕获的前向中做模块遍历
        modules = dict(core.named_modules())
        self.state_attention_modules = tuple(modules[path] for path in self.layout.state_attention_paths)
        # 条件化槽位：layer 级 state_attn 用 active_c_norm；
        # ouro_stm.state_attn / stm.state_attn 用对应槽位的 recurrent
        self.conditioning_slots = tuple(
            (-1 if path == "stm.state_attn" else int(path.split(".")[1]))
            if ".ouro_stm.state_attn" in path or path == "stm.state_attn"
            else None
            for path in self.layout.state_attention_paths
        )

    def _causal_mask(self, device: torch.device) -> torch.Tensor:
        mask = self._masks.get(device)
        if mask is None:
            mask = torch.ones(self.patch_size, self.patch_size, dtype=torch.bool, device=device).tril_()
            self._masks[device] = mask
        return mask

    def _memory(self, path: str, memories: tuple[torch.Tensor, ...]):
        return memories[self.layout.memory_indices[path + ".mem"]]

    # ---- 每 patch 一次的状态条件化（lock_mem=True 直调模块，禁传 logical_seq_len）----

    def prepare(
        self,
        memory: torch.Tensor,
        recurrent: tuple[torch.Tensor, ...],
        queue: torch.Tensor,
        mtm: torch.Tensor,
    ) -> PreparedOuroState:
        core = self.core
        memories = core.state_layout.views(memory)
        batch = recurrent[-1].shape[0]

        base_active_c = recurrent[-1].to(torch.bfloat16)
        _, mid_state = core.mtm.read(base_active_c, mtm, self._memory("mtm.w_q", memories), lock_mem=True)
        temporal_input = torch.cat([mid_state, queue[:batch]], dim=1)
        queue_history = temporal_input[:, :-1, :]
        m_temporal = tuple(
            self._memory(f"temporal_attn.{name}", memories) for name in ("q_proj", "kv_proj", "o_proj")
        )
        _, temporal = core.temporal_attn(base_active_c, queue_history, m_temporal, lock_mem=True)
        active_c = base_active_c + temporal
        m_state_ffn = tuple(self._memory(f"state_ffn.{name}", memories) for name in ("w12", "w3"))
        _, h = core.state_ffn(core.c_state_norm(active_c), m_state_ffn, lock_mem=True)
        active_c = active_c + h
        active_c_norm = core.active_c_norm(active_c)

        prepared_k, prepared_v = [], []
        for path, module, slot in zip(
            self.layout.state_attention_paths, self.state_attention_modules, self.conditioning_slots
        ):
            condition = active_c_norm if slot is None else recurrent[slot]
            _, projected = module.state_proj(
                condition.unsqueeze(1),
                self._memory(path + ".state_proj", memories),
                lock_mem=True,
            )
            gate = module.gate_attn
            _, kv = gate.kv_proj(
                projected,
                self._memory(path + ".gate_attn.kv_proj", memories),
                lock_mem=True,
            )
            kv = kv.view(batch, 1, 2, gate.num_heads, gate.head_dim).permute(2, 0, 3, 1, 4)
            prepared_k.append(gate.k_norm(kv[0]))
            prepared_v.append(kv[1])

        depth = core.final_depth_attn
        _, depth_q = depth.q_proj(
            depth.norm_q(active_c_norm),
            self._memory("final_depth_attn.q_proj", memories),
            lock_mem=True,
        )
        depth_q = depth_q.view(batch, 1, depth.num_heads, 1, depth.head_dim)
        return PreparedOuroState(torch.stack(prepared_k), torch.stack(prepared_v), depth_q)

    # ---- 逐位置求值原语 ----

    @staticmethod
    def _clean_history(value: torch.Tensor, valid: torch.Tensor, patch: int):
        return torch.where(valid.view(1, patch, 1), value, 0.0)

    def _delta(self, path, q, k, e, context: _DecodeContext):
        patch = self.patch_size
        kv = context.delta[self.layout.delta_indices[path]]
        if context.prefill:
            kv[0].copy_(k)
            kv[1].copy_(e)
            scores = torch.bmm(q, k.transpose(-1, -2)).masked_fill(~self._causal_mask(q.device), 0.0)
            return torch.bmm(scores, e), scores
        kv[0].index_copy_(1, context.position, k.to(kv[0].dtype))
        kv[1].index_copy_(1, context.position, e.to(kv[1].dtype))
        keys = self._clean_history(kv[0], context.valid, patch)
        values = self._clean_history(kv[1], context.valid, patch)
        scores = torch.bmm(q, keys.transpose(-1, -2))
        scores = scores.masked_fill(~context.valid.view(1, 1, patch), 0.0)
        return torch.bmm(scores, values), scores

    def _cell(self, module, path: str, x: torch.Tensor, context: _DecodeContext):
        # 与 OuroCell.forward(lock_mem=True) 逐式一致；K/E 写缓存替代全序列打分
        memory = self._memory(path, context.memory)
        patch = self.patch_size
        base = module.linear(x)
        low_rank = x @ module.lora_A.T
        q, k, v, g, d = module.w_qkvgd(module.mem_norm(low_rank)).chunk(5, dim=-1)
        q = F.normalize(q, p=2, dim=-1, eps=1e-5)
        k = F.normalize(k, p=2, dim=-1, eps=1e-5)
        e = (torch.sigmoid(g) * (1.0 / patch)) * (v - k @ memory)
        delta, _ = self._delta(path, q, k, e, context)
        value = (q @ memory + delta) * module.act(d)
        value = module.hidden_norm(value)
        value = module.act(module.w_o(value))
        return base + value @ module.lora_B.T

    def _attention(self, module, path: str, x: torch.Tensor, context: _DecodeContext):
        batch, sequence = x.shape[:2]
        patch = self.patch_size
        qkv = self._cell(module.qkv_proj, path + ".qkv_proj", x, context)
        qkv = qkv.view(batch, sequence, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = module.q_norm(qkv[0]), module.k_norm(qkv[1]), qkv[2]
        index = self.layout.attention_indices[path]
        kv = (context.attention[0][index], context.attention[1][index])
        if context.prefill:
            kv[0].copy_(k)
            kv[1].copy_(v)
            value = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        else:
            kv[0].index_copy_(2, context.position, k.to(kv[0].dtype))
            kv[1].index_copy_(2, context.position, v.to(kv[1].dtype))
            valid = context.valid.view(1, 1, patch, 1)
            keys = torch.where(valid, kv[0], 0.0)
            values = torch.where(valid, kv[1], 0.0)
            value = F.scaled_dot_product_attention(
                q,
                keys,
                values,
                attn_mask=context.valid.view(1, 1, 1, patch),
                dropout_p=0.0,
                is_causal=False,
            )
        value = value.transpose(1, 2).contiguous().view(batch, sequence, module.embed_dim)
        gate = torch.sigmoid(self._cell(module.gate, path + ".gate", x, context))
        return self._cell(module.out_proj, path + ".out_proj", gate * value, context)

    def _state_attention(self, module, path: str, x: torch.Tensor, context: _DecodeContext):
        gate = module.gate_attn
        batch, sequence = x.shape[:2]
        q = self._cell(gate.q_proj, path + ".gate_attn.q_proj", module.in_norm(x), context)
        q = gate.q_norm(q.view(batch, sequence, gate.num_heads, gate.head_dim).transpose(1, 2))
        index = self.layout.state_attention_indices[path]
        kv = (context.prepared.gate_k[index], context.prepared.gate_v[index])
        scores = torch.matmul(q, kv[0].transpose(-1, -2)) / math.sqrt(gate.head_dim)
        injection = (
            (torch.sigmoid(scores) * kv[1])
            .transpose(1, 2)
            .contiguous()
            .view(batch, sequence, module.embed_dim)
        )
        output = self._attention(module.attn, path + ".attn", module.norm(x + injection), context)
        return injection + output

    def _ffn(self, module, path: str, x: torch.Tensor, context: _DecodeContext):
        a, b = self._cell(module.w12, path + ".w12", x, context).chunk(2, dim=-1)
        value = module.act(a) * b
        return self._cell(module.w3, path + ".w3", value, context)

    def _stm(self, module, path: str, x: torch.Tensor, scan_slot: int, context: _DecodeContext):
        # commit 专属的 state_proj/ouro_norm 不在此处（记忆固化由训练前向完成）
        x = x + self._state_attention(module.state_attn, path + ".state_attn", x, context)
        gates = self._cell(module.w, path + ".w", x, context)
        i, f, g, o = gates.chunk(4, dim=-1)
        v = torch.sigmoid(i) * module.act(g)
        if context.prefill:
            c = sequential_scan(torch.sigmoid(f), v, context.scan[scan_slot])
            # scan 进位恢复到最后一个真实 token，补零尾部不进入进位
            context.scan[scan_slot].copy_(c.index_select(1, context.position)[:, 0])
            h = torch.sigmoid(o) * c
        else:
            c = torch.sigmoid(f[:, 0]) * context.scan[scan_slot] + v[:, 0]
            context.scan[scan_slot].copy_(c)
            h = torch.sigmoid(o) * c.unsqueeze(1)
        return self._cell(module.out_proj, path + ".out_proj", module.out_norm(h), context)

    def _full_layer(self, module, path: str, x: torch.Tensor, context: _DecodeContext):
        memory = self._memory(path, context.memory)
        patch = self.patch_size
        qkvgd = self._cell(module.w_qkvgd, path + ".w_qkvgd", module.mem_norm(x), context)
        q, k, v, g, d = qkvgd.chunk(5, dim=-1)
        q = F.normalize(q, p=2, dim=-1, eps=1e-5)
        k = F.normalize(k, p=2, dim=-1, eps=1e-5)
        e = (torch.sigmoid(g) * (1.0 / patch)) * (v - k @ memory)
        delta, scores = self._delta(path, q, k, e, context)
        value = (q @ memory + delta) * module.act(d)
        value = self._cell(module.w_o, path + ".w_o", module.out_norm(value), context)
        value = self._cell(module.o, path + ".o", module.act(value), context)
        return x + value, scores

    def _block(self, module, path: str, x: torch.Tensor, index: int, context: _DecodeContext):
        patch = self.patch_size
        residual = x
        out_list = []
        score_sum = None
        for layer_index, layer in enumerate(module.ouro_layers):
            layer_path = path + f".ouro_layers.{layer_index}"
            x = x + self._state_attention(layer.state_attn, layer_path + ".state_attn", x, context)
            if layer.need_mem:
                x, scores = self._full_layer(layer, layer_path, x, context)
                score_sum = scores if score_sum is None else score_sum + scores
            else:
                x = x + self._stm(layer.ouro_stm, layer_path + ".ouro_stm", x, index, context)
            out_list.append(x)
        inner_residual = x
        scale = module.embed_dim**-0.5
        history = tuple(arena[index] for arena in context.emergent)
        projected = (x @ module.ouro_self_attn_proj) * scale
        if context.prefill:
            history[0].copy_(projected)
            emergent = torch.bmm(score_sum, projected)
        else:
            history[0].index_copy_(1, context.position, projected.to(history[0].dtype))
            emergent = torch.bmm(score_sum, self._clean_history(history[0], context.valid, patch))
        normalized = module.ouro_self_attn_norm(module.act(emergent))
        values = self._cell(module.w_v, path + ".w_v", module.v_norm(residual), context)
        if context.prefill:
            history[1].copy_(normalized)
            history[2].copy_(values)
            keys = normalized
            bias = (score_sum * scale).masked_fill(~self._causal_mask(x.device), float("-inf"))
        else:
            history[1].index_copy_(1, context.position, normalized.to(history[1].dtype))
            history[2].index_copy_(1, context.position, values.to(history[2].dtype))
            keys = self._clean_history(history[1], context.valid, patch)
            values = self._clean_history(history[2], context.valid, patch)
            bias = (score_sum * scale).masked_fill(~context.valid.view(1, 1, patch), float("-inf"))
        output = F.scaled_dot_product_attention(
            normalized.unsqueeze(1),
            keys.unsqueeze(1),
            values.unsqueeze(1),
            attn_mask=bias.unsqueeze(1),
            dropout_p=0.0,
            scale=scale,
        ).squeeze(1)
        gate = torch.sigmoid(
            self._cell(module.ouro_self_attn_gate, path + ".ouro_self_attn_gate", residual, context)
        )
        x = inner_residual + self._cell(
            module.ouro_self_attn_output_proj, path + ".ouro_self_attn_output_proj", gate * output, context
        )
        x = x + self._ffn(module.ffn, path + ".ffn", module.norm(x), context)
        out_list.append(x)
        return x, out_list

    def _depth(self, history_states, context: _DecodeContext):
        module = self.core.final_depth_attn
        batch, sequence = history_states[0].shape[:2]
        depth = len(history_states)
        history = torch.stack(history_states, dim=2)
        # 深度 KV 始终处理完整深度轴（直调模块，无时间缓存）
        _, kv = module.kv_proj(
            module.norm_kv(history),
            self._memory("final_depth_attn.kv_proj", context.memory),
            lock_mem=True,
        )
        kv = kv.view(batch, sequence, depth, 2, module.num_heads, module.head_dim).permute(3, 0, 1, 4, 2, 5)
        scores = torch.matmul(context.prepared.depth_q, kv[0].transpose(-1, -2)) / math.sqrt(module.head_dim)
        value = (
            torch.matmul(F.softmax(scores, dim=-1), kv[1])
            .squeeze(-2)
            .contiguous()
            .view(batch, sequence, module.embed_dim)
        )
        return self._cell(module.o_proj, "final_depth_attn.o_proj", value, context)

    def decode(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        prepared: PreparedOuroState,
        delta: tuple[torch.Tensor, torch.Tensor],
        attention: tuple[torch.Tensor, torch.Tensor],
        emergent: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        scan: torch.Tensor,
        position: torch.Tensor,
        positions: torch.Tensor,
        prefill: bool = False,
    ):
        core = self.core
        context = _DecodeContext(
            core.state_layout.views(memory),
            self.layout.delta_views(delta, x.shape[0]),
            attention,
            emergent,
            scan,
            prepared,
            position,
            positions <= position,
            prefill,
        )
        original = x
        x = x + self._attention(core.in_attn, "in_attn", core.in_norm(x), context)
        x = x + self._ffn(core.in_ffn, "in_ffn", core.in_ffn_norm(x), context)
        history = [original, x]
        for index, block in enumerate(core.ouro_blocks):
            if index > 0:
                stacked = torch.stack(history, dim=2)
                keys = core.attnres_k_norm(stacked)
                scores = torch.matmul(keys, core.attnres_queries[index - 1]) / (core.embed_dim**0.5)
                x = torch.sum(F.softmax(scores, dim=-1).unsqueeze(-1) * stacked, dim=2)
            x, outputs = self._block(block, f"ouro_blocks.{index}", x, index, context)
            history.extend(outputs)
        x = x + self._depth(history, context)
        x = x + self._ffn(core.m_ffn, "m_ffn", core.m_ffn_norm(x), context)
        x = x + self._attention(core.out_attn, "out_attn", core.out_norm(x), context)
        x = x + self._ffn(core.out_ffn, "out_ffn", core.out_ffn_norm(x), context)
        return self._stm(core.stm, "stm", x, core.blocks, context)
