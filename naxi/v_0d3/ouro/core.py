import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint


HEAD_DIM = 64


class OuroState(NamedTuple):
    """显式状态容器（驻留 fp32；窗口内 recurrent 可为 bf16，边界由 detach_state 收敛）

    memory:    [total_numel] 扁平打包的全局方阵记忆（batch 共享，StateLayout 布局）
    recurrent: tuple，长度 blocks+1，每项 [B, D] 的 STM c_state
               （[i] 为 block i 的 STM，[-1] 为顶层 STM；不堆叠为张量——
               保持对象级跨步图拓扑，存档时按 list 存储）
    queue:     [cap, temporal_queue_len, D] c 历史队列（cap: 训练=max_batch，推理=batch）
    mtm:       [cap, D, D] MTM 专属记忆（per-batch，不跨 rank 平均）
    """
    memory: torch.Tensor
    recurrent: tuple[torch.Tensor, ...]
    queue: torch.Tensor
    mtm: torch.Tensor


class StateLayout:
    """memory 扁平打包布局。

    spec path 为稳定的模块路径命名（`in_attn.qkv_proj.mem` 形态），
    供调试与外部工具按名寻址；布局顺序 = 模块树声明顺序（DFS）。
    """

    def __init__(self, specs: list[tuple[str, tuple[int, ...]]]):
        self.specs = specs
        self.sizes = [math.prod(shape) for _, shape in specs]
        self.total_numel = sum(self.sizes)

    def views(self, memory: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # split 出的视图共享底层 storage（只读用途）
        return tuple(
            seg.view(shape)
            for seg, (_, shape) in zip(torch.split(memory, self.sizes), self.specs)
        )

    def flatten(self, memories) -> torch.Tensor:
        return torch.cat([m.reshape(-1) for m in memories])


def _memory_specs(ouro: 'Ouro') -> list[tuple[str, tuple[int, ...]]]:
    """按模块树 DFS（与 named_modules 同序）收集全局记忆条目：
    每个 OuroCell 的 mem [1, rank, rank] 与每个 need_mem OuroLayer 的 mem [1, D, D]。
    MTM 记忆 / STM c_state / c_state_queue 不属于此布局
    （分别进 OuroState.mtm / recurrent / queue）。"""
    specs = []
    for name, module in ouro.named_modules():
        if isinstance(module, OuroLayer) and module.need_mem:
            specs.append((f'{name}.mem', (1, module.embed_dim, module.embed_dim)))
        elif isinstance(module, OuroCell):
            specs.append((f'{name}.mem', (1, module.rank, module.rank)))
    return specs


class OuroNorm(nn.Module):
    def __init__(self, embed_dim: int, init_bias: float = 0):
        super().__init__()
        self.embed_dim = embed_dim

        self.k_proj = nn.Linear(embed_dim, 1)
        self.act = nn.Sigmoid()

        nn.init.zeros_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, init_bias)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.act(self.k_proj(x))
        x_normed = F.normalize(x, p=2.0, dim=-1) * (self.embed_dim ** 0.5)
        return k * x_normed


class OuroCell(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
    ):
        super().__init__()
        self.intrinsic_loss = 0.0
        self.linear = nn.Linear(in_features, out_features, bias)

        self.multiple_of = 16
        hidden_dim = int(out_features / 16)
        self.rank = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank)
        )

        self.mem_norm = nn.LayerNorm(self.rank)
        self.w_qkvgd = nn.Linear(self.rank, self.rank * 5)
        self.act = nn.SiLU()
        self.hidden_norm = nn.LayerNorm(self.rank)
        self.w_o = nn.Linear(self.rank, self.rank, bias=False)

        with torch.no_grad():
            torch.nn.init.normal_(
                self.w_qkvgd.weight[3 * self.rank : 4 * self.rank, :],
                mean=0.0, std=0.02
            )
            torch.nn.init.constant_(
                self.w_qkvgd.bias[3 * self.rank : 4 * self.rank],
                -6.0
            )

            nn.init.kaiming_uniform_(
                self.lora_A,
                a=math.sqrt(5),
            )

            nn.init.normal_(
            self.lora_B,
                mean=0.0,
                std=1e-2,
            )

    # 记忆形状 [1, rank, rank]（batch 共享，提交时跨 batch 求均值）
    @property
    def mem_shape(self) -> tuple[int, ...]:
        return (1, self.rank, self.rank)

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor,
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape

        if len(orig_shape) == 2:
            x = x.unsqueeze(1)
        elif len(orig_shape) > 3:
            x = x.contiguous().view(-1, orig_shape[-2], orig_shape[-1])

        _, seq_len, _ = x.shape
        # 推理传入静态 logical_seq_len(=patch_size)与训练定长对齐; None 用物理长度
        eff_seq_len = logical_seq_len if logical_seq_len is not None else seq_len

        base_output = self.linear(x)

        x_low_rank = x @ self.lora_A.T

        qkvgd: torch.Tensor = self.w_qkvgd(self.mem_norm(x_low_rank))
        context_q, context_k, context_v, context_g, context_d = qkvgd.chunk(5, dim=-1)

        context_q = F.normalize(context_q, p=2, dim=-1, eps=1e-5)
        context_k = F.normalize(context_k, p=2, dim=-1, eps=1e-5)

        mem_g = torch.sigmoid(context_g) * (1.0 / eff_seq_len)
        # 预测的 V
        v_retrieved: torch.Tensor = context_k @ mem

        # 计算真实 V 与预测 V 的 Delta
        delta_v = context_v - v_retrieved

        v_dyn = mem_g * delta_v

        # 外积更新
        delta_mem: torch.Tensor = torch.bmm(context_k.transpose(-1, -2), v_dyn)

        # 记忆更新（lock_mem 时原样直通）
        next_mem: torch.Tensor = mem + delta_mem
        if not lock_mem:
            next_mem = next_mem.mean(0, keepdim=True)
        else:
            next_mem = mem

        # 历史记忆
        mem_out_prev = context_q @ mem

        # QK 的标准注意力打分矩阵
        scores = torch.bmm(context_q, context_k.transpose(-1, -2))

        # 动态生成 Causal Mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril_()
        scores.masked_fill_(~mask, 0.0)

        mem_out_delta: torch.Tensor = torch.bmm(scores, v_dyn)

        # 合并输出
        mem_out = mem_out_prev
        mem_out = mem_out + mem_out_delta

        # 动态门控
        mem_out = mem_out * self.act(context_d)
        mem_out = self.hidden_norm(mem_out)
        mem_out = self.act(self.w_o(mem_out))

        lora_output = mem_out @ self.lora_B.T
        output = base_output + lora_output

        if len(orig_shape) == 2:
            output = output.squeeze(1)
        elif len(orig_shape) > 3:
            out_shape = orig_shape[:-1] + (output.shape[-1],)
            output = output.view(out_shape)

        return next_mem, output


class Attention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim
        self.dropout = dropout

        self.qkv_proj = OuroCell(embed_dim, 3 * embed_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.gate = OuroCell(embed_dim, embed_dim, bias=False)
        self.out_proj = OuroCell(embed_dim, embed_dim, bias=False)

    def forward(
        self, x: torch.Tensor, mems: tuple[torch.Tensor, ...],
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        qkv_mem, gate_mem, out_mem = mems
        batch_size, seq_len, _ = x.shape

        next_qkv_mem, qkv = self.qkv_proj(x, qkv_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)

        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k  = self.q_norm(q), self.k_norm(k)

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        next_gate_mem, gate = self.gate(x, gate_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        gate = torch.sigmoid(gate)
        next_out_mem, output = self.out_proj(gate * attn_output, out_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)

        return (next_qkv_mem, next_gate_mem, next_out_mem), output


class GateAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim

        self.dropout = dropout

        self.q_proj = OuroCell(embed_dim, embed_dim, bias=False)
        self.kv_proj = OuroCell(embed_dim, 2 * embed_dim, bias=False)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, mems: tuple[torch.Tensor, ...],
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        q_mem, kv_mem = mems
        batch_size, seq_len, _ = x.shape

        next_q_mem, q = self.q_proj(x, q_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if state.dim() == 2:
            state = state.unsqueeze(1)

        # kv_proj(state) 保持自然长度（禁传 logical_seq_len）
        next_kv_mem, kv = self.kv_proj(state, kv_mem, lock_mem=lock_mem)
        kv = kv.view(batch_size, 1, 2, self.num_heads, self.head_dim)

        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q, k = self.q_norm(q), self.k_norm(k)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        gate = torch.sigmoid(scores)

        if self.training and self.dropout > 0.0:
            gate = F.dropout(gate, p=self.dropout)

        attn_output = gate * v

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return (next_q_mem, next_kv_mem), attn_output


class OuroStateAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim

        self.dropout = dropout

        self.state_proj = OuroCell(self.embed_dim, self.embed_dim)

        self.gate_attn = GateAttention(self.embed_dim, self.dropout)
        self.in_norm = nn.LayerNorm(self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(self.embed_dim, self.dropout)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, mems: tuple[torch.Tensor, ...],
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        sp_mem, gq_mem, gkv_mem, aqkv_mem, ag_mem, ao_mem = mems

        state = state.unsqueeze(1)
        # state_proj 保持自然长度（禁传 logical_seq_len）
        next_sp_mem, state = self.state_proj(state, sp_mem, lock_mem=lock_mem)

        next_ga_mems, state_injection = self.gate_attn(
            self.in_norm(x), state, (gq_mem, gkv_mem),
            lock_mem=lock_mem, logical_seq_len=logical_seq_len,
        )
        x = x + state_injection
        next_a_mems, attn_out = self.attn(
            self.norm(x), (aqkv_mem, ag_mem, ao_mem),
            lock_mem=lock_mem, logical_seq_len=logical_seq_len,
        )
        return (next_sp_mem, *next_ga_mems, *next_a_mems), state_injection + attn_out


class OuroDepthAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = embed_dim // self.head_dim

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.q_proj = OuroCell(embed_dim, embed_dim, bias=False)
        self.kv_proj = OuroCell(embed_dim, 2 * embed_dim, bias=False)
        self.o_proj = OuroCell(embed_dim, embed_dim, bias=False)

        with torch.no_grad():
            nn.init.zeros_(self.o_proj.linear.weight)

    def forward(
        self, active_c: torch.Tensor, history_states: list[torch.Tensor],
        mems: tuple[torch.Tensor, ...], lock_mem: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        q_mem, kv_mem, o_mem = mems
        batch_size = active_c.shape[0]
        seq_len = history_states[0].shape[1]
        num_layers = len(history_states)

        H = torch.stack(history_states, dim=2)

        q_norm = self.norm_q(active_c)
        next_q_mem, q = self.q_proj(q_norm, q_mem, lock_mem=lock_mem)

        q = q.view(batch_size, 1, self.num_heads, 1, self.head_dim)

        H_norm = self.norm_kv(H)
        next_kv_mem, kv = self.kv_proj(H_norm, kv_mem, lock_mem=lock_mem)

        kv = kv.view(batch_size, seq_len, num_layers, 2, self.num_heads, self.head_dim)
        kv = kv.permute(3, 0, 1, 4, 2, 5)
        k, v = kv[0], kv[1]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.squeeze(-2).contiguous().view(batch_size, seq_len, self.embed_dim)

        next_o_mem, output = self.o_proj(out, o_mem, lock_mem=lock_mem)
        return (next_q_mem, next_kv_mem, next_o_mem), output


class OuroTemporalAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = embed_dim // self.head_dim

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.q_proj = OuroCell(embed_dim, embed_dim, bias=False)
        self.kv_proj = OuroCell(embed_dim, 2 * embed_dim, bias=False)
        self.o_proj = OuroCell(embed_dim, embed_dim, bias=False)

        with torch.no_grad():
            nn.init.zeros_(self.o_proj.linear.weight)

    def forward(
        self, current_c: torch.Tensor, state_queue: torch.Tensor,
        mems: tuple[torch.Tensor, ...], lock_mem: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        q_mem, kv_mem, o_mem = mems
        batch_size = current_c.shape[0]

        q_norm = self.norm_q(current_c)

        next_q_mem, q = self.q_proj(q_norm, q_mem, lock_mem=lock_mem)
        q = q.view(batch_size, self.num_heads, 1, self.head_dim)

        kv_norm = self.norm_kv(state_queue)
        next_kv_mem, kv = self.kv_proj(kv_norm, kv_mem, lock_mem=lock_mem)
        kv = kv.view(batch_size, state_queue.shape[1], 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.squeeze(2).contiguous().view(batch_size, self.embed_dim)
        next_o_mem, output = self.o_proj(out, o_mem, lock_mem=lock_mem)
        return (next_q_mem, next_kv_mem, next_o_mem), output


class FFN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim

        self.multiple_of = 256
        hidden_dim = int(2 * (4 * self.embed_dim) / 3)
        self.hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        self.w12 = OuroCell(self.embed_dim, 2 * self.hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.w3 = OuroCell(self.hidden_dim, self.embed_dim, bias=False)

    def forward(
        self, x: torch.Tensor, mems: tuple[torch.Tensor, ...],
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        w12_mem, w3_mem = mems
        next_w12_mem, combined_projected = self.w12(x, w12_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        x_w1, x_v = torch.chunk(combined_projected, chunks=2, dim=-1)

        swiglu_out = self.act(x_w1) * x_v
        next_w3_mem, output = self.w3(swiglu_out, w3_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        return (next_w12_mem, next_w3_mem), output


class OuroSTM(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        self.state_attn = OuroStateAttention(self.embed_dim)
        self.act = nn.SiLU()

        self.w = OuroCell(self.embed_dim, self.embed_dim * 4)

        self.state_proj = OuroCell(self.embed_dim, self.embed_dim)
        self.ouro_norm = OuroNorm(self.embed_dim)

        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.out_proj = OuroCell(self.embed_dim, self.embed_dim)

    def forward(
        self, x: torch.Tensor, active_c: torch.Tensor, mems: tuple[torch.Tensor, ...],
        lock_mem: bool = False, need_print: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        sa_mems = mems[:6]
        w_mem, sp_mem, op_mem = mems[6], mems[7], mems[8]
        batch_size, seq_len, _ = x.shape

        if need_print:
            with torch.no_grad():
                print('输入x: ', x.norm(), ' 状态范数: ', active_c.norm())

        next_sa_mems, state_attn = self.state_attn(
            x, active_c, sa_mems, lock_mem=lock_mem, logical_seq_len=logical_seq_len,
        )
        x = x + state_attn

        next_w_mem, gates = self.w(x, w_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        i, f, g, o = torch.chunk(gates, 4, dim=-1)

        i = torch.sigmoid(i)
        g: torch.Tensor = self.act(g)
        o = torch.sigmoid(o)

        v = i * g
        f_gate = torch.sigmoid(f)

        c_states = []
        curr_c = active_c

        # 线性时序扫描
        for t in range(seq_len):
            curr_c = f_gate[:, t, :] * curr_c + v[:, t, :]
            c_states.append(curr_c)

        c = torch.stack(c_states, dim=1)
        h: torch.Tensor = o * c

        # 状态更新（lock_mem 时直通）
        if not lock_mem:
            c_last = c[:, -1, :]
            # state_proj 保持自然长度（禁传 logical_seq_len）
            next_sp_mem, c_last = self.state_proj(c_last, sp_mem, lock_mem=lock_mem)
            next_c = self.ouro_norm(c_last)
        else:
            next_sp_mem = sp_mem
            next_c = active_c

        next_op_mem, output = self.out_proj(
            self.out_norm(h), op_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len,
        )

        return (*next_sa_mems, next_w_mem, next_sp_mem, next_op_mem), next_c, output


class OuroMTM(nn.Module):
    """MTM：专属历史记忆 [cap, D, D]（per-batch，不跨 rank 平均，提交无 mean）"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        self.w_q = OuroCell(self.embed_dim, self.embed_dim, bias=False)
        self.w_kvg = OuroCell(self.embed_dim, self.embed_dim * 3, bias=False)

    def read(
        self, active_c: torch.Tensor, mtm_mem: torch.Tensor,
        w_q_mem: torch.Tensor, lock_mem: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = active_c.shape[0]

        next_w_q_mem, q = self.w_q(active_c, w_q_mem, lock_mem=lock_mem)
        q = F.normalize(q, p=2, dim=-1, eps=1e-5).unsqueeze(1) # [B, 1, D]

        # 切片出当前 batch 的专属记忆进行读取
        curr_mem = mtm_mem[:batch_size] # [B, D, D]

        return next_w_q_mem, q @ curr_mem

    def write(
        self, dropped_state: torch.Tensor, mtm_mem: torch.Tensor,
        w_kvg_mem: torch.Tensor, lock_mem: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lock_mem:
            return w_kvg_mem, mtm_mem

        batch_size = dropped_state.shape[0]
        x = dropped_state.unsqueeze(1) # [B, 1, D]

        next_w_kvg_mem, kvg = self.w_kvg(x, w_kvg_mem, lock_mem=lock_mem)
        k, v, g = kvg.chunk(3, dim=-1)

        k = F.normalize(k, p=2, dim=-1, eps=1e-5)
        g = torch.sigmoid(g)

        curr_mem = mtm_mem[:batch_size] # 取出专属记忆 [B, D, D]

        v_retrieved = k @ curr_mem
        delta_v = v - v_retrieved
        v_dyn = g * delta_v

        delta_mem = torch.bmm(k.transpose(-1, -2), v_dyn)

        # 核心：直接保存形状为 [B, D, D] 的矩阵，去掉了 .mean(0)；尾行原样保留
        # clone+index_put_ 提交：原地写入使目标附上计算图，跨步图边得以保留
        next_mtm = mtm_mem.clone()
        next_mtm[:batch_size] = curr_mem + delta_mem
        return next_w_kvg_mem, next_mtm


class OuroLayer(nn.Module):
    def __init__(self, embed_dim: int, max_batch: int, need_mem: bool = False, need_stm: bool = False):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // HEAD_DIM

        self.max_batch = max_batch

        self.need_mem = need_mem
        self.need_stm = need_stm

        self.act = nn.SiLU()
        self.state_attn = OuroStateAttention(self.embed_dim)

        if self.need_stm:
            self.ouro_stm = OuroSTM(self.embed_dim)

        # 开启标准的 Delte Rule 实现
        if self.need_mem:
            self._causal_mask: torch.Tensor
            self.register_buffer('_causal_mask', torch.ones(self.embed_dim, self.embed_dim, dtype=torch.bool).tril_(), persistent=False)

            self.mem_norm = nn.LayerNorm(embed_dim)

            self.w_qkvgd = OuroCell(embed_dim, embed_dim * 5)

            self.out_norm = nn.LayerNorm(embed_dim)
            self.w_o = OuroCell(self.embed_dim, self.embed_dim, bias=False)
            self.o = OuroCell(self.embed_dim, self.embed_dim, bias=False)

            with torch.no_grad():
                torch.nn.init.normal_(
                    self.w_qkvgd.linear.weight[3 * embed_dim : 4 * embed_dim, :],
                    mean=0.0, std=0.02
                )
                torch.nn.init.constant_(
                    self.w_qkvgd.linear.bias[3 * embed_dim : 4 * embed_dim],
                    -6.0
                )

    @property
    def memory_count(self) -> int:
        # need_mem: 自身 layer mem 1 + state_attn 6 + qkvgd/w_o/o 3 = 10
        # need_stm: state_attn 6 + ouro_stm 9 = 15
        return 10 if self.need_mem else 15

    def forward(
        self, x: torch.Tensor, last_state: torch.Tensor | None,
        state: tuple[torch.Tensor, ...], active_c: torch.Tensor | None = None,
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        _, seq_len, _ = x.shape
        # 推理传入静态 logical_seq_len(=patch_size)与训练定长对齐; None 用物理长度
        eff_seq_len = logical_seq_len if logical_seq_len is not None else seq_len

        # state 布局（与 StateLayout DFS 同序）：
        #   need_mem: layer_mem, state_attn(6), qkvgd, w_o, o（OuroLayer 自身 mem 先于子模块）
        #   need_stm: state_attn(6), ouro_stm(9)
        sa_mems = tuple(state[1:7]) if self.need_mem else tuple(state[:6])
        next_sa_mems, sa_out = self.state_attn(
            x, last_state, sa_mems, lock_mem=lock_mem, logical_seq_len=logical_seq_len,
        )
        x = x + sa_out

        if self.need_mem:
            layer_mem = state[0]
            qkvgd_mem, w_o_mem, o_mem = state[7], state[8], state[9]

            mem_context: torch.Tensor = self.mem_norm(x)
            next_qkvgd_mem, qkvgd = self.w_qkvgd(mem_context, qkvgd_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
            context_q, context_k, context_v, context_g, context_d = qkvgd.chunk(5, dim=-1)

            context_q = F.normalize(context_q, p=2, dim=-1, eps=1e-5)
            context_k = F.normalize(context_k, p=2, dim=-1, eps=1e-5)

            mem_g = torch.sigmoid(context_g) * (1.0 / eff_seq_len)

            # 预测的 V
            v_retrieved: torch.Tensor = context_k @ layer_mem

            # 计算真实 V 与预测 V 的 Delta
            delta_v = context_v - v_retrieved

            v_dyn = mem_g * delta_v

            # 外积更新
            delta_mem: torch.Tensor = torch.bmm(context_k.transpose(-1, -2), v_dyn)

            # 记忆更新（lock_mem 时原样直通）
            next_layer_mem: torch.Tensor = layer_mem + delta_mem
            if not lock_mem:
                next_layer_mem = next_layer_mem.mean(0, keepdim=True)
            else:
                next_layer_mem = layer_mem

            # 历史记忆
            mem_out_prev = context_q @ layer_mem

            # QK 的标准注意力打分矩阵
            scores = torch.bmm(context_q, context_k.transpose(-1, -2))

            # 标准注意力
            mask = self._causal_mask[:seq_len, :seq_len]
            scores.masked_fill_(~mask, 0.0)
            mem_out_delta: torch.Tensor = torch.bmm(scores, v_dyn)

            # 合并输出
            mem_out = mem_out_prev
            mem_out = mem_out + mem_out_delta

            # 动态门控
            mem_out = mem_out * self.act(context_d)
            next_w_o_mem, mem_out = self.w_o(self.out_norm(mem_out), w_o_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
            next_o_mem, mem_out = self.o(self.act(mem_out), o_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
            x = x + mem_out

            next_state = (next_layer_mem, *next_sa_mems,
                          next_qkvgd_mem, next_w_o_mem, next_o_mem)
            return next_state, None, x, scores, x

        if self.need_stm:
            next_stm_mems, next_c, stm = self.ouro_stm(
                x, active_c, state[6:], lock_mem=lock_mem, logical_seq_len=logical_seq_len,
            )
            # 两次独立相加：保持两个 add 节点的图拓扑（梯度累积结构对节点形态敏感）
            return (*next_sa_mems, *next_stm_mems), next_c, x + stm, None, x + stm


class OuroBlock(nn.Module):
    def __init__(self, embed_dim: int, max_batch: int, block_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_batch = max_batch
        self.block_layers = block_layers

        self.act = nn.SiLU()

        self.ouro_self_attn_proj = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self.ouro_self_attn_norm = nn.LayerNorm(self.embed_dim)

        self.w_v = OuroCell(embed_dim, embed_dim, bias=False)
        self.v_norm = nn.LayerNorm(embed_dim)

        self.ouro_self_attn_output_proj = OuroCell(embed_dim, embed_dim, bias=False)
        self.ouro_self_attn_gate = OuroCell(embed_dim, embed_dim, bias=False)

        self.ouro_layers: nn.ModuleList[OuroLayer] = nn.ModuleList([
            OuroLayer(self.embed_dim, self.max_batch, (_!=0), _==0) for _ in range(self.block_layers)
        ])

        self.ffn = FFN(self.embed_dim)

        self.norm = nn.LayerNorm(self.embed_dim)

    @property
    def memory_count(self) -> int:
        # w_v / output_proj / gate 3 + 各 layer + ffn 2
        return 3 + sum(layer.memory_count for layer in self.ouro_layers) + 2

    def forward(
        self, x: torch.Tensor, last_state: torch.Tensor | None,
        state: tuple[torch.Tensor, ...], active_c: torch.Tensor,
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        _, seq_len, _ = x.shape
        residual = x

        # state 布局（与 StateLayout DFS 同序）：w_v, output_proj, gate, 各 layer..., ffn(2)
        w_v_mem, outp_mem, gate_mem = state[0], state[1], state[2]
        ffn_mems = state[-2:]
        layer_states = state[3:-2]

        ouro_self_attn = torch.tensor(0.0)

        next_layer_states = []
        next_c = None
        out_list = []
        si = 0
        for layer in self.ouro_layers:
            layer: OuroLayer
            n = layer.memory_count
            sub = tuple(layer_states[si:si + n]); si += n
            nsub, c_l, x, attn, out = layer(
                x, last_state, sub, active_c=active_c,
                lock_mem=lock_mem, logical_seq_len=logical_seq_len,
            )
            next_layer_states.extend(nsub)
            if layer.need_mem:
                ouro_self_attn = ouro_self_attn + attn
            if layer.need_stm:
                next_c = c_l

            out_list.append(out)
            inner_residual = x

        # 涌现注意力 (Emergent Attention)
        scale_factor: torch.Tensor = self.embed_dim**(-0.5)

        ouro_self_attn_residual = ouro_self_attn

        x_proj: torch.Tensor = torch.matmul(x, self.ouro_self_attn_proj) * scale_factor
        ouro_self_attn = torch.bmm(ouro_self_attn_residual, x_proj)

        ouro_self_attn: torch.Tensor = self.act(ouro_self_attn)
        ouro_self_attn_normed: torch.Tensor = self.ouro_self_attn_norm(ouro_self_attn)

        next_w_v_mem, v_states = self.w_v(self.v_norm(residual), w_v_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        attn_bias: torch.Tensor = (ouro_self_attn_residual * scale_factor).masked_fill(~causal_mask, float('-inf'))

        ouro_self_attn_output = F.scaled_dot_product_attention(
            ouro_self_attn_normed.unsqueeze(1),
            ouro_self_attn_normed.unsqueeze(1),
            v_states.unsqueeze(1),
            attn_mask=attn_bias.unsqueeze(1),
            scale=scale_factor
        ).squeeze(1)

        next_gate_mem, gate = self.ouro_self_attn_gate(residual, gate_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        gate = torch.sigmoid(gate)
        next_outp_mem, ouro_self_attn_output = self.ouro_self_attn_output_proj(
            gate * ouro_self_attn_output, outp_mem, lock_mem=lock_mem, logical_seq_len=logical_seq_len,
        )

        x = inner_residual + ouro_self_attn_output

        # 标准输出
        next_ffn_mems, h = self.ffn(self.norm(x), ffn_mems, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        x = x + h
        out_list.append(x)

        next_state = (next_w_v_mem, next_outp_mem, next_gate_mem, *next_layer_states, *next_ffn_mems)
        return next_state, next_c, x, out_list


class Ouro(nn.Module):
    """
    Ouro 标准模型（无状态版：全部记忆状态经 OuroState 显式进出）
    """
    def __init__(self, embed_dim: int, max_batch: int, blocks: int, block_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_batch = max_batch
        self.blocks = blocks

        self.in_norm = nn.LayerNorm(self.embed_dim)
        self.in_attn = Attention(self.embed_dim)
        self.in_ffn_norm = nn.LayerNorm(self.embed_dim)
        self.in_ffn = FFN(self.embed_dim)

        self.temporal_queue_len = 64
        self.temporal_attn = OuroTemporalAttention(self.embed_dim)
        self.c_state_norm = nn.LayerNorm(self.embed_dim)
        self.active_c_norm = nn.LayerNorm(self.embed_dim)
        self.state_ffn = FFN(self.embed_dim)

        self.mtm = OuroMTM(self.embed_dim)

        self.ouro_blocks = nn.ModuleList([
            OuroBlock(self.embed_dim, self.max_batch, block_layers) for _ in range(blocks)
        ])

        self.attnres_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(self.embed_dim)) for _ in range(self.blocks - 1)
        ])
        self.attnres_k_norm = nn.LayerNorm(self.embed_dim)
        self.final_depth_attn = OuroDepthAttention(self.embed_dim)

        self.m_ffn_norm = nn.LayerNorm(self.embed_dim)
        self.m_ffn = FFN(self.embed_dim)

        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.out_attn = Attention(self.embed_dim)
        self.out_ffn_norm = nn.LayerNorm(self.embed_dim)
        self.out_ffn = FFN(self.embed_dim)

        self.stm = OuroSTM(self.embed_dim)

        # 状态布局（构建于全部子模块之后）；激活重算开关由 Gridman 按 config 设置
        self.state_layout = StateLayout(_memory_specs(self))
        self.grad_checkpoint = False

    # ---------------- 状态初始化 ----------------

    def init_state(self, batch_size: int, device) -> OuroState:
        memory = torch.zeros(self.state_layout.total_numel, device=device)
        # 恒等初始化：每个方阵视图对角线填 1
        for view in self.state_layout.views(memory):
            view.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        return OuroState(
            memory=memory,
            recurrent=tuple(
                torch.zeros(batch_size, self.embed_dim, device=device)
                for _ in range(self.blocks + 1)
            ),
            queue=torch.zeros(self.max_batch, self.temporal_queue_len, self.embed_dim, device=device),
            mtm=torch.eye(self.embed_dim, device=device).unsqueeze(0).repeat(self.max_batch, 1, 1),
        )

    def init_inference_state(self, batch_size: int, device) -> OuroState:
        state = self.init_state(batch_size, device)
        return state._replace(
            queue=torch.zeros(batch_size, self.temporal_queue_len, self.embed_dim, device=device),
            mtm=torch.eye(self.embed_dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
        )

    @staticmethod
    def detach_state(state: OuroState) -> OuroState:
        # 仅 detach 不切 dtype：窗口边界处 c_state 保持 bf16 驻留
        return OuroState(
            state.memory.detach(),
            tuple(r.detach() for r in state.recurrent),
            state.queue.detach(),
            state.mtm.detach(),
        )

    # ---------------- 前向 ----------------

    def _forward_packed(self, x, memory, recurrent, queue, mtm, logical_seq_len):
        # checkpoint 边界内固定 lock_mem=False
        return self._forward_impl(x, memory, recurrent, queue, mtm, False, logical_seq_len)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, recurrent: torch.Tensor,
        queue: torch.Tensor, mtm: torch.Tensor,
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and not lock_mem and self.grad_checkpoint:
            # 激活重算：状态张量同时是边界的输入和输出；反向时重放同数值计算
            return _torch_checkpoint(
                self._forward_packed, x, memory, recurrent, queue, mtm, logical_seq_len,
                use_reentrant=False, preserve_rng_state=True,
            )
        return self._forward_impl(x, memory, recurrent, queue, mtm, lock_mem, logical_seq_len)

    def _forward_impl(
        self, x: torch.Tensor, memory: torch.Tensor, recurrent: torch.Tensor,
        queue: torch.Tensor, mtm: torch.Tensor,
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, _ = x.shape

        # ---- memory 解包（顺序 = StateLayout 声明序，即模块树 DFS）----
        views = self.state_layout.views(memory)
        off = 0
        def take(n: int) -> tuple[torch.Tensor, ...]:
            nonlocal off
            s = tuple(views[off:off + n]); off += n; return s

        m_in_attn = take(3); m_in_ffn = take(2)
        m_temporal = take(3); m_state_ffn = take(2)
        m_mtm_q = take(1)[0]; m_mtm_kvg = take(1)[0]
        block_count = self.ouro_blocks[0].memory_count
        m_blocks = [take(block_count) for _ in range(self.blocks)]
        m_depth = take(3); m_m_ffn = take(2)
        m_out_attn = take(3); m_out_ffn = take(2)
        m_stm = take(9)
        assert off == len(views), f'memory 切片数不符: {off} != {len(views)}'

        x0 = x

        # ---- 标准输入 ----
        nm_in_attn, h = self.in_attn(self.in_norm(x), m_in_attn, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        x = x + h
        nm_in_ffn, h = self.in_ffn(self.in_ffn_norm(x), m_in_ffn, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        x = x + h

        # ---- 状态获取 ----
        base_active_c = recurrent[-1].to(torch.bfloat16)
        nm_mtm_q, mid_state = self.mtm.read(base_active_c, mtm, m_mtm_q, lock_mem=lock_mem)

        active_queue = queue[:batch_size]
        temporal_input = torch.cat([mid_state, active_queue], dim=1)
        queue_history = temporal_input[:, :-1, :] # 切片为 64，供给 temporal_attn

        nm_temporal, temporal_context = self.temporal_attn(base_active_c, queue_history, m_temporal, lock_mem=lock_mem)
        active_c = base_active_c + temporal_context
        # state_ffn 为禁传点：保持自然长度，不传 logical_seq_len
        nm_state_ffn, h = self.state_ffn(self.c_state_norm(active_c), m_state_ffn, lock_mem=lock_mem)
        active_c = active_c + h
        active_c_norm = self.active_c_norm(active_c)

        # ---- 计算核心 ----
        history_states = [x0, x]

        nm_blocks = []
        next_block_c = []
        for i, block in enumerate(self.ouro_blocks):
            block: OuroBlock
            if i > 0:
                stacked_history = torch.stack(history_states, dim=2)

                keys = self.attnres_k_norm(stacked_history)
                values = stacked_history
                q = self.attnres_queries[i - 1]

                scores = torch.matmul(keys, q) / (self.embed_dim ** 0.5)
                alpha = F.softmax(scores, dim=-1)
                x = torch.sum(alpha.unsqueeze(-1) * values, dim=2)

            nm_block, c_i, x, out_list = block(
                x, active_c_norm, m_blocks[i], recurrent[i],
                lock_mem=lock_mem, logical_seq_len=logical_seq_len,
            )
            nm_blocks.append(nm_block)
            next_block_c.append(c_i)
            history_states.extend(out_list)
            residual = x

        nm_depth, depth_attn_out = self.final_depth_attn(active_c_norm, history_states, m_depth, lock_mem=lock_mem)

        x = residual + depth_attn_out
        nm_m_ffn, h = self.m_ffn(self.m_ffn_norm(x), m_m_ffn, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        x = x + h

        # ---- 标准输出 ----
        nm_out_attn, h = self.out_attn(self.out_norm(x), m_out_attn, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        x = x + h
        nm_out_ffn, h = self.out_ffn(self.out_ffn_norm(x), m_out_ffn, lock_mem=lock_mem, logical_seq_len=logical_seq_len)
        x = x + h

        nm_stm, next_c_top, out = self.stm(
            x, recurrent[-1], m_stm, lock_mem=lock_mem, logical_seq_len=logical_seq_len,
        )

        # ---- 状态更新 ----
        if not lock_mem:
            dropped_state = queue[:batch_size, 0, :]
            nm_mtm_kvg, next_mtm = self.mtm.write(dropped_state, mtm, m_mtm_kvg, lock_mem=lock_mem)

            # 常规短时队列更新：clone→roll(-1)→index_put_ 末尾写入
            # （roll 的环绕等价于尾行追加自身旧 slot0；原地写入保留跨步图边）
            new_c = next_c_top
            next_queue = torch.roll(queue.clone(), shifts=-1, dims=1)
            next_queue[:batch_size, -1, :] = new_c
        else:
            nm_mtm_kvg = m_mtm_kvg
            next_mtm = mtm
            next_queue = queue

        # 不堆叠：保持对象级跨步图拓扑（存档时按 list 存储）
        next_recurrent = (*next_block_c, next_c_top)

        # 按 StateLayout 声明序重新打包
        next_memory = self.state_layout.flatten([
            *nm_in_attn, *nm_in_ffn, *nm_temporal, *nm_state_ffn,
            nm_mtm_q, nm_mtm_kvg,
            *(m for nm_block in nm_blocks for m in nm_block),
            *nm_depth, *nm_m_ffn, *nm_out_attn, *nm_out_ffn, *nm_stm,
        ])

        return next_memory, next_recurrent, next_queue, next_mtm, out
