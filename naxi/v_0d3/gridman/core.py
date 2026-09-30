import torch
import torch.nn as nn

from naxi.v_0d3.ouro.core import Ouro, OuroState
from naxi.v_0d3.gridman.config import Config


class Gridman(nn.Module):
    """
    基于 Ouro 架构的自回归语言模型 Gridman（无状态版：状态经 OuroState 显式进出）
    """
    def __init__(self, config=Config()):
        super().__init__()

        self.config = config
        self.embed_dim = config.embed_dim

        # 嵌入层
        self.byte_emb = nn.Embedding(config.tokenizer.vocab_size, config.embed_dim)

        # Ouro
        self.core_ouro = Ouro(self.embed_dim, config.chunk_size, config.blocks, config.block_layers)
        self.core_ouro.grad_checkpoint = config.grad_checkpoint

        # 输出头
        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, config.tokenizer.vocab_size)

        torch.nn.init.normal_(self.byte_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.out_proj.weight, mean=0.1, std=0.02)
        torch.nn.init.zeros_(self.out_proj.bias)

    # 状态助手透传
    def init_state(self, batch_size: int, device) -> OuroState:
        return self.core_ouro.init_state(batch_size, device)

    def init_inference_state(self, batch_size: int, device) -> OuroState:
        return self.core_ouro.init_inference_state(batch_size, device)

    @staticmethod
    def detach_state(state: OuroState) -> OuroState:
        return Ouro.detach_state(state)

    def forward(
        self, x: torch.Tensor, state: OuroState,
        lock_mem: bool = False, logical_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, OuroState]:
        # DDP/compile 途经可能将 NamedTuple 展平为普通 tuple，防御性复原
        if not isinstance(state, OuroState):
            state = OuroState(*state)

        x = self.byte_emb(x)

        next_memory, next_recurrent, next_queue, next_mtm, hidden = self.core_ouro(
            x, state.memory, state.recurrent, state.queue, state.mtm,
            lock_mem, logical_seq_len=logical_seq_len,
        )
        ar_logits = self.out_proj(self.out_norm(hidden))
        return ar_logits, OuroState(next_memory, next_recurrent, next_queue, next_mtm)
