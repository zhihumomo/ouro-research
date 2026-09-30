import codecs
from collections.abc import Iterator

import torch

from naxi.v_0d3.ouro.core import OuroState
from naxi.v_0d3.gridman.config import RUNNING_CONFIG, Config
from naxi.v_0d3.gridman.core import Gridman
from naxi.v_0d3.gridman.inference import GridmanInference
from naxi.v_0d3.gridman.tools import load_model_checkpoint


class GridmanChat:
    def __init__(self, model: Gridman, is_sft: bool = True, config: Config = RUNNING_CONFIG,
                 init_state: OuroState | None = None):
        self.model = model
        self.is_sft = is_sft
        self.config = config
        self.tokenizer = config.tokenizer
        self.patch_size = config.patch_size
        self.device = config.device
        self.device_type = config.device_type

        self.is_first_turn = True
        self.total_gen_tokens_num = 0
        self.current_patch = []
        # 用于存储最近一次记忆固化（lock_mem=False）产生的最后一个 logit
        self.cached_logits = None
        model.eval()  # 缓存推理要求 eval 模式(GridmanInference 构造期硬校验)
        # 推理会话：增量解码 + 每 patch 一次状态条件化 + bulk prefill（CUDA 下默认
        # 叠加 bf16 权重驻留与 CUDA Graph 重放）。会话持有热启动状态快照——
        # delta-rule 记忆依赖训练稳态，空白起步为分布外（实测 teacher-forced loss 1.96→1.04）
        self.inference = GridmanInference(
            model,
            self.device,
            init_state=init_state,
            cuda_graph=config.inference_cuda_graph,
            bf16_weights=config.inference_bf16_weights,
            prefill_min=config.inference_prefill_min,
        )
        self.inference.warmup()

    def _commit_current_patch(self):
        """固化完整 patch 进记忆 (commit 仅允许完整 patch_size token)。

        恒整 patch_size 固化 = 与训练前向对齐: 边界为任意字节, 不做 UTF-8 对齐;
        logical_seq_len 静态取 patch_size, 归一化与训练定长一致。
        """
        if len(self.current_patch) != self.patch_size:
            raise ValueError(f'commit 需要完整 {self.patch_size} token, 实际 {len(self.current_patch)}')
        self.cached_logits = self.inference.forward_prefix(self.current_patch, False)
        self.current_patch = []

    def _forward_step(self) -> torch.Tensor:
        """部分 patch 步进前向 (不固化记忆); 增量解码仅计算新增位置。"""
        if not 0 < len(self.current_patch) < self.patch_size:
            raise ValueError(f'步进前向需要 1..{self.patch_size - 1} token, 实际 {len(self.current_patch)}')
        return self.inference.forward_prefix(self.current_patch, True)

    def _next_logits(self) -> torch.Tensor | None:
        # 上一步采样可能恰好填满 patch: 仅在确实需要模型结果时才固化一次
        if len(self.current_patch) == self.patch_size:
            self._commit_current_patch()
        if self.cached_logits is not None:
            logits, self.cached_logits = self.cached_logits, None
            return logits
        if self.current_patch:
            return self._forward_step()
        return None

    def chat(self, user_input: str | None, max_len: int = 512, temperature: float = 0.7) -> Iterator[str]:
        """流式对话: 逐字节自回归生成, 经增量 UTF-8 解码器解码, 可解码即 yield。

        显示流(仅 <256 字节)与模型历史(全部 token)分离; 多字节字符跨 patch /
        跨 yield 由增量解码器安全拼接; 非法字节产出 U+FFFD, 流结束时残缺尾部
        显式解码(final=True), 不静默丢弃。生成器惰性求值: 产出字符前不做多余前向。
        """
        self.model.eval()

        # 处理 Prompt
        if user_input is not None:
            # 构造 prefix
            if self.is_sft:
                prefix = [self.tokenizer.eos_token_id, self.tokenizer.user_token_id] if self.is_first_turn else [self.tokenizer.user_token_id]
                input_ids = prefix + self.tokenizer.encode(user_input) + [self.tokenizer.eos_token_id, self.tokenizer.assistant_token_id]
                self.is_first_turn = False
            else:
                input_ids = ([self.tokenizer.eos_token_id] if self.is_first_turn else []) + self.tokenizer.encode(user_input)
                self.is_first_turn = False

            # 消费输入: 上下文一变即作废缓存 logits; 填满即固化
            for token in input_ids:
                self.current_patch.append(token)
                self.cached_logits = None
                if len(self.current_patch) == self.patch_size:
                    self._commit_current_patch()

            self.total_gen_tokens_num = 0

        # 增量 UTF-8 解码器: 每次回答新建, 缓存跨边界的残缺多字节序列
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        # 自回归生成
        for _ in range(max_len):
            current_logits = self._next_logits()
            if current_logits is None:
                # 理论上不应到达这里
                break

            # 采样
            if temperature <= 0.0:
                next_token = torch.argmax(current_logits, dim=-1).item()
            else:
                probs = torch.nn.functional.softmax(current_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            self.current_patch.append(next_token)
            self.total_gen_tokens_num += 1

            # 普通字节进增量解码器, 可解码即产出 (先产出后计算)
            if next_token < 256:
                piece = decoder.decode(bytes([next_token]))
                if piece:
                    yield piece

            # EOS / 达到最大长度: 终态 token 留在模型历史中, 不固化不完整 patch;
            # 冲刷解码器残缺尾部(若有则显式变为 U+FFFD)后结束
            if next_token == self.tokenizer.eos_token_id or self.total_gen_tokens_num >= max_len:
                tail = decoder.decode(b'', final=True)
                if tail:
                    yield tail
                self.total_gen_tokens_num = 0
                return


def gridman_chat(is_sft: bool = True):
    config = RUNNING_CONFIG
    device = config.device

    # 加载模型（推理专用入口: 载权重, 不恢复训练现场/RNG; 返回完整热启动初始状态）
    grid_man = Gridman(config).to(device)
    init_state = load_model_checkpoint(grid_man, is_sft)

    # 实例化对话系统
    chat_bot = GridmanChat(grid_man, is_sft, config, init_state=init_state)

    print('\n开启对话 (输入 "quit" 或 "exit" 退出)')
    while True:
        user_input = input('User: ')
        if user_input.strip().lower() in ['exit', 'quit']:
            break

        for piece in chat_bot.chat(user_input, max_len=4096, temperature=0.7):
            print(piece, end='', flush=True)
        print() # 结束换行
