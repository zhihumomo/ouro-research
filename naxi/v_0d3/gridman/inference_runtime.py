"""推理视图：共享只读参数，matmul 类权重预转 bf16 副本。

复制模块容器、共享只读参数与静态缓冲。BF16 副本与 autocast 在 matmul 处的
舍入结果一致，且由 decode/commit 两个视图共享（weight_memo 去重）。
Embedding/LayerNorm/持久状态保持原 dtype 共享，不转换。

视图永不训练；dtype/device/结构变更后需重建；会话侧换权重后必须经
refresh_weights 显式刷新副本。
"""

import copy

import torch
import torch.nn as nn


def _inference_view(model: nn.Module, weight_dtype, weight_memo) -> nn.Module:
    memo = {}
    weight_memo = {} if weight_memo is None else weight_memo

    def visit(module, path=""):
        if id(module) in memo:
            return memo[id(module)]
        result = copy.copy(module)
        memo[id(module)] = result
        result._modules = {
            name: visit(child, f"{path}.{name}") if child is not None else None
            for name, child in module._modules.items()
        }
        result._parameters = module._parameters.copy()
        result._buffers = module._buffers.copy()
        result.training = False
        if weight_dtype is not None:
            # 仅转换被 autocast matmul/linear 消费的参数：
            # nn.Linear 权重/偏置、LoRA 参数、涌现注意力投影与 attnres 查询
            keys = set()
            if isinstance(module, nn.Linear):
                keys.update(("weight", "bias"))
            keys.update(("lora_A", "lora_B", "ouro_self_attn_proj"))
            if isinstance(module, nn.ParameterList) and path.endswith(".attnres_queries"):
                keys.update(module._parameters)
            for name, parameter in module._parameters.items():
                if name not in keys or parameter is None or parameter.dtype != torch.float32:
                    continue
                if id(parameter) not in weight_memo:
                    weight_memo[id(parameter)] = nn.Parameter(
                        parameter.to(weight_dtype), requires_grad=parameter.requires_grad
                    )
                result._parameters[name] = weight_memo[id(parameter)]
        return result

    return visit(model)


def decode_view(model, weight_dtype=None, weight_memo=None):
    return _inference_view(model, weight_dtype, weight_memo)


def commit_view(model, weight_dtype=None, weight_memo=None):
    return _inference_view(model, weight_dtype, weight_memo)
