"""checkpoint：训练存档/恢复/推理加载。

存档为单文件 payload：纯权重 weights + global_memory
+ per-rank runtime_states(recurrent/queue/mtm/RNG×4/dataloader) + 结构配置快照。
"""
import os
import platform
import random
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist

from naxi.v_0d3.ouro.core import OuroState
from naxi.v_0d3.gridman.config import Config, RUNNING_CONFIG
from naxi.v_0d3.gridman.checkpoint_storage import atomic_save, load_payload

try:
    import numpy as np
except ImportError:
    np = None


# 结构校验字段：不一致则加载必然错误, 直接拒绝
_STRUCT_FIELDS = ('embed_dim', 'blocks', 'block_layers', 'patch_size', 'chunk_size', 'bptt_size')


def print_model_parameters(model: nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    core = model.core_ouro
    embed_dim = core.embed_dim
    max_batch = core.max_batch
    # 无状态架构: 记忆状态不进 state_dict, 按布局理论值统计
    mem_size = core.state_layout.total_numel
    queue_size = max_batch * core.temporal_queue_len * embed_dim
    mtm_size = max_batch * embed_dim * embed_dim
    recurrent_size = (core.blocks + 1) * max_batch * embed_dim
    state_total = mem_size + queue_size + mtm_size + recurrent_size

    total_tracked = trainable_params + state_total

    # 打印汇总信息
    print("\n" + "="*60)
    print(f"Gridman 🤖 模型体积统计:")
    print(f" ├─ 总规模: {total_tracked / 1e6:.2f} M")
    print(f" ├─ 可训练参数: {trainable_params / 1e6:.2f} M")
    print(f" └─ 记忆状态量: {state_total / 1e6:.2f} M")
    print("="*60)


def _checkpoint_path(config: Config, stage: str) -> str:
    return os.path.join(config.checkpoint_dir, f'{config.name}_{config.version}_{stage}.pt')


def _core(model: nn.Module):
    return model.core_ouro if hasattr(model, 'core_ouro') else model


def _state_to(state: OuroState, device) -> OuroState:
    return OuroState(
        state.memory.to(device),
        tuple(r.to(device) for r in state.recurrent),
        state.queue.to(device),
        state.mtm.to(device),
    )


def _dump_numpy_rng():
    """numpy RNG 的 weights_only 兼容编码（uint32 视图转 int64 张量）"""
    if np is None:
        return None
    st = np.random.get_state()
    return (st[0], torch.from_numpy(st[1].astype(np.int64)), int(st[2]), int(st[3]), float(st[4]))


def _restore_numpy_rng(saved):
    if np is None or saved is None:
        return
    name, arr, pos, has_gauss, cached = saved
    np.random.set_state((name, arr.to(torch.uint32).numpy(), pos, has_gauss, cached))


def _training_metadata(config: Config, use_dist: bool) -> dict:
    return {
        'torch': str(torch.__version__),  # str() 退化 TorchVersion 子类, 兼容 weights_only
        'python': platform.python_version(),
        'world_size': dist.get_world_size() if use_dist else 1,
        'parameter_dtype': 'float32',
        'autocast_dtype': str(config.dtype),
    }


def _check_metadata(saved: dict | None, config: Config, use_dist: bool):
    """训练元数据不一致时默认仅警告（模型处于活跃迭代期, 不设硬门禁）"""
    if not saved:
        return
    current = _training_metadata(config, use_dist)
    diffs = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
    if diffs:
        warnings.warn(f'checkpoint 训练元数据与当前环境不一致（默认容忍）: {diffs}')


def save_checkpoint(
        model: nn.Module,
        is_sft: bool = False,
        config: Config = RUNNING_CONFIG,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        step: int = 0,
        dataloader: Any = None,
        state: OuroState | None = None,
):
    """保存完整训练状态: 纯权重 + 优化器 + 调度器 + 步数 + 全局 memory
    + 各 rank 的 recurrent/queue/mtm/RNG×4/数据流现场。

    所有 rank 都必须调用(内部要收集各 rank 的运行时状态), 仅 rank0 写盘。
    权重剥离 torch.compile 的 _orig_mod. 前缀; tmp+fsync+os.replace 原子写入。
    """
    if state is None:
        raise ValueError('无状态架构下 save_checkpoint 必须传入当前 OuroState')

    use_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_dist else 0
    stage = 'sft' if is_sft else 'pretrain'
    core = _core(model)

    # 无状态架构的 state_dict 天然只含权重
    weights = {k.removeprefix('_orig_mod.'): v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 本 rank 运行时状态
    runtime_state = {
        'rank': rank,
        'recurrent': [r.detach().cpu().clone() for r in state.recurrent],
        'queue': state.queue.detach().cpu().clone(),
        'mtm': state.mtm.detach().cpu().clone(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'python_rng_state': random.getstate(),
        'numpy_rng_state': _dump_numpy_rng(),
        'dataloader': dataloader.state_dict() if dataloader is not None else None,
    }
    if use_dist:
        runtime_states = [None] * dist.get_world_size()
        dist.all_gather_object(runtime_states, runtime_state)
    else:
        runtime_states = [runtime_state]

    checkpoint = {
        'producer': 'naxi.v_0d3.gridman',
        'step': step,
        'config': {k: getattr(config, k) for k in _STRUCT_FIELDS},
        'training_metadata': _training_metadata(config, use_dist),
        'weights': weights,
        # 全局记忆: 各 rank 每 patch 已均值同步, 存本 rank(rank0 写盘)值即可
        'global_memory': state.memory.detach().cpu().clone(),
        'runtime_states': runtime_states,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if rank == 0:
        atomic_save(checkpoint, _checkpoint_path(config, stage))


def load_checkpoint(
        model: nn.Module,
        is_sft: bool = False,
        need_print: bool = True,
        config: Config = RUNNING_CONFIG,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        dataloader: Any = None,
        device=None,
) -> tuple[int, OuroState]:
    """断点续训统一入口, 返回 (已完成 step 数, OuroState)。

    - 本阶段检查点存在: 完整恢复 模型/优化器/调度器/本 rank 的 RNG×4、数据流与记忆状态
    - is_sft=True 且无 SFT 检查点: 回退加载 pretrain 权重, 返回 (0, 初始状态)
    - 无检查点: 返回 (0, 初始状态), 从头训练

    model 请传入未编译的原始模块(compile/DDP 只是包装, 共享同一批参数)。
    """
    use_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_dist else 0
    stage = 'sft' if is_sft else 'pretrain'
    device = device if device is not None else config.device
    core = _core(model)
    checkpoint_path = _checkpoint_path(config, stage)

    resume = os.path.exists(checkpoint_path)

    if not resume:
        if is_sft:
            # SFT 首次启动: 回退到 pretrain 权重做初始化
            pretrain_path = _checkpoint_path(config, 'pretrain')
            if not os.path.exists(pretrain_path):
                raise FileNotFoundError(f'⚠️ 未找到检查点: {checkpoint_path}, 也未找到预训练权重: {pretrain_path}')
            checkpoint_path = pretrain_path
        else:
            if need_print:
                print(f'⚠️ 未找到检查点, 从头开始训练: {checkpoint_path}')
            return 0, core.init_state(config.chunk_size, device)

    payload = load_payload(checkpoint_path)
    if 'weights' not in payload or 'runtime_states' not in payload:
        raise ValueError(f'存档内容无法识别: {checkpoint_path}')

    # 结构与布局硬校验
    saved_cfg = payload.get('config', {})
    diffs = {k: (saved_cfg.get(k), getattr(config, k)) for k in _STRUCT_FIELDS
             if saved_cfg.get(k) != getattr(config, k)}
    if diffs:
        raise ValueError(f'checkpoint 结构配置不匹配（存档 vs 当前）: {diffs}')
    _check_metadata(payload.get('training_metadata'), config, use_dist)

    model.load_state_dict(payload['weights'], strict=True)

    if not resume:
        # SFT 权重初始化: 只取模型权重, 不恢复任何训练状态
        if need_print:
            print(f'✅ 已从 {checkpoint_path} 加载预训练权重, SFT 将从头开始训练')
        return 0, core.init_state(config.chunk_size, device)

    if optimizer is not None:
        if 'optimizer_state_dict' not in payload:
            raise RuntimeError('checkpoint 缺 optimizer_state_dict, 无法恢复优化器')
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    if scheduler is not None:
        if 'scheduler_state_dict' not in payload:
            raise RuntimeError('checkpoint 缺 scheduler_state_dict, 无法恢复调度器')
        scheduler.load_state_dict(payload['scheduler_state_dict'])

    # 恢复本 rank 的 RNG×4、数据流与记忆状态
    runtime_states = payload['runtime_states']
    if rank >= len(runtime_states):
        raise RuntimeError(f'checkpoint 仅含 {len(runtime_states)} 个 rank 的运行时状态, 当前 rank={rank}')
    rs = runtime_states[rank]

    if rs.get('rng_state') is not None:
        torch.set_rng_state(rs['rng_state'])
    if rs.get('cuda_rng_state_all') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rs['cuda_rng_state_all'])
    if rs.get('python_rng_state') is not None:
        random.setstate(rs['python_rng_state'])
    _restore_numpy_rng(rs.get('numpy_rng_state'))
    if dataloader is not None and rs.get('dataloader') is not None:
        dataloader.load_state_dict(rs['dataloader'])

    state = OuroState(payload['global_memory'], tuple(rs['recurrent']), rs['queue'], rs['mtm'])

    start_step = payload.get('step', 0)
    if need_print:
        print(f'🔄 已从 {checkpoint_path} 加载, 当前 step: {start_step}')
    return start_step, _state_to(state, device)


def load_model_checkpoint(model: nn.Module, is_sft: bool = False, need_print: bool = True, config: Config = RUNNING_CONFIG) -> OuroState | None:
    """推理专用入口：载权重, 不恢复训练现场或 RNG。

    返回推理初始状态（完整热启动: 训练稳态 global_memory + rank0 第 0 条流的
    recurrent/queue/mtm 切片至 batch=1）——delta-rule 记忆依赖训练稳态, 空白起步
    为分布外（实测 teacher-forced loss 1.96→1.04）。存档不含状态时返回 None,
    会话空白起步。"""
    stage = 'sft' if is_sft else 'pretrain'
    checkpoint_path = _checkpoint_path(config, stage)
    if not os.path.exists(checkpoint_path):
        if is_sft:
            checkpoint_path = _checkpoint_path(config, 'pretrain')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'⚠️ 未找到可用检查点: {checkpoint_path}')

    payload = load_payload(checkpoint_path)
    if 'weights' not in payload:
        raise ValueError(f'存档内容无法识别: {checkpoint_path}')

    model.load_state_dict(payload['weights'], strict=True)

    if need_print:
        print(f'✅ 已从 {checkpoint_path} 加载模型权重')

    memory = payload.get('global_memory')
    runtime_states = payload.get('runtime_states')
    if memory is None or not runtime_states:
        return None
    rs = runtime_states[0]
    state = OuroState(
        memory,
        tuple(r[0:1] for r in rs['recurrent']),
        rs['queue'][0:1],
        rs['mtm'][0:1],
    )
    return _state_to(state, config.device)
