"""checkpoint 底层存储：原子写（tmp → fsync → os.replace → 目录 fsync）。"""
import os
from typing import Any

import torch


def atomic_save(value: dict[str, Any], path: str):
    """tmp 写入 → flush+fsync → os.replace → 目录 fsync（崩溃安全）"""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    _fsync_directory(directory)


def _fsync_directory(path: str):
    """目录 fsync：Windows 下目录不可 fsync, 尽力而为即可"""
    try:
        if os.name == 'nt':
            return
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def load_payload(path: str) -> dict:
    return torch.load(path, map_location='cpu', weights_only=True)
