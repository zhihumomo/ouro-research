"""会话持有的 CUDA Graph 捕获/重放（torch 原生，无编译器改写，无隐藏状态提交）。"""

import torch


class StaticCudaGraph:
    """捕获一个张量参数地址固定的 callable。

    可变缓存输入在 warmup/捕获后恢复原值。返回输出为图所有；
    需要跨 replay 存活的公共结果由调用方 clone。
    """

    def __init__(self, function, args, mutable=()):
        self.args = args
        self.device = args[0].device
        with torch.cuda.device(self.device):
            self._capture(function, args, mutable)

    def _capture(self, function, args, mutable):
        self.graph = torch.cuda.CUDAGraph()
        saved = tuple(t.clone() for t in mutable)
        stream = torch.cuda.Stream(device=args[0].device)
        stream.wait_stream(torch.cuda.current_stream(args[0].device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                function(*args)
        torch.cuda.current_stream(args[0].device).wait_stream(stream)
        for tensor, original in zip(mutable, saved):
            tensor.copy_(original)
        with torch.cuda.graph(self.graph):
            self.output = function(*args)
        for tensor, original in zip(mutable, saved):
            tensor.copy_(original)

    def replay(self):
        with torch.cuda.device(self.device):
            self.graph.replay()
        return self.output
