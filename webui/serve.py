"""gridman WebUI 后端: 标准库实现的 SSE 流式对话 server。

路由:
  GET  /           -> 同目录 index.html
  GET  /api/info   -> {"session_id", "model", "device", "patch_size"}
  GET  /api/localimg?path= -> 本地图片直出(仅绝对路径 + 图片扩展名白名单)
  POST /api/reset  -> 清空对话状态(新会话)
  POST /api/chat   -> SSE 流式生成; 帧格式: data: {"delta"|"done"|"error": ...}

--mock 模式不加载模型, 产出内置样例文本, 用于无 checkpoint 时的前端联调。
仅依赖标准库; 对 naxi/ 树零改动(本项目根插入 sys.path 后只读 import)。
"""
import argparse
import getpass
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SESSION_ID = f'{int(time.time())}'
GEN_LOCK = threading.Lock()
IMG_MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml'}

MOCK_REPLY = r"""# Mock 全特性验证

这是一条来自 **mock 模式** 的流式回复,覆盖 markdown 引擎的全部特性:`inline code`、*斜体*、***粗斜体***、~~删除线~~ 与 [链接](https://github.com/anomalyco/opencode)。

## 标题阶梯

#### 四级标题

###### 六级标题

## 列表

- 无序项 **加粗**
  - 嵌套无序
    1. 嵌套有序
    2. 第二项
- 回到一级

3. 起始编号三
4. 编号四

- [x] 已完成任务
- [ ] 待办任务

## 表格

| 特性 | 语法 | 状态 |
| :- | :-: | -: |
| 表格 | `\| a \| b \|` | 支持 |
| 转义竖线 | `a \| b` | **支持** |

## 引用

> 一级引用。
>> 二级引用。

---

数学: 行内 $E=mc^2$、句中 $$\sqrt{x}$$ 与 \(\alpha+\beta\),块级:

$$\int_0^1 x^2\,dx = \frac{1}{3}$$

\[ e^{i\pi} + 1 = 0 \]

价格 $5 和 $10 元不触发公式; snake_case_var 不触发斜体; \*字面星号\*; 裸链接 https://github.com/anomalyco/opencode 自动识别。

```python
def chat(message):
    for piece in model.chat(message):  # 逐片段产出
        yield piece
```

![示例图片](C:\Users\EDY\Desktop\v2-16063333340052312413537be8c4cb32_1440w.jpg)

缺失图片回退为占位说明:

![缺失图片](C:\Users\EDY\Desktop\__no_such__.png)

未写 alt 的图片自动编号:

![](C:\Users\EDY\Desktop\v2-16063333340052312413537be8c4cb32_1440w.jpg)

真实模式下,这段文字将由 gridman 模型生成。"""


class MockChat:
    """与 GridmanChat 同接口的假对话器。"""
    is_sft = True
    patch_size = 64
    device = 'mock'

    def chat(self, user_input, max_len=4096, temperature=0.7):
        for ch in MOCK_REPLY:
            yield ch
            time.sleep(0.015)

    def reset(self):
        pass


def build_real_chat(is_sft: bool):
    """加载真实模型; import 延迟到此处, 保证 --mock 不依赖 torch。"""
    from naxi.v_0d3.gridman.config import RUNNING_CONFIG
    from naxi.v_0d3.gridman.core import Gridman
    from naxi.v_0d3.gridman.chat import GridmanChat
    from naxi.v_0d3.gridman.tools import load_model_checkpoint

    class WebChat(GridmanChat):
        def reset(self):
            self.is_first_turn = True
            self.current_patch = []
            self.cached_logits = None
            self.inference.reset()  # 回热启动快照

    config = RUNNING_CONFIG
    model = Gridman(config).to(config.device)
    model.eval()  # GridmanInference 缓存推理要求 eval 模式
    init_state = load_model_checkpoint(model, is_sft)
    bot = WebChat(model, is_sft, config, init_state=init_state)
    bot.model_name = f'gridman-{"sft" if is_sft else "pretrain"}'
    return bot


def _ckpt_dir(bot) -> str | None:
    return getattr(getattr(bot, 'config', None), 'checkpoint_dir', None)


def _resolve_ckpt_name(is_sft: bool) -> str | None:
    """与 load_model_checkpoint 相同的文件名解析(含 sft -> pretrain 回退)。"""
    from naxi.v_0d3.gridman.config import RUNNING_CONFIG as config
    stage = 'sft' if is_sft else 'pretrain'
    path = os.path.join(config.checkpoint_dir, f'{config.name}_{config.version}_{stage}.pt')
    if not os.path.exists(path) and is_sft:
        path = os.path.join(config.checkpoint_dir, f'{config.name}_{config.version}_pretrain.pt')
    return os.path.basename(path) if os.path.exists(path) else None


class Handler(BaseHTTPRequestHandler):
    server_version = 'gridman-webui/1.0'
    bot = None          # 启动时注入
    ckpt_name = None    # 当前加载的 checkpoint 文件名
    default_max_len = 4096
    default_temperature = 0.7

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return None

    def _send_image(self):
        """受控本地图片直出: 绝对路径 + 扩展名白名单 + 文件存在, 否则 404(前端 onerror 回退)。"""
        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        img_path = qs.get('path', [''])[0]
        ext = os.path.splitext(img_path)[1].lower()
        if not img_path or not os.path.isabs(img_path) or ext not in IMG_MIME \
                or not os.path.isfile(img_path):
            self._send_json(404, {'error': f'图片不存在或类型不允许: {img_path}'})
            return
        try:
            f = open(img_path, 'rb')
        except OSError:
            self._send_json(404, {'error': f'图片读取失败: {img_path}'})
            return
        try:
            self.send_response(200)
            self.send_header('Content-Type', IMG_MIME[ext])
            self.send_header('Content-Length', str(os.fstat(f.fileno()).st_size))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            while True:
                chunk = f.read(1 << 16)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except OSError:  # 客户端中断
                    break
        finally:
            f.close()

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
            try:
                body = (Path(__file__).with_name('index.html')).read_bytes()
            except OSError:
                self._send_json(404, {'error': 'index.html missing'})
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == '/api/info':
            self._send_json(200, {
                'session_id': SESSION_ID,
                'model': getattr(self.bot, 'model_name', 'gridman'),
                'device': str(getattr(self.bot, 'device', '-')),
                'patch_size': getattr(self.bot, 'patch_size', None),
                'user': getpass.getuser(),
                'cwd': os.getcwd(),
                'checkpoint': type(self).ckpt_name,
                'checkpoint_dir': _ckpt_dir(self.bot),
            })
        elif path == '/api/checkpoints':
            self._send_json(200, {'checkpoints': self._list_checkpoints(),
                                  'current': type(self).ckpt_name})
        elif path == '/api/localimg':
            self._send_image()
        else:
            self._send_json(404, {'error': 'not found'})

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        payload = self._read_body()
        if payload is None:
            self._send_json(400, {'error': 'invalid json body'})
            return
        if path == '/api/reset':
            with GEN_LOCK:
                self.bot.reset()
            self._send_json(200, {'ok': True, 'session_id': SESSION_ID})
        elif path == '/api/chat':
            self._stream_chat(payload)
        elif path == '/api/checkpoint':
            self._switch_checkpoint(payload)
        else:
            self._send_json(404, {'error': 'not found'})

    def _list_checkpoints(self):
        if isinstance(self.bot, MockChat):
            names = ['mock_pretrain.pt', 'mock_sft.pt']
            return [{'name': n, 'size_mb': 0, 'current': n == type(self).ckpt_name} for n in names]
        ckpt_dir = _ckpt_dir(self.bot)
        items = []
        if ckpt_dir and os.path.isdir(ckpt_dir):
            for fn in sorted(os.listdir(ckpt_dir)):
                if not fn.endswith('.pt'):
                    continue
                st = os.stat(os.path.join(ckpt_dir, fn))
                items.append({'name': fn, 'size_mb': round(st.st_size / 1e6),
                              'current': fn == type(self).ckpt_name})
        return items

    def _switch_checkpoint(self, payload):
        """热切换 checkpoint: 全量重建对话器(权重 + 热启动状态), 旧实例显式释放显存。"""
        name = str(payload.get('name') or '')
        if not name or name != os.path.basename(name) or not name.endswith('.pt'):
            self._send_json(400, {'error': 'invalid checkpoint name'})
            return
        if isinstance(self.bot, MockChat):
            type(self).ckpt_name = name
            self._send_json(200, {'ok': True, 'name': name, 'model': self.bot.model_name})
            return
        ckpt_dir = _ckpt_dir(self.bot)
        if not ckpt_dir or not os.path.isfile(os.path.join(ckpt_dir, name)):
            self._send_json(404, {'error': f'checkpoint 不存在: {name}'})
            return
        if name == type(self).ckpt_name:
            self._send_json(200, {'ok': True, 'name': name, 'unchanged': True})
            return
        if not GEN_LOCK.acquire(blocking=False):
            self._send_json(409, {'error': 'busy'})
            return
        try:
            new_bot = build_real_chat(is_sft=name.endswith('_sft.pt'))
            old_bot = self.bot
            type(self).bot = new_bot
            type(self).ckpt_name = name
            del old_bot
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            self._send_json(200, {'ok': True, 'name': name, 'model': new_bot.model_name})
        except Exception as exc:
            self._send_json(500, {'error': f'{type(exc).__name__}: {exc}'})
        finally:
            GEN_LOCK.release()

    def _stream_chat(self, payload):
        if not GEN_LOCK.acquire(blocking=False):
            self._send_json(409, {'error': 'busy'})
            return
        try:
            message = str(payload.get('message') or '').strip()
            if not message:
                self._send_json(400, {'error': 'empty message'})
                return
            max_len = max(1, min(int(payload.get('max_len') or self.default_max_len), 8192))
            temperature = max(0.0, min(float(payload.get('temperature', self.default_temperature)), 2.0))

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True  # 关闭连接即流结束, 规避 chunked 编码

            def emit(obj):
                self.wfile.write(b'data: ' + json.dumps(obj, ensure_ascii=False).encode('utf-8') + b'\n\n')
                self.wfile.flush()

            try:
                for piece in self.bot.chat(message, max_len=max_len, temperature=temperature):
                    emit({'delta': piece})
                emit({'done': True})
            except (BrokenPipeError, ConnectionResetError):
                pass  # 前端停止/断开: 生成器随本次迭代退出
            except Exception as exc:  # 模型侧错误如实回传
                try:
                    emit({'error': f'{type(exc).__name__}: {exc}'})
                except OSError:
                    pass
        finally:
            GEN_LOCK.release()


def main():
    parser = argparse.ArgumentParser(description='gridman WebUI SSE server')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8901)
    parser.add_argument('--max-len', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--pretrain', action='store_true', help='使用预训练模型(默认 SFT)')
    parser.add_argument('--mock', action='store_true', help='不加载模型, 产出内置样例')
    args = parser.parse_args()

    if args.mock:
        bot = MockChat()
        bot.model_name = 'mock'
        ckpt_name = 'mock_pretrain.pt'
        print('[webui] mock 模式: 不加载模型')
    else:
        bot = build_real_chat(is_sft=not args.pretrain)
        ckpt_name = _resolve_ckpt_name(is_sft=not args.pretrain)

    Handler.bot = bot
    Handler.ckpt_name = ckpt_name
    Handler.default_max_len = args.max_len
    Handler.default_temperature = args.temperature

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f'[webui] 监听 http://{args.host}:{args.port} (session {SESSION_ID})')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
