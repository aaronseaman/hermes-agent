"""A local OpenAI-compatible ``/v1/chat/completions`` stub for E2E tests of the real auxiliary
resolution chain (no real keys, no network).

``reply(payload) -> (status, content)`` decides each answer: a 200 carries ``content`` as the
assistant message; any other status returns an OpenAI-shaped error body with ``content`` as its
message. Every request is recorded as ``(path, payload, headers)``.
"""

from __future__ import annotations

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator, List, Tuple

Reply = Callable[[dict], Tuple[int, str]]


@contextlib.contextmanager
def fake_chat_server(reply: Reply) -> Iterator[Tuple[str, List[tuple]]]:
    """Yield ``(base_url, requests)``; ``base_url`` ends in ``/v1``."""
    requests: List[tuple] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload, dict(self.headers)))
            status, content = reply(payload)
            if status == 200:
                body = {
                    "id": "chatcmpl-fake", "object": "chat.completion", "created": 1, "model": payload.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }
            else:
                body = {"error": {"message": content, "type": "invalid_request_error", "code": str(status)}}
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
