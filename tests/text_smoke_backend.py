"""Deterministic OpenAI-compatible backend for the public CI smokes.

It answers only for the one expected model, so a smoke also proves which model
chord asked for: `example-local-model` by default (the container smoke), or the
model named as the first argument (the README quickstart walk uses llama3.1).
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MODEL = "example-local-model"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        if body.get("model") != MODEL:
            self.send_error(400, "unexpected model")
            return
        reply = {
            "id": "chatcmpl-local-smoke", "object": "chat.completion", "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "text-only smoke passed"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


if __name__ == "__main__":
    MODEL = sys.argv[1] if len(sys.argv) > 1 else MODEL
    ThreadingHTTPServer(("127.0.0.1", 11434), Handler).serve_forever()
