"""Credential-free baseline: real turns against a local wire, read back through the call ledger.

Runs real ``AIAgent.run_conversation`` turns against a local OpenAI-compatible server that
scripts the tool calls and reports usage (including a growing cached prefix) and publishes
``/models`` pricing, so ``agent.usage_pricing`` prices the run through its normal path. No API
key, no network. The point is not the model's answers — they are fixtures — but that the
ledger and ``hermes insights --ledger`` produce a coherent baseline over a real turn loop.

    python evals/call_ledger/baseline.py            # print the report
    python evals/call_ledger/baseline.py --out /tmp/baseline.json

The scripted session is designed to exercise what the baseline is meant to measure: a parallel
batch, a call repeated inside a turn, the same call repeated in a later turn, a mutating
(non-idempotent) tool, and a failing call.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = "ledger-baseline-local"
# Per-token rates in the OpenAI-compatible ``/models`` shape (≈ $3 / $15 / $0.30 per M tokens).
PRICING = {"prompt": "0.000003", "completion": "0.000015", "cache_read": "0.0000003"}


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _script(workdir: Path) -> list:
    """The scripted assistant responses, in order; ``None`` content means "final answer"."""
    a, b = str(workdir / "alpha.txt"), str(workdir / "beta.txt")
    note = str(workdir / "note.txt")
    return [
        # Turn 1: a parallel batch of two distinct reads, then one of them again, then answer.
        [_tool_call("t1a", "read_file", {"path": a}), _tool_call("t1b", "read_file", {"path": b})],
        [_tool_call("t1c", "read_file", {"path": a})],
        None,
        # Turn 2: a mutating call, a read of a file that does not exist (failure), then answer.
        [_tool_call("t2a", "write_file", {"path": note, "content": "baseline\n"})],
        [_tool_call("t2b", "read_file", {"path": str(workdir / "missing.txt")})],
        None,
        # Turn 3: a read already made in turn 1 — a cross-turn repeat, not a within-turn one.
        [_tool_call("t3a", "read_file", {"path": a})],
        None,
    ]


class _Wire:
    """``/v1/models`` with pricing + ``/v1/chat/completions`` replaying *script* in order.

    Usage grows with the conversation and reports the previous request's prompt as cached, the
    way a provider with a warm per-conversation prefix does.
    """

    def __init__(self, script: list) -> None:
        self.script = deque(script)
        self.requests = 0
        self._last_prompt = 0
        wire = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - parent's name
                pass

            def _json(self, payload: dict, status: int = 200) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                # Exactly the OpenAI-compatible route: /api/v1/models would make the endpoint
                # probe classify this server as LM Studio and take its native metadata path.
                if self.path.rstrip("/") == "/v1/models":
                    self._json({"data": [{"id": MODEL, "context_length": 200_000, "pricing": PRICING}]})
                else:
                    self._json({"error": "not found"}, status=404)

            def _sse(self, chunks: list) -> None:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                messages = body.get("messages") or []
                prompt = 900 + 140 * len(messages)
                cached = min(wire._last_prompt, prompt)
                wire._last_prompt = prompt
                wire.requests += 1
                tool_calls = wire.script.popleft() if wire.script else None
                finish = "tool_calls" if tool_calls else "stop"
                usage = {"prompt_tokens": prompt, "completion_tokens": 60, "total_tokens": prompt + 60,
                         "prompt_tokens_details": {"cached_tokens": cached}}
                if body.get("stream") is True:
                    base = {"id": f"cmpl-{wire.requests}", "object": "chat.completion.chunk",
                            "created": 0, "model": MODEL}
                    delta: dict[str, Any] = {"role": "assistant"}
                    if tool_calls:
                        delta["tool_calls"] = [
                            {"index": i, "id": tc["id"], "type": "function", "function": tc["function"]}
                            for i, tc in enumerate(tool_calls)
                        ]
                    else:
                        delta["content"] = "done."
                    self._sse([
                        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
                    ])
                    return
                message = {"role": "assistant", "content": None if tool_calls else "done."}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                self._json({
                    "id": f"cmpl-{wire.requests}", "object": "chat.completion", "created": 0, "model": MODEL,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage,
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _prepare(root: Path) -> tuple[Path, Path]:
    """A HERMES_HOME with the ledger on, and a working directory with the files to read."""
    home, workdir = root / "home", root / "work"
    home.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "agent:\n  call_ledger:\n    enabled: true\n", encoding="utf-8")
    (workdir / "alpha.txt").write_text("alpha contents\n" * 20, encoding="utf-8")
    (workdir / "beta.txt").write_text("beta contents\n" * 20, encoding="utf-8")
    return home, workdir


def run(root: Path) -> tuple[str, dict]:
    from agent.call_ledger_report import build_report, format_report, load_records
    from agent.call_ledger_store import LEDGER_DIRNAME, WRITER
    from run_agent import AIAgent

    home, workdir = _prepare(root)
    wire = _Wire(_script(workdir))
    agent = AIAgent(
        api_key="not-a-real-key", base_url=wire.base_url, provider="custom", model=MODEL,
        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
        enabled_toolsets=["file"], session_id="ledger-baseline", platform="cli",
    )
    history: list = []
    try:
        for prompt in ("read alpha and beta", "leave a note, then read the missing file", "re-read alpha"):
            history = agent.run_conversation(prompt, conversation_history=history)["messages"]
    finally:
        agent.close()
        wire.close()

    WRITER.flush(timeout=5.0)
    sources = [home / LEDGER_DIRNAME]
    report = build_report(load_records(sources))
    return format_report(report, sources=sources), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the report JSON here")
    parser.add_argument("--keep", action="store_true", help="keep the temporary HERMES_HOME")
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="call-ledger-baseline-"))
    os.environ["HERMES_HOME"] = str(root / "home")
    try:
        text, report = run(root)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    # The run is only a baseline if the ledger saw the turns the loop actually ran.
    agg = report["aggregate"]
    return 0 if (agg["turns"], agg["tool_calls"], agg["model_calls"]) == (3, 6, 8) else 1


if __name__ == "__main__":
    raise SystemExit(main())
