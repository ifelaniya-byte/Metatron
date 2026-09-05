#!/usr/bin/env python3
"""Tiny Ollama-compatible HTTP bridge for the persisted Metatron champion.

The custom NumPy geometry is not a native Ollama/GGUF architecture, so this
exposes the real champion through Ollama's HTTP API shape
(``/api/tags``, ``/api/generate``, ``/api/chat``) instead of claiming a GGUF
conversion that does not exist.

Run on the worker:

    python daycare/ollama_bridge.py --checkpoint daycare_state/champion/model.pkl --port 11435
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Handler(BaseHTTPRequestHandler):
    model = None

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._send(200, {"models": [{
                "name": "metatron:champion",
                "model": "metatron:champion",
                "size": 0,
                "digest": "local-champion",
            }]})
        if self.path == "/api/version":
            return self._send(200, {"version": "0.1.0-metatron"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        prompt = body.get("prompt", "")
        if body.get("messages"):
            prompt = "\n".join(str(x.get("content", ""))
                               for x in body["messages"]
                               if x.get("role") != "system")
        opts = body.get("options") or {}
        try:
            text = self.model.generate(
                prompt or "the flower of life",
                max_new=int(opts.get("num_predict", 96)),
                temperature=float(opts.get("temperature", 0.7)),
            )
        except Exception as e:
            return self._send(500, {"error": f"generation failed: {e}"})
        if self.path in ("/api/generate", "/api/chat"):
            out = {"model": "metatron:champion", "created_at": "",
                   "done": True, "response": text}
            if self.path.endswith("/chat"):
                out["message"] = {"role": "assistant", "content": text}
            return self._send(200, out)
        return self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="daycare_state/champion/model.pkl")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=11435)
    a = ap.parse_args()

    p = Path(a.checkpoint)
    if not p.is_file():
        raise SystemExit("champion checkpoint missing: " + str(p))

    # Importing the vendored package makes pickle deserialization work even
    # though the checkpoint was written by the daycare adapters.
    try:
        import metatron  # noqa: F401
    except ImportError:
        pass

    with open(p, "rb") as f:
        Handler.model = pickle.load(f)
    print(f"Serving metatron:champion from {p} on {a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
