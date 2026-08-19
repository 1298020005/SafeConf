#!/usr/bin/env python3
"""Local HTTP copy bridge. Windows clip_sync.ps1 polls this via SSH port-forward."""
from __future__ import annotations

import hashlib
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GROK = Path(__file__).resolve().parent.parent
COPY_FILE = GROK / "COPY.txt"
HOST = "127.0.0.1"
PORT = int(os.environ.get("COPY_SERVER_PORT", "18765"))


def read_text() -> str:
    if COPY_FILE.exists():
        return COPY_FILE.read_text(encoding="utf-8-sig")
    return ""


def write_text(body: bytes) -> None:
    text = body.decode("utf-8")
    COPY_FILE.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    global _digest
    _digest = hashlib.sha256(body).hexdigest()


_digest = hashlib.sha256(read_text().encode("utf-8")).hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def do_GET(self):
        if self.path in ("/copy", "/copy/"):
            text = read_text()
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Copy-Hash", _digest)
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path in ("/health", "/health/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path not in ("/copy", "/copy/"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        write_text(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"copy_server listening on {HOST}:{PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()