#!/usr/bin/env python3
"""Write COPY.txt / COPY.gbk / COPY.html from buffer."""
import html
import sys
from pathlib import Path

GROK = Path(__file__).resolve().parent.parent
BUFFER = GROK / ".copy_buffer.txt"
SANITIZE = GROK / "scripts" / "sanitize_copy.py"


def load_text() -> str:
    if not BUFFER.exists() or BUFFER.stat().st_size == 0:
        return "[empty buffer]"
    raw = BUFFER.read_text(encoding="utf-8-sig")
    if SANITIZE.exists():
        import subprocess
        p = subprocess.run(
            [sys.executable, str(SANITIZE)],
            input=raw,
            capture_output=True,
            text=True,
        )
        if p.returncode == 0 and p.stdout.strip():
            raw = p.stdout
            if raw.startswith("\ufeff"):
                raw = raw[1:]
    return raw.strip()


def main() -> None:
    text = load_text()
    (GROK / "COPY.txt").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    (GROK / "COPY.gbk").write_bytes(text.encode("gbk", errors="replace"))
    body = html.escape(text)
    page = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="UTF-8">'
        "<title>SafeConf Copy</title>"
        "<style>body{font:16px/1.6 sans-serif;max-width:720px;margin:2em auto;padding:0 1em}"
        "pre{white-space:pre-wrap;word-break:break-word}</style>"
        "</head><body><pre>"
        f"{body}</pre></body></html>"
    )
    (GROK / "COPY.html").write_text(page, encoding="utf-8")
    print(f"variants: COPY.txt COPY.gbk COPY.html ({len(text)} chars)")


if __name__ == "__main__":
    main()