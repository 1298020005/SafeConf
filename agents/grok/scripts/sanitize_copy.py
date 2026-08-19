#!/usr/bin/env python3
"""Sanitize text for Windows paste via /copy. UTF-8 BOM output."""
import re
import sys

REPLACEMENTS = {
    "\u03c1": "rho",
    "\u2013": "-",
    "\u2014": "-",
    "\uff1a": ":",
    "\uff1b": ";",
    "\u300c": '"',
    "\u300d": '"',
    "\u300e": '"',
    "\u300f": '"',
    "\u2026": "...",
    "\u00b7": "-",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2713": "[OK]",
    "\u2717": "[X]",
    "\u26a0\ufe0f": "[!]",
    "\u26a0": "[!]",
    "\u2705": "[OK]",
    "\u274c": "[X]",
}


def sanitize(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-|:\s]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    raw = sys.stdin.read() if not sys.argv[1:] else open(sys.argv[1], encoding="utf-8-sig").read()
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    out = sanitize(raw)
    sys.stdout.buffer.write(b"\xef\xbb\xbf")
    sys.stdout.buffer.write(out.encode("utf-8"))


if __name__ == "__main__":
    main()