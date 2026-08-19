#!/usr/bin/env python3
"""Build a pure-ASCII paste package that decodes to full Chinese instructions.

Why: selecting Chinese in Grok TUI/chat often copies UTF-8 bytes as Latin-1
mojibake. Pure ASCII hex cannot be corrupted that way.
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("/home/yyf/proj/agents/grok/粘贴给Codex.txt")
OUT = Path("/home/yyf/proj/agents/grok/粘贴_可复制_纯ASCII解码中文.txt")
OUT2 = Path("/home/yyf/proj/agents/粘贴_可复制_纯ASCII解码中文.txt")


def main() -> None:
    src = SRC.read_text(encoding="utf-8")
    assert "周老师" in src and "硬目标" in src

    # --- recovery tests ---
    def recover(body: str) -> str:
        if any(0x4E00 <= ord(c) <= 0x9FFF for c in body):
            return body
        return body.encode("latin-1").decode("utf-8")

    assert recover(src) == src
    mojibake = src.encode("utf-8").decode("latin-1")
    assert recover(mojibake) == src

    hexbody = src.encode("utf-8").hex()
    assert bytes.fromhex(hexbody).decode("utf-8") == src
    hex_lines = "\n".join(hexbody[i : i + 80] for i in range(0, len(hexbody), 80))

    final = f"""# Codex instruction package (PURE ASCII — will not garble when copied)

STEP 0: run this exactly, then obey the printed Chinese text.

python3 <<'PY'
from pathlib import Path
hex_lines = r'''
{hex_lines}
'''.split()
text = bytes.fromhex("".join(hex_lines)).decode("utf-8")
assert "\\u5468\\u8001\\u5e08"  # placeholder replaced below
print(text)
Path("/tmp/codex_instr.txt").write_text(text, encoding="utf-8")
print("SAVED /tmp/codex_instr.txt")
print("Execute every rule in that Chinese text.")
print("First reply: Goal / Gaps / Single next action")
print("Start: gene-side hard-setting summary table")
PY

STEP 1: open /tmp/codex_instr.txt and do everything it says.
"""

    # fix assert to real Chinese check without putting Chinese in final if we want pure ASCII
    # final must stay pure ASCII — assert "Zhou" markers via unicode escapes in the generated script
    final = f"""# Codex instruction package (PURE ASCII - safe to mouse-select and copy)

STEP 0: run this exactly, then obey the printed Chinese text.

python3 <<'PY'
from pathlib import Path
hex_lines = r'''
{hex_lines}
'''.split()
text = bytes.fromhex("".join(hex_lines)).decode("utf-8")
assert any(0x4E00 <= ord(c) <= 0x9FFF for c in text), "decode failed"
print(text)
Path("/tmp/codex_instr.txt").write_text(text, encoding="utf-8")
print("SAVED /tmp/codex_instr.txt")
print("Execute every rule in that Chinese text.")
print("First reply: Goal / Gaps / Single next action")
print("Start: gene-side hard-setting summary table")
PY

STEP 1: open /tmp/codex_instr.txt and do everything it says.
"""

    assert all(ord(c) < 128 for c in final), "not pure ASCII"

    # simulate broken terminal copy of the package: ASCII unchanged under latin-1 reinterpret of utf-8 of ascii
    as_bytes = final.encode("utf-8")
    # if someone wrongly decodes package as latin-1 for display then copies "characters"...
    # for pure ASCII utf-8 == ascii, so latin-1 misread of the same bytes still yields same string
    wrong_view = as_bytes.decode("latin-1")
    assert wrong_view == final

    # execute package decode path
    hx = "".join(
        final.split("hex_lines = r'''", 1)[1]
        .split("'''.split()", 1)[0]
        .split()
    )
    decoded = bytes.fromhex(hx).decode("utf-8")
    assert decoded == src

    OUT.write_text(final, encoding="ascii", newline="\n")
    OUT2.write_text(final, encoding="ascii", newline="\n")
    print("SELFTEST_PASS")
    print("OUT", OUT)
    print("CHARS_PACKAGE", len(final))
    print("CHARS_CHINESE", len(src))


if __name__ == "__main__":
    main()
