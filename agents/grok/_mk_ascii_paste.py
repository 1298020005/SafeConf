#!/usr/bin/env python3
from pathlib import Path
import base64

src = Path("/home/yyf/proj/agents/grok/粘贴给Codex.txt")
body = src.read_text(encoding="utf-8")
b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
assert base64.b64decode(b64).decode("utf-8") == body

chunks = [b64[i : i + 80] for i in range(0, len(b64), 80)]
joined = "\n".join(f'    "{c}"' for c in chunks)

final = f"""Run this exactly, then obey the printed Chinese text in full:

python3 <<'PY'
import base64
from pathlib import Path
b64 = (
{joined}
)
text = base64.b64decode("".join(b64)).decode("utf-8")
print(text)
Path("/tmp/codex_zh_instructions.txt").write_text(text, encoding="utf-8")
print("---")
print("SAVED /tmp/codex_zh_instructions.txt")
PY

Then open /tmp/codex_zh_instructions.txt and execute every rule.
First reply: Goal / Gaps / Single next action.
Start with gene-side hard-setting summary table.
"""

assert all(ord(c) < 128 for c in final)

# verify executable part
code = final.split("python3 <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
ns = {}
exec(code, ns)
assert Path("/tmp/codex_zh_instructions.txt").read_text(encoding="utf-8") == body

out = Path("/home/yyf/proj/agents/grok/粘贴_纯ASCII.txt")
out.write_text(final, encoding="ascii", newline="\n")
Path("/home/yyf/proj/agents/粘贴_纯ASCII.txt").write_text(final, encoding="ascii", newline="\n")
print(final)
print("VERIFY_OK", out, "bytes", out.stat().st_size)
