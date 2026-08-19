#!/usr/bin/env python3
"""Export each native HTML slide to a consistent 16:9 high-resolution PNG."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
HTML = ROOT / "SafeConf_阶段进展汇报_20260709.html"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

NAMES = [
    "01_封面",
    "02_当前进度总览",
    "03_上次沟通后的进展",
    "04_系统与实际用途",
    "05_任务与防泄漏",
    "06_核心风险证据",
    "07_证据链全景",
    "08_七主数据集结果",
    "09_E1至E4稳健性",
    "10_E8b外部基准",
    "11_Tahoe实验链",
    "12_Tahoe定量结果",
    "13_实用价值与异质性",
    "14_已解决与未解决",
    "15_下一决定性实验",
    "16_周四需要确认的决定",
    "17_七数据集数据地图",
    "18_老师六问逐项回答",
]

for old in OUT.glob("*.png"):
    old.unlink()

for index, name in enumerate(NAMES, start=1):
    raw = OUT / f".{name}.raw.png"
    output = OUT / f"{name}.png"
    url = f"{HTML.resolve().as_uri()}?slide={index}"
    subprocess.run(
        [
            "google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1.5",
            "--window-size=1440,897",
            f"--screenshot={raw}",
            url,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with Image.open(raw) as image:
        image.convert("RGB").crop((0, 0, 2160, 1215)).save(output, optimize=True)
    raw.unlink()

thumb_w, thumb_h = 540, 304
rows = (len(NAMES) + 1) // 2
sheet = Image.new("RGB", (thumb_w * 2 + 30, thumb_h * rows + 10 * (rows + 1)), "white")
draw = ImageDraw.Draw(sheet)
for index, name in enumerate(NAMES):
    with Image.open(OUT / f"{name}.png") as image:
        image.thumbnail((thumb_w, thumb_h))
        x = 10 + (index % 2) * (thumb_w + 10)
        y = 10 + (index // 2) * (thumb_h + 10)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline="#d5dce1", width=1)
sheet.save(OUT / "00_全部页面总览.png", optimize=True)

print(f"exported={len(NAMES)}")
print(f"output={OUT}")
