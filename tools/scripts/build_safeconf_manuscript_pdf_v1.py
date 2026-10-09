#!/usr/bin/env python3
"""Render the frozen SafeConf manuscript draft to a complete review PDF.

This is a reproducible internal manuscript rendering step. It reads only the
draft text and frozen figure files; it does not train models, read raw truth,
or alter evidence artifacts.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
from pathlib import Path
import re
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/研究推进/20261009_投稿证据_v21"
DRAFT = BASE / "submission_draft_v1"
EVID = BASE / "evidence_freeze_v1"
# The manuscript is English. DejaVu keeps the PDF text layer searchable;
# Chinese labels remain in the separate review packet rendered with Noto CJK.
FONT = FontProperties(family="DejaVu Sans")


def paragraphs_from_markdown(text: str):
    blocks = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            title = re.sub(r"^#+\s*", "", block.splitlines()[0]).strip()
            body = " ".join(line.strip() for line in block.splitlines()[1:] if line.strip())
            blocks.append(("heading", title))
            if body:
                blocks.append(("body", body))
        else:
            blocks.append(("body", " ".join(line.strip() for line in block.splitlines())))
    return blocks


def text_page(pdf, title, blocks, start=0, fontsize=10.2):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(.08, .95, title, fontsize=17, fontweight="bold", va="top", color="#1e2d35", fontproperties=FONT)
    y = .90
    for kind, content in blocks[start:]:
        if kind == "heading":
            lines = textwrap.wrap(content, width=86) or [""]
            size = 13 if content[:1].isdigit() else 11.5
            weight = "bold"
            gap = .02
        else:
            lines = textwrap.wrap(content, width=100) or [""]
            size = fontsize
            weight = "normal"
            gap = .025
        height = .026 * len(lines) + gap
        if y - height < .07:
            return fig, start
        fig.text(.08, y, "\n".join(lines), fontsize=size, fontweight=weight, va="top",
                 linespacing=1.35, color="#263238", fontproperties=FONT)
        y -= height
        start += 1
    return fig, start


def figure_page(pdf, image_path, title, caption):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([.04, .12, .92, .80])
    ax.imshow(imread(image_path))
    ax.axis("off")
    fig.text(.05, .96, title, fontsize=16, fontweight="bold", va="top", color="#1e2d35", fontproperties=FONT)
    fig.text(.05, .045, caption, fontsize=9, va="bottom", color="#4d5c63", fontproperties=FONT)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_text(pdf, title, text):
    blocks = paragraphs_from_markdown(text)
    cursor = 0
    first = True
    while cursor < len(blocks):
        fig, cursor2 = text_page(pdf, title if first else title + " (continued)", blocks, cursor)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        if cursor2 <= cursor:
            raise RuntimeError("manuscript renderer made no progress")
        cursor = cursor2
        first = False


def main():
    out_main = DRAFT / "MANUSCRIPT_SUBMISSION.pdf"
    out_supp = DRAFT / "SUPPLEMENT_SUBMISSION.pdf"
    manuscript = (DRAFT / "MANUSCRIPT_DRAFT.md").read_text()
    supplement = (DRAFT / "SUPPLEMENT_DRAFT.md").read_text()
    with PdfPages(out_main) as pdf:
        render_text(pdf, "SafeConf manuscript", manuscript)
        figdir = EVID / "figures"
        figure_page(pdf, figdir / "FIG1_ZERO_ERROR.png", "Figure 1 | Cold-start risk auditing", "Public evidence enters before target-screen error labels; missing-history tasks use the fixed magnitude fallback.")
        figure_page(pdf, figdir / "FIG2_CONTENT.png", "Figure 2 | Public evidence controls", "Support-matched content permutations and public-history controls are reported without post-hoc mechanism selection.")
        figure_page(pdf, figdir / "FIG3_FEEDBACK.png", "Figure 3 | Feedback budget", "Global and context-macro endpoints remain separate; feedback is an optional extension to the public start.")
        figure_page(pdf, figdir / "FIG4_KOLF.png", "Figure 4 | Frozen independent KOLF evaluation", "Ridge passes the predictor gate; the public-versus-magnitude point estimate is positive with a broad cluster interval.")
        figure_page(pdf, figdir / "FIG5_FRANGIEH_STRESS.png", "Figure 5 | Cross-family stress evidence", "Frangieh provides stress evidence only because all upstream competence checks failed.")
    with PdfPages(out_supp) as pdf:
        render_text(pdf, "SafeConf supplement", supplement)
    print(out_main)
    print(out_supp)


if __name__ == "__main__":
    main()
