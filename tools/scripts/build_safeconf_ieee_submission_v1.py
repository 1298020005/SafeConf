#!/usr/bin/env python3
"""Build a portable IEEEtran submission draft from the frozen SafeConf text."""
from __future__ import annotations
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/研究推进/20261009_投稿证据_v21"
DRAFT = BASE / "submission_draft_v1"
OUT = DRAFT / "IEEE_SUBMISSION"
SRC = OUT / "main.tex"
BUILD = OUT / "build"
EVID_FIG = BASE / "evidence_freeze_v1" / "figures"
TOOL = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/paper/build/toolchain/tectonic"
CLASS = Path("/home/yyf/师姐论文/报告正文latex模板/IEEEtran.cls")


def tex_escape(s: str) -> str:
    s = s.replace("–", "--").replace("—", "---").replace("“", "``").replace("”", "''").replace("’", "'")
    placeholders = {}
    def hold(value):
        key = f"ZZHOLD{len(placeholders)}ZZ"
        placeholders[key] = value
        return key
    s = re.sub(r"\\\((.*?)\\\)", lambda m: hold("\\(" + m.group(1) + "\\)"), s)
    s = re.sub(r"\\\[(.*?)\\\]", lambda m: hold("\\[" + m.group(1) + "\\]"), s, flags=re.S)
    s = re.sub(r"\[@([A-Za-z0-9:_-]+)(?:;\s*@([A-Za-z0-9:_-]+))*\]", lambda m: hold("\\cite{" + ",".join(x for x in m.groups() if x) + "}"), s)
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
        s = s.replace(a, b)
    for key, value in placeholders.items():
        s = s.replace(key, value)
    return s


def parse_blocks(text: str):
    return [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]


def render_body(text: str) -> str:
    out = []
    for block in parse_blocks(text):
        lines = block.splitlines()
        first = lines[0].strip()
        if first.startswith("# "):
            continue
        if first.startswith("## "):
            out.append("\\section{" + tex_escape(first[3:]) + "}")
            rest = " ".join(x.strip() for x in lines[1:] if x.strip())
            if rest:
                out.append(tex_escape(rest))
            continue
        if first.startswith("### "):
            out.append("\\subsection{" + tex_escape(first[4:]) + "}")
            rest = " ".join(x.strip() for x in lines[1:] if x.strip())
            if rest:
                out.append(tex_escape(rest))
            continue
        if block.startswith("\\[") and block.endswith("\\]"):
            out.append(block)
            continue
        if all(x.lstrip().startswith(("- ", "* ")) for x in lines):
            items = "\n".join("\\item " + tex_escape(x.lstrip()[2:]) for x in lines)
            out.append("\\begin{itemize}\n" + items + "\n\\end{itemize}")
            continue
        out.append(tex_escape(" ".join(x.strip() for x in lines if x.strip())))
    return "\n\n".join(out)


def bibliography() -> str:
    entries = [
        ("officialbishal2025pertema", "B. Shrestha, ``PertEMA: Perturbation Error Meta-Assessment,'' GitHub repository, 2026."),
        ("roohani2023gears", "Y. Roohani, K. Huang, and J. Leskovec, ``Predicting transcriptional outcomes of novel multigene perturbations with GEARS,'' Nature Biotechnology, 2023."),
        ("cui2024scgpt", "H. Cui et al., ``scGPT: toward building a foundation model for single-cell multi-omics using generative AI,'' Nature Methods, vol. 21, 2024."),
        ("replogle2022perturbseq", "J. M. Replogle et al., ``Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq,'' Cell, vol. 185, 2022."),
        ("frangieh2021perturbcite", "C. J. Frangieh et al., ``Multimodal pooled Perturb-CITE-seq screens in patient models define mechanisms of cancer immune evasion,'' Nature Genetics, vol. 53, 2021."),
        ("cui2026perturbmap", "P. Cui, Y. Liu, and W. Sun, ``PerturbMap: Cross-Context Transfer of Single-Cell Perturbation Responses,'' arXiv:2607.28090, 2026."),
        ("nourreddine2026kolf", "S. Nourreddine et al., ``A genome-scale CRISPRi perturbation atlas of human induced pluripotent stem cells,'' Nature Biotechnology, 2026."),
        ("cheng2025prescribe", "J. Cheng et al., ``PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation,'' NeurIPS, 2025."),
        ("nicol2026spurious", "P. B. Nicol, S. Shivakumar, and R. A. Irizarry, ``Spurious correlation inflates performance in single-cell perturbation prediction,'' bioRxiv, 2026."),
    ]
    body = ["\\begin{thebibliography}{99}"]
    for key, value in entries:
        body.append("\\bibitem{" + key + "}" + value)
    body.append("\\end{thebibliography}")
    return "\n".join(body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLASS, OUT / "IEEEtran.cls")
    for name in ["FIG1_ZERO_ERROR.png", "FIG2_CONTENT.png", "FIG3_FEEDBACK.png", "FIG4_KOLF.png", "FIG5_FRANGIEH_STRESS.png"]:
        shutil.copy2(EVID_FIG / name, OUT / name)
    manuscript = (DRAFT / "MANUSCRIPT_DRAFT.md").read_text()
    abstract = manuscript.split("## 1. Introduction", 1)[0]
    abstract = re.sub(r"^# .*?\n", "", abstract, count=1).strip()
    abstract = " ".join(x.strip() for x in abstract.splitlines() if x.strip())
    abstract = re.sub(r"\[@[^\]]+\]", "", abstract)
    body_text = manuscript.split("## 1. Introduction", 1)[1].split("## References", 1)[0]
    tex = r"""\documentclass[journal]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref}
\hypersetup{hidelinks}
\title{Public perturbation evidence enables cold-start risk auditing for single-cell perturbation prediction}
\author{AUTHOR NAMES\\AFFILIATION\\Corresponding author: AUTHOR NAME, EMAIL}
\begin{document}
\maketitle
\begin{abstract}
""" + tex_escape(abstract) + r"""
\end{abstract}
\begin{IEEEkeywords}
single-cell perturbation prediction, reliability auditing, public experimental evidence, selective review, target feedback
\end{IEEEkeywords}
""" + render_body(body_text) + r"""
\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{FIG1_ZERO_ERROR.png}\hfill
\includegraphics[width=0.48\textwidth]{FIG2_CONTENT.png}
\caption{Public evidence and content controls.}
\label{fig:public}
\end{figure*}
\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{FIG3_FEEDBACK.png}\hfill
\includegraphics[width=0.48\textwidth]{FIG4_KOLF.png}
\caption{Feedback budgets and frozen independent KOLF evaluation.}
\label{fig:feedback}
\end{figure*}
\begin{figure*}[t]
\centering
\includegraphics[width=0.82\textwidth]{FIG5_FRANGIEH_STRESS.png}
\caption{Frangieh cross-family stress evidence.}
\label{fig:stress}
\end{figure*}
""" + bibliography() + r"""
\end{document}
"""
    SRC.write_text(tex)
    if not TOOL.exists():
        raise SystemExit(f"missing tectonic tool: {TOOL}")
    cmd = [str(TOOL), "--keep-logs", "--keep-intermediates", "--outdir", str(BUILD), str(SRC)]
    env = os.environ.copy()
    env["TEXMFVAR"] = str(BUILD / "texmf-var")
    proc = subprocess.run(cmd, cwd=OUT, env=env, text=True, capture_output=True)
    (BUILD / "TECTONIC_STDOUT.log").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    pdf = BUILD / "main.pdf"
    if not pdf.exists():
        raise SystemExit("tectonic completed without main.pdf")
    final = OUT / "SafeConf_TCBB_IEEE.pdf"
    shutil.copy2(pdf, final)
    print(final)


if __name__ == "__main__":
    main()
