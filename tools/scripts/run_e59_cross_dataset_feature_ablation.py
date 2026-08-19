#!/usr/bin/env python3
"""E59: isolate the contribution of predicted magnitude in cross-dataset risk.

The cross-dataset score used in E55/E57 is a sum of four deployable terms:
low source support, low control-state similarity, predictor disagreement, and
*predicted* effect magnitude.  The advisor specifically asked whether the
effect-size term is computed from unavailable held-out truth.  It is not, but
we still need to show whether the composite score contributes beyond that
single term.

This audit compares, within each source -> target direction:
  full composite; predicted magnitude only; and the composite with magnitude
  removed.  ``true_l2_diagnostic`` remains marked oracle-only throughout and
is never evaluated as a deployable candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E59_cross_dataset_feature_ablation_20260711"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
INPUTS = {
    "E55": ROOT / "docs" / "实验结果" / "E55_cross_dataset_transfer_20260710" / "tables" / "E55_CROSS_DATASET_SCORE_TABLE.csv",
    "E57": ROOT / "docs" / "实验结果" / "E57_dataset_expansion_cross_dataset_20260710" / "tables" / "E57_DATASET_EXPANSION_SCORE_TABLE.csv",
}
ERROR = "error_combined_rmse"
SCORES = [
    "risk_cross_dataset",
    "risk_predicted_magnitude",
    "risk_without_predicted_magnitude",
    "risk_disagreement",
    "risk_low_source_support",
    "risk_low_context_similarity",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def top20(score: np.ndarray, error: np.ndarray) -> float:
    if len(score) < 5:
        return float("nan")
    k = max(1, int(math.ceil(len(score) * 0.2)))
    mean = float(error.mean())
    return float(error[np.argsort(-score, kind="stable")[:k]].mean()) / mean if mean > 1e-12 else float("nan")


def ci95(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan")
    return float(np.quantile(x, .025)), float(np.quantile(x, .975))


def pair_rows(group: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[list[dict], dict]:
    clean = group[[*SCORES, ERROR]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(clean)
    base = group.iloc[0]
    rows = []
    by_score = {}
    for col in SCORES:
        score = clean[col].to_numpy(float)
        error = clean[ERROR].to_numpy(float)
        rho = spearman(score, error)
        enrich = top20(score, error)
        by_score[col] = rho
        rows.append({
            "input_batch": base["input_batch"], "pair_group": base["pair_group"],
            "directional_pair": base["directional_pair"], "source_dataset": base["source_dataset"],
            "target_dataset": base["target_dataset"], "n_tasks": n, "score_name": col,
            "spearman_vs_error": rho, "top20_error_enrichment": enrich,
            "deployable": True,
        })

    full = clean["risk_cross_dataset"].to_numpy(float)
    mag = clean["risk_predicted_magnitude"].to_numpy(float)
    no_mag = clean["risk_without_predicted_magnitude"].to_numpy(float)
    err = clean[ERROR].to_numpy(float)
    delta_mag = float("nan")
    delta_no_mag = float("nan")
    delta_mag_ci = (float("nan"), float("nan"))
    delta_no_mag_ci = (float("nan"), float("nan"))
    if n >= 3:
        draw_mag, draw_no_mag = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            r_full = spearman(full[idx], err[idx])
            draw_mag[i] = r_full - spearman(mag[idx], err[idx])
            draw_no_mag[i] = r_full - spearman(no_mag[idx], err[idx])
        delta_mag = by_score["risk_cross_dataset"] - by_score["risk_predicted_magnitude"]
        delta_no_mag = by_score["risk_cross_dataset"] - by_score["risk_without_predicted_magnitude"]
        delta_mag_ci, delta_no_mag_ci = ci95(draw_mag), ci95(draw_no_mag)

    if n < 30:
        verdict = "任务数不足 30：只作探索性比较"
    elif np.isfinite(delta_mag_ci[0]) and delta_mag_ci[0] > 0:
        verdict = "总分优于预测幅度：组合特征有额外贡献"
    elif np.isfinite(delta_mag_ci[1]) and delta_mag_ci[1] < 0:
        verdict = "预测幅度优于总分：组合项未带来增益"
    else:
        verdict = "总分与预测幅度差异不明确"
    comparison = {
        "input_batch": base["input_batch"], "pair_group": base["pair_group"],
        "directional_pair": base["directional_pair"], "source_dataset": base["source_dataset"],
        "target_dataset": base["target_dataset"], "n_tasks": n,
        "rho_full": by_score["risk_cross_dataset"],
        "rho_predicted_magnitude": by_score["risk_predicted_magnitude"],
        "rho_without_predicted_magnitude": by_score["risk_without_predicted_magnitude"],
        "delta_full_minus_predicted_magnitude": delta_mag,
        "bootstrap_delta_full_minus_predicted_magnitude_ci95_low": delta_mag_ci[0],
        "bootstrap_delta_full_minus_predicted_magnitude_ci95_high": delta_mag_ci[1],
        "delta_full_minus_without_magnitude": delta_no_mag,
        "bootstrap_delta_full_minus_without_magnitude_ci95_low": delta_no_mag_ci[0],
        "bootstrap_delta_full_minus_without_magnitude_ci95_high": delta_no_mag_ci[1],
        "verdict": verdict,
    }
    return rows, comparison


def table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "（无行）"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        cells=[]
        for c in cols:
            v=r[c]
            cells.append(f"{v:.3f}" if isinstance(v,float) and np.isfinite(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_svg(comp: pd.DataFrame) -> None:
    view = comp[comp["n_tasks"] >= 30].sort_values("delta_full_minus_predicted_magnitude", ascending=False)
    width, top, row_h, left, right = 1640, 118, 48, 560, 130
    height = max(280, top + (len(view) or 1) * row_h + 82)
    x0, x1 = left, width-right
    def sx(v: float) -> float:
        return x0 + (max(-1., min(1., v))+1)/2*(x1-x0)
    lines=[
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#27323c}.t{font-size:27px;font-weight:700}.s{font-size:16px;fill:#5d6b78}.l{font-size:14px}.sm{font-size:12px;fill:#5d6b78}</style>',
        '<text class="t" x="55" y="45">E59｜总分相对预测幅度的增益</text>',
        '<text class="s" x="55" y="75">点为 ρ(full risk) − ρ(predicted magnitude)，线为 bootstrap 95% CI；正值说明总分排序更好。</text>'
    ]
    for tick in [-1,-.5,0,.5,1]:
        x=sx(tick); stroke='#94a3b8' if tick==0 else '#e2e8f0'
        lines.append(f'<line x1="{x:.1f}" y1="{top-20}" x2="{x:.1f}" y2="{height-45}" stroke="{stroke}" stroke-width="{2 if tick==0 else 1}"/>')
        lines.append(f'<text class="sm" x="{x:.1f}" y="{top-30}" text-anchor="middle">{tick:g}</text>')
    if view.empty:
        lines.append('<text class="l" x="55" y="150">没有任务数 ≥ 30 的方向。</text>')
    for i,(_,r) in enumerate(view.iterrows()):
        y=top+i*row_h; lbl=str(r['directional_pair']); lbl=lbl if len(lbl)<66 else lbl[:63]+'…'
        lo=float(r['bootstrap_delta_full_minus_predicted_magnitude_ci95_low']); hi=float(r['bootstrap_delta_full_minus_predicted_magnitude_ci95_high']); val=float(r['delta_full_minus_predicted_magnitude'])
        color='#137c8b' if lo>0 else ('#b45309' if hi<0 else '#64748b')
        lines += [
          f'<text class="l" x="{left-13}" y="{y+5}" text-anchor="end">{escape(lbl)}</text>',
          f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>',
          f'<circle cx="{sx(val):.1f}" cy="{y}" r="7" fill="{color}"/>',
          f'<text class="sm" x="{width-70}" y="{y+5}" text-anchor="middle">n={int(r["n_tasks"])}</text>'
        ]
    lines.append('</svg>')
    (FIGURES/'F1_full_risk_vs_predicted_magnitude.svg').write_text('\n'.join(lines),encoding='utf-8')


def write_report(scores: pd.DataFrame, comp: pd.DataFrame, n_rows: int, args: argparse.Namespace) -> None:
    big=comp[comp.n_tasks>=30].copy()
    full_win=big[big.verdict.eq('总分优于预测幅度：组合特征有额外贡献')]
    mag_win=big[big.verdict.eq('预测幅度优于总分：组合项未带来增益')]
    unclear=big[big.verdict.eq('总分与预测幅度差异不明确')]
    lines=[
      '# E59｜跨数据集分数构成审计', '',
      '## 问题', '',
      '老师问到效应幅度时，必须分清两件事：`predicted_l2_combined` 是只由源数据构造的预测向量长度，可在打分时得到；`true_l2_diagnostic` 来自目标真实效应，只作 oracle 诊断，完全不进入 `risk_cross_dataset`。', '',
      '跨数据集总分由四项相加：低历史支持、低控制状态相似度、两个源域参考预测器的分歧、预测效应幅度。本轮比较总分与单独预测幅度，检查组合项有没有实际贡献。', '',
      f'- 输入任务行：{n_rows:,}', f'- 方向数：{len(comp)}', f'- bootstrap 次数：{args.n_boot}', '',
      '## 任务数 ≥ 30 的比较', '',
      table(big.sort_values('delta_full_minus_predicted_magnitude',ascending=False),['directional_pair','n_tasks','rho_full','rho_predicted_magnitude','rho_without_predicted_magnitude','delta_full_minus_predicted_magnitude','bootstrap_delta_full_minus_predicted_magnitude_ci95_low','bootstrap_delta_full_minus_predicted_magnitude_ci95_high','verdict']), '',
      '## 如何使用这张表', '',
      f'- 总分显著优于预测幅度的方向数：{len(full_win)}。', f'- 预测幅度显著优于总分的方向数：{len(mag_win)}。', f'- 差异不明确的方向数：{len(unclear)}。', '',
      '这张表不支持把“预测幅度”说成 SafeConf 的独立贡献。若某个方向预测幅度更强，就按结果原样保留；若总分有额外贡献，才说明支持度、上下文和分歧提供了幅度以外的信息。', '',
      '## 文件', '',
      '- 图：`figures/F1_full_risk_vs_predicted_magnitude.svg`',
      '- 各成分相关：`tables/E59_FEATURE_SCORE_SUMMARY.csv`',
      '- 总分对幅度的 bootstrap 差值：`tables/E59_FULL_VS_MAGNITUDE_COMPARISON.csv`',
    ]
    (REPORTS/'E59_FEATURE_ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--n-boot',type=int,default=2000); p.add_argument('--seed',type=int,default=20260711); args=p.parse_args()
    if args.n_boot<100: raise ValueError('Use at least 100 bootstrap draws.')
    for d in [TABLES,REPORTS,FIGURES]: d.mkdir(parents=True,exist_ok=True)
    frames=[]
    for name,path in INPUTS.items():
        df=pd.read_csv(path); df['input_batch']=name
        df['risk_without_predicted_magnitude']=df['risk_low_source_support']+df['risk_low_context_similarity']+df['risk_disagreement']
        frames.append(df)
    all_df=pd.concat(frames,ignore_index=True)
    rng=np.random.default_rng(args.seed); srows=[]; crows=[]
    groups=all_df.groupby(['input_batch','pair_group','directional_pair'],sort=True)
    for i,(_,g) in enumerate(groups,1):
        one,comparison=pair_rows(g,rng,args.n_boot); srows.extend(one); crows.append(comparison)
        print(f"[E59] {i}/{len(groups)} {comparison['directional_pair']}: delta={comparison['delta_full_minus_predicted_magnitude']:.3f}",flush=True)
    scores=pd.DataFrame(srows).sort_values(['input_batch','pair_group','directional_pair','score_name'])
    comp=pd.DataFrame(crows).sort_values(['input_batch','pair_group','directional_pair'])
    scores.to_csv(TABLES/'E59_FEATURE_SCORE_SUMMARY.csv',index=False)
    comp.to_csv(TABLES/'E59_FULL_VS_MAGNITUDE_COMPARISON.csv',index=False)
    write_svg(comp); write_report(scores,comp,len(all_df),args)
    status={'experiment':'E59_cross_dataset_feature_ablation','created_at':now(),'git_head_before_run':git_head(),'n_input_task_rows':int(len(all_df)),'n_directional_pairs':int(len(comp)),'n_boot':args.n_boot,'seed':args.seed,'deployable_score_terms':['risk_low_source_support','risk_low_context_similarity','risk_disagreement','risk_predicted_magnitude'],'oracle_only_column':'true_l2_diagnostic','outputs':['tables/E59_FEATURE_SCORE_SUMMARY.csv','tables/E59_FULL_VS_MAGNITUDE_COMPARISON.csv','figures/F1_full_risk_vs_predicted_magnitude.svg','reports/E59_FEATURE_ABLATION_REPORT.md']}
    (OUT/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'README_先看这个.md').write_text('# E59 先看这个\n\n先读 `reports/E59_FEATURE_ABLATION_REPORT.md`。\n\n图 `figures/F1_full_risk_vs_predicted_magnitude.svg` 为白底 SVG，可直接放 PPT。\n\n`true_l2_diagnostic` 在本实验中不参与任何可部署分数。\n',encoding='utf-8')


if __name__=='__main__': main()
