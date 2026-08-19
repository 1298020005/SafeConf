#!/usr/bin/env python3
"""E61: source-calibrated, cross-domain task-risk audit.

E59 showed that the fixed equal-weight cross-domain score often does not beat
predicted effect magnitude.  E61 tests a falsifiable repair: learn the risk
weights *only in the source domain*, using five source-internal held-out folds,
then freeze the fitted calibrator and apply it to a target dataset.

The target truth is never read when constructing the target risk score.  It is
used only afterwards to calculate the simple source-only predictor's RMSE.
The model is deliberately a small ridge regression rather than a new opaque
deep model: every feature and fitted coefficient is saved.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "scripts"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from run_e55_cross_dataset_transfer import (  # noqa: E402
    DatasetSpec,
    build_specs as e55_specs,
    build_tasks_for_genes,
    choose_common_genes,
    cosine,
    rmse,
    source_indices,
    vec_l2,
)
from run_e57_dataset_expansion_cross_dataset import specs as e57_specs  # noqa: E402


OUT = ROOT / "docs" / "实验结果" / "E61_source_calibrated_cross_domain_risk_20260711"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
FEATURES = ["log1p_source_support", "nearest_context_similarity", "prediction_disagreement_rmse", "predicted_l2_combined"]
TARGET_ERROR = "error_combined_rmse"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def make_pair_plan() -> list[tuple[str, DatasetSpec, DatasetSpec, int]]:
    e57 = e57_specs(); e55 = e55_specs()
    return [
        ("lara_cross_condition", e57["Lara_exvivo"], e57["Lara_invivo"], 500),
        ("lara_cross_condition", e57["Lara_invivo"], e57["Lara_exvivo"], 500),
        ("lara_cross_condition", e57["Lara_exvivo"], e57["Lara_leukemia"], 500),
        ("lara_cross_condition", e57["Lara_leukemia"], e57["Lara_exvivo"], 500),
        ("lara_cross_condition", e57["Lara_invivo"], e57["Lara_leukemia"], 500),
        ("lara_cross_condition", e57["Lara_leukemia"], e57["Lara_invivo"], 500),
        ("tian_crispra_crispri", e57["TianActivation"], e57["TianInhibition"], 500),
        ("tian_crispra_crispri", e57["TianInhibition"], e57["TianActivation"], 500),
        ("chemical_pbmc_to_cancer", e55["KaggleCrossCell_celltype"], e55["McFarland_cellline"], 1000),
        ("same_panel_new_donor", e55["KaggleCrossCell_celltype"], e55["KaggleCrossPatient_donor"], 1000),
    ]


def raw_feature_rows(history: list[dict], queries: list[dict]) -> pd.DataFrame:
    """Calculate source-only raw features; query truth is stored separately."""
    if not history:
        raise ValueError("empty history")
    idx = source_indices(history)
    contexts = sorted(idx["context_control"].keys())
    rows = []
    for q in queries:
        pert = q["perturbation_key"]
        support_ids = idx["by_pert"].get(pert, [])
        p_pert = idx["pert_mean"].get(pert, idx["global_mean"])
        sims = [(ctx, cosine(q["control_mean"], idx["context_control"][ctx])) for ctx in contexts]
        nearest_ctx, nearest_sim = max(sims, key=lambda z: z[1]) if sims else ("", 0.0)
        if support_ids:
            best = max(support_ids, key=lambda i: cosine(q["control_mean"], history[i]["control_mean"]))
            p_ctx = history[best]["effect"]
            predictor_mode = "nearest_context_same_perturbation"
        elif nearest_ctx in idx["context_mean"]:
            p_ctx = idx["context_mean"][nearest_ctx]
            predictor_mode = "nearest_context_mean_effect"
        else:
            p_ctx = idx["global_mean"]
            predictor_mode = "global_fallback"
        combined = .5 * (np.asarray(p_pert) + np.asarray(p_ctx))
        rows.append({
            "context": q["context"], "perturbation": q["perturbation"], "perturbation_key": pert,
            "source_support_count": len(support_ids), "log1p_source_support": math.log1p(len(support_ids)),
            "nearest_context_similarity": nearest_sim, "nearest_source_context": nearest_ctx,
            "prediction_disagreement_rmse": rmse(p_pert, p_ctx), "predicted_l2_combined": vec_l2(combined),
            TARGET_ERROR: rmse(combined, q["effect"]), "true_l2_diagnostic": vec_l2(q["effect"]),
            "context_predictor_mode": predictor_mode,
        })
    return pd.DataFrame(rows)


def source_oof_rows(tasks: list[dict], n_folds: int, seed: int) -> pd.DataFrame:
    if len(tasks) < n_folds + 3:
        raise ValueError(f"too few source tasks for {n_folds}-fold calibration: {len(tasks)}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(tasks))
    fold_ids = np.empty(len(tasks), dtype=int)
    for fold, ids in enumerate(np.array_split(shuffled, n_folds)):
        fold_ids[ids] = fold
    frames = []
    for fold in range(n_folds):
        query_ids = np.where(fold_ids == fold)[0]
        history = [t for i, t in enumerate(tasks) if fold_ids[i] != fold]
        frame = raw_feature_rows(history, [tasks[i] for i in query_ids])
        frame["source_oof_fold"] = fold
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 3 or len(np.unique(xx[ok])) < 2 or len(np.unique(yy[ok])) < 2:
        return float("nan")
    return float(pd.Series(xx[ok]).corr(pd.Series(yy[ok]), method="spearman"))


def bootstrap_delta(full: np.ndarray, magnitude: np.ndarray, error: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    values = []
    for _ in range(n):
        ix = rng.integers(0, len(error), len(error))
        values.append(spearman(full[ix], error[ix]) - spearman(magnitude[ix], error[ix]))
    x = np.asarray(values); x = x[np.isfinite(x)]
    return (float(np.quantile(x, .025)), float(np.quantile(x, .975))) if len(x) else (float("nan"), float("nan"))


def fit_and_score(source_oof: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x_train, y_train = source_oof[FEATURES].to_numpy(float), source_oof[TARGET_ERROR].to_numpy(float)
    model = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    model.fit(x_train, y_train)
    out = target.copy()
    out["risk_source_calibrated"] = model.predict(out[FEATURES].to_numpy(float))
    out["risk_predicted_magnitude"] = out["predicted_l2_combined"]
    scaler = model.named_steps["scale"]; ridge = model.named_steps["ridge"]
    coefficients = pd.DataFrame({"feature": FEATURES, "standardized_ridge_coefficient": ridge.coef_})
    metadata = {"ridge_alpha": float(ridge.alpha_), "intercept_standardized_space": float(ridge.intercept_), "source_oof_n": int(len(source_oof)), "source_oof_error_mean": float(y_train.mean())}
    return out, coefficients, metadata


def run_pair(group: str, source: DatasetSpec, target: DatasetSpec, n_genes: int, n_folds: int, seed: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    src_head, tgt_head = ad.read_h5ad(source.path, backed="r"), ad.read_h5ad(target.path, backed="r")
    genes = choose_common_genes(src_head, tgt_head, n_genes)
    if len(genes) < 100:
        raise ValueError(f"only {len(genes)} common genes")
    source_tasks, source_meta = build_tasks_for_genes(source, genes, min_cells=20, max_cells_per_group=400, seed=seed)
    target_tasks, target_meta = build_tasks_for_genes(target, genes, min_cells=20, max_cells_per_group=400, seed=seed + 61)
    if len(source_tasks) < n_folds + 3 or len(target_tasks) < 10:
        raise ValueError(f"insufficient tasks: source={len(source_tasks)}, target={len(target_tasks)}")
    oof = source_oof_rows(source_tasks, n_folds=n_folds, seed=seed)
    target_raw = raw_feature_rows(source_tasks, target_tasks)
    target_scored, coefficients, fit = fit_and_score(oof, target_raw)
    target_scored["pair_group"] = group; target_scored["source_dataset"] = source.name; target_scored["target_dataset"] = target.name; target_scored["directional_pair"] = f"{source.name} -> {target.name}"; target_scored["n_common_genes"] = len(genes)
    coefficients["pair_group"] = group; coefficients["directional_pair"] = f"{source.name} -> {target.name}"
    rho_full = spearman(target_scored["risk_source_calibrated"], target_scored[TARGET_ERROR])
    rho_mag = spearman(target_scored["risk_predicted_magnitude"], target_scored[TARGET_ERROR])
    lo, hi = bootstrap_delta(target_scored["risk_source_calibrated"].to_numpy(float), target_scored["risk_predicted_magnitude"].to_numpy(float), target_scored[TARGET_ERROR].to_numpy(float), np.random.default_rng(seed + 1000), n_boot)
    if len(target_scored) < 30:
        verdict = "探索性：目标任务不足 30"
    elif np.isfinite(lo) and lo > 0:
        verdict = "校准总分优于预测幅度"
    elif np.isfinite(hi) and hi < 0:
        verdict = "预测幅度优于校准总分"
    else:
        verdict = "差异不明确"
    summary = pd.DataFrame([{
        "pair_group": group, "directional_pair": f"{source.name} -> {target.name}", "source_dataset": source.name, "target_dataset": target.name,
        "n_common_genes": len(genes), "source_n_tasks": len(source_tasks), "target_n_tasks": len(target_tasks), "source_oof_folds": n_folds,
        "spearman_source_calibrated": rho_full, "spearman_predicted_magnitude": rho_mag, "delta_calibrated_minus_magnitude": rho_full-rho_mag,
        "bootstrap_delta_ci95_low": lo, "bootstrap_delta_ci95_high": hi, "verdict": verdict, **fit,
    }])
    oof["pair_group"] = group; oof["directional_pair"] = f"{source.name} -> {target.name}"
    return target_scored, oof, coefficients, {"summary": summary, "source_meta": source_meta, "target_meta": target_meta}


def write_svg(summary: pd.DataFrame) -> None:
    view=summary.sort_values("delta_calibrated_minus_magnitude",ascending=False); width,left,right,top,row=1580,570,110,118,48; height=max(270,top+len(view)*row+75); x0,x1=left,width-right
    def sx(v: float) -> float: return x0+(max(-1.,min(1.,v))+1)/2*(x1-x0)
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="#ffffff"/>','<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#23313d}.t{font-size:27px;font-weight:700}.s{font-size:16px;fill:#5c6a75}.l{font-size:14px}.sm{font-size:12px;fill:#5c6a75}</style>','<text class="t" x="55" y="45">E61｜源域校准总分相对预测幅度的变化</text>','<text class="s" x="55" y="75">点为 ρ(source-calibrated risk) − ρ(predicted magnitude)，线为 bootstrap 95% CI；目标真值未参与校准。</text>']
    for tick in [-1,-.5,0,.5,1]:
        x=sx(tick); lines += [f'<line x1="{x:.1f}" y1="{top-20}" x2="{x:.1f}" y2="{height-44}" stroke="{"#94a3b8" if tick==0 else "#e2e8f0"}" stroke-width="{2 if tick==0 else 1}"/>',f'<text class="sm" x="{x:.1f}" y="{top-30}" text-anchor="middle">{tick:g}</text>']
    for i,(_,r) in enumerate(view.iterrows()):
        y=top+i*row; label=str(r.directional_pair); label=label if len(label)<67 else label[:64]+'…'; lo,hi,val=float(r.bootstrap_delta_ci95_low),float(r.bootstrap_delta_ci95_high),float(r.delta_calibrated_minus_magnitude); color='#137c8b' if lo>0 else ('#b45309' if hi<0 else '#64748b')
        lines += [f'<text class="l" x="{left-14}" y="{y+5}" text-anchor="end">{escape(label)}</text>',f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>',f'<circle cx="{sx(val):.1f}" cy="{y}" r="7" fill="{color}"/>',f'<text class="sm" x="{width-62}" y="{y+5}" text-anchor="middle">n={int(r.target_n_tasks)}</text>']
    lines.append('</svg>'); (FIGURES/"F1_calibrated_vs_magnitude.svg").write_text('\n'.join(lines),encoding='utf-8')


def write_report(summary: pd.DataFrame) -> None:
    cols=["directional_pair","source_n_tasks","target_n_tasks","spearman_source_calibrated","spearman_predicted_magnitude","delta_calibrated_minus_magnitude","bootstrap_delta_ci95_low","bootstrap_delta_ci95_high","verdict"]
    lines=['# E61｜源域校准的跨数据集风险审计','', '## 设计','', '固定等权总分在 E59 中经常不如预测幅度。E61 不在目标域选择权重：每个源数据集先按任务切成 5 折，轮流留一折，得到源域内部“模拟未见任务”的特征和误差；用这些源域 OOF 行拟合 ridge calibrator。随后用全部源数据构建目标任务的四个输入特征，直接输出预测风险。目标真实 effect 只在最后计算 `error_combined_rmse`。','', '特征为：`log1p_source_support`、`nearest_context_similarity`、`prediction_disagreement_rmse`、`predicted_l2_combined`。系数在 `tables/E61_CALIBRATOR_COEFFICIENTS.csv` 中逐方向保存。','', '## 结果','', '| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in summary.sort_values('delta_calibrated_minus_magnitude',ascending=False).iterrows():
        vals=[]
        for c in cols:
            v=r[c]; vals.append(f'{v:.3f}' if isinstance(v,(float,np.floating)) and np.isfinite(v) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['', '## 口径','', '这个实验只回答一件事：源域可见任务上学到的四项权重，换到目标数据集后，能否比单独的预测幅度更好地排序高误差任务。它不是用目标真值重新调参，也不把 oracle true magnitude 当作输入。若结果仍不稳定，就说明这四个特征的跨域可迁移性有限，应该如实作为方法边界。','', '## 文件','', '- 目标任务分数：`tables/E61_TARGET_TASK_SCORES.csv`','- 源域 OOF 校准行：`tables/E61_SOURCE_OOF_CALIBRATION_ROWS.csv`','- 各方向系数：`tables/E61_CALIBRATOR_COEFFICIENTS.csv`','- 汇总：`tables/E61_SUMMARY.csv`','- 图：`figures/F1_calibrated_vs_magnitude.svg`']
    (REPORTS/'E61_SOURCE_CALIBRATED_RISK_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'README_先看这个.md').write_text('# E61 先看这个\n\n先读 `reports/E61_SOURCE_CALIBRATED_RISK_REPORT.md`。\n\n本轮把固定等权总分改为源域 OOF 校准后的透明 ridge 分数，目标真值没有进入校准或打分。\n',encoding='utf-8')


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--n-folds',type=int,default=5); p.add_argument('--n-boot',type=int,default=2000); p.add_argument('--max-pairs',type=int,default=0); args=p.parse_args()
    for d in [TABLES,REPORTS,FIGURES]: d.mkdir(parents=True,exist_ok=True)
    outputs=[]; oofs=[]; coefs=[]; summaries=[]; statuses=[]
    plan=make_pair_plan(); plan=plan[:args.max_pairs] if args.max_pairs else plan
    for i,(group,src,tgt,n_genes) in enumerate(plan,1):
        print(f'[E61] {i}/{len(plan)} {src.name} -> {tgt.name}',flush=True)
        try:
            target,oof,coef,meta=run_pair(group,src,tgt,n_genes,args.n_folds,20260711+i,args.n_boot)
            outputs.append(target); oofs.append(oof); coefs.append(coef); summaries.append(meta['summary']); statuses.append({'directional_pair':f'{src.name} -> {tgt.name}','status':'ok','source_meta':meta['source_meta'],'target_meta':meta['target_meta']})
        except Exception as exc:
            statuses.append({'directional_pair':f'{src.name} -> {tgt.name}','status':'failed','error':repr(exc)})
            print(f'  failed: {exc!r}',flush=True)
    target_df=pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame(); oof_df=pd.concat(oofs,ignore_index=True) if oofs else pd.DataFrame(); coef_df=pd.concat(coefs,ignore_index=True) if coefs else pd.DataFrame(); summary=pd.concat(summaries,ignore_index=True) if summaries else pd.DataFrame()
    target_df.to_csv(TABLES/'E61_TARGET_TASK_SCORES.csv',index=False); oof_df.to_csv(TABLES/'E61_SOURCE_OOF_CALIBRATION_ROWS.csv',index=False); coef_df.to_csv(TABLES/'E61_CALIBRATOR_COEFFICIENTS.csv',index=False); summary.to_csv(TABLES/'E61_SUMMARY.csv',index=False); pd.DataFrame(statuses).to_json(TABLES/'E61_PAIR_STATUS.json',force_ascii=False,orient='records',indent=2)
    if not summary.empty: write_svg(summary); write_report(summary)
    status={'experiment':'E61_source_calibrated_cross_domain_risk','generated_at':now(),'git_head_before_run':git_head(),'n_planned_pairs':len(plan),'n_scored_pairs':len(summary),'n_target_task_rows':int(len(target_df)),'n_source_oof_rows':int(len(oof_df)),'n_folds':args.n_folds,'n_boot':args.n_boot,'target_truth_used_in_training_or_scoring':False,'features':FEATURES,'outputs':['tables/E61_TARGET_TASK_SCORES.csv','tables/E61_SOURCE_OOF_CALIBRATION_ROWS.csv','tables/E61_CALIBRATOR_COEFFICIENTS.csv','tables/E61_SUMMARY.csv','tables/E61_PAIR_STATUS.json','figures/F1_calibrated_vs_magnitude.svg','reports/E61_SOURCE_CALIBRATED_RISK_REPORT.md']}
    (OUT/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
