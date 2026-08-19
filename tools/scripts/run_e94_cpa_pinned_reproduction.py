#!/usr/bin/env python3
"""Reproduce the frozen E83 pilot in CPA's pinned dependency environment."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = ROOT / "tools/scripts/run_e83_cpa_rdkit_pilot.py"
ORIGINAL = ROOT / "docs/实验结果/E83_cpa_rdkit_pilot_20260712"
OUT = ROOT / "docs/实验结果/E94_cpa_pinned_reproduction_20260712"


def main() -> None:
    spec = importlib.util.spec_from_file_location("e83_frozen", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.OUT = OUT
    # PredictionRecord v1 currently permits only formal/smoke; this is a
    # non-independent environment check and therefore remains smoke.
    module.RUN_TYPE = "smoke"
    args = SimpleNamespace(
        manifest_id="E81_r1_p75",
        seed=20260712,
        epochs=10,
        max_cells=32,
        control_cells=64,
        pseudo_cells=16,
        batch_size=64,
        eval_batch_size=128,
        device=0,
    )
    if not (OUT / "PREDICT_STATUS.json").exists():
        module.predict(args)
    if not (OUT / "tables/E83_TASK_SCORES.csv").exists():
        module.evaluate(args)

    old = pd.read_csv(ORIGINAL / "tables/E83_TASK_SCORES.csv").sort_values("task_key")
    new = pd.read_csv(OUT / "tables/E83_TASK_SCORES.csv").sort_values("task_key")
    if old["task_key"].tolist() != new["task_key"].tolist():
        raise RuntimeError("Pinned reproduction task keys changed")
    comparison = {
        "n_tasks": len(new),
        "rho_disagreement_old_vs_pinned": float(stats.spearmanr(old["cpa_ridge_disagreement_rmse"], new["cpa_ridge_disagreement_rmse"]).statistic),
        "rho_pair_mean_error_old_vs_pinned": float(stats.spearmanr(old["pair_mean_rmse"], new["pair_mean_rmse"]).statistic),
        "rho_pinned_disagreement_vs_pair_mean_error": float(stats.spearmanr(new["cpa_ridge_disagreement_rmse"], new["pair_mean_rmse"]).statistic),
        "rho_original_disagreement_vs_pair_mean_error": float(stats.spearmanr(old["cpa_ridge_disagreement_rmse"], old["pair_mean_rmse"]).statistic),
        "median_abs_change_disagreement": float(np.median(np.abs(old["cpa_ridge_disagreement_rmse"].to_numpy() - new["cpa_ridge_disagreement_rmse"].to_numpy()))),
        "median_abs_change_pair_mean_error": float(np.median(np.abs(old["pair_mean_rmse"].to_numpy() - new["pair_mean_rmse"].to_numpy()))),
    }
    pd.DataFrame([comparison]).to_csv(OUT / "tables/E94_ENVIRONMENT_COMPARISON.csv", index=False)
    status = json.loads((OUT / "RUN_STATUS.json").read_text())
    status.update(
        {
            "experiment": "E94_cpa_pinned_reproduction",
            "phase": "reproduction_complete",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_experiment": "E83_cpa_rdkit_pilot_20260712",
            "environment": "/home/yyf/.conda/envs/cpa_env",
            "torch": "2.0.0+cu117",
            "scvi_tools": "0.20.3",
            "scanpy": "1.10.3",
            "anndata": "0.9.2",
            "ray": "2.9.3",
            "pyarrow": "14.0.2",
            "hyperparameters_changed": False,
            **comparison,
        }
    )
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = f"""# E94｜CPA pinned 环境复现

E83 的 manifest、种子、细胞抽样、网络、剂量变换和 10 epochs 全部保持不变，只把运行环境从兼容环境切换到 CPA 0.8.8 官方依赖组合。该测试集已在 E83 查看，因此 E94 只检查环境再现性，不增加新的独立证据。

- tasks：{comparison['n_tasks']}
- disagreement 排序 old vs pinned：ρ={comparison['rho_disagreement_old_vs_pinned']:.3f}
- pair-mean error 排序 old vs pinned：ρ={comparison['rho_pair_mean_error_old_vs_pinned']:.3f}
- pinned 内 disagreement–error：ρ={comparison['rho_pinned_disagreement_vs_pair_mean_error']:.3f}
- original 内 disagreement–error：ρ={comparison['rho_original_disagreement_vs_pair_mean_error']:.3f}
- disagreement 中位绝对变化：{comparison['median_abs_change_disagreement']:.6f}
- pair-mean error 中位绝对变化：{comparison['median_abs_change_pair_mean_error']:.6f}
"""
    (OUT / "reports/E94_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E94 先看这个\n\n先读 `reports/E94_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
