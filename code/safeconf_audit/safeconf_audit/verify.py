#!/usr/bin/env python3
"""One-command verification of the frozen SafeConf numbers (E199 / E200; E201 hook).

The point estimates below are recomputed independently from the committed
task-level tables and compared against the frozen report tables. Structural
invariants (task counts, identity residual, violation counts) are exact checks.
Cluster-bootstrap CIs are recomputed informationally: RNG seeds of the original
runs are not part of the freeze, so CI checks are sign-consistency checks
(whether the interval excludes zero), never numeric-equality checks.

Usage:
    python -m safeconf_audit.verify --repo /path/to/SafeConf
    safeconf-audit --repo /path/to/SafeConf   (after pip install -e .)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

E199_TABLE = Path("docs/实验结果/E199_txpert_public_k562_20260802/formal_evaluation/tables")
E200_TABLE = Path("docs/实验结果/E200_txpert_cross_context_k562_20260802/formal_evaluation/tables")
E201_DIR = Path("docs/实验结果/E201_txpert_multitarget_retraining_20260802")

POINT_TOL = 5e-4
BOOT_N = 5000
BOOT_SEED = 20260817


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def add(self, name: str, expected: str, got, ok: bool) -> None:
        self.rows.append((name, expected, f"{got}", bool(ok)))

    def numeric(self, name: str, expected: float, got: float, tol: float = POINT_TOL) -> None:
        self.add(name, f"{expected:.4f}", f"{got:.4f}", abs(got - expected) <= tol)

    def report(self, title: str) -> int:
        print(f"\n== {title} ==")
        fails = 0
        for name, expected, got, ok in self.rows:
            mark = "PASS" if ok else "FAIL"
            fails += (not ok)
            print(f"[{mark}] {name}: expected {expected} | recomputed {got}")
        return fails


def cluster_bootstrap_ci(x, y, clusters, n_boot: int = BOOT_N, seed: int = BOOT_SEED):
    """Spearman CI by resampling whole clusters (perturbation conditions).

    Clusters are pre-grouped into numpy arrays once, so each draw is a cheap
    concatenate instead of a frame filter.
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(clusters)
    groups = [np.flatnonzero(c == u) for u in np.unique(c)]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stats = np.empty(n_boot)
    n_valid = 0
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        bx, by = x[idx], y[idx]
        if np.unique(bx).size < 2 or np.unique(by).size < 2:
            continue
        stats[n_valid] = spearmanr(bx, by).statistic
        n_valid += 1
    if n_valid == 0:
        return (float("nan"), float("nan"), 0)
    lo, hi = np.percentile(stats[:n_valid], [2.5, 97.5])
    return (float(lo), float(hi), int(n_valid))


def frozen_rho(table: pd.DataFrame, predictor: str, outcome: str) -> float:
    row = table[(table.predictor == predictor) & (table.outcome == outcome)]
    if len(row) != 1:
        raise ValueError(f"frozen row not unique for {predictor}/{outcome}")
    return float(row.spearman.iloc[0])


def verify_e199(repo: Path, check: Check) -> None:
    t = pd.read_csv(repo / E199_TABLE / "E199_TASK_CERTIFICATE.csv")
    frozen = pd.read_csv(repo / E199_TABLE / "E199_RISK_ASSOCIATIONS.csv")
    main = t[t.n_cells >= 30]
    check.add("E199 main tasks (n_cells>=30)", "263", len(main), len(main) == 263)
    for col, frozen_name in (
        ("diversity_lower_bound", "diversity_lower_bound"),
        ("predicted_magnitude", "predicted_magnitude"),
    ):
        got = float(spearmanr(main[col], main.family_rms_error).statistic)
        check.numeric(f"E199 ρ({frozen_name}, family_rms_error)",
                      frozen_rho(frozen, frozen_name, "family_rms_error"), got)
    resid_max = float(t.rms_identity_residual.abs().max())
    check.add("E199 identity residual max ≤ 1e-10", "≤1e-10", f"{resid_max:.2e}", resid_max <= 1e-10)
    check.add("E199 lower-bound violations", "0",
              int(t.family_rms_lower_violation.sum() + t.family_worst_lower_violation.sum()),
              (t.family_rms_lower_violation.sum() + t.family_worst_lower_violation.sum()) == 0)
    lo, hi, n = cluster_bootstrap_ci(main.diversity_lower_bound, main.family_rms_error, main.condition_label)
    check.add("E199 disagreement CI excludes 0 (sign-consistent)", "yes",
              f"[{lo:.3f},{hi:.3f}] n={n}", lo > 0)
    lo2, hi2, _ = cluster_bootstrap_ci(main.predicted_magnitude, main.family_rms_error, main.condition_label)
    frozen_ci = frozen[(frozen.predictor == "predicted_magnitude") & (frozen.outcome == "family_rms_error")]
    frozen_crosses = float(frozen_ci.ci95_lower.iloc[0]) < 0
    mine_crosses = lo2 < 0
    check.add("E199 magnitude CI sign-consistency (crosses 0)", "crosses" if frozen_crosses else "excludes",
              "crosses" if mine_crosses else "excludes", frozen_crosses == mine_crosses)


def verify_e200(repo: Path, check: Check) -> None:
    t = pd.read_csv(repo / E200_TABLE / "E200_TASK_METRICS.csv")
    frozen = pd.read_csv(repo / E200_TABLE / "E200_RISK_ASSOCIATIONS.csv")
    main = t[t.analysis_stratum == "primary_ge30"]
    check.add("E200 primary tasks", "566", len(main), len(main) == 566)
    for col, name in (("transfer_risk", "transfer_risk"),
                      ("predicted_magnitude", "predicted_magnitude"),
                      ("training_delta_dispersion", "training_delta_dispersion")):
        got = float(spearmanr(main[col], main.gat_centroid_rmse).statistic)
        check.numeric(f"E200 ρ({name}, gat_centroid_rmse)",
                      frozen_rho(frozen, name, "gat_centroid_rmse"), got)
    lo, hi, n = cluster_bootstrap_ci(main.transfer_risk, main.gat_centroid_rmse, main.condition_label)
    check.add("E200 transfer-risk CI excludes 0 (sign-consistent)", "yes",
              f"[{lo:.3f},{hi:.3f}] n={n}", lo > 0)


def verify_e201(repo: Path, check: Check) -> None:
    status = repo / E201_DIR / "formal_core_evaluation" / "E201_CORE_FINAL_STATUS.json"
    if status.is_file():
        check.add("E201 core evaluation status", "released", "released", True)
    else:
        check.add("E201 core evaluation status", "pending(blind)", "pending(blind)", True)
        print("[NOTE] E201 sealed evaluation not yet released; nothing to verify.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args(argv)
    repo = args.repo.resolve()
    total = 0
    for title, fn in (("E199 unseen-gene audit (K562, public TxPert STRING-GAT)", verify_e199),
                      ("E200 full-context holdout audit (K562)", verify_e200),
                      ("E201 four-context blind audit", verify_e201)):
        check = Check()
        try:
            fn(repo, check)
        except Exception as exc:  # surface, never silently pass
            check.add(f"{title} runner", "no exception", f"{type(exc).__name__}: {exc}", False)
        total += check.report(title)
    print(f"\nTOTAL: {'ALL PASS' if total == 0 else f'{total} FAIL'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
