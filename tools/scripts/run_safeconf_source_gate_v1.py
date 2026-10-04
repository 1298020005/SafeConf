#!/usr/bin/env python3
"""Nested Source/Public complementarity and no-switch gate audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC, SEEDS, bootstrap_u20, fit_risk, rank_labels

RISK_CACHE = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache")
OUT_DEFAULT = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/source_gate")
GATE_CANDIDATES = ("always_public", "source_top25_unreliable", "source_top50_unreliable", "source_top75_unreliable")
SHUFFLE_SEEDS = (20260930, 20261001, 20261002, 20261003, 20261004)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def ids_hash(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def metric(part: pd.DataFrame, score: np.ndarray) -> tuple[float, float, float]:
    y = part.true_error_rmse.to_numpy(float)
    score = np.asarray(score, float)
    ids = part.task_id.astype(str).to_numpy()
    k = max(1, int(np.ceil(0.2 * len(y))))
    chosen = np.lexsort((ids, -score))[:k]
    oracle = np.lexsort((ids, -y))[:k]
    denom = y[oracle].mean() - y.mean()
    u20 = (y[chosen].mean() - y.mean()) / denom if len(y) >= 20 and denom > 1e-12 else np.nan
    low = np.lexsort((ids, score))
    aurc = float(np.mean(np.cumsum(y[low]) / np.arange(1, len(y) + 1)))
    return float(u20), aurc, float(np.nanmean(y[chosen]))


def macro_metric(frame: pd.DataFrame, score: np.ndarray) -> tuple[float, float]:
    frame = frame.copy().reset_index(drop=True)
    frame["_score_for_metric"] = np.asarray(score, float)
    vals = [metric(part, part["_score_for_metric"].to_numpy(float))[:2]
            for _, part in frame.groupby("target", sort=True)]
    arr = np.asarray(vals, float)
    return float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1]))


def gate_score(public: np.ndarray, source: np.ndarray, bad: np.ndarray,
               threshold: float | None) -> np.ndarray:
    if threshold is None:
        return public.copy()
    use_source = np.isfinite(bad) & (bad >= threshold)
    out = public.copy()
    out[use_source] = source[use_source]
    return out


def thresholds(train_bad: np.ndarray) -> dict[str, float | None]:
    finite = train_bad[np.isfinite(train_bad)]
    if not len(finite):
        return {name: None for name in GATE_CANDIDATES}
    return {
        "always_public": None,
        "source_top25_unreliable": float(np.quantile(finite, 0.75)),
        "source_top50_unreliable": float(np.quantile(finite, 0.50)),
        "source_top75_unreliable": float(np.quantile(finite, 0.25)),
    }


def inner_oof(train: pd.DataFrame, seed: int, shuffle: bool) -> tuple[pd.DataFrame, dict]:
    """Generate OOF Public/Source scores used only to choose a gate."""
    genes = sorted(train.gene.astype(str).unique(), key=lambda g: hashlib.sha256(
        f"safeconf-source-gate-inner|{seed}|{g}".encode()).hexdigest())
    assignment = {g: i % 3 for i, g in enumerate(genes)}
    chunks = []
    for held in range(3):
        fit = train[train.gene.astype(str).map(assignment).ne(held)].reset_index(drop=True)
        val = train[train.gene.astype(str).map(assignment).eq(held)].reset_index(drop=True)
        labels, _ = rank_labels(fit, f"source_gate/inner/{seed}/{held}")
        if shuffle:
            rng = np.random.default_rng(seed + held)
            labels = labels.copy()
            valid = np.isfinite(labels)
            labels[valid] = labels[valid][rng.permutation(valid.sum())]
        model = fit_risk(fit, labels, P + PUBLIC, "hgb", seed=seed, weighted=True)
        out = val[["task_id", "target", "gene", "true_error_rmse",
                   "prior_uncertainty", "prediction_prior_rmse"]].copy()
        out["public_score"] = np.sqrt(np.maximum(
            val.prediction_prior_rmse.to_numpy(float) ** 2 + val.prior_uncertainty.to_numpy(float) ** 2, 0.0))
        out["source_score"] = model.predict(val)
        out["gate_feature"] = val.prior_uncertainty.to_numpy(float)
        out["inner_fold"] = held
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True), {"seed": seed, "shuffle": shuffle,
        "n_rows": len(train), "n_genes": train.gene.nunique()}


def select_candidate(oof: pd.DataFrame, train_bad: np.ndarray) -> tuple[str, dict, pd.DataFrame]:
    th = thresholds(train_bad)
    rows = []
    for name in GATE_CANDIDATES:
        score = gate_score(oof.public_score.to_numpy(float), oof.source_score.to_numpy(float),
                           oof.gate_feature.to_numpy(float), th[name])
        u, a = macro_metric(oof, score)
        rows.append({"candidate": name, "threshold": th[name], "u20": u, "aurc": a,
                     "n_source_selected": int(np.isfinite(oof.gate_feature).sum()) if name != "always_public" else 0})
    table = pd.DataFrame(rows).sort_values(["u20", "aurc", "candidate"], ascending=[False, True, True])
    selected = str(table.iloc[0].candidate)
    return selected, th, table


def run_direction(source: str, target: str, out: Path, shuffle: bool = False,
                  shuffle_seed: int = SEEDS[0]) -> tuple[list[dict], list[dict]]:
    records, audits = [], []
    for fold in range(5):
        src_path = RISK_CACHE / f"nested_{fold}_{source}_Manual.parquet"
        tgt_path = RISK_CACHE / f"nested_{fold}_{target}_Manual.parquet"
        src = pd.read_parquet(src_path)
        tgt = pd.read_parquet(tgt_path)
        train = src[src.fold.ne(fold)].reset_index(drop=True)
        query = tgt[tgt.fold.eq(fold)].reset_index(drop=True)
        if set(train.gene.astype(str)) & set(query.gene.astype(str)):
            raise RuntimeError("Source gate crossed outer gene split")
        oof, audit = inner_oof(train, shuffle_seed, shuffle)
        selected, th, selection = select_candidate(oof, train.prior_uncertainty.to_numpy(float))
        chosen_threshold = th[selected]
        labels, cdf = rank_labels(train, f"source_gate/outer/{source}/{target}/{fold}/{shuffle}/{shuffle_seed}")
        if shuffle:
            rng = np.random.default_rng(shuffle_seed + 1000 + fold)
            valid = np.isfinite(labels)
            labels = labels.copy(); labels[valid] = labels[valid][rng.permutation(valid.sum())]
        model = fit_risk(train, labels, P + PUBLIC, "hgb", seed=shuffle_seed, weighted=True)
        public = np.sqrt(np.maximum(query.prediction_prior_rmse.to_numpy(float) ** 2 + query.prior_uncertainty.to_numpy(float) ** 2, 0.0))
        source_score = model.predict(query)
        score = gate_score(public, source_score, query.prior_uncertainty.to_numpy(float), chosen_threshold)
        for method, values in [("always_public", public), ("always_source", source_score),
                               ("selected_gate", score)]:
            part = query[["task_id", "target", "gene", "fold", "upstream", "true_error_rmse"]].copy()
            part["risk"] = values; part["method"] = method; part["source"] = source
            part["target_model"] = target; part["outer_fold"] = fold
            part["shuffle"] = shuffle; part["shuffle_seed"] = shuffle_seed
            part["selected_candidate"] = selected; part["threshold"] = chosen_threshold
            records.extend(part.to_dict("records"))
        audits.append({"source": source, "target": target, "outer_fold": fold,
                       "shuffle": shuffle, "shuffle_seed": shuffle_seed,
                       "selected_candidate": selected, "threshold": chosen_threshold,
                       "training_rows": len(train), "training_genes": train.gene.nunique(),
                       "query_rows": len(query), "query_genes": query.gene.nunique(),
                       "train_id_hash": ids_hash(train.task_id),
                       "query_id_hash": ids_hash(query.task_id),
                       "selection_table": selection.to_dict("records"),
                       "cdf_groups": len(cdf)})
    return records, audits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "preflight"], default="dev")
    ap.add_argument("--source-cache", type=Path, default=RISK_CACHE)
    ap.add_argument("--reliability-root", type=Path)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.phase == "preflight":
        write_json(out / "SOURCE_GATE_PREFLIGHT.json", {
            "status": "PASS", "gate_candidates": GATE_CANDIDATES,
            "source_cache": str(args.source_cache), "no_holdout_truth_opened": True,
            "learner": "fixed HGB P+PUBLIC", "shuffle_seeds": SHUFFLE_SEEDS,
        })
        print(json.dumps({"status": "PASS", "output": str(out)})); return 0
    all_records, all_audits = [], []
    directions = [("TxPert_GAT", "TxPert_Exphormer"), ("TxPert_Exphormer", "TxPert_GAT")]
    for source, target in directions:
        rec, aud = run_direction(source, target, out, False, SEEDS[0])
        all_records.extend(rec); all_audits.extend(aud)
        for ss in SHUFFLE_SEEDS:
            rec, aud = run_direction(source, target, out, True, ss)
            all_records.extend(rec); all_audits.extend(aud)
    pred = pd.DataFrame(all_records)
    write_csv(out / "SOURCE_GATE_TASK_PREDICTIONS.csv", pred)
    write_csv(out / "SOURCE_GATE_FIT_AUDIT.csv", pd.DataFrame(all_audits))
    metrics = []
    for keys, part in pred.groupby(["source", "target_model", "shuffle", "shuffle_seed", "method"], sort=True):
        u, a = macro_metric(part, part.risk.to_numpy(float))
        metrics.append({"source": keys[0], "target": keys[1], "shuffle": keys[2],
                        "shuffle_seed": keys[3], "method": keys[4], "u20": u, "aurc": a,
                        "rows": len(part), "genes": part.gene.nunique()})
    metrics = pd.DataFrame(metrics)
    write_csv(out / "SOURCE_GATE_METRICS.csv", metrics)
    paired = []
    for (source, target), part in pred[pred.shuffle.eq(False)].groupby(["source", "target_model"], sort=True):
        base = part[part.method.eq("always_public")].drop_duplicates("task_id")
        gate = part[part.method.eq("selected_gate")].drop_duplicates("task_id")
        source_score = part[part.method.eq("always_source")].drop_duplicates("task_id")
        if len(base) != len(gate) or len(base) != len(source_score):
            raise RuntimeError("real source gate task coverage differs")
        wide = base[["task_id", "target", "gene", "true_error_rmse", "risk"]].copy()
        wide = wide.merge(gate[["task_id", "risk"]].rename(columns={"risk": "gate_risk"}), on="task_id", validate="one_to_one")
        wide = wide.merge(source_score[["task_id", "risk"]].rename(columns={"risk": "source_risk"}), on="task_id", validate="one_to_one")
        for method, values in [("selected_gate", wide.gate_risk.to_numpy(float)), ("always_source", wide.source_risk.to_numpy(float))]:
            result = bootstrap_u20(wide, values, wide.risk.to_numpy(float), 5000, SEEDS[0])
            paired.append({"source": source, "target": target, "method": method,
                           "comparison": "vs_always_public", **result})
    write_csv(out / "SOURCE_GATE_PAIRED_BOOTSTRAP.csv", pd.DataFrame(paired))
    write_json(out / "SOURCE_GATE_RUN_STATUS.json", {
        "status": "COMPLETE", "directions": directions,
        "candidate_set": GATE_CANDIDATES, "shuffle_seeds": SHUFFLE_SEEDS,
        "rows": len(pred), "permanent_test_truth_opened": False,
        "learner": "HistGradientBoostingRegressor fixed P+PUBLIC",
        "gate_feature": "prior_uncertainty",
        "gate_feature_reason": "source nested cache has no experiment-level history vectors for PublicMeanJackknife; E258 J is retained as a validation-only reliability result",
        "reliability_root": str(args.reliability_root) if args.reliability_root else None,
        "bootstrap_replicates": 5000,
    })
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
