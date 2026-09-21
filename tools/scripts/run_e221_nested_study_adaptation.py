#!/usr/bin/env python3
"""Leave-one-study-out selection of a finite, non-fitted score dictionary.

The adaptive candidate is selected only inside the outer training studies.  The
outer study is never used to choose weights.  This is a retrospective development
audit and does not replace the frozen E208 confirmation protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables/E153_ABSOLUTE_TASK_INPUT.csv"
DEFAULT_OUTPUT = ROOT / "docs/实验结果/E221_nested_study_adaptation_20260921"
INPUT_SHA256 = "b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a"
BUDGET = 0.20
SEED = 20260921


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if len(values) < 4 or not np.isfinite(values).all():
        raise ContractError("invalid ranking block")
    return rankdata(values, method="average") / len(values)


def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for _, block in frame.groupby(["dataset", "fold_id"], sort=True):
        block = block.copy()
        block["m"] = percentile(block.baseline_predicted_magnitude.to_numpy())
        block["s"] = percentile(block.safeconf_calibrated_pair_risk.to_numpy())
        block["d"] = percentile(block.risk_model_disagreement.to_numpy())
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def load(path: Path) -> pd.DataFrame:
    if not path.is_file() or sha256(path) != INPUT_SHA256:
        raise ContractError("E153 input is missing or has changed")
    frame = pd.read_csv(path)
    required = {"dataset", "fold_id", "task_id", "error_two_predictor_mean_rmse",
                "safeconf_calibrated_pair_risk", "risk_model_disagreement",
                "baseline_predicted_magnitude"}
    if not required.issubset(frame.columns) or len(frame) != 3465:
        raise ContractError("E153 schema or row count changed")
    if frame.duplicated(["dataset", "fold_id", "task_id"]).any() or frame.dataset.nunique() != 8:
        raise ContractError("task/study identity changed")
    return add_ranks(frame)


def simplex_candidates() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    for wm in grid:
        for ws in grid:
            wd = 1.0 - wm - ws
            if wd in grid:
                out.append((f"simplex_M{wm:.2f}_S{ws:.2f}_D{wd:.2f}", np.array([wm, ws, wd])))
    return out


def candidates() -> list[tuple[str, tuple]]:
    result: list[tuple[str, tuple]] = [
        ("magnitude_percentile", ("linear", np.array([1.0, 0.0, 0.0]))),
        ("safeconf_percentile", ("linear", np.array([0.0, 1.0, 0.0]))),
        ("disagreement_percentile", ("linear", np.array([0.0, 0.0, 1.0]))),
        ("fixed_80m_20s", ("linear", np.array([0.8, 0.2, 0.0]))),
    ]
    result.extend((name, ("linear", w)) for name, w in simplex_candidates())
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    for ls in grid:
        for ld in grid:
            result.append((f"one_sided_S{ls:.2f}_D{ld:.2f}", ("one_sided", np.array([ls, ld]))))
    # stable order and unique labels
    seen = set()
    unique = []
    for item in result:
        if item[0] not in seen:
            unique.append(item)
            seen.add(item[0])
    return unique


def score(block: pd.DataFrame, spec: tuple) -> np.ndarray:
    kind, weights = spec
    m, s, d = (block[c].to_numpy(float) for c in ("m", "s", "d"))
    if kind == "linear":
        return weights[0] * m + weights[1] * s + weights[2] * d
    if kind == "one_sided":
        return m + weights[0] * np.maximum(s - m, 0) + weights[1] * np.maximum(d - m, 0)
    raise ContractError(f"unknown candidate kind: {kind}")


def metrics(score_values: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    score_values, outcome = np.asarray(score_values, float), np.asarray(outcome, float)
    if len(score_values) < 4 or not np.isfinite(score_values).all() or not np.isfinite(outcome).all():
        raise ContractError("invalid metric input")
    rs, ry = rankdata(score_values, method="average"), rankdata(outcome, method="average")
    rho = float(np.corrcoef(rs, ry)[0, 1]) if np.std(rs) > 0 and np.std(ry) > 0 else float("nan")
    k = max(1, math.ceil(BUDGET * len(outcome)))
    selected = np.argsort(-score_values, kind="mergesort")[:k]
    oracle = np.argsort(-outcome, kind="mergesort")[:k]
    mean = float(outcome.mean())
    denom = float(outcome[oracle].mean()) - mean
    utility = float((outcome[selected].mean() - mean) / denom) if denom > 1e-15 else float("nan")
    capture = float(outcome[selected].sum() / outcome.sum()) if outcome.sum() > 0 else float("nan")
    remain = float((outcome.sum() - outcome[selected].sum()) / max(1, len(outcome) - k))
    return {"spearman": rho, "utility_20": utility, "capture_20": capture, "remaining_error_80": remain}


def fold_metrics(frame: pd.DataFrame, candidate: tuple[str, tuple]) -> dict[str, float]:
    rows = []
    for _, block in frame.groupby("fold_id", sort=True):
        rows.append(metrics(score(block, candidate[1]), block.error_two_predictor_mean_rmse.to_numpy()))
    averaged = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], float)
        finite = values[np.isfinite(values)]
        averaged[key] = float(finite.mean()) if len(finite) else float("nan")
    return averaged


def select_candidate(training: pd.DataFrame) -> tuple[str, tuple, pd.DataFrame]:
    rows = []
    candidate_list = candidates()
    studies = sorted(training.dataset.unique())
    for heldout in studies:
        inner_test = training.loc[training.dataset.eq(heldout)]
        for name, spec in candidate_list:
            values = fold_metrics(inner_test, (name, spec))
            rows.append({"inner_heldout_study": heldout, "candidate": name, **values})
    table = pd.DataFrame(rows)
    # These scores have no fitted parameters: this is study-balanced development
    # selection, not six-study fitting followed by one-study validation.
    # Preserve the initial deterministic tie rule for reproducibility.
    ranking = []
    for name, spec in candidate_list:
        subset = table.loc[table.candidate.eq(name)]
        active = int(np.sum(np.asarray(spec[1]) != 0))
        ranking.append((float(subset.utility_20.mean()), float(subset.spearman.mean()), -active, name, spec))
    chosen = max(ranking, key=lambda row: (row[0], row[1], row[2], tuple(reversed(row[3]))))
    return chosen[3], (chosen[4][0], chosen[4][1]), table


def evaluate_outer(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, inner_rows, selections = [], [], []
    for heldout in sorted(frame.dataset.unique()):
        train = frame.loc[frame.dataset.ne(heldout)]
        test = frame.loc[frame.dataset.eq(heldout)].copy()
        name, spec, inner = select_candidate(train)
        inner["outer_heldout_study"] = heldout
        inner_rows.append(inner)
        selections.append({"outer_heldout_study": heldout, "selected_candidate": name,
                           "selected_spec": repr(spec), "n_train_tasks": len(train), "n_test_tasks": len(test)})
        methods = [("adaptive_nested", spec), ("magnitude_percentile", candidates()[0][1]),
                   ("fixed_80m_20s", candidates()[3][1]), ("safeconf_percentile", candidates()[1][1]),
                   ("disagreement_percentile", candidates()[2][1])]
        for method, candidate in methods:
            for fold, block in test.groupby("fold_id", sort=True):
                rows.append({"heldout_study": heldout, "fold_id": fold, "method": method,
                             "n_tasks": len(block), **metrics(score(block, candidate), block.error_two_predictor_mean_rmse.to_numpy())})
    return pd.DataFrame(rows), pd.concat(inner_rows, ignore_index=True), pd.DataFrame(selections)


def bootstrap(results: pd.DataFrame, n: int) -> pd.DataFrame:
    study_rows = results.groupby(["heldout_study", "method"], sort=True)[["spearman", "utility_20", "capture_20", "remaining_error_80"]].mean().reset_index()
    baseline = study_rows.loc[study_rows.method.eq("magnitude_percentile")].set_index("heldout_study")
    rows = []
    rng = np.random.default_rng(SEED)
    studies = sorted(baseline.index)
    for method in sorted(study_rows.method.unique()):
        block = study_rows.loc[study_rows.method.eq(method)].set_index("heldout_study").loc[studies]
        deltas = block[["spearman", "utility_20", "capture_20", "remaining_error_80"]].to_numpy() - baseline[["spearman", "utility_20", "capture_20", "remaining_error_80"]].to_numpy()
        draws = np.empty((n, deltas.shape[1]))
        for i in range(n):
            draws[i] = deltas[rng.integers(0, len(studies), len(studies))].mean(axis=0)
        for j, metric_name in enumerate(["spearman", "utility_20", "capture_20", "remaining_error_80"]):
            point = float(deltas[:, j].mean())
            rows.append({"method": method, "metric": metric_name, "delta_vs_magnitude": point,
                         "ci95_lower": float(np.quantile(draws[:, j], .025)),
                         "ci95_upper": float(np.quantile(draws[:, j], .975)),
                         "positive_studies": int((deltas[:, j] > 0).sum()), "n_studies": len(studies)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("use at least 1000 study bootstrap draws")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise ContractError(f"refusing to overwrite non-empty {out}")
    frame = load(args.input.resolve())
    results, inner, selections = evaluate_outer(frame)
    intervals = bootstrap(results, args.bootstrap)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "E221_OUTER_FOLD_RESULTS.csv", index=False)
    inner.to_csv(out / "E221_INNER_SELECTION_RESULTS.csv", index=False)
    selections.to_csv(out / "E221_OUTER_SELECTIONS.csv", index=False)
    intervals.to_csv(out / "E221_STUDY_BOOTSTRAP_INTERVALS.csv", index=False)
    adaptive = intervals.loc[intervals.method.eq("adaptive_nested")].set_index("metric")
    status = {"experiment": "E221_nested_study_adaptation", "status": "COMPLETE",
              "evidence_class": "retrospective_leave_one_study_out_dictionary_selection",
              "inner_model_fitting": False,
              "script_sha256": sha256(Path(__file__)),
              "input_sha256": sha256(args.input.resolve()), "n_tasks": len(frame),
              "n_studies": int(frame.dataset.nunique()), "n_folds": int(frame.groupby(["dataset", "fold_id"]).ngroups),
              "n_candidates": len(candidates()), "bootstrap": args.bootstrap,
              "adaptive_delta_spearman": float(adaptive.loc["spearman", "delta_vs_magnitude"]),
              "adaptive_delta_utility20": float(adaptive.loc["utility_20", "delta_vs_magnitude"]),
              "adaptive_utility20_ci95": [float(adaptive.loc["utility_20", "ci95_lower"]), float(adaptive.loc["utility_20", "ci95_upper"])],
              "outer_truth_used_for_selection": False, "source_data_modified": False,
              "all_candidates_retained": True, "seed": SEED}
    (out / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# E221｜留一研究外推的自适应风险排序\n", "本实验使用 E153 八个公开研究。每次整项留出一个研究，公式只在其余研究的内层留一研究结果上选择；外层研究的真实误差没有参与选参。\n", "## 当前结果\n", f"- 候选公式：{len(candidates())} 个，外层研究：{frame.dataset.nunique()} 个，任务：{len(frame)} 个。\n", f"- 自适应分数相对同口径幅度的平均 ΔSpearman：{status['adaptive_delta_spearman']:+.5f}。\n", f"- 平均 Δutility@20%：{status['adaptive_delta_utility20']:+.5f}，研究簇 bootstrap 95% 区间 [{status['adaptive_utility20_ci95'][0]:+.5f}, {status['adaptive_utility20_ci95'][1]:+.5f}]。\n", "\n## 证据解释\n", "这里的自适应分数需要在开发期看到带标签的历史误差；它不能被改名为无需标签的 SafeConf-M。若区间跨 0，只能说明目前八个研究不足以支持稳定的实际复核收益；若区间为正，也必须冻结选择规则后在新数据上再次确认。\n", "所有候选、每个外层选择、负结果和每个 fold 均已输出；没有删除研究、任务或真实误差。\n"]
    (out / "E221_REPORT.md").write_text("".join(report), encoding="utf-8")
    with (out / "E221_REPORT.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## 算法名称与区间说明（审计补充）\n\n"
                     "本轮是固定公式字典的留一研究选参；44 个条目含等价表达，并非 44 个独立算法。"
                     "候选公式无需拟合，所谓内层实际是七个开发研究等权打分，没有六研究训练步骤。"
                     "文件名保留以便追溯，不能描述成训练了 44 个评分模型。真正的嵌套评分模型训练另见 E222。\n"
                     "输入仅移除了从未被选参或打分使用的误差秩缓存列，原指标和选择结果不变。"
                     "研究重采样固定外层预测，不包含重叠训练研究引起的全部选参不确定性。"
                     "这是反复使用公开开发数据后的探索，不是新的独立确证。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
