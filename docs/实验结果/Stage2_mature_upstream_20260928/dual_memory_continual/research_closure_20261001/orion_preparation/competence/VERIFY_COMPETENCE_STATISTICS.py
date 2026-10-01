#!/usr/bin/env python3
"""Verify Orion competence statistics using invented task RMSE summaries only.

No expression, predictions, source vectors, or SafeConf scores are loaded. The
only imported project API is paired_competence; its CLI is never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXTS = ("HCT116", "HEK293T")
REPLICATES = 5000
SEED = 20260929
TOLERANCE = 2e-14


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: object, expected: object, message: str) -> None:
    if not np.allclose(actual, expected, rtol=TOLERANCE, atol=TOLERANCE):
        raise AssertionError(f"{message}: actual={actual!r}, expected={expected!r}")


def input_frame(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["context_id", "target_gene_id", "model_rmse", "baseline_rmse"]
    )


def constant_contexts(first: float, second: float, baseline: float = 1.0) -> pd.DataFrame:
    return input_frame([
        (context, f"synthetic_gene_{gene:03d}", model, baseline)
        for context, model in zip(CONTEXTS, (first, second))
        for gene in range(4)
    ])


def independent_oracle(tasks: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Recompute the declared estimand directly from task rows and draw weights."""
    genes = sorted(tasks.target_gene_id.unique())
    gene_index = {gene: index for index, gene in enumerate(genes)}
    groups = [tasks.loc[tasks.context_id == context] for context in CONTEXTS]
    model_context = np.array([group.model_rmse.mean() for group in groups])
    baseline_context = np.array([group.baseline_rmse.mean() for group in groups])
    model_macro = model_context.mean()
    baseline_macro = baseline_context.mean()
    require(baseline_macro > 0, "oracle needs a positive baseline macro")
    noninferior = model_context <= 1.02 * baseline_context
    indices = [group.target_gene_id.map(gene_index).to_numpy(int) for group in groups]
    errors = [group[["model_rmse", "baseline_rmse"]].to_numpy(float) for group in groups]
    rng = np.random.default_rng(SEED)
    records = []
    for replicate in range(REPLICATES):
        draw = rng.integers(0, len(genes), size=len(genes))
        multiplicities = np.bincount(draw, minlength=len(genes))
        means = []
        for task_indices, task_errors in zip(indices, errors):
            task_weights = multiplicities[task_indices]
            denominator = int(task_weights.sum())
            require(denominator > 0, "oracle fixture unexpectedly has an empty context draw")
            # Expand integer task weights instead of sharing evaluator aggregation code.
            expanded = np.repeat(task_errors, task_weights, axis=0)
            means.append(expanded.mean(axis=0))
        macro_model, macro_baseline = np.mean(means, axis=0)
        require(macro_baseline > 0, "oracle fixture has a nonpositive draw baseline")
        ratio = macro_model / macro_baseline
        draw_hash = hashlib.sha256(draw.astype("<i8").tobytes()).hexdigest()
        records.append((replicate, macro_model, macro_baseline, ratio, draw_hash))
    draws = pd.DataFrame(records, columns=[
        "replicate", "macro_model_rmse", "macro_baseline_rmse", "ratio", "gene_draw_sha256"
    ])
    lower_ratio, upper_ratio = np.quantile(draws.ratio, [.025, .975], method="linear")
    stable = bool(lower_ratio > 1.02)
    expected = {
        "model_macro_rmse": float(model_macro),
        "baseline_macro_rmse": float(baseline_macro),
        "relative_macro_error_gap": float(model_macro / baseline_macro - 1),
        "noninferior_strata_fraction": float(noninferior.mean()),
        "noninferior_contexts": int(noninferior.sum()),
        "actual_context_count": 2,
        "lower95_relative_gap": float(lower_ratio - 1),
        "upper95_relative_gap": float(upper_ratio - 1),
        "stable_disadvantage": stable,
        "passes_competence": bool(
            model_macro <= 1.02 * baseline_macro
            and noninferior.mean() >= .6
            and not stable
        ),
    }
    return expected, draws


def csv_digest(frame: pd.DataFrame) -> str:
    encoded = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("synthetic_orion_competence_evaluator", path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(callable(getattr(module, "paired_competence", None)), "paired_competence API missing")
    return module.paired_competence


def default_evaluator() -> Path:
    for ancestor in Path(__file__).resolve().parents:
        script_directory = ancestor / "tools" / "scripts"
        if script_directory.is_dir():
            return script_directory / "evaluate_safeconf_orion_competence_agent.py"
    raise RuntimeError("cannot locate repository tools/scripts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path)
    args = parser.parse_args()
    evaluator_path = (args.evaluator or default_evaluator()).resolve()
    require(evaluator_path.is_file(), f"evaluator does not exist: {evaluator_path}")
    require(not args.output.exists(), f"refusing to overwrite output: {args.output}")
    evaluate = load_evaluator(evaluator_path)
    results = []
    fixtures = []
    examples = []

    def valid_case(name: str, tasks: pd.DataFrame):
        expected, oracle_draws = independent_oracle(tasks)
        summary, context_rows, actual_draws = evaluate(tasks.copy(deep=True))
        require(len(actual_draws) == REPLICATES, f"{name}: wrong draw count")
        require(np.isfinite(actual_draws.ratio).all(), f"{name}: nonfinite ratios")
        require(actual_draws.replicate.is_unique, f"{name}: repeated replicate identifiers")
        require(np.all(np.diff(actual_draws.replicate) == 1), f"{name}: replicate order")
        for column in ("macro_model_rmse", "macro_baseline_rmse", "ratio"):
            close(actual_draws[column], oracle_draws[column], f"{name}: all draws {column}")
        require(actual_draws.gene_draw_sha256.tolist() == oracle_draws.gene_draw_sha256.tolist(), f"{name}: seeded gene draw hashes")
        close(actual_draws.relative_gap, actual_draws.ratio - 1, f"{name}: draw gap")
        for key, value in expected.items():
            require(key in summary, f"{name}: missing summary field {key}")
            if isinstance(value, (bool, int)):
                require(summary[key] == value, f"{name}: {key} differs from independent oracle")
            else:
                close(summary[key], value, f"{name}: {key}")
        require(set(context_rows.context_id) == set(CONTEXTS), f"{name}: contexts changed")
        require(len(context_rows) == 2, f"{name}: duplicate context output")
        for context in CONTEXTS:
            source = tasks.loc[tasks.context_id == context]
            row = context_rows.loc[context_rows.context_id == context].iloc[0]
            require(row.n_tasks == len(source), f"{name}: {context} task count")
            close(row.model_macro_rmse, source.model_rmse.mean(), f"{name}: {context} model")
            close(row.baseline_macro_rmse, source.baseline_rmse.mean(), f"{name}: {context} baseline")
            require(bool(row.noninferior) == bool(source.model_rmse.mean() <= 1.02 * source.baseline_rmse.mean()), f"{name}: {context} noninferiority")
        results.append({"case": name, "status": "PASS", **expected})
        fixtures.append(tasks.assign(case=name))
        return summary, actual_draws

    # Unequal context sizes must not let the larger context dominate the endpoint.
    unequal = input_frame([
        (context, f"synthetic_gene_{gene:03d}", model, 1.0)
        for context, count, model in ((CONTEXTS[0], 120, 1.0), (CONTEXTS[1], 100, 1.10))
        for gene in range(count)
    ])
    summary, _ = valid_case("equal_context_weighting", unequal)
    close(summary["relative_macro_error_gap"], .05, "equal context gap")
    pooled_gap = unequal.model_rmse.mean() / unequal.baseline_rmse.mean() - 1
    require(abs(pooled_gap - .05) > .001, "unequal-count fixture must distinguish task pooling")
    examples.append({"case": "equal_context_weighting", "pooled_gap": float(pooled_gap), "correct_gap": .05})

    ratio_fixture = input_frame([
        (context, f"synthetic_gene_{gene:03d}", model, baseline)
        for context, model, baseline in ((CONTEXTS[0], 1.0, 1.0), (CONTEXTS[1], 101.0, 100.0))
        for gene in range(4)
    ])
    summary, _ = valid_case("ratio_of_context_macros", ratio_fixture)
    close(summary["relative_macro_error_gap"], 1 / 101, "ratio of macros")
    require(abs(summary["relative_macro_error_gap"] - .005) > .001, "fixture must reject mean of context relative gaps")

    cancellation = input_frame([
        (CONTEXTS[0], "synthetic_gene_000", .8, 1.0),
        (CONTEXTS[0], "synthetic_gene_001", 1.2, 1.0),
        (CONTEXTS[1], "synthetic_gene_000", 1.2, 1.0),
        (CONTEXTS[1], "synthetic_gene_001", .8, 1.0),
    ])
    summary, draws = valid_case("paired_gene_cancellation", cancellation)
    close(draws.ratio, np.ones(REPLICATES), "shared gene multiplicities must cancel")
    close([summary["lower95_relative_gap"], summary["upper95_relative_gap"]], [0, 0], "paired cancellation CI")

    summary, _ = valid_case("exact_two_percent_boundary", constant_contexts(1.02, 1.02))
    require(bool(summary["passes_competence"]), "exact 2% margin must pass")
    require(not summary["stable_disadvantage"], "exact lower-bound margin is not stable disadvantage")

    summary, _ = valid_case("both_contexts_required", constant_contexts(1.0, 1.03))
    close(summary["relative_macro_error_gap"], .015, "point macro should pass in 2/2 counterexample")
    require(summary["noninferior_contexts"] == 1 and not summary["passes_competence"], "60% of two requires two contexts")

    summary, _ = valid_case("stable_three_percent_disadvantage", constant_contexts(1.03, 1.03))
    require(summary["stable_disadvantage"] and not summary["passes_competence"], "3% constant disadvantage must fail")

    # Missing genes differ by context; row multiplicities must weight available tasks.
    weighted = input_frame([
        (context, f"synthetic_gene_{gene:03d}", 1 + .013 * gene + .1 * offset, .9 + .009 * gene + .2 * offset)
        for context, offset, start, stop in ((CONTEXTS[0], 0, 0, 45), (CONTEXTS[1], 1, 5, 50))
        for gene in range(start, stop)
    ])
    weighted_summary, weighted_draws = valid_case("weighted_available_tasks", weighted)
    repeat_summary, _, repeat_draws = evaluate(weighted.copy(deep=True))
    require(csv_digest(weighted_draws) == csv_digest(repeat_draws), "fixed seed must give byte-identical draw CSV")
    for key in weighted_summary:
        require(weighted_summary[key] == repeat_summary[key], f"fixed seed changed summary {key}")
    shuffled_summary, _, shuffled_draws = evaluate(weighted.sample(frac=1, random_state=SEED))
    close(shuffled_draws.ratio, weighted_draws.ratio, "sorted gene clusters must ignore input row order")
    close(shuffled_summary["relative_macro_error_gap"], weighted_summary["relative_macro_error_gap"], "row ordering point estimate")
    require(shuffled_draws.gene_draw_sha256.tolist() == weighted_draws.gene_draw_sha256.tolist(), "gene draw hashes changed with row order")
    results.append({"case": "fixed_seed_draw_hash_and_row_order", "status": "PASS", "draw_csv_sha256": csv_digest(weighted_draws)})

    def invalid_case(name: str, tasks: pd.DataFrame):
        try:
            evaluate(tasks.copy(deep=True))
        except RuntimeError as error:
            results.append({"case": name, "status": "PASS", "guard_message": str(error)})
            fixtures.append(tasks.assign(case=name))
        else:
            raise AssertionError(f"{name}: expected RuntimeError")

    invalid_case("missing_expected_context_guard", constant_contexts(1, 1).query("context_id == 'HCT116'"))
    invalid_case("zero_baseline_macro_guard", constant_contexts(1, 1, baseline=0))
    invalid_case("negative_baseline_macro_guard", constant_contexts(1, 1, baseline=-1))
    invalid_case("duplicate_gene_context_guard", pd.concat([cancellation, cancellation.iloc[[0]]], ignore_index=True))
    nonfinite = constant_contexts(1, 1)
    nonfinite.loc[0, "model_rmse"] = np.nan
    invalid_case("nonfinite_model_error_guard", nonfinite)
    nonfinite_baseline = constant_contexts(1, 1)
    nonfinite_baseline.loc[0, "baseline_rmse"] = np.inf
    invalid_case("nonfinite_baseline_error_guard", nonfinite_baseline)
    invalid_case("zero_baseline_bootstrap_guard", input_frame([
        (context, f"synthetic_gene_{gene:03d}", 1, float(gene))
        for context in CONTEXTS for gene in range(2)
    ]))
    invalid_case("empty_context_bootstrap_guard", input_frame([
        (CONTEXTS[0], "synthetic_gene_000", 1, 1),
        (CONTEXTS[0], "synthetic_gene_001", 1, 1),
        (CONTEXTS[0], "synthetic_gene_002", 1, 1),
        (CONTEXTS[1], "synthetic_gene_000", 1, 1),
    ]))

    # The API consumes an already frozen baseline; this illustration is not a
    # test of TRAIN OOF baseline selection, which occurs outside the API.
    synthetic_train_oof = np.array([[1., 2.], [2., 1.]])
    fixed_choices = synthetic_train_oof.argmin(axis=1)
    examples.append({
        "case": "illustrative_percontext_baseline_selection",
        "selected_indices": fixed_choices.tolist(),
        "percontext_selected_macro": float(synthetic_train_oof[np.arange(2), fixed_choices].mean()),
        "single_global_baseline_macro": float(synthetic_train_oof.mean(axis=0).min()),
        "baseline_selection_api_tested": False,
    })

    args.output.mkdir(parents=True)
    report = {
        "schema": "orion_competence_independent_synthetic_statistics_v1",
        "status": "PASS", "checks": len(results),
        "synthetic_only": True, "actual_vectors_read": False,
        "safeconf_scores_produced": False,
        "model_family_count_increased": False,
        "bootstrap_replicates": REPLICATES, "bootstrap_seed": SEED,
        "quantile_method": "linear", "quantile_probabilities": [.025, .975],
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "weighted_draw_csv_sha256": csv_digest(weighted_draws),
        "results": results, "illustrations": examples,
    }
    (args.output / "VERIFY_COMPETENCE_STATISTICS.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    pd.concat(fixtures, ignore_index=True).to_csv(args.output / "SYNTHETIC_TASK_ERRORS.csv", index=False)
    weighted_draws.head(12).to_csv(args.output / "WEIGHTED_DRAW_SAMPLE.csv", index=False, float_format="%.17g")
    print(f"PASS: {len(results)} synthetic checks; 5000 paired draws checked per valid scenario")
    print(f"draw CSV SHA256: {report['weighted_draw_csv_sha256']}")
    print(f"report: {args.output.resolve() / 'VERIFY_COMPETENCE_STATISTICS.json'}")


if __name__ == "__main__":
    main()
