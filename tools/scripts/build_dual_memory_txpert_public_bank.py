#!/usr/bin/env python3
"""Build item-level, source-only TxPert Public Memory without target truth.

The builder reconstructs the 5,238 source-context perturbation effects from
the already-audited E201 physical blind views.  It deliberately does not open
target treatment rows or evaluation truth.  Large vectors live under the data
root; a compact integrity audit is written to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.safeconf_continual import PublicMemoryItem, PublicMemoryStore
E201 = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
SUPPORT = E201 / "tables/E201_SOURCE_CONTEXT_SUPPORT.csv"
TASKS = E201 / "tables/E201_PRETRUTH_TASK_BASE.csv"
DEFAULT_CACHE = Path("/home/yyf/data/txpert_official_20260802/cache")
DEFAULT_STORE = Path("/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201")
DEFAULT_AUDIT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/PUBLIC_MEMORY_TXPERT_AUDIT.json"
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")


class BuildFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_mean(matrix, indices: np.ndarray) -> np.ndarray:
    if not len(indices):
        raise BuildFailure("cannot average an empty block")
    return np.asarray(matrix[indices].mean(axis=0), dtype=np.float64).ravel()


def perturbation_target(condition: str) -> str:
    tokens = [token for token in str(condition).split("+") if token.lower() != "ctrl"]
    if len(tokens) != 1:
        raise BuildFailure(f"expected a single genetic perturbation: {condition}")
    return tokens[0]


def build_target(
    target: str,
    cache_root: Path,
    expected: pd.DataFrame,
    start_row: int,
) -> tuple[list[PublicMemoryItem], list[np.ndarray], list[np.ndarray], list[str]]:
    path = cache_root / f"E201_blind_{target}" / "de_adata_test.h5ad"
    manifest_path = path.parent / "E201_BLIND_VIEW_MANIFEST.json"
    if not path.is_file() or not manifest_path.is_file():
        raise BuildFailure(f"missing physical blind view: {target}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("target") != target or int(manifest.get("n_target_treatments", -1)) != 0:
        raise BuildFailure(f"physical isolation contract failed: {target}")
    dataset = ad.read_h5ad(path, backed="r")
    obs = dataset.obs.copy()
    genes = list(map(str, dataset.var_names))
    if len(genes) != 3352:
        dataset.file.close()
        raise BuildFailure(f"unexpected gene axis: {target}/{len(genes)}")
    is_control = obs.control.astype(bool)
    if int((~is_control & obs.cell_line.astype(str).eq(target)).sum()) != 0:
        dataset.file.close()
        raise BuildFailure(f"target perturbations survived blind view: {target}")
    items: list[PublicMemoryItem] = []
    effects: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    try:
        for source_context in sorted(set(TARGETS) - {target}):
            context = obs.cell_line.astype(str).eq(source_context)
            control_global = np.flatnonzero((context & is_control).to_numpy())
            perturb_global = np.flatnonzero((context & ~is_control).to_numpy())
            control_x = sparse.csr_matrix(dataset.X[control_global])
            perturb_x = sparse.csr_matrix(dataset.X[perturb_global])
            control_obs = obs.iloc[control_global].reset_index(drop=True)
            perturb_obs = obs.iloc[perturb_global].reset_index(drop=True)
            control_batches = control_obs.batch.astype(str).to_numpy()
            control_means = {
                batch: row_mean(control_x, np.flatnonzero(control_batches == batch))
                for batch in sorted(pd.unique(control_batches))
            }
            conditions = perturb_obs.condition.astype(str).to_numpy()
            batches = perturb_obs.batch.astype(str).to_numpy()
            allowed = set(expected.loc[expected.source_context.eq(source_context), "condition"].astype(str))
            for condition in sorted(allowed):
                local = np.flatnonzero(conditions == condition)
                if not len(local):
                    raise BuildFailure(f"source item disappeared: {target}/{source_context}/{condition}")
                used_batches, counts = np.unique(batches[local], return_counts=True)
                if any(batch not in control_means for batch in used_batches):
                    raise BuildFailure(f"matched control missing: {target}/{source_context}/{condition}")
                matched_control = sum(
                    control_means[batch] * int(count)
                    for batch, count in zip(used_batches, counts)
                ) / len(local)
                effect = row_mean(perturb_x, local) - matched_control
                row = start_row + len(items)
                items.append(PublicMemoryItem(
                    experiment_id=f"E201::{target}::{source_context}::{condition}",
                    study_id="E201_TxPert_official_multicontext",
                    context=source_context,
                    perturbation_type="genetic_single_gene",
                    perturbation_target=perturbation_target(condition),
                    condition=condition,
                    effect_vector_row=row,
                    effect_contract_id="E201_log1p_matched_batch_delta_v1",
                    control_source="source_context_batch_matched_control",
                    gene_space_id="E201_GEARS_3352",
                    n_cells=int(len(local)),
                    n_guides=None,
                    n_plates=None,
                    n_batches=int(len(used_batches)),
                    guide_reproducibility=None,
                    plate_reproducibility=None,
                    split_half_stability=None,
                    batch_agreement=None,
                    provenance=str(path),
                    eligibility=True,
                    timestamp="2026-08-02T00:00:00+00:00",
                ))
                effects.append(effect.astype(np.float32))
                controls.append(matched_control.astype(np.float32))
    finally:
        dataset.file.close()
    return items, effects, controls, genes


def main() -> None:
    args = parse_args()
    if args.audit.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.audit}")
    support = pd.read_csv(SUPPORT)
    tasks = pd.read_csv(TASKS)
    expected_conditions = {
        target: set(tasks.loc[tasks.target.eq(target), "condition"].astype(str))
        for target in TARGETS
    }
    items: list[PublicMemoryItem] = []
    effects: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    gene_ids: list[str] | None = None
    for target in TARGETS:
        expected = support[support.target.eq(target)].copy()
        if set(expected.condition.astype(str)) != expected_conditions[target]:
            raise BuildFailure(f"task/support condition mismatch: {target}")
        block_items, block_effects, block_controls, genes = build_target(
            target, args.cache_root, expected, len(items)
        )
        if gene_ids is None:
            gene_ids = genes
        elif gene_ids != genes:
            raise BuildFailure(f"gene axis changed across targets: {target}")
        items.extend(block_items)
        effects.extend(block_effects)
        controls.extend(block_controls)
        print(f"[PublicMemory] {target}: {len(block_items)} items", flush=True)
    if len(items) != len(support) or len(items) != 5238:
        raise BuildFailure(f"expected 5238 source items, observed {len(items)}")
    built = pd.DataFrame([item.as_record() for item in items])
    # target is encoded in experiment_id and multiple target blocks can share
    # source_context/condition, so validate with an explicit parsed target.
    built["target"] = built.experiment_id.str.split("::").str[1]
    check = built.merge(
        support,
        left_on=["target", "context", "condition"],
        right_on=["target", "source_context", "condition"],
        validate="one_to_one",
    )
    if not np.array_equal(check.n_cells.to_numpy(int), check.n_source_perturbed_cells.to_numpy(int)):
        raise BuildFailure("source cell support changed")
    if not np.array_equal(check.n_batches.to_numpy(int), check.n_source_batches.to_numpy(int)):
        raise BuildFailure("source batch support changed")
    # The 5,238 rows describe target-to-source eligibility.  Several blind
    # views contain the same physical source experiment; storing each copy as
    # a memory item would falsely multiply biological support.  Collapse to
    # the 2,008 unique source-context/condition experiments and retain the
    # many-to-many eligibility edges separately.
    candidate_effects = np.stack(effects)
    candidate_controls = np.stack(controls)
    canonical_items: list[PublicMemoryItem] = []
    canonical_effects: list[np.ndarray] = []
    canonical_controls: list[np.ndarray] = []
    eligibility_rows: list[dict[str, str]] = []
    for (context, condition), group in built.groupby(["context", "condition"], sort=True):
        indices = group.index.to_numpy(int)
        first = int(indices[0])
        if not np.allclose(candidate_effects[indices], candidate_effects[first], atol=1e-6, rtol=1e-6):
            raise BuildFailure(f"duplicate blind views disagree on effect: {context}/{condition}")
        if not np.allclose(candidate_controls[indices], candidate_controls[first], atol=1e-6, rtol=1e-6):
            raise BuildFailure(f"duplicate blind views disagree on control: {context}/{condition}")
        if group.n_cells.nunique() != 1 or group.n_batches.nunique() != 1:
            raise BuildFailure(f"duplicate blind views disagree on support: {context}/{condition}")
        memory_id = f"E201::{context}::{condition}"
        item = items[first]
        canonical_items.append(replace(
            item,
            experiment_id=memory_id,
            effect_vector_row=len(canonical_items),
            provenance=";".join(sorted(group.provenance.astype(str).unique())),
        ))
        canonical_effects.append(candidate_effects[first])
        canonical_controls.append(candidate_controls[first])
        for target in sorted(group.target.astype(str).unique()):
            eligibility_rows.append({
                "target_context": target,
                "condition": str(condition),
                "public_experiment_id": memory_id,
                "eligibility_rule": "E201_physical_target_exclusion_v1",
            })
    eligibility = pd.DataFrame(eligibility_rows)
    if len(canonical_items) != 2008 or len(eligibility) != 5238:
        raise BuildFailure(
            f"canonical memory/eligibility mismatch: {len(canonical_items)}/{len(eligibility)}"
        )
    source_manifest = {
        "support_path": str(SUPPORT),
        "support_sha256": sha256_file(SUPPORT),
        "task_path": str(TASKS),
        "task_sha256": sha256_file(TASKS),
        "target_truth_opened": False,
        "target_treatment_rows_opened": 0,
    }
    store = PublicMemoryStore(args.store_root)
    manifest = store.create(
        canonical_items,
        np.stack(canonical_effects),
        np.stack(canonical_controls),
        gene_ids or [],
        source_manifest,
        eligibility,
    )
    audit = {
        "experiment": "DualMemory_TxPert_PublicBank",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "store_root": str(args.store_root),
        "n_items": len(canonical_items),
        "n_eligibility_edges": len(eligibility),
        "n_tasks_with_history": int(built[["target", "condition"]].drop_duplicates().shape[0]),
        "n_source_contexts": int(built.context.nunique()),
        "quality_available": False,
        "support_available": True,
        "item_level_effect_available": True,
        "target_truth_opened": False,
        "target_treatment_rows_opened": 0,
        "manifest": manifest,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
