from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import scanpy as sc

from safetrans_confidence.data.dataset_spec import DatasetSpec
from safetrans_confidence.scoring.protocol_v0_2 import assign_dataset_family

CONTEXT_CANDIDATES = ["cell_type", "cell_label", "condition1", "cell_line", "patient", "donor_id", "batch", "condition"]
PERT_CANDIDATES = ["perturbation", "condition2", "sgRNA", "GenePair", "treatment"]
# Never use "gene" as context when it duplicates perturbation (Norman/Adamson generalization trap).
FORBIDDEN_CONTEXT_COLS = frozenset({"gene"})


@dataclass
class EligibilityRow:
    dataset_name: str
    dataset_family: str
    h5ad_path: str
    context_col: str
    perturbation_col: str
    n_contexts: int
    n_perturbations: int
    has_control: bool
    n_eligible_pairs: int
    cross_context_eligible: bool
    reason_if_not: str


def _pick_col(obs: pd.DataFrame, candidates: list[str], forbidden: frozenset[str] = frozenset()) -> str | None:
    for col in candidates:
        if col in forbidden:
            continue
        if col in obs.columns and obs[col].nunique(dropna=True) >= 2:
            return col
    low = {c.lower(): c for c in obs.columns}
    for cand in candidates:
        if cand.lower() in forbidden:
            continue
        for key, real in low.items():
            if cand.lower() in key and real not in forbidden and obs[real].nunique(dropna=True) >= 2:
                return real
    return None


def _has_control(obs: pd.DataFrame, pert_col: str, control_values: list[str]) -> bool:
    vals = obs[pert_col].astype(str).str.strip().str.lower()
    ctrl = {v.lower() for v in control_values}
    return bool(vals.isin(ctrl).any())


def audit_h5ad(
    path: Path,
    dataset_name: str | None = None,
    context_col: str | None = None,
    perturbation_col: str | None = None,
    min_cells: int = 6,
) -> EligibilityRow:
    path = Path(path)
    name = dataset_name or path.stem
    family = assign_dataset_family(name)
    reason_parts: list[str] = []

    if not path.exists():
        return EligibilityRow(
            name, family, str(path), "", "", 0, 0, False, 0, False, "file_missing"
        )

    adata = sc.read_h5ad(path, backed="r")
    obs = adata.obs

    ctx = context_col or _pick_col(obs, CONTEXT_CANDIDATES, FORBIDDEN_CONTEXT_COLS)
    pert = perturbation_col or _pick_col(obs, PERT_CANDIDATES, frozenset())

    if ctx is None:
        reason_parts.append("no_context_col>=2")
        ctx = ""
    if pert is None:
        reason_parts.append("no_perturbation_col>=2")
        pert = ""

    n_ctx = int(obs[ctx].nunique()) if ctx else 0
    n_pert = int(obs[pert].nunique()) if pert else 0
    has_ctrl = _has_control(obs, pert, ["control", "ctrl", "non-targeting", "nt", "vehicle"]) if pert else False

    if ctx and pert and ctx == pert:
        reason_parts.append("context_equals_perturbation")
    if n_ctx < 2:
        reason_parts.append("n_contexts<2")
    # GEARS processed files use condition≈perturbation combinatorics — not true cross-context.
    if ctx == "condition" and n_ctx >= 2 and n_pert >= 2 and abs(n_ctx - n_pert) <= max(2, int(0.05 * n_pert)):
        reason_parts.append("condition_perturbation_combinatorial_not_cross_context")

    n_pairs = 0
    if ctx and pert:
        grp = obs.groupby([ctx, pert], observed=False).size()
        n_pairs = int((grp >= min_cells).sum())

    if not has_ctrl:
        reason_parts.append("no_control_in_perturbation_col")
    if n_pairs < 8:
        reason_parts.append(f"n_eligible_pairs={n_pairs}<8")

    eligible = (
        n_ctx >= 2
        and n_pert >= 2
        and has_ctrl
        and n_pairs >= 8
        and ctx != pert
        and ctx not in FORBIDDEN_CONTEXT_COLS
        and not reason_parts
    )

    return EligibilityRow(
        dataset_name=name,
        dataset_family=family,
        h5ad_path=str(path),
        context_col=ctx or "",
        perturbation_col=pert or "",
        n_contexts=n_ctx,
        n_perturbations=n_pert,
        has_control=has_ctrl,
        n_eligible_pairs=n_pairs,
        cross_context_eligible=eligible,
        reason_if_not="" if eligible else ";".join(reason_parts) or "unknown",
    )


def audit_from_scan(atlas_root: Path, dataset_names: list[str]) -> pd.DataFrame:
    scan = pd.read_csv(atlas_root / "metadata" / "h5ad_scan.tsv", sep="\t")
    rows = []
    for name in dataset_names:
        sub = scan[scan["study_family"].astype(str) == name]
        if sub.empty:
            rows.append(
                asdict(
                    EligibilityRow(
                        name, assign_dataset_family(name), "", "", "", 0, 0, False, 0, False, "not_in_scan"
                    )
                )
            )
            continue
        path = Path(str(sub.iloc[0]["local_path"]))
        if not path.exists():
            old_root = Path("/home/yyf/datasets/singlecell_perturbation_atlas")
            try:
                rel = path.relative_to(old_root)
            except ValueError:
                rel = None
            if rel is not None:
                repaired = atlas_root / rel
                if repaired.exists():
                    path = repaired
        rows.append(asdict(audit_h5ad(path, name)))
    return pd.DataFrame(rows)
