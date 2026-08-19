#!/usr/bin/env python3
"""E150: build the raw-count Replogle asset fixed by the E149 contract.

The E149 selection and split artifacts are treated as immutable inputs.  This
step reads expression counts only after verifying all frozen hashes.  It does
not normalize expression, fit a model, calculate an effect, or change a task.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
from datetime import datetime
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
E149 = ROOT / "docs/实验结果/E149_replogle_two_cellline_contract_20260714"
E149_STATUS = E149 / "RUN_STATUS.json"
SELECTION = E149 / "tables/E149_SELECTED_PERTURBATIONS.csv"
MANIFEST = E149 / "manifests/E149_TASK_MANIFEST.csv"
SOURCES = {
    "K562": Path(
        "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
        "ReplogleWeissman2022_K562_essential.h5ad"
    ),
    "RPE1": Path(
        "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
        "ReplogleWeissman2022_rpe1.h5ad"
    ),
}
DATA_OUT = Path("/home/yyf/data/safeconf_e150_replogle")
ASSET = DATA_OUT / "Replogle_two_cellline_E149_selected_raw_counts.h5ad"
TEMP_ASSET = DATA_OUT / "Replogle_two_cellline_E149_selected_raw_counts.tmp.h5ad"
OUT = ROOT / "docs/实验结果/E150_replogle_combined_asset_20260714"
TABLES, REPORTS = OUT / "tables", OUT / "reports"
CONTROL_LABEL = "control"
READ_CHUNK_ROWS = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_memory_bytes(matrix: sp.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def verify_frozen_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    status = json.loads(E149_STATUS.read_text())
    if status["status"] != "frozen_before_expression_values_predictions_or_errors":
        raise RuntimeError("E149 is not in the expected pre-expression frozen state")

    expected_artifacts = status["artifact_sha256"]
    required = {
        "tables/E149_SELECTED_PERTURBATIONS.csv": SELECTION,
        "manifests/E149_TASK_MANIFEST.csv": MANIFEST,
    }
    artifact_hashes = {}
    for relative, path in required.items():
        observed = sha256(path)
        expected = expected_artifacts[relative]
        if observed != expected:
            raise RuntimeError(f"E149 frozen artifact hash changed: {relative}")
        artifact_hashes[relative] = observed

    source_hashes = {}
    for context, path in SOURCES.items():
        observed = sha256(path)
        expected = status["sources"][context]["sha256"]
        if observed != expected:
            raise RuntimeError(f"E149 source hash changed: {context}")
        source_hashes[context] = observed

    selection = pd.read_csv(SELECTION, keep_default_na=False)
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    if len(selection) != status["n_selected_perturbations"]:
        raise RuntimeError("selection row count differs from E149 status")
    if len(manifest) != status["n_manifest_rows"]:
        raise RuntimeError("manifest row count differs from E149 status")
    if selection["perturbation"].duplicated().any():
        raise RuntimeError("E149 selection contains duplicate perturbations")
    if set(manifest["perturbation"].astype(str)) != set(selection["perturbation"].astype(str)):
        raise RuntimeError("manifest and selection perturbation sets differ")
    return status, selection, manifest, {
        "artifact_hashes": artifact_hashes,
        "source_hashes": source_hashes,
    }


def common_gene_axis() -> tuple[list[str], dict]:
    axes = {}
    shapes = {}
    for context, path in SOURCES.items():
        data = ad.read_h5ad(path, backed="r")
        axes[context] = data.var_names.astype(str).tolist()
        shapes[context] = [int(data.n_obs), int(data.n_vars)]
        data.file.close()
    common = set(axes["K562"]) & set(axes["RPE1"])
    # Preserve K562 source order instead of sorting by expression-derived values.
    ordered = [gene for gene in axes["K562"] if gene in common]
    if len(ordered) != len(common) or len(ordered) == 0:
        raise RuntimeError("failed to construct a unique non-empty common gene axis")
    return ordered, {"source_shapes": shapes, "source_gene_counts": {k: len(v) for k, v in axes.items()}}


def raw_count_block_to_csr(
    data: ad.AnnData,
    row_indices: np.ndarray,
    gene_indices: np.ndarray,
) -> tuple[sp.csr_matrix, dict]:
    chunks = []
    nonzero = 0
    negative = 0
    maximum = 0.0
    minimum_nonzero = float("inf")
    maximum_integer_deviation = 0.0
    for start in range(0, len(row_indices), READ_CHUNK_ROWS):
        selected_rows = row_indices[start : start + READ_CHUNK_ROWS]
        # h5py supports one ordered fancy row index.  Select common columns only
        # after loading this bounded row block.
        dense = np.asarray(data.X[selected_rows, :], dtype=np.float32)[:, gene_indices]
        values = dense[dense != 0]
        if values.size:
            nonzero += int(values.size)
            negative += int(np.count_nonzero(values < 0))
            maximum = max(maximum, float(values.max()))
            minimum_nonzero = min(minimum_nonzero, float(values.min()))
            maximum_integer_deviation = max(
                maximum_integer_deviation,
                float(np.max(np.abs(values - np.rint(values)))),
            )
        chunks.append(sp.csr_matrix(dense))
    matrix = sp.vstack(chunks, format="csr")
    matrix.eliminate_zeros()
    if matrix.shape != (len(row_indices), len(gene_indices)):
        raise RuntimeError("expression block shape changed during sparse conversion")
    audit = {
        "nnz": int(matrix.nnz),
        "nonzero_values_audited": nonzero,
        "negative_nonzero_values": negative,
        "minimum_nonzero_value": minimum_nonzero if np.isfinite(minimum_nonzero) else 0.0,
        "maximum_value": maximum,
        "maximum_integer_deviation": maximum_integer_deviation,
        "all_nonzero_values_nonnegative_integer_counts": bool(
            negative == 0 and maximum_integer_deviation <= 1e-6
        ),
    }
    return matrix, audit


def build_context_piece(
    context: str,
    source: Path,
    selected: set[str],
    common_genes: list[str],
    expected_selection: pd.DataFrame,
    expected_manifest: pd.DataFrame,
) -> tuple[ad.AnnData, pd.DataFrame, dict]:
    data = ad.read_h5ad(source, backed="r")
    source_genes = data.var_names.astype(str).tolist()
    source_gene_index = {gene: index for index, gene in enumerate(source_genes)}
    gene_indices = np.asarray([source_gene_index[gene] for gene in common_genes], dtype=np.int64)
    perturbations = data.obs["perturbation"].astype(str)
    keep = perturbations.isin(selected | {CONTROL_LABEL}).to_numpy()
    row_indices = np.flatnonzero(keep).astype(np.int64)
    obs = data.obs.iloc[row_indices].copy()
    obs["source_obs_name"] = data.obs_names[row_indices].astype(str)
    obs["context"] = context
    obs["source_cell_line"] = context
    obs.index = pd.Index(
        [f"{context}::{value}" for value in obs["source_obs_name"].astype(str)],
        name="cell_id",
    )
    matrix, count_audit = raw_count_block_to_csr(data, row_indices, gene_indices)
    original_x_storage = type(data.X).__name__
    original_x_dtype = str(data.X.dtype)
    data.file.close()

    var = pd.DataFrame(
        {"gene_symbol": common_genes},
        index=pd.Index(common_genes, name="gene_symbol_index"),
    )
    piece = ad.AnnData(X=matrix, obs=obs, var=var)
    piece.uns["e149_contract"] = {
        "selection_file_sha256": sha256(SELECTION),
        "manifest_file_sha256": sha256(MANIFEST),
        "normalization_applied": False,
        "x_semantics": "source raw nonnegative integer UMI counts, common-gene subset",
    }

    observed = obs.groupby(obs["perturbation"].astype(str), observed=True).agg(
        observed_cells=("perturbation", "size"),
        observed_batches=("batch", "nunique"),
        observed_guide_entities=("guide_id", "nunique"),
        observed_transcript_groups=("transcript", "nunique"),
    )
    rows = []
    selection_lookup = expected_selection.set_index("perturbation")
    manifest_lookup = expected_manifest.set_index(["context", "perturbation"])["n_cells"]
    for perturbation in [CONTROL_LABEL, *sorted(selected)]:
        is_selected = perturbation in selected
        expected_from_selection = (
            int(selection_lookup.loc[perturbation, f"n_cells_{context}"])
            if is_selected
            else pd.NA
        )
        expected_from_manifest = (
            int(manifest_lookup.loc[(context, perturbation)]) if is_selected else pd.NA
        )
        actual = int(observed.loc[perturbation, "observed_cells"])
        coverage_pass = bool(
            not is_selected
            or (actual == expected_from_selection == expected_from_manifest)
        )
        rows.append(
            {
                "context": context,
                "perturbation": perturbation,
                "is_e149_selected_perturbation": is_selected,
                "expected_cells_selection": expected_from_selection,
                "expected_cells_manifest": expected_from_manifest,
                "observed_cells_asset": actual,
                "observed_batches": int(observed.loc[perturbation, "observed_batches"]),
                "observed_guide_entities": int(observed.loc[perturbation, "observed_guide_entities"]),
                "observed_transcript_groups": int(observed.loc[perturbation, "observed_transcript_groups"]),
                "coverage_exact_match": coverage_pass,
            }
        )
    coverage = pd.DataFrame(rows)
    if not coverage["coverage_exact_match"].all():
        raise RuntimeError(f"task coverage mismatch in {context}")

    audit = {
        "context": context,
        "source_path": str(source),
        "source_file_bytes": source.stat().st_size,
        "source_shape_cells": int(len(perturbations)),
        "source_shape_genes": len(source_genes),
        "source_x_storage": original_x_storage,
        "source_x_dtype": original_x_dtype,
        "selected_cells_including_control": int(piece.n_obs),
        "selected_perturbation_cells": int(
            piece.obs["perturbation"].astype(str).ne(CONTROL_LABEL).sum()
        ),
        "control_cells": int(piece.obs["perturbation"].astype(str).eq(CONTROL_LABEL).sum()),
        "selected_perturbations_present": int(
            piece.obs.loc[
                piece.obs["perturbation"].astype(str).ne(CONTROL_LABEL), "perturbation"
            ].nunique()
        ),
        "common_genes": int(piece.n_vars),
        "asset_x_storage": "scipy.sparse.csr_matrix",
        "asset_x_dtype": str(piece.X.dtype),
        "asset_nnz": int(piece.X.nnz),
        "asset_density": float(piece.X.nnz / (piece.n_obs * piece.n_vars)),
        "asset_sparse_memory_bytes": sparse_memory_bytes(piece.X),
        "normalization_applied": False,
        **count_audit,
    }
    if not audit["all_nonzero_values_nonnegative_integer_counts"]:
        raise RuntimeError(f"{context} X failed raw-count integer audit")
    return piece, coverage, audit


def main() -> None:
    for directory in [DATA_OUT, OUT, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    if TEMP_ASSET.exists():
        TEMP_ASSET.unlink()

    e149_status, selection, manifest, verified = verify_frozen_inputs()
    selected = set(selection["perturbation"].astype(str))
    manifest_unique = manifest[["context", "perturbation", "n_cells"]].drop_duplicates()
    duplicate_counts = manifest_unique.groupby(["context", "perturbation"])["n_cells"].nunique()
    if duplicate_counts.max() != 1:
        raise RuntimeError("manifest has inconsistent cell counts across folds")
    manifest_unique = manifest_unique.drop_duplicates(["context", "perturbation"])

    common_genes, axis_audit = common_gene_axis()
    pieces, coverage_frames, context_audits = [], [], []
    for context, source in SOURCES.items():
        piece, coverage, audit = build_context_piece(
            context,
            source,
            selected,
            common_genes,
            selection,
            manifest_unique,
        )
        pieces.append(piece)
        coverage_frames.append(coverage)
        context_audits.append(audit)
        print(
            f"[E150] {context}: {piece.n_obs} cells x {piece.n_vars} genes, "
            f"nnz={piece.X.nnz}, rss={peak_rss_mb():.1f} MB",
            flush=True,
        )

    combined = ad.concat(pieces, axis=0, join="inner", merge="same")
    combined.X = sp.csr_matrix(combined.X)
    combined.var_names = pd.Index(common_genes)
    combined.var["gene_symbol"] = common_genes
    combined.uns["e150_provenance"] = {
        "experiment": "E150_replogle_combined_asset",
        "e149_status_file": str(E149_STATUS),
        "e149_selection_sha256": verified["artifact_hashes"][
            "tables/E149_SELECTED_PERTURBATIONS.csv"
        ],
        "e149_manifest_sha256": verified["artifact_hashes"][
            "manifests/E149_TASK_MANIFEST.csv"
        ],
        "source_sha256": verified["source_hashes"],
        "normalization_applied": False,
        "x_semantics": "unmodified source UMI counts after row/gene subsetting",
    }
    if not sp.isspmatrix_csr(combined.X):
        raise RuntimeError("combined X is not CSR")
    if combined.n_obs != sum(piece.n_obs for piece in pieces):
        raise RuntimeError("combined cell count differs from context pieces")
    if combined.n_vars != len(common_genes):
        raise RuntimeError("combined gene axis differs from frozen common axis")

    combined.write_h5ad(TEMP_ASSET, compression="gzip")
    os.replace(TEMP_ASSET, ASSET)
    output_sha256 = sha256(ASSET)
    with h5py.File(ASSET, "r") as handle:
        h5_x_encoding = handle["X"].attrs.get("encoding-type", "")
        if isinstance(h5_x_encoding, bytes):
            h5_x_encoding = h5_x_encoding.decode()
    check = ad.read_h5ad(ASSET, backed="r")
    output_shape = [int(check.n_obs), int(check.n_vars)]
    output_contexts = sorted(check.obs["context"].astype(str).unique().tolist())
    output_perturbations = sorted(check.obs["perturbation"].astype(str).unique().tolist())
    check.file.close()
    if output_shape != [combined.n_obs, combined.n_vars]:
        raise RuntimeError("written asset shape verification failed")
    if output_contexts != sorted(SOURCES):
        raise RuntimeError("written asset context verification failed")
    if set(output_perturbations) != selected | {CONTROL_LABEL}:
        raise RuntimeError("written asset perturbation verification failed")
    if h5_x_encoding != "csr_matrix":
        raise RuntimeError(f"written X encoding is not CSR: {h5_x_encoding}")

    coverage = pd.concat(coverage_frames, ignore_index=True)
    context_audit = pd.DataFrame(context_audits)
    gene_axis = pd.DataFrame(
        {
            "gene_order_index": np.arange(len(common_genes), dtype=int),
            "gene_symbol": common_genes,
            "present_in_K562": True,
            "present_in_RPE1": True,
            "selection_used_expression_value": False,
        }
    )
    coverage.to_csv(TABLES / "E150_TASK_COVERAGE_AUDIT.csv", index=False)
    context_audit.to_csv(TABLES / "E150_CONTEXT_ASSET_AUDIT.csv", index=False)
    gene_axis.to_csv(TABLES / "E150_COMMON_GENE_AXIS.csv", index=False)

    status = {
        "experiment": "E150_replogle_combined_asset",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "source_contract": str(E149.relative_to(ROOT)),
        "e149_status_at_build": e149_status["status"],
        "verified_source_sha256": verified["source_hashes"],
        "verified_frozen_artifact_sha256": verified["artifact_hashes"],
        "asset_path": str(ASSET),
        "asset_sha256": output_sha256,
        "asset_file_bytes": ASSET.stat().st_size,
        "asset_shape": output_shape,
        "asset_x_h5ad_encoding": h5_x_encoding,
        "asset_x_in_memory_encoding": "scipy.sparse.csr_matrix",
        "asset_x_dtype": str(combined.X.dtype),
        "asset_nnz": int(combined.X.nnz),
        "asset_density": float(combined.X.nnz / (combined.n_obs * combined.n_vars)),
        "asset_sparse_memory_bytes": sparse_memory_bytes(combined.X),
        "peak_process_rss_mb": peak_rss_mb(),
        "source_gene_counts": axis_audit["source_gene_counts"],
        "common_gene_count": len(common_genes),
        "contexts": output_contexts,
        "n_selected_perturbations": len(selected),
        "n_control_cells": int(
            combined.obs["perturbation"].astype(str).eq(CONTROL_LABEL).sum()
        ),
        "n_selected_perturbation_cells": int(
            combined.obs["perturbation"].astype(str).ne(CONTROL_LABEL).sum()
        ),
        "task_coverage_rows": len(coverage),
        "all_selected_task_counts_match_e149": bool(
            coverage.loc[coverage["is_e149_selected_perturbation"], "coverage_exact_match"].all()
        ),
        "all_x_values_nonnegative_integer_counts": bool(
            context_audit["all_nonzero_values_nonnegative_integer_counts"].all()
        ),
        "normalization_applied": False,
        "effect_vectors_computed": False,
        "model_training_executed": False,
        "selection_or_split_modified": False,
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )
    report = [
        "# E150｜Replogle 原始计数组合资产",
        "",
        "## 完成状态",
        "",
        f"E149 的源文件、选择表和任务 manifest 哈希全部通过后，按固定的 128 个 perturbations 加 control 读取 K562/RPE1。两个源文件取共同 {combined.n_vars} 个基因，合并得到 {combined.n_obs} 个细胞。",
        "",
        f"- K562：{int(context_audit.set_index('context').loc['K562','selected_cells_including_control'])} cells；RPE1：{int(context_audit.set_index('context').loc['RPE1','selected_cells_including_control'])} cells。",
        f"- perturbation cells={status['n_selected_perturbation_cells']}，control cells={status['n_control_cells']}。",
        f"- X 为 CSR float32，nnz={status['asset_nnz']}，density={status['asset_density']:.3f}；非零值逐块审计均为非负整数计数。",
        f"- CSR 内存占用约 {status['asset_sparse_memory_bytes']/1024**3:.2f} GiB；h5ad 磁盘占用约 {status['asset_file_bytes']/1024**3:.2f} GiB；构建过程 peak RSS={status['peak_process_rss_mb']:.1f} MiB。",
        f"- 256 个 selected context × perturbation 组合的细胞数全部与 E149 selection/manifest 精确一致；另记录两个 control 组合。",
        "",
        "## 信息边界",
        "",
        "本步骤只作行过滤、共同基因列过滤、稀疏格式转换和两个细胞系拼接。没有 normalize_total、log1p、scale 或其他数值变换；没有计算 perturbation effect、模型预测或误差，也没有改变 E149 的任务选择与划分。",
    ]
    (REPORTS / "E150_REPORT.md").write_text("\n".join(report) + "\n")
    (OUT / "README_先看这个.md").write_text(
        "# E150 先看这个\n\n"
        "先读 `reports/E150_REPORT.md`。大体积原始计数组合 h5ad 不进入 Git，其绝对路径与 SHA-256 见 `RUN_STATUS.json`。\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(context_audit.to_string(index=False))


if __name__ == "__main__":
    main()
