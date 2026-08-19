#!/usr/bin/env python3
"""Freeze the expression-blind E168 Primary CD4 confirmation contract.

Only an explicit allowlist of HDF5 metadata datasets is decoded.  The script
never indexes ``X``, ``layers``, DE outputs, guide efficacy outputs, or any
``keep_*``/count-derived outcome filter.  Its outputs define the rows that a
later isolated asset builder may read before and after the truth gate.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import platform
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
TABLES = OUT / "tables"
MANIFESTS = OUT / "manifests"
REPORTS = OUT / "reports"

DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
LOCAL_METADATA = DATA_ROOT / "source_metadata"
SAMPLE_METADATA = LOCAL_METADATA / "sample_metadata.suppl_table.csv"
DATA_README = LOCAL_METADATA / "data_sharing_readme.md"
GUIDE_LIBRARY = LOCAL_METADATA / "sgRNA_library_curated.csv"
LOCAL_H5AD = DATA_ROOT / "source/GWCD4i.pseudobulk_merged.h5ad"

S3_BUCKET = "genome-scale-tcell-perturb-seq"
S3_KEY = "marson2025_data/GWCD4i.pseudobulk_merged.h5ad"
SOURCE_URL = f"https://{S3_BUCKET}.s3.amazonaws.com/{S3_KEY}"
SOURCE_LOCK = {
    "s3_uri": f"s3://{S3_BUCKET}/{S3_KEY}",
    "https_url": SOURCE_URL,
    "content_length_bytes": 44_566_657_140,
    "last_modified_utc": "2026-05-28T23:56:40Z",
    "etag": "010c14e0af0dccbc2524529d28ca517e-5313",
    "etag_is_multipart_not_md5": True,
    "version_id": "BWCjgMRhH80BOFIid2.0kbCr2o8wNVmn",
    "checksum_crc64nvme_base64": "E2slkXBEb2c=",
    "checksum_crc64nvme_hex": "136b259170446f67",
    "official_repository": "https://github.com/emdann/GWT_perturbseq_analysis_2025",
    "official_repository_commit": "848d62fc2b7027f7218d6fc5f5b0c37255dc94af",
    "official_data_card": "https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq?access_dataset=true",
}

SCGPT_DIR = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
SCGPT_FILES = [SCGPT_DIR / name for name in ["args.json", "vocab.json", "best_model.pt"]]
GEARS_GO = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")
METHOD_REFERENCE_FILES = [
    ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py",
    ROOT / "tools/scripts/run_e107_frangieh_context_gears.py",
    ROOT / "tools/scripts/run_e112_external_formal_dual_models.py",
    ROOT / "tools/scripts/run_e167a_riag_resolution_correction.py",
    ROOT
    / "code/20260426_154505_perturb_transport_final_push/"
    "safetrans_confidence/scoring/protocol_v0_2.py",
]

ALLOWED_VALUE_PATHS = (
    "obs/10xrun_id",
    "obs/donor_id",
    "obs/culture_condition",
    "obs/guide_id",
    "obs/perturbed_gene_name",
    "obs/perturbed_gene_id",
    "obs/guide_type",
    "var/gene_ids",
    "var/gene_name",
)
FORBIDDEN_VALUE_PATHS = (
    "X",
    "layers",
    "obs/n_cells",
    "obs/total_counts",
    "obs/log10_n_cells",
    "obs/keep_min_cells",
    "obs/keep_effective_guides",
    "obs/keep_total_counts",
    "obs/keep_for_DE",
    "obs/keep_test_genes",
)
FORBIDDEN_SOURCE_KEYS = (
    "marson2025_data/GWCD4i.DE_stats.h5ad",
    "marson2025_data/GWCD4i.DE_stats.by_donors.h5mu",
    "marson2025_data/GWCD4i.DE_stats.by_guide.h5mu",
    "marson2025_data/suppl_tables/DE_stats.suppl_table.csv",
    "marson2025_data/suppl_tables/guide_kd_efficiency.suppl_table.csv",
    "marson2025_data/suppl_tables/sgrna_library_metadata.suppl_table.csv",
)

STATES = ("Rest", "Stim8hr", "Stim48hr")
N_TARGETS = 200
N_COLUMN_UNSEEN = 40
MIN_COMMON_GUIDES_PER_TARGET = 2
TARGET_SALT = "E168_TARGET_PANEL_V1"
UNSEEN_SALT = "E168_COLUMN_UNSEEN_V1"
DONOR_SALT = "E168_DONOR_ROLE_V1"
MODEL_SEEDS = (3407, 3408, 3409)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(salt: str, value: str) -> str:
    return hashlib.sha256(salt.encode() + b"\0" + str(value).encode()).hexdigest()


def decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in "OSU":
        return values
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values],
        dtype=object,
    )


def read_h5_value(root: h5py.File, path: str, access_log: list[dict]) -> np.ndarray:
    if path not in ALLOWED_VALUE_PATHS:
        raise PermissionError(f"E168 metadata reader denied HDF5 value path: {path}")
    node = root[path]
    if isinstance(node, h5py.Group):
        if set(node.keys()) != {"categories", "codes"}:
            raise RuntimeError(f"unexpected categorical encoding at {path}: {list(node.keys())}")
        categories = decode_array(node["categories"][...])
        codes = np.asarray(node["codes"][...], dtype=np.int64)
        values = np.asarray(
            [categories[code] if code >= 0 else "" for code in codes], dtype=object
        )
        encoding = "categorical/categories+codes"
    else:
        values = decode_array(node[...])
        encoding = str(node.attrs.get("encoding-type", "array"))
    access_log.append(
        {
            "hdf5_path": path,
            "value_access": True,
            "n_values": int(len(values)),
            "encoding": encoding,
            "purpose": "expression-blind task and row-access freeze",
        }
    )
    return values


def head_source() -> dict[str, object]:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"HEAD"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    response = session.head(
        SOURCE_URL,
        headers={"x-amz-checksum-mode": "ENABLED"},
        timeout=(30, 60),
    )
    response.raise_for_status()
    observed = {
        "content_length_bytes": int(response.headers["Content-Length"]),
        "etag": response.headers["ETag"].strip('"'),
        "last_modified_http": response.headers["Last-Modified"],
        "version_id": response.headers.get("x-amz-version-id", ""),
        "checksum_crc64nvme_base64": response.headers.get("x-amz-checksum-crc64nvme", ""),
        "checksum_type": response.headers.get("x-amz-checksum-type", ""),
        "accept_ranges": response.headers.get("Accept-Ranges", ""),
    }
    expected = {
        "content_length_bytes": SOURCE_LOCK["content_length_bytes"],
        "etag": SOURCE_LOCK["etag"],
        "last_modified_http": "Thu, 28 May 2026 23:56:40 GMT",
        "version_id": SOURCE_LOCK["version_id"],
        "checksum_crc64nvme_base64": SOURCE_LOCK["checksum_crc64nvme_base64"],
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"source HEAD lock mismatch: {mismatches}")
    return observed


class RobustHttpRangeReader(io.RawIOBase):
    """Small retrying block cache for deterministic remote HDF5 metadata reads."""

    def __init__(self, url: str, size: int, block_size: int = 16 * 1024 * 1024):
        self.url = url
        self.size = int(size)
        self.block_size = int(block_size)
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.max_cached_blocks = 32
        self.network_bytes_fetched = 0
        retry = Retry(
            total=8,
            connect=8,
            read=8,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = int(offset)
        elif whence == io.SEEK_CUR:
            position = self.position + int(offset)
        elif whence == io.SEEK_END:
            position = self.size + int(offset)
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def _fetch_block(self, block_index: int) -> bytes:
        if block_index in self.cache:
            payload = self.cache.pop(block_index)
            self.cache[block_index] = payload
            return payload
        start = block_index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 120),
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"range request returned {response.status_code}")
                expected_length = end - start + 1
                if len(response.content) != expected_length:
                    raise RuntimeError(
                        f"range length mismatch {len(response.content)} != {expected_length}"
                    )
                observed_etag = response.headers.get("ETag", "").strip('"')
                if observed_etag != SOURCE_LOCK["etag"]:
                    raise RuntimeError(f"range ETag changed: {observed_etag}")
                payload = response.content
                self.network_bytes_fetched += len(payload)
                self.cache[block_index] = payload
                while len(self.cache) > self.max_cached_blocks:
                    self.cache.popitem(last=False)
                return payload
            except Exception as exc:  # network retries are intentionally explicit
                last_error = exc
                if attempt < 8:
                    time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"failed range block {block_index}: {last_error}")

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(int(size), self.size - self.position)
        start = self.position
        stop = start + size
        parts = []
        cursor = start
        while cursor < stop:
            block_index = cursor // self.block_size
            block = self._fetch_block(block_index)
            within = cursor % self.block_size
            take = min(stop - cursor, len(block) - within)
            parts.append(block[within : within + take])
            cursor += take
        self.position = stop
        return b"".join(parts)

    def readinto(self, buffer) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)

    def close(self) -> None:
        if not self.closed:
            self.session.close()
        super().close()


def load_remote_metadata() -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict, dict]:
    access_log: list[dict] = []
    head_before = head_source()
    remote = RobustHttpRangeReader(SOURCE_URL, SOURCE_LOCK["content_length_bytes"])
    try:
        remote_size = int(remote.size)
        if remote_size != SOURCE_LOCK["content_length_bytes"]:
            raise RuntimeError(f"source size changed: {remote_size}")
        with h5py.File(remote, "r") as handle:
            root_keys = sorted(map(str, handle.keys()))
            expected_roots = ["X", "layers", "obs", "obsm", "obsp", "uns", "var", "varm", "varp"]
            if root_keys != expected_roots:
                raise RuntimeError(f"unexpected AnnData roots: {root_keys}")
            obs = pd.DataFrame(
                {
                    name.removeprefix("obs/"): read_h5_value(handle, name, access_log)
                    for name in ALLOWED_VALUE_PATHS
                    if name.startswith("obs/")
                }
            )
            var = pd.DataFrame(
                {
                    name.removeprefix("var/"): read_h5_value(handle, name, access_log)
                    for name in ALLOWED_VALUE_PATHS
                    if name.startswith("var/")
                }
            )
            schema = {
                "anndata_root_keys_metadata_only": root_keys,
                "n_obs": int(len(obs)),
                "n_vars": int(len(var)),
                "expression_matrix_X_decoded": False,
                "layers_decoded": False,
                "forbidden_column_values_decoded": False,
                "remote_range_bytes_fetched": int(remote.network_bytes_fetched),
            }
    finally:
        remote.close()
    head_after = head_source()
    if head_before != head_after:
        raise RuntimeError(f"source HEAD changed during metadata read: {head_before} != {head_after}")
    obs.insert(0, "metadata_row_index", np.arange(len(obs), dtype=np.int64))
    return obs, var, access_log, schema, {
        "before_metadata_read": head_before,
        "after_metadata_read": head_after,
        "identical_before_after": True,
    }


def canonical_type(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def choose_donor_roles(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"donor_id", "culture_condition", "10xrun_id"}
    if not required.issubset(sample.columns):
        raise RuntimeError(f"sample metadata missing {sorted(required - set(sample.columns))}")
    core = sample[list(required)].drop_duplicates()
    donors = sorted(core.donor_id.astype(str).unique())
    if len(donors) != 4 or set(core.culture_condition.astype(str)) != set(STATES):
        raise RuntimeError("expected exactly four donors and three frozen states")
    runs = {
        donor: {
            state: sorted(
                core.loc[
                    core.donor_id.astype(str).eq(donor)
                    & core.culture_condition.astype(str).eq(state),
                    "10xrun_id",
                ].astype(str).unique()
            )
            for state in STATES
        }
        for donor in donors
    }
    candidates = []
    for train in itertools.combinations(donors, 2):
        remaining = [donor for donor in donors if donor not in train]
        for validation, test in itertools.permutations(remaining, 2):
            mapping = (
                f"train={','.join(sorted(train))};validation={validation};test={test}"
            )
            train_runs = {
                state: sorted({run for donor in train for run in runs[donor][state]})
                for state in STATES
            }
            heldout_runs_covered = all(
                set(runs[heldout][state]).issubset(train_runs[state])
                for heldout in (validation, test)
                for state in STATES
            )
            complete = all(len(runs[donor][state]) >= 1 for donor in donors for state in STATES)
            feasible = bool(heldout_runs_covered and complete)
            candidates.append(
                {
                    "train_donors": "+".join(sorted(train)),
                    "validation_donor": validation,
                    "test_donor": test,
                    "canonical_mapping": mapping,
                    "mapping_sha256": stable_hash(DONOR_SALT, mapping),
                    "all_donor_state_combinations_present": complete,
                    "validation_and_test_runs_covered_by_train_within_each_state": heldout_runs_covered,
                    "feasible": feasible,
                }
            )
    audit = pd.DataFrame(candidates).sort_values(["feasible", "mapping_sha256"], ascending=[False, True])
    feasible = audit.loc[audit.feasible].sort_values("mapping_sha256")
    if feasible.empty:
        raise RuntimeError("ABSTAIN_SPLIT_RUN_CONFOUNDED")
    winner = feasible.iloc[0]
    role = {donor: "train" for donor in winner.train_donors.split("+")}
    role[str(winner.validation_donor)] = "validation"
    role[str(winner.test_donor)] = "test"
    rows = []
    for donor in donors:
        for state in STATES:
            rows.append(
                {
                    "donor_id": donor,
                    "donor_role": role[donor],
                    "culture_condition": state,
                    "source_runs": "+".join(runs[donor][state]),
                    "mapping_sha256": winner.mapping_sha256,
                    "selected_before_expression_truth": True,
                }
            )
    return pd.DataFrame(rows), audit


def build_target_tables(
    obs: pd.DataFrame,
    var: pd.DataFrame,
    guide_library: pd.DataFrame,
    vocab: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select an assay-available universe without abundance or expression values."""
    frame = obs.copy()
    frame["guide_type_normalized"] = frame.guide_type.map(canonical_type)
    observed_types = sorted(frame.guide_type_normalized.unique())
    if not set(observed_types).issubset({"targeting", "non-targeting"}):
        raise RuntimeError(f"unexpected guide types: {observed_types}")

    ntc = frame.loc[frame.guide_type_normalized.eq("non-targeting")].copy()
    ntc_summary = (
        ntc.groupby(["donor_id", "culture_condition"], as_index=False)
        .agg(n_ntc_rows=("metadata_row_index", "size"), n_ntc_guides=("guide_id", "nunique"))
    )
    expected_pairs = pd.MultiIndex.from_product(
        [sorted(frame.donor_id.astype(str).unique()), STATES],
        names=["donor_id", "culture_condition"],
    )
    observed_pairs = pd.MultiIndex.from_frame(ntc_summary[["donor_id", "culture_condition"]])
    if not expected_pairs.isin(observed_pairs).all():
        raise RuntimeError("ABSTAIN_MISSING_NTC_DONOR_STATE")

    required_library = {
        "sgrna_id",
        "perturbed_gene_name",
        "perturbed_gene_id",
    }
    if not required_library.issubset(guide_library.columns):
        raise RuntimeError(
            f"guide library missing {sorted(required_library - set(guide_library.columns))}"
        )
    library = guide_library[sorted(required_library)].copy()
    library = library.loc[
        library.perturbed_gene_id.astype(str).str.strip().ne("")
        & library.perturbed_gene_name.astype(str).str.strip().ne("NTC")
    ].copy()
    library["ensembl_core"] = library.perturbed_gene_id.astype(str).str.split(".").str[0]
    guide_audit = (
        library.groupby("sgrna_id", as_index=False)
        .agg(
            n_target_ids=("ensembl_core", "nunique"),
            n_target_names=("perturbed_gene_name", "nunique"),
            ensembl_core=("ensembl_core", "first"),
            perturbed_gene_name=("perturbed_gene_name", "first"),
        )
    )
    guide_audit["guide_mapping_unique"] = (
        guide_audit.n_target_ids.eq(1) & guide_audit.n_target_names.eq(1)
    )
    guide_audit["selection_information_source"] = (
        "official_expression_independent_sgrna_design_annotation"
    )

    targeting = frame.loc[frame.guide_type_normalized.eq("targeting")].copy()
    targeting["observed_ensembl_core"] = (
        targeting.perturbed_gene_id.astype(str).str.split(".").str[0]
    )
    observed_mapping = (
        targeting.groupby("guide_id", as_index=False)
        .agg(
            observed_target_id_count=("observed_ensembl_core", "nunique"),
            observed_target_name_count=("perturbed_gene_name", "nunique"),
            observed_ensembl_core=("observed_ensembl_core", "first"),
            observed_gene_name=("perturbed_gene_name", "first"),
        )
    )
    observed_pairs = (
        targeting.groupby(["guide_id", "donor_id", "culture_condition"], as_index=False)
        .agg(n_identity_rows=("metadata_row_index", "size"))
    )
    observed_coverage = (
        observed_pairs.groupby("guide_id", as_index=False)
        .agg(
            n_donor_state_pairs_present=("donor_id", "size"),
            minimum_identity_rows_per_donor_state=("n_identity_rows", "min"),
        )
    )
    guide_audit = (
        guide_audit.merge(
            observed_mapping,
            left_on="sgrna_id",
            right_on="guide_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            observed_coverage,
            left_on="sgrna_id",
            right_on="guide_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_coverage"),
        )
    )
    guide_audit["observed_mapping_unique"] = (
        guide_audit.observed_target_id_count.fillna(0).eq(1)
        & guide_audit.observed_target_name_count.fillna(0).eq(1)
    )
    guide_audit["design_observed_mapping_match"] = (
        guide_audit.ensembl_core.eq(guide_audit.observed_ensembl_core)
        & guide_audit.perturbed_gene_name.eq(guide_audit.observed_gene_name)
    )
    guide_audit["complete_12_donor_state_identity"] = (
        guide_audit.n_donor_state_pairs_present.fillna(0).eq(12)
        & guide_audit.minimum_identity_rows_per_donor_state.fillna(0).ge(1)
    )
    guide_audit["eligible_identity_guide"] = (
        guide_audit.guide_mapping_unique
        & guide_audit.observed_mapping_unique
        & guide_audit.design_observed_mapping_match
        & guide_audit.complete_12_donor_state_identity
    )
    guide_audit["targeting_abundance_or_expression_used"] = False

    var_axis = var.copy()
    var_axis["ensembl_core"] = var_axis.gene_ids.astype(str).str.split(".").str[0]
    var_axis["gene_name"] = var_axis.gene_name.astype(str)
    id_counts = var_axis.ensembl_core.value_counts()
    name_counts = var_axis.gene_name.value_counts()
    unique_id_to_name = (
        var_axis.loc[
            var_axis.ensembl_core.map(id_counts).eq(1), ["ensembl_core", "gene_name"]
        ]
        .drop_duplicates("ensembl_core")
        .set_index("ensembl_core")
        .gene_name.to_dict()
    )

    unique_guides = guide_audit.loc[guide_audit.eligible_identity_guide].copy()
    target = (
        unique_guides.groupby(["ensembl_core", "perturbed_gene_name"], as_index=False)
        .agg(
            n_identity_complete_guides=("sgrna_id", "nunique"),
            eligible_guide_ids=("sgrna_id", lambda values: "+".join(sorted(map(str, values)))),
        )
    )
    target["expression_axis_id_unique"] = target.ensembl_core.map(id_counts).fillna(0).eq(1)
    target["expression_axis_gene_name"] = target.ensembl_core.map(unique_id_to_name).fillna("")
    target["target_name_matches_expression_axis"] = target.perturbed_gene_name.astype(str).eq(
        target.expression_axis_gene_name
    )
    target["expression_axis_name_unique"] = (
        target.expression_axis_gene_name.map(name_counts).fillna(0).eq(1)
    )
    target["scgpt_token"] = target.expression_axis_gene_name.astype(str).str.upper()
    target["in_scgpt_vocabulary"] = target.scgpt_token.isin(vocab)
    target["eligible_target"] = (
        target.n_identity_complete_guides.ge(MIN_COMMON_GUIDES_PER_TARGET)
        & target.expression_axis_id_unique
        & target.target_name_matches_expression_axis
        & target.expression_axis_name_unique
        & target.in_scgpt_vocabulary
    )
    target["target_selection_sha256"] = target.ensembl_core.map(
        lambda value: stable_hash(TARGET_SALT, value)
    )
    target["column_unseen_sha256"] = target.ensembl_core.map(
        lambda value: stable_hash(UNSEEN_SALT, value)
    )
    selected = (
        target.loc[target.eligible_target]
        .sort_values(["target_selection_sha256", "ensembl_core"])
        .head(N_TARGETS)
        .copy()
    )
    if len(selected) != N_TARGETS:
        raise RuntimeError(f"ABSTAIN_INSUFFICIENT_TARGETS: {len(selected)}")
    selected.insert(0, "target_selection_rank", np.arange(1, len(selected) + 1))
    unseen_ids = set(
        selected.sort_values(["column_unseen_sha256", "ensembl_core"])
        .head(N_COLUMN_UNSEEN)
        .ensembl_core
    )
    selected["target_stratum"] = np.where(
        selected.ensembl_core.isin(unseen_ids), "COLUMN_UNSEEN", "DONOR_UNSEEN_ONLY"
    )
    selected["perturbation_support_contexts_pretruth"] = np.where(
        selected.target_stratum.eq("COLUMN_UNSEEN"), 0, 6
    )
    selected["selected_from_label_free_identity_availability_without_abundance_or_expression"] = True

    selected_guides = {
        guide
        for packed in selected.eligible_guide_ids.astype(str)
        for guide in packed.split("+")
    }
    coverage_summary = guide_audit.loc[
        guide_audit.sgrna_id.isin(selected_guides),
        [
            "sgrna_id",
            "ensembl_core",
            "perturbed_gene_name",
            "observed_ensembl_core",
            "observed_gene_name",
            "design_observed_mapping_match",
            "n_donor_state_pairs_present",
            "minimum_identity_rows_per_donor_state",
            "complete_12_donor_state_identity",
        ],
    ].rename(columns={"sgrna_id": "guide_id"})
    coverage_summary["used_for_assay_available_universe"] = True
    return guide_audit, target, selected, ntc_summary, coverage_summary


def build_manifests(
    obs: pd.DataFrame,
    donor_roles: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_role = donor_roles.drop_duplicates("donor_id").set_index("donor_id").donor_role.to_dict()
    target_info = selected.set_index("ensembl_core").to_dict("index")
    task_rows = []
    for ensembl_id, info in target_info.items():
        for donor in sorted(donor_role):
            for state in STATES:
                role = donor_role[donor]
                stratum = info["target_stratum"]
                if role == "train" and stratum == "DONOR_UNSEEN_ONLY":
                    split, truth_phase = "train", "PRETRUTH_SUPERVISED"
                elif role == "validation" and stratum == "DONOR_UNSEEN_ONLY":
                    split, truth_phase = "validation", "PRETRUTH_VALIDATION"
                elif role == "test":
                    split, truth_phase = "test", "POSTGATE_TEST_TRUTH"
                else:
                    split, truth_phase = f"{role}_query_only", "FORBIDDEN_COLUMN_UNSEEN_TRUTH"
                task_rows.append(
                    {
                        "task_id": f"E168::{donor}::{state}::{ensembl_id}",
                        "fold_id": "E168_primary_CD4_donor_holdout_01",
                        "donor_id": donor,
                        "donor_role": role,
                        "culture_condition": state,
                        "perturbed_gene_id": ensembl_id,
                        "perturbed_gene_name": info["expression_axis_gene_name"],
                        "scgpt_token": info["scgpt_token"],
                        "target_stratum": stratum,
                        "split": split,
                        "truth_access_phase": truth_phase,
                        "prediction_query_required": (
                            role in {"validation", "test"}
                            or (role == "train" and stratum == "DONOR_UNSEEN_ONLY")
                        ),
                        "risk_reference_required": (
                            role == "train" and stratum == "DONOR_UNSEEN_ONLY"
                        ),
                        "primary_test_task": role == "test",
                        "training_support_contexts": int(info["perturbation_support_contexts_pretruth"]),
                    }
                )
    tasks = pd.DataFrame(task_rows)

    selected_guide_ids = {
        guide
        for packed in selected.eligible_guide_ids.astype(str)
        for guide in packed.split("+")
    }
    expected_by_guide = {
        guide: ensembl_id
        for ensembl_id, info in target_info.items()
        for guide in str(info["eligible_guide_ids"]).split("+")
    }
    frame = obs.copy()
    frame["guide_type_normalized"] = frame.guide_type.map(canonical_type)
    relevant = frame.loc[
        frame.guide_type_normalized.eq("non-targeting")
        | frame.guide_id.astype(str).isin(selected_guide_ids)
    ].copy()
    relevant["observed_ensembl_core"] = (
        relevant.perturbed_gene_id.astype(str).str.split(".").str[0]
    )
    relevant["expected_ensembl_core"] = relevant.guide_id.astype(str).map(expected_by_guide).fillna("")
    targeting_relevant = relevant.guide_type_normalized.eq("targeting")
    mismatch = relevant.loc[
        targeting_relevant
        & relevant.observed_ensembl_core.ne(relevant.expected_ensembl_core)
    ]
    if not mismatch.empty:
        raise RuntimeError(
            "selected guide design-to-observed target mapping changed: "
            + mismatch[["guide_id", "expected_ensembl_core", "observed_ensembl_core"]]
            .drop_duplicates()
            .head(20)
            .to_json(orient="records")
        )
    relevant["ensembl_core"] = relevant.expected_ensembl_core
    x_phase = []
    for row in relevant.itertuples(index=False):
        role = donor_role[str(row.donor_id)]
        if row.guide_type_normalized == "non-targeting":
            phase = "PRETRUTH_CONTROL_X"
            stratum = "CONTROL"
        else:
            stratum = target_info[str(row.ensembl_core)]["target_stratum"]
            if role == "test":
                phase = "POSTGATE_TEST_TRUTH_X"
            elif stratum == "COLUMN_UNSEEN":
                phase = "FORBIDDEN_COLUMN_UNSEEN_X"
            elif role == "train":
                phase = "PRETRUTH_TRAIN_X"
            elif role == "validation":
                phase = "PRETRUTH_VALIDATION_X"
            else:
                raise RuntimeError(f"unexpected role {role}")
        x_phase.append((phase, stratum, role))
    relevant[["x_access_phase", "target_stratum", "donor_role"]] = pd.DataFrame(x_phase, index=relevant.index)
    access = relevant[
        [
            "metadata_row_index",
            "10xrun_id",
            "donor_id",
            "donor_role",
            "culture_condition",
            "guide_id",
            "guide_type_normalized",
            "perturbed_gene_id",
            "perturbed_gene_name",
            "observed_ensembl_core",
            "expected_ensembl_core",
            "ensembl_core",
            "target_stratum",
            "x_access_phase",
        ]
    ].sort_values("metadata_row_index")
    if tasks.loc[tasks.primary_test_task].shape[0] != 600:
        raise RuntimeError("primary test manifest is not 600 tasks")
    return tasks, access


def write_protocol() -> None:
    text = f"""# E168｜Primary Human CD4+ T-cell fresh external confirmation

## 研究问题

在 SafeConf、RIAG v2、上游模型和所有判定线固定后，使用一位完整留出的真实供体，检验 SafeConf 能否比预测幅度更有效地优先发现 scGPT–GEARS 双模型的高误差扰动预测。

本实验是一个新的公开数据集上的 prospective-style sealed evaluation。它不是新做湿实验；四位供体中只有一位最终 test donor，因此 600 个任务不能当成 600 位独立受试者。

## 信息边界

- metadata freeze 只解码 `{'`, `'.join(ALLOWED_VALUE_PATHS)}`。
- `X`、`layers`、`n_cells`、`total_counts`、`log10_n_cells`、全部 `keep_*` 值均未读取。
- DE、guide knockdown、显著性和效应筛选文件不下载、不打开。
- 下载 H5AD 和计算整文件哈希只搬运字节；首次解析表达值发生在下一阶段的隔离 asset builder。
- test donor 的 non-targeting control 是预测时给定的基础状态，可在 pretruth 阶段读取；test donor 的 targeting rows 必须等 RIAG v2 snapshot 提交后才解封。

## 固定任务

- 状态：Rest、Stim8hr、Stim48hr。
- donor：2 train、1 validation、1 test；角色通过 run-aware 可行分配中的最小 SHA-256 决定。
- 靶基因：用官方 expression-independent sgRNA design/annotation、表达基因身份轴、冻结 scGPT 词表，以及每条 guide 在 12 个 donor×state 中是否存在身份行，定义 label-free assay-available universe；随后按 SHA-256 固定 200 个。H5AD 中 test targeting 的 `n_cells`、表达、效应和显著性均不参与选择。身份可用性可能产生轻微 availability selection，必须披露。
- 其中 40 个 `COLUMN_UNSEEN` 在 train/validation 阶段都不读取 targeting X；160 个 `DONOR_UNSEEN_ONLY` 用两个 train donor 监督训练、一个 validation donor 选 epoch，最终检验整供体迁移。
- test：200 targets × 3 states = 600 tasks；三种状态分别形成实际排序 batch。

## 固定表达定义

同一 guide×donor×state 的原始 UMI 行先求和，再计算 `log1p(1e4 * counts / library_sum)`；同 donor×state 的 non-targeting guide 原始计数先合并再归一化。单 guide effect 是 targeting profile 减 matched NTC，target effect 是预先合格的共同 guides 等权平均。test guide 一致性不用于删任务。

## 固定 512-gene panel

200 个 target genes 全部纳入；其余 312 个只根据两个 train donors 的六个 NTC 状态平均表达补足，并限制在固定 scGPT vocab。相同表达按 Ensembl ID 排序。该结果只称为 512-gene reduced-panel pseudobulk benchmark。

## 上游预测器

- scGPT：whole-human checkpoint，seeds {MODEL_SEEDS}，最多 10 epochs，Adam lr=1e-4，batch=16，patience=3。
- GEARS：E112 架构，seeds {MODEL_SEEDS}，最多 40 epochs，Adam lr=1e-3，batch=16，patience=6；GO 图来自冻结的外部 E107 GO 文件，coexpression 只用两个 train donors 的 NTC profiles 构建。
- test graph 不含 `y`；必须使用 query-only forward，禁止复用 E112 同时构造 test truth 的接口。
- 强基线：NoChange；160 seen targets 另报同 state 的 TrainDonorEffectMean。

## SafeConf 与 pretruth gate

固定 confidence 为 `z(context_similarity_max) + z(log1p(perturbation_support_count)) - z(model_disagreement_rmse)`，risk 为其相反数。它直接调用 `protocol_v0_2.zscore_by_ref`，reference 是 960 个监督训练 tasks 的部署前特征，不改成 validation reference。`perturbation_support_count` 是进入监督训练的 donor×state contexts 数，因此 seen=6、column-unseen=0，不用细胞数替换。z 的中心为 train median，scale 为 train IQR；IQR 无效或 ≤1e-9 时退到 train std，std 仍无效时 scale=1。40 个 unseen validation truth 保持不读。预测幅度是双模型 seed-mean ensemble 的 512-gene RMS。

`context_similarity_max` 明确定义为 query donor×state 的 512-gene NTC 向量，对六个 train donor×state NTC 向量的最大 cosine similarity；`model_disagreement_rmse` 是 scGPT 三 seed 均值效应与 GEARS 三 seed 均值效应的 512-gene RMSE。正式 risk 使用两个 family seed-mean；G4 的 seed-risk 则固定配对同编号 scGPT/GEARS seed。所有 z 在三种 state 合并的 train reference 上一次性计算，之后才按 state 分批排序。

RIAG v2 按每个 state 分别检查：源文件/访问隔离、risk 非退化、cutoff ties、每个 predictor 的任务依赖、三 seed 排名稳定性及与 magnitude 的同序性。risk/prediction/magnitude 的 operational tolerance 都固定为 1e-6，rounding 为 NumPy ties-to-even。low-risk coverage 登记 20%、25%……95%、100%，另登记最高风险 review 20%。cutoff 穿过并列组时记 `TIEBREAK_REQUIRED`，不加 jitter。

G2 要求每个 state 至少两个量化 risk levels 且 population std>1e-6。G3 对 3 个 scGPT seed、3 个 GEARS seed及两个 family seed-mean 分别要求向量有限、至少两个量化后不同的任务向量、至少一个坐标跨任务 std>1e-6。G4 将同 seed 的 scGPT/GEARS 组成 seed-risk，要求三组 pairwise Spearman 中位数≥0.5，并以 target gene 整簇 bootstrap 2,000 次后的 95% CI 下界>0。G2 与 G4 必须在 `all_200` 和 `seen_160` 两个 registered strata × 3 states 分别通过，防止 support 的 160/40 二分掩盖 seen targets 内的分歧塌缩；column-unseen 40 只作描述。G1/G2/G3/G4 任一失败，test targeting X 保持未读并记录 `PRETRUTH_ABORTED`；不换 seed、不抖动分数、不降低门槛。risk 与 magnitude 同序不阻止解封，但状态必须写成 `EVALUABLE_BASELINE_EQUIVALENT`，不得宣称增量价值。

## 主要终点

每个 task 的 loss 是 ensemble effect 与 truth effect 的 512-gene RMSE。coverage 固定为 0.20–1.00、步长 0.05，AURC 对 coverage 宽度 0.8 归一化；ties 使用 E167a 的 tie-average/best/worst legal order，primary 使用 tie-average。主要效应为三个 state 等权平均的 `AURC_magnitude - AURC_SafeConf`。

- 按 target gene 整簇 bootstrap 10,000 次，seed=2026071681；一个 target 的三个 states 同进同出。
- 按 target gene 整簇交换 candidate/magnitude，单侧 permutation 100,000 次，seed=2026071682。
- `CONFIRMATION_PASS_NONTRIVIAL` 要求：全部 pretruth gates 通过；三个 states 上 ensemble 胜 NoChange 的任务比例都 >0.5；全 200 targets 的 delta>0、CI 下界>0、p<0.05、至少 2/3 state delta>0；随后 160 seen targets 也需 CI 下界>0 且 p<0.05。
- 全 200 通过但 160 seen 不通过，记为 `PARTIAL_SUPPORT_STRATIFICATION_ONLY`；其余为 `NO_CONFIRMATION`。

所有 ties、失败状态和负结果原样保留。`deployment_authorized=false`；本实验不授权临床或自动湿实验决策。

## 解释边界

三种 state 分批排序后，context similarity 在单个 state 的 200 个 targets 内完全相同，因此不会改变该 batch 的名次；E168 实际检验 disagreement 与 support 两部分，不能单独证明完整 context component。必须同时报告 disagreement-only、support-only、context-only 与 magnitude 的 tie-aware AURC。TrainDonorEffectMean 若优于深度模型，只能把结论写成“对指定上游预测器的风险分诊”，不能写成预测模型达到 SOTA。
"""
    (OUT / "PREREG_ANALYSIS_PLAN.md").write_text(text, encoding="utf-8")
    (OUT / "CANDIDATE_SELECTION_NOTE.md").write_text(
        "# E168 candidate selection note\n\n"
        "TianKampmann2019 已出现在 E129/E153 和历史 SafeTrans 资产中，因此只能作为桥接重分析，不能再称 untouched confirmation。"
        "Primary Human CD4 数据替换发生在任何表达 X 被读取之前；理由是 freshness、真实 donor 重复和三个状态，而不是效果结果。"
        "E167 v1 的 FAIL 与 E167a 的回顾性修正保持不变。\n",
        encoding="utf-8",
    )
    (OUT / "PREFREEZE_CORRECTION_LOG.md").write_text(
        "# E168 pre-freeze correction log\n\n"
        "独立审计在任何 freeze commit 和任何 X 访问之前，否决了一版未提交的候选清单：它曾用四位 donor 的 targeting `n_cells` 和 12-combination coverage 作 eligibility。"
        "`n_cells` 可能携带扰动后存活信息，故该候选输出已删除，未用于本合同。最终版本用官方 expression-independent guide design/annotation、基因身份轴、scGPT 词表和 label-free guide identity presence 定义 assay-available universe，再做 hash 选择。"
        "该 availability eligibility 会如实披露；它不使用 abundance、表达或效应。整个修正过程中 test targeting X、DE、guide efficacy 和 keep_* 值均未读取。\n",
        encoding="utf-8",
    )
    (OUT / "DOWNLOAD_AND_VERIFY.md").write_text(
        "# E168 byte acquisition\n\n"
        "公开对象支持匿名 HTTP Range。VersionId 查询本身不允许匿名读取，因此下载未带 versionId；下载前后都必须重新 HEAD，并要求"
        " Content-Length、ETag、VersionId、CRC64NVME 与 `SOURCE_LOCK.json` 完全一致。\n\n"
        "```bash\n"
        "mkdir -p /home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/source\n"
        "aria2c --continue=true --max-connection-per-server=8 --split=8 --min-split-size=16M "
        "--file-allocation=none --dir=/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/source "
        "--out=GWCD4i.pseudobulk_merged.h5ad "
        f"'{SOURCE_URL}'\n"
        "sha256sum /home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/source/GWCD4i.pseudobulk_merged.h5ad\n"
        "```\n\n"
        "下载和哈希不解析 HDF5。全文件 SHA-256 写入新的 byte attestation；冻结的 SOURCE_LOCK 不回写。\n",
        encoding="utf-8",
    )


def main() -> None:
    for directory in [OUT, TABLES, MANIFESTS, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    missing = [
        str(path)
        for path in [
            SAMPLE_METADATA,
            DATA_README,
            GUIDE_LIBRARY,
            GEARS_GO,
            *SCGPT_FILES,
            *METHOD_REFERENCE_FILES,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing frozen metadata/model inputs: {missing}")
    local_size = int(LOCAL_H5AD.stat().st_size) if LOCAL_H5AD.exists() else 0
    aria2_control = Path(str(LOCAL_H5AD) + ".aria2")
    if aria2_control.exists():
        local_byte_state = "partial_download_unparsed"
    elif local_size == 0:
        local_byte_state = "absent"
    elif local_size < SOURCE_LOCK["content_length_bytes"]:
        local_byte_state = "partial_download_unparsed"
    elif local_size == SOURCE_LOCK["content_length_bytes"]:
        local_byte_state = "complete_byte_copy_unparsed"
    else:
        raise RuntimeError(f"local H5AD is larger than frozen source: {local_size}")

    sample = pd.read_csv(SAMPLE_METADATA, keep_default_na=False)
    donor_roles, split_audit = choose_donor_roles(sample)
    obs, var, access_log, schema, observed_headers = load_remote_metadata()
    vocab_payload = json.loads((SCGPT_DIR / "vocab.json").read_text(encoding="utf-8"))
    vocab = set(vocab_payload if isinstance(vocab_payload, dict) else map(str, vocab_payload))
    guide_library = pd.read_csv(GUIDE_LIBRARY, keep_default_na=False)
    guide_audit, target_audit, selected, ntc_summary, selected_guide_coverage = (
        build_target_tables(obs, var, guide_library, vocab)
    )
    tasks, row_access = build_manifests(obs, donor_roles, selected)
    if not selected_guide_coverage.complete_12_donor_state_identity.all():
        raise RuntimeError("ABSTAIN_SELECTED_GUIDE_METADATA_INCOMPLETE")

    source_lock = {
        **SOURCE_LOCK,
        "local_target_path": str(LOCAL_H5AD),
        "local_byte_copy_state_at_metadata_freeze": local_byte_state,
        "local_logical_size_bytes_at_metadata_freeze": local_size,
        "local_h5ad_opened_by_freeze_script": False,
        "official_data_readme_sha256": sha256(DATA_README),
        "official_sample_metadata_sha256": sha256(SAMPLE_METADATA),
        "official_expression_independent_guide_library_sha256": sha256(GUIDE_LIBRARY),
        "observed_source_headers": observed_headers,
        "forbidden_source_keys": list(FORBIDDEN_SOURCE_KEYS),
        "allowed_hdf5_value_paths": list(ALLOWED_VALUE_PATHS),
        "forbidden_hdf5_value_paths": list(FORBIDDEN_VALUE_PATHS),
        "source_full_sha256": "PENDING_AFTER_BYTE_DOWNLOAD_BEFORE_X_ACCESS",
    }
    (OUT / "SOURCE_LOCK.json").write_text(json.dumps(source_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (OUT / "HDF5_VALUE_ACCESS_LOG.json").write_text(json.dumps(access_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (OUT / "MODEL_INPUT_LOCK.json").write_text(
        json.dumps(
            {
                "scgpt_checkpoint_files": {str(path): sha256(path) for path in SCGPT_FILES},
                "gears_external_go_file": {
                    "path": str(GEARS_GO),
                    "sha256": sha256(GEARS_GO),
                    "test_dataset_derived": False,
                },
                "method_reference_files": {
                    str(path.relative_to(ROOT)): sha256(path) for path in METHOD_REFERENCE_FILES
                },
                "metadata_freeze_script_sha256": sha256(Path(__file__).resolve()),
                "model_seeds": list(MODEL_SEEDS),
                "gene_panel_size": 512,
                "target_gene_count": N_TARGETS,
                "column_unseen_count": N_COLUMN_UNSEEN,
                "pretruth_test_targeting_x_access_count_required": 0,
                "deployment_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (OUT / "STATISTICAL_ANALYSIS_LOCK.json").write_text(
        json.dumps(
            {
                "primary_risk": "negative_frozen_protocol_v0_2_confidence",
                "confidence_formula": "z(context_similarity_max)+z(log1p(perturbation_support_count))-z(model_disagreement_rmse)",
                "z_reference": "all_960_supervised_train_tasks_pooled_across_three_states",
                "z_implementation": "protocol_v0_2.zscore_by_ref_train_median_iqr_std_fallback",
                "context_similarity_definition": "max_cosine_query_donor_state_NTC_to_six_train_donor_state_NTC_vectors_on_frozen_512_panel",
                "support_definition": "number_of_supervised_train_donor_state_contexts_seen_6_unseen_0",
                "disagreement_definition": "rmse_512_between_scgpt_three_seed_mean_and_gears_three_seed_mean_effects",
                "seed_risk_pairing_for_g4": "same_seed_scgpt_and_gears_then_frozen_formula",
                "z_scope": "pooled_once_before_state_specific_ranking",
                "primary_comparator": "ensemble_predicted_magnitude_rms_512",
                "loss": "ensemble_effect_rmse_512",
                "ranking_batches": list(STATES),
                "coverage_grid": [round(value, 2) for value in np.arange(0.20, 1.001, 0.05)],
                "aurc_normalization_width": 0.8,
                "tie_primary": "tie_average",
                "tie_bounds": ["best_legal_order", "worst_legal_order"],
                "operational_tolerance": 1e-6,
                "rounding": "numpy_ties_to_even",
                "riag_g2_min_levels": 2,
                "riag_g2_population_std_min_exclusive": 1e-6,
                "riag_g2_registered_strata": ["all_200", "seen_160"],
                "riag_g4_pairwise_spearman_median_min": 0.5,
                "riag_g4_cluster_bootstrap_draws": 2000,
                "riag_g4_cluster_bootstrap_ci_lower_min_exclusive": 0.0,
                "riag_g4_registered_strata": ["all_200", "seen_160"],
                "primary_cluster": "perturbed_gene_id_keep_three_states_together",
                "bootstrap_draws": 10000,
                "bootstrap_seed": 2026071681,
                "permutation_draws": 100000,
                "permutation_seed": 2026071682,
                "permutation_alternative": "SafeConf_delta_greater_than_zero",
                "alpha": 0.05,
                "primary_strata_order": ["all_200", "seen_160", "column_unseen_40_descriptive"],
                "secondary_ablations": [
                    "disagreement_only",
                    "support_only",
                    "context_only",
                    "magnitude",
                ],
                "deployment_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    donor_roles.to_csv(MANIFESTS / "E168_DONOR_STATE_ROLES.csv", index=False)
    split_audit.to_csv(TABLES / "E168_DONOR_SPLIT_AUDIT.csv", index=False)
    ntc_summary.to_csv(TABLES / "E168_NTC_METADATA_AUDIT.csv", index=False)
    guide_audit.to_csv(TABLES / "E168_GUIDE_ELIGIBILITY_AUDIT.csv", index=False)
    selected_guide_coverage.to_csv(
        TABLES / "E168_SELECTED_GUIDE_METADATA_COVERAGE.csv", index=False
    )
    target_audit.to_csv(TABLES / "E168_TARGET_ELIGIBILITY_AUDIT.csv", index=False)
    selected.to_csv(MANIFESTS / "E168_SELECTED_TARGETS.csv", index=False)
    tasks.to_csv(MANIFESTS / "E168_TASK_MANIFEST.csv", index=False)
    row_access.to_csv(MANIFESTS / "E168_ROW_ACCESS_MANIFEST.csv", index=False)
    pd.DataFrame(access_log).to_csv(TABLES / "E168_HDF5_VALUE_ACCESS_AUDIT.csv", index=False)
    write_protocol()

    chosen = donor_roles.drop_duplicates("donor_id").sort_values("donor_role")
    primary = tasks.loc[tasks.primary_test_task]
    report = f"""# E168 metadata freeze report

- Source shape from allowlisted metadata: {schema['n_obs']:,} pseudobulk rows × {schema['n_vars']:,} genes.
- Decoded expression values: **0**; decoded forbidden-column values: **0**.
- Guide types observed: {', '.join(sorted(obs.guide_type.map(canonical_type).unique()))}.
- Eligible targets in the label-free assay-available universe before hash selection: {int(target_audit.eligible_target.sum()):,}; selected: {len(selected)}; column-unseen: {int(selected.target_stratum.eq('COLUMN_UNSEEN').sum())}.
- The universe requires at least two design-matched guides with identity rows in all 12 donor-state combinations. This is availability selection and is disclosed; test targeting `n_cells`, expression, DE, guide efficacy and `keep_*` values were not used.
- Final test manifest: {len(primary)} tasks in {primary.culture_condition.nunique()} state-specific ranking batches; test donor `{primary.donor_id.iloc[0]}`. The three batches share one donor and the same target genes; they are not independent biological replicates.
- Donor roles: {', '.join(f'{row.donor_id}={row.donor_role}' for row in chosen.itertuples(index=False))}.
- Relevant row-access manifest: {len(row_access):,} HDF5 rows. Any row absent from this manifest is default-deny.
- Local source-byte state during this freeze: `{local_byte_state}` ({local_size:,} bytes). The freeze script did not open the local H5AD. Byte download/hash do not decode X; test targeting X remains sealed until the pretruth snapshot is committed to both remotes.
"""
    (REPORTS / "E168_METADATA_FREEZE_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E168 先看这个\n\n先读 `reports/E168_METADATA_FREEZE_REPORT.md` 和 `PREREG_ANALYSIS_PLAN.md`。当前只冻结 metadata、任务和访问边界，尚未读取任何表达值。\n",
        encoding="utf-8",
    )

    artifacts = sorted(
        path for path in OUT.rglob("*") if path.is_file() and path.name != "RUN_STATUS.json"
    )
    status = {
        "experiment": "E168_primary_human_cd4_fresh_confirmation",
        "stage": "F1_METADATA_FREEZE",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS",
        "python": platform.python_version(),
        "source_schema": schema,
        "n_eligible_targets": int(target_audit.eligible_target.sum()),
        "n_selected_targets": int(len(selected)),
        "n_column_unseen_targets": int(selected.target_stratum.eq("COLUMN_UNSEEN").sum()),
        "n_primary_test_tasks": int(len(primary)),
        "n_selected_guides": int(len(selected_guide_coverage)),
        "all_selected_guides_present_in_12_donor_states": bool(
            selected_guide_coverage.complete_12_donor_state_identity.all()
        ),
        "test_donor": str(primary.donor_id.iloc[0]),
        "test_targeting_x_values_read": 0,
        "all_expression_x_values_read": 0,
        "forbidden_hdf5_values_read": 0,
        "local_byte_copy_state": local_byte_state,
        "local_logical_size_bytes": local_size,
        "local_h5ad_opened_by_freeze_script": False,
        "deployment_authorized": False,
        "artifact_sha256": {str(path.relative_to(OUT)): sha256(path) for path in artifacts},
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
