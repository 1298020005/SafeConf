"""The E247 first gate is label-only and uses a stable perturbation split."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_e247_kaden_labels.py"
SPEC = importlib.util.spec_from_file_location("audit_e247_kaden_labels", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_partition_is_stable_and_label_specific():
    labels = [f"TF{i:04d}" for i in range(1000)]
    first = [MODULE.split_label(label) for label in labels]
    assert first == [MODULE.split_label(label) for label in labels]
    assert set(first) == {"train", "validation", "test"}
    assert 550 < first.count("train") < 650


def test_count_reader_never_needs_expression_values(tmp_path):
    path = tmp_path / "labels_only.h5ad"
    with h5py.File(path, "w") as handle:
        x = handle.create_group("X")
        x.attrs["shape"] = np.asarray([68, 2])
        obs = handle.create_group("obs").create_group("perturbation")
        obs.create_dataset("categories", data=np.asarray([b"control", b"A", b"B"]))
        obs.create_dataset("codes", data=np.asarray([0] * 2 + [1] * 35 + [2] * 31))
        handle.create_group("var").create_dataset("_index", data=np.asarray([b"G1", b"G2"]))
    rows, n_cells, n_genes, controls = MODULE.read_label_counts(path)
    assert (n_cells, n_genes, controls) == (68, 2, 2)
    assert [(row["perturbation"], row["n_cells"], row["eligible_ge30"]) for row in rows] == [
        ("control", 2, 0), ("A", 35, 1), ("B", 31, 1)
    ]


def test_missing_label_fails_closed(tmp_path):
    path = tmp_path / "missing_label.h5ad"
    with h5py.File(path, "w") as handle:
        x = handle.create_group("X")
        x.attrs["shape"] = np.asarray([2, 1])
        obs = handle.create_group("obs").create_group("perturbation")
        obs.create_dataset("categories", data=np.asarray([b"control"]))
        obs.create_dataset("codes", data=np.asarray([0, -1]))
        handle.create_group("var").create_dataset("_index", data=np.asarray([b"G1"]))
    with pytest.raises(RuntimeError, match="unlabeled cells"):
        MODULE.read_label_counts(path)
