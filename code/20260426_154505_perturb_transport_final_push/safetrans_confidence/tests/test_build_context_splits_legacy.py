from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SPLITS = PROJECT_ROOT / "03_code" / "build_context_splits.py"


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_build_context_splits", LEGACY_SPLITS)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_or_blank_perturbation_labels_are_control_like() -> None:
    legacy = _load_legacy_module()

    for value in [None, np.nan, pd.NA, "", " ", "nan", "NaN", "none", "NULL"]:
        assert legacy.is_control_value(value)
        assert legacy.is_missing_label(value)

    assert not legacy.is_control_value("TP53")
    assert not legacy.is_control_value("Trametinib")
    assert not legacy.is_missing_label("TP53")
    assert not legacy.is_missing_label("control")


def test_obs_label_normalization_preserves_real_labels_without_stringifying_nan() -> None:
    legacy = _load_legacy_module()

    assert legacy.normalize_obs_label(np.nan) == ""
    assert legacy.normalize_obs_label(pd.NA) == ""
    assert legacy.normalize_obs_label("  TP53 ") == "TP53"
