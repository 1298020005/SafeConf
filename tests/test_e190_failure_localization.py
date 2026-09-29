"""Unit tests for E190 failure-localization decision rules."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.scripts.diagnose_e190_safeconf_v4 import classify_abnormalities


def test_calibration_abnormality_does_not_automatically_imply_gate_failure() -> None:
    result = classify_abnormalities(
        raw_delta_u20=0.10,
        calibrated_delta_u20=0.10,
        p_slope=-0.1,
        h_slope=0.5,
        v2_loss_from_best=0.0,
        mean_weight=0.9,
    )
    assert result["calibration_abnormality"] is True
    assert result["gate_misrouting"] is False


def test_gate_misrouting_requires_better_branch_and_wrong_direction() -> None:
    result = classify_abnormalities(
        raw_delta_u20=0.10,
        calibrated_delta_u20=0.10,
        p_slope=0.5,
        h_slope=0.5,
        v2_loss_from_best=0.08,
        mean_weight=0.01,
    )
    assert result["gate_misrouting"] is True
    assert result["calibration_abnormality"] is False


def test_history_underperformance_is_distinct_from_calibration() -> None:
    result = classify_abnormalities(
        raw_delta_u20=-0.10,
        calibrated_delta_u20=-0.10,
        p_slope=0.5,
        h_slope=0.5,
        v2_loss_from_best=0.0,
        mean_weight=0.1,
    )
    assert result["history_branch_underperformance"] is True
    assert result["calibration_abnormality"] is False
    assert result["gate_misrouting"] is False
