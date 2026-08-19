from __future__ import annotations

import numpy as np
import pytest

from safetrans_confidence.eval.metrics import (
    compute_aurc,
    compute_excess_aurc,
    compute_oracle_aurc,
    compute_random_aurc,
)


def test_aurc_risk_score_direction_matches_oracle_for_perfect_ranking() -> None:
    errors = np.array([0.1, 0.2, 1.0, 2.0])
    risk_scores = errors.copy()

    assert compute_aurc(errors, risk_scores, "risk") == pytest.approx(compute_oracle_aurc(errors))
    assert compute_excess_aurc(errors, risk_scores, "risk") == pytest.approx(0.0)


def test_aurc_confidence_score_direction_matches_oracle_for_perfect_ranking() -> None:
    errors = np.array([0.1, 0.2, 1.0, 2.0])
    confidence_scores = -errors

    assert compute_aurc(errors, confidence_scores, "confidence") == pytest.approx(
        compute_oracle_aurc(errors)
    )
    assert compute_excess_aurc(errors, confidence_scores, "confidence") == pytest.approx(0.0)


def test_reverse_ranking_has_larger_aurc_than_random_baseline() -> None:
    errors = np.array([0.1, 0.2, 1.0, 2.0])
    reversed_risk_scores = -errors

    assert compute_aurc(errors, reversed_risk_scores, "risk") > compute_random_aurc(errors)
    assert compute_excess_aurc(errors, reversed_risk_scores, "risk") > 0.0

