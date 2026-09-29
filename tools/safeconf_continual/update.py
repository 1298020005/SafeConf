"""Trigger and release policy for continual SafeConf updates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseMetrics:
    delta_u20_new_tasks: float
    relative_aurc_degradation_anchor: float
    miss_rate_degradation: float
    nonnegative_strata_fraction: float


@dataclass(frozen=True)
class UpdateDecision:
    trigger: bool
    reasons: tuple[str, ...]


class ContinualUpdateManager:
    """Preregistered update triggers and rollback-compatible release gates."""

    ERROR_THRESHOLDS = (10, 25, 50, 100)

    @staticmethod
    def public_update_due(current_clusters: int, new_clusters: int, new_studies: int) -> UpdateDecision:
        reasons = []
        if new_clusters >= 25:
            reasons.append("at_least_25_new_clusters")
        if current_clusters > 0 and new_clusters / current_clusters >= 0.10:
            reasons.append("at_least_10_percent_bank_growth")
        if new_studies >= 1:
            reasons.append("new_independent_study")
        return UpdateDecision(bool(reasons), tuple(reasons))

    @classmethod
    def error_update_due(cls, previous_clusters: int, current_clusters: int) -> UpdateDecision:
        crossed = [
            value for value in cls.ERROR_THRESHOLDS
            if previous_clusters < value <= current_clusters
        ]
        reasons = [f"crossed_{value}_feedback_clusters" for value in crossed]
        if current_clusters > 100 and current_clusters // 50 > previous_clusters // 50:
            reasons.append("additional_50_feedback_clusters")
        return UpdateDecision(bool(reasons), tuple(reasons))

    @staticmethod
    def release(metrics: ReleaseMetrics) -> UpdateDecision:
        failed = []
        if metrics.delta_u20_new_tasks < -0.005:
            failed.append("new_task_u20_noninferiority_failed")
        if metrics.relative_aurc_degradation_anchor > 0.05:
            failed.append("anchor_aurc_failed")
        if metrics.miss_rate_degradation > 0.02:
            failed.append("high_risk_miss_rate_failed")
        if metrics.nonnegative_strata_fraction < 0.60:
            failed.append("strata_consistency_failed")
        return UpdateDecision(not failed, tuple(failed) if failed else ("release_gate_passed",))
