"""Align a predicted delta expression to a registered common response axis.

The original predictor endpoint is unchanged.  The returned view is used only
for comparison with a public response normalized on the smaller common axis.
It depends on the frozen prediction, NTC control and metadata; no query truth.
"""
from __future__ import annotations

import numpy as np


def common_axis_effect(prediction, control_log_expression, axis_mask):
    p = np.asarray(prediction, dtype=np.float64)
    control = np.asarray(control_log_expression, dtype=np.float64)
    mask = np.asarray(axis_mask, dtype=bool)
    if p.ndim != 2 or mask.shape != (p.shape[1],) or mask.sum() < 2:
        raise ValueError('aligned two-dimensional prediction and nonempty common axis required')
    if control.ndim == 1:
        control = np.broadcast_to(control, p.shape)
    if control.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(control).all():
        raise ValueError('finite, aligned frozen prediction and control required')
    log_level = p[:, mask] + control[:, mask]
    # A regression predictor can emit an impossible negative log expression.
    # Enforce its physical lower bound in this comparison view only, retaining
    # the original output and reporting the projected fraction per task.
    clipped_fraction = (log_level < 0).mean(axis=1)
    treated = np.expm1(np.maximum(log_level, 0.))
    ntc = np.expm1(control[:, mask])
    if not np.isfinite(treated).all() or not np.isfinite(ntc).all():
        raise ValueError('inverse-log common-axis view is nonfinite')
    treated_total, ntc_total = treated.sum(axis=1), ntc.sum(axis=1)
    if (treated_total <= 0).any() or (ntc_total <= 0).any():
        raise ValueError('common-axis view has empty expression support')
    transformed = np.log1p(1e4 * treated / treated_total[:, None])
    transformed -= np.log1p(1e4 * ntc / ntc_total[:, None])
    return transformed, clipped_fraction
