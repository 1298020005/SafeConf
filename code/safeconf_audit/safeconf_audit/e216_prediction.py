"""Prediction adapters frozen for the resource-bounded E216 experiment."""

from __future__ import annotations


def counterfactual_from_fixed_controls(
    _training_data_handle,
    prediction_dataframe,
    *,
    fixed_control_h5: str,
    **kwargs,
):
    """Load only the pre-frozen control asset for counterfactual prediction.

    PerturBench normally passes its training H5 handle as the first positional
    argument.  E216 intentionally ignores that handle during prediction so the
    93GB source file is not rescanned for every model.  The asset contains only
    control cells and therefore cannot expose perturbed test outcomes.
    """

    from perturbench.data.datasets import Counterfactual

    return Counterfactual.from_h5(
        fixed_control_h5,
        prediction_dataframe,
        **kwargs,
    )
