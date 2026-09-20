import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "scripts"
    / "run_e216_pretruth_scoring.py"
)
SPEC = importlib.util.spec_from_file_location("run_e216_pretruth_scoring", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_architecture_balancing_prevents_seed_replication_weight():
    # Four latent seeds predict 2; one linear architecture predicts 10.
    members = np.asarray([[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[10.0]]])
    result = MODULE.architecture_balanced_family(members)
    np.testing.assert_allclose(result["centroid"], [[6.0]])
    np.testing.assert_allclose(result["weights"], [0.125, 0.125, 0.125, 0.125, 0.5])
    np.testing.assert_allclose(result["lower_bound"], [4.0])


def test_architecture_family_rejects_nonfinite_predictions():
    members = np.zeros((5, 2, 3))
    members[0, 0, 0] = np.nan
    try:
        MODULE.architecture_balanced_family(members)
    except MODULE.PretruthFailure as exc:
        assert "five finite" in str(exc)
    else:
        raise AssertionError("non-finite family must fail")
