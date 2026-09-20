import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "scripts"
    / "run_e216_fixed_tau_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("run_e216_fixed_tau_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_longdouble_identity_and_fixed_certificate_curve():
    members = np.asarray(
        [
            [[0.0, 2.0], [1.0, 3.0]],
            [[2.0, 0.0], [3.0, 1.0]],
        ],
        dtype=np.float32,
    )
    truth = np.asarray([[3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    result = MODULE.recompute_identity(members, np.asarray([0.5, 0.5]), truth)
    assert result["max_abs_identity_residual"] < 1e-15
    curve = MODULE.certificate_curve(result["lower_bound"], result["family_rms_error"])
    assert curve.n_false_certificates.sum() == 0
    assert curve.tau.tolist() == list(MODULE.FIXED_E211_TAU)
