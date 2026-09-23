import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_e232_similarity_robustness as e232


def test_corruption_is_deterministic_and_changes_selected_labels():
    source = pd.DataFrame({
        "target": ["T"] * 20,
        "context": ["A"] * 10 + ["B"] * 10,
        "condition": [f"g{i}" for i in range(20)],
    })
    first, mapping = e232.corrupt_indices(source, 0.25, 101)
    second, _ = e232.corrupt_indices(source, 0.25, 101)
    assert np.array_equal(first, second)
    assert np.any(first != np.arange(len(source)))
    assert all(x["condition"] != x["donor_condition"] for x in mapping)


def test_all_registered_fractions_are_bounded():
    assert e232.FRACTIONS == (0.10, 0.25, 0.50, 1.00)
    assert all(0 < value <= 1 for value in e232.FRACTIONS)
    assert e232.CANDIDATE == "similarity_variance_shrink"
