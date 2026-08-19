from pathlib import Path

import pandas as pd
import pytest

from safetrans_confidence.split.leakage import assert_no_test_leakage, check_split_leakage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V21 = PROJECT_ROOT / "outputs" / "confidence_task_mvp_v2_1"


@pytest.mark.skipif(not (V21 / "tables" / "HELDOUT_PAIR_SPLITS.csv").exists(), reason="v2_1 outputs missing")
def test_v21_split_has_zero_leakage():
    split_df = pd.read_csv(V21 / "tables" / "HELDOUT_PAIR_SPLITS.csv")
    stats = check_split_leakage(split_df)
    assert stats["test_pair_leakage_rows"] == 0
    assert_no_test_leakage(split_df)
