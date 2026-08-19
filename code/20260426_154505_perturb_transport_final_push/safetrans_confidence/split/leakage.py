from __future__ import annotations

import pandas as pd


def check_split_leakage(split_df: pd.DataFrame) -> dict[str, int]:
    """Return leakage counts; all should be zero for PASS."""
    test = split_df[split_df["split"] == "test"].copy()
    leakage = test[test.get("pair_seen_in_train", False) == True]  # noqa: E712
    return {
        "test_pair_leakage_rows": int(len(leakage)),
        "n_test": int(len(test)),
    }


def assert_no_test_leakage(split_df: pd.DataFrame) -> None:
    stats = check_split_leakage(split_df)
    if stats["test_pair_leakage_rows"] != 0:
        raise AssertionError(f"Leakage detected: {stats}")
