from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.scripts.audit_mcfaline_metadata import history_eligibility, summarize_tasks, task_frame


def test_history_eligibility_uses_train_metadata_only() -> None:
    obs = pd.DataFrame(
        {
            "condition": ["control", "G1", "control", "G1", "G2"],
            "cell_type": ["A", "A", "B", "B", "B"],
            "treatment": ["T", "T", "T", "T", "T"],
        },
        index=["c0", "c1", "c2", "c3", "c4"],
    )
    split = pd.Series(["train", "train", "test", "test", "test"], index=obs.index)
    frame = task_frame(obs, split)
    task_counts, summary = summarize_tasks(frame)
    eligible = history_eligibility(task_counts).set_index("condition")
    assert summary.set_index("split").loc["test", "n_treated_tasks"] == 2
    assert bool(eligible.loc["G1", "has_internal_history"])
    assert not bool(eligible.loc["G2", "has_internal_history"])
    assert eligible.loc["G1", "train_history_cells"] == 1


def test_task_frame_rejects_incomplete_split() -> None:
    obs = pd.DataFrame(
        {"condition": ["G1"], "cell_type": ["A"], "treatment": ["T"]},
        index=["c1"],
    )
    try:
        task_frame(obs, pd.Series(dtype=str))
    except ValueError as exc:
        assert "missing from split manifest" in str(exc)
    else:
        raise AssertionError("incomplete split must fail closed")
