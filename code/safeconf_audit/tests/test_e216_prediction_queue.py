import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "scripts"
    / "run_e216_prediction_queue.py"
)
SPEC = importlib.util.spec_from_file_location("run_e216_prediction_queue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_completed_training_runs_requires_all_five(tmp_path):
    for architecture, seed in MODULE.EXPECTED_RUNS[:-1]:
        directory = tmp_path / architecture / f"seed_{seed}"
        directory.mkdir(parents=True)
        (directory / "E216_RUN_STATUS.json").write_text(
            json.dumps({"status": "COMPLETE"}), encoding="utf-8"
        )
    complete, waiting = MODULE.completed_training_runs(tmp_path)
    assert complete == list(MODULE.EXPECTED_RUNS[:-1])
    assert waiting == [MODULE.EXPECTED_RUNS[-1]]
