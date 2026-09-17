"""Contract tests for the truth-isolated E208 formal training automation."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
JOB_SCRIPT = ROOT / "tools/scripts/run_e208_formal_training_job.py"
QUEUE_SCRIPT = ROOT / "tools/scripts/run_e208_formal_training_queue.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


JOB = load("e208_formal_job", JOB_SCRIPT)
QUEUE = load("e208_formal_queue", QUEUE_SCRIPT)


class E208FormalTrainingTests(unittest.TestCase):
    def test_command_preserves_registered_training_and_disables_test(self) -> None:
        command = JOB.build_command(
            python=Path("/env/python"),
            perturbench_repo=Path("/pb"),
            data_dir=Path("/data"),
            run_dir=Path("/run"),
            architecture="latent",
            seed=4,
            batch_size=1000,
        )
        joined = "\n".join(command)
        self.assertIn("test=false", command)
        self.assertIn("trainer.max_epochs=400", command)
        self.assertIn("trainer.min_epochs=5", command)
        self.assertIn("callbacks.early_stopping.patience=50", command)
        self.assertIn("SingleCellPerturbationWithControls.from_h5", joined)
        self.assertNotIn("trainer.test", joined)
        self.assertNotIn("target_truth", joined)

    def test_unregistered_architecture_seed_pair_is_rejected(self) -> None:
        with self.assertRaises(JOB.FormalJobFailure):
            JOB.build_command(
                python=Path("/env/python"),
                perturbench_repo=Path("/pb"),
                data_dir=Path("/data"),
                run_dir=Path("/run"),
                architecture="linear",
                seed=2,
                batch_size=2000,
            )

    def test_resolved_config_rejects_test_true(self) -> None:
        config = {
            "train": True,
            "test": True,
            "seed": 1,
            "model": {"_target_": "perturbench.modelcore.models.LinearAdditive"},
            "data": {
                "_target_": "perturbench.data.modules.H5LitModule",
                "data_iter_factory": {
                    "_target_": (
                        "perturbench.data.datasets.h5."
                        "SingleCellPerturbationWithControls.from_h5"
                    ),
                    "cache_size": 0,
                },
                "loader": {"batch_size": 2000, "num_workers": 8},
            },
            "trainer": {
                "max_epochs": 400,
                "min_epochs": 5,
                "deterministic": True,
            },
            "callbacks": {
                "early_stopping": {"patience": 50},
                "model_checkpoint": {"save_last": True},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(JOB.FormalJobFailure):
                JOB.validate_resolved_config(path, "linear", 1, 2000)

    def test_smoke_gate_wait_pass_and_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            self.assertEqual(QUEUE.smoke_gate(path)[0], "WAIT")
            path.write_text(
                json.dumps(
                    {
                        "status": "SMOKE_PASS_FORMAL_QUEUE_NOT_STARTED",
                        "selected_batch_size": 1000,
                        "test_truth_access": "NOT_AUTHORIZED",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(QUEUE.smoke_gate(path), ("GO", "smoke passed at batch size 1000", 1000))
            path.write_text(
                json.dumps({"status": "BLOCKED_NON_OOM_SMOKE_FAILURE"}),
                encoding="utf-8",
            )
            self.assertEqual(QUEUE.smoke_gate(path)[0], "FAIL")

    def test_completion_gate_requires_zero_truth_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            last = run_dir / "checkpoints/last.ckpt"
            best = run_dir / "checkpoints/epoch=1.ckpt"
            last.parent.mkdir()
            last.write_bytes(b"last")
            best.write_bytes(b"best")
            status = {
                "experiment": "E208_jiang24_external_confirmation",
                "stage": "D1_FORMAL_TRAINING",
                "status": "COMPLETE",
                "architecture": "latent",
                "seed": 1,
                "batch_size": 2000,
                "test_truth_access": "AUTHORIZED",
                "test_perturbed_expression_rows_read": 0,
                "last_checkpoint": str(last),
                "best_checkpoint": str(best),
                "resolved_config_sha256": "a" * 64,
            }
            (run_dir / "E208_RUN_STATUS.json").write_text(
                json.dumps(status), encoding="utf-8"
            )
            with self.assertRaises(QUEUE.QueueFailure):
                QUEUE.validate_complete(run_dir, "latent", 1, 2000)


if __name__ == "__main__":
    unittest.main()
