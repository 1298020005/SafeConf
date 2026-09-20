"""Contract tests for the resource-bounded E216 training wrapper."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e216_training_job.py"
SPEC = importlib.util.spec_from_file_location("e216_job", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E216TrainingJobTests(unittest.TestCase):
    def test_formal_command_is_truth_isolated_and_resource_bounded(self) -> None:
        command = MODULE.build_command(
            python=Path("/env/python"),
            perturbench_repo=Path("/pb"),
            data_dir=Path("/data"),
            assets_dir=Path("/assets"),
            run_dir=Path("/run"),
            architecture="latent",
            seed=4,
            mode="formal",
        )
        joined = "\n".join(command)
        self.assertIn("test=false", command)
        self.assertIn("trainer.max_epochs=10", command)
        self.assertIn("trainer.min_epochs=5", command)
        self.assertIn("callbacks.early_stopping.patience=3", command)
        self.assertIn("jiang24_e216_resource_split.csv", joined)
        self.assertIn("jiang24_official_hvg4000.csv", joined)
        self.assertNotIn("trainer.test", joined)

    def test_unregistered_linear_seed_is_rejected(self) -> None:
        with self.assertRaises(MODULE.TrainingFailure):
            MODULE.build_command(
                python=Path("/env/python"),
                perturbench_repo=Path("/pb"),
                data_dir=Path("/data"),
                assets_dir=Path("/assets"),
                run_dir=Path("/run"),
                architecture="linear",
                seed=2,
                mode="formal",
            )


if __name__ == "__main__":
    unittest.main()
