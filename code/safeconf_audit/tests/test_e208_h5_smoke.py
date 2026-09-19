"""Contract tests for the truth-isolated E208 H5 smoke runner."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e208_jiang24_h5_smoke.py"
SPEC = importlib.util.spec_from_file_location("e208_h5_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E208H5SmokeTests(unittest.TestCase):
    def test_disjoint_preflight_requires_an_unused_gpu_and_blind_e205(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e205.json"
            path.write_text(
                json.dumps(
                    {
                        "experiment": "E205_cross_family_exphormer",
                        "status": "RUNNING",
                        "completed": 15,
                        "waiting": 0,
                        "active": [{"device": "0"}],
                        "detached": [],
                        "permanent_failures": [],
                        "target_truth_access": "NOT_AUTHORIZED",
                    }
                ),
                encoding="utf-8",
            )
            _, mode = MODULE.check_e205_gate(
                path, cuda_device="1", allow_disjoint_running=True
            )
            self.assertEqual(mode, "DISJOINT_ENGINEERING_PREFLIGHT")
            with self.assertRaises(MODULE.SmokeFailure):
                MODULE.check_e205_gate(
                    path, cuda_device="0", allow_disjoint_running=True
                )
            with self.assertRaises(MODULE.SmokeFailure):
                MODULE.check_e205_gate(
                    path, cuda_device="1", allow_disjoint_running=False
                )

    def test_command_is_one_batch_h5_and_truth_isolated(self) -> None:
        command = MODULE.build_command(
            python=Path("/env/python"),
            perturbench_repo=Path("/pb"),
            data_dir=Path("/data"),
            run_dir=Path("/run"),
            architecture="latent",
            batch_size=2000,
        )
        joined = "\n".join(command)
        self.assertIn("test=false", command)
        self.assertIn("trainer.limit_train_batches=1", joined)
        self.assertIn("trainer.limit_val_batches=1", joined)
        self.assertIn("H5LitModule", joined)
        self.assertIn("SingleCellPerturbationWithControls.from_h5", joined)
        self.assertNotIn("trainer.test", joined)
        self.assertNotIn("target_truth", joined)
        self.assertNotIn("trainer.enable_progress_bar=false", joined)

    def test_only_registered_batch_fallbacks_are_allowed(self) -> None:
        for batch_size in (2000, 1000, 500, 250):
            MODULE.build_command(
                python=Path("/env/python"),
                perturbench_repo=Path("/pb"),
                data_dir=Path("/data"),
                run_dir=Path("/run"),
                architecture="linear",
                batch_size=batch_size,
            )
        with self.assertRaises(MODULE.SmokeFailure):
            MODULE.build_command(
                python=Path("/env/python"),
                perturbench_repo=Path("/pb"),
                data_dir=Path("/data"),
                run_dir=Path("/run"),
                architecture="linear",
                batch_size=128,
            )

    def test_resolved_config_rejects_test_true(self) -> None:
        config = {
            "train": True,
            "test": True,
            "seed": 1,
            "model": {"_target_": "perturbench.modelcore.models.LatentAdditive"},
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
                "max_epochs": 1,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "deterministic": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(MODULE.SmokeFailure):
                MODULE.validate_resolved_config(path, "latent", 2000)

    def test_truth_named_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "target_truth" / "values.csv"
            bad.parent.mkdir()
            bad.write_text("forbidden\n", encoding="utf-8")
            with self.assertRaises(MODULE.SmokeFailure):
                MODULE.reject_truth_or_test_outputs(root)


if __name__ == "__main__":
    unittest.main()
