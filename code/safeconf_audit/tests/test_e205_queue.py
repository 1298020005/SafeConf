from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e205_exphormer_formal_queue.py"
SPEC = importlib.util.spec_from_file_location("e205_queue", SCRIPT)
E205 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E205)


class E205QueueTests(unittest.TestCase):
    def write_status(self, run_dir: Path, payload: dict) -> Path:
        run_dir.mkdir(parents=True)
        path = run_dir / "E205_RUN_STATUS.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_profile_gate_accepts_complete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "last.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            self.write_status(
                root / "profile",
                {
                    "status": "COMPLETE",
                    "kind": "profile",
                    "target": "RPE1",
                    "seed": 1,
                    "batch_size": 64,
                    "model_family": "TxPert-Exphormer",
                    "architecture_change_only": True,
                    "base_config": "config-x-cell-gat",
                    "architecture_config": "config-exphormer",
                    "current_epoch": 1,
                    "target_perturbed_cells_accessed": 0,
                    "target_test_dataset_constructed": False,
                    "last_model_path": str(checkpoint),
                    "cuda_peak_memory_reserved_bytes": 100,
                    "fit_wall_seconds": 10.0,
                },
            )
            result = E205.validate_profile(root / "profile")
            self.assertEqual(result["model_family"], "TxPert-Exphormer")

    def test_profile_gate_rejects_wrong_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "last.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            self.write_status(
                root / "profile",
                {
                    "status": "COMPLETE",
                    "kind": "profile",
                    "target": "RPE1",
                    "seed": 1,
                    "batch_size": 64,
                    "model_family": "TxPert-STRING-GAT",
                    "architecture_change_only": True,
                    "base_config": "config-x-cell-gat",
                    "architecture_config": "config-exphormer",
                    "current_epoch": 1,
                    "target_perturbed_cells_accessed": 0,
                    "target_test_dataset_constructed": False,
                    "last_model_path": str(checkpoint),
                    "cuda_peak_memory_reserved_bytes": 100,
                    "fit_wall_seconds": 10.0,
                },
            )
            with self.assertRaisesRegex(E205.QueueFailure, "family"):
                E205.validate_profile(root / "profile")

    def test_formal_gate_and_attempt_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "last.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            run_dir = root / "formal/K562/seed_1"
            self.write_status(
                run_dir,
                {
                    "status": "COMPLETE",
                    "kind": "formal",
                    "target": "K562",
                    "seed": 1,
                    "model_family": "TxPert-Exphormer",
                    "architecture_change_only": True,
                    "base_config": "config-x-cell-gat",
                    "resolved_model_config": {
                        "pert_model": {"model_type": "exphormer"}
                    },
                    "current_epoch": 80,
                    "target_perturbed_cells_accessed": 0,
                    "target_test_dataset_constructed": False,
                    "last_model_path": str(checkpoint),
                },
            )
            E205.validate_complete(run_dir, "K562", 1)
            logs = root / "logs"
            logs.mkdir()
            (logs / "K562_seed1_attempt1.log").write_text("one", encoding="utf-8")
            (logs / "K562_seed1_attempt2.log").write_text("two", encoding="utf-8")
            self.assertEqual(E205.attempts_from_logs(logs, "K562", 1), 2)

    def test_stale_formal_run_can_resume_from_last_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "formal/RPE1/seed_1"
            checkpoint = run_dir / "checkpoints/last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            (run_dir / "E205_RUN_STATUS.json").write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "kind": "formal",
                        "target": "RPE1",
                        "seed": 1,
                        "model_family": "TxPert-Exphormer",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                E205.resume_checkpoint_for(run_dir, "RPE1", 1), checkpoint.resolve()
            )
            self.assertIsNone(E205.resume_checkpoint_for(run_dir, "K562", 1))


if __name__ == "__main__":
    unittest.main()
