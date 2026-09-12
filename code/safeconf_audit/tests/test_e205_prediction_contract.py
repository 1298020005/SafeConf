from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e201_txpert_sealed_prediction.py"
SPEC = importlib.util.spec_from_file_location("sealed_prediction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SealedPredictionContractTests(unittest.TestCase):
    def test_original_e201_contract_is_preserved(self) -> None:
        contract = MODULE.FAMILY_CONTRACTS["gat"]
        self.assertEqual(contract["seal_status"], "SEALED_16_CHECKPOINTS")
        self.assertEqual(contract["status_file"], "E201_PREDICTION_RUN.json")
        self.assertEqual(contract["model_type"], "gnn")
        self.assertEqual(contract["layer_type"], "gat_v2")
        self.assertIsNone(contract["architecture_config"])

    def test_e205_contract_requires_exphormer_identity(self) -> None:
        contract = MODULE.FAMILY_CONTRACTS["exphormer"]
        self.assertEqual(
            contract["seal_status"], "SEALED_16_EXPHORMER_CHECKPOINTS"
        )
        self.assertEqual(contract["model_family"], "TxPert-Exphormer")
        self.assertEqual(contract["model_type"], "exphormer")
        self.assertEqual(contract["layer_type"], "exphormer_w_mpnn")
        self.assertEqual(contract["architecture_config"], "config-exphormer")


if __name__ == "__main__":
    unittest.main()
