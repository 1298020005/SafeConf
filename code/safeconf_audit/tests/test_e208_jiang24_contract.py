from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/audit_e208_jiang24_contract.py"
SPEC = importlib.util.spec_from_file_location("jiang24_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Jiang24ContractTests(unittest.TestCase):
    def test_headerless_split_is_counted_without_dropping_first_barcode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_text(
                "cell-a,train\ncell-b,val\ncell-c,test\n", encoding="utf-8"
            )
            record, barcodes = MODULE.audit_split(path)
        self.assertEqual(record["valid_rows"], 3)
        self.assertEqual(record["unique_barcodes"], 3)
        self.assertEqual(barcodes, {"cell-a", "cell-b", "cell-c"})
        self.assertEqual(
            record["split_counts"], {"test": 1, "train": 1, "val": 1}
        )
        self.assertTrue(record["gates"]["well_formed"])

    def test_duplicate_and_unknown_split_are_not_silently_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_text(
                "cell-a,train\ncell-a,test\ncell-b,unknown\n", encoding="utf-8"
            )
            record, _ = MODULE.audit_split(path)
        self.assertEqual(record["duplicate_barcodes"], 1)
        self.assertFalse(record["gates"]["unique"])
        self.assertFalse(record["gates"]["well_formed"])


if __name__ == "__main__":
    unittest.main()
