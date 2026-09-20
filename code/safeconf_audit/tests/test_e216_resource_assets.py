"""Tests for the expression-blind E216 resource selection."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/build_e216_jiang24_resource_assets.py"
SPEC = importlib.util.spec_from_file_location("e216_assets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E216ResourceAssetTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        rows = []
        for split in ("train", "val", "test"):
            for condition in ("control", "GENE1"):
                for index in range(9):
                    rows.append(
                        {
                            "cell_barcode": f"{split}-{condition}-{index}",
                            "split": split,
                            "cell_type": "k562",
                            "treatment": "IFNG",
                            "condition": condition,
                        }
                    )
        return pd.DataFrame(rows)

    def test_caps_development_and_preserves_every_test_row(self) -> None:
        frame = self.frame()
        selected = MODULE.resource_split(frame, perturbed_cap=3, control_cap=5)
        self.assertTrue(selected.loc[frame.split.eq("test")].eq("test").all())
        for split in ("train", "val"):
            block = frame.assign(selected=selected).loc[
                frame.split.eq(split) & selected.eq(split)
            ]
            self.assertEqual(block.condition.eq("GENE1").sum(), 3)
            self.assertEqual(block.condition.eq("control").sum(), 5)

    def test_selection_is_deterministic_and_order_independent_by_barcode(self) -> None:
        frame = self.frame()
        first = MODULE.resource_split(frame, perturbed_cap=3, control_cap=5)
        shuffled = frame.sample(frac=1.0, random_state=7)
        second = MODULE.resource_split(shuffled, perturbed_cap=3, control_cap=5)
        first_map = dict(zip(frame.cell_barcode, first, strict=True))
        second_map = dict(zip(shuffled.cell_barcode, second, strict=True))
        self.assertEqual(first_map, second_map)


if __name__ == "__main__":
    unittest.main()
