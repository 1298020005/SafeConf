"""Metadata-only E260/E261 panel and barcode-sampling guards."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/scripts'))
from run_e260_tahoe_metadata_panel import choose_panel, eligible_pairs  # noqa: E402
from run_e261_tahoe_pretruth_cell_selection import priority, push_smallest  # noqa: E402


def test_locked_panel_uses_coverage_only_and_seals_six_lines() -> None:
    data = []
    for i in range(24):
        line = f'line_{i:02d}'
        data.append((line, 'DMSO', 'DMSO_TF', 'plate1', 1000))
        for j in range(100):
            data.append((line, f'drug_{j:03d}', f'drug_{j:03d}', 'plate1',
                         64 + i + j))
    counts = pd.DataFrame(data, columns=['cell_line', 'drug_dose', 'drug',
                                         'plate', 'n_full_cells'])
    eligible = eligible_pairs(counts)
    panel, audit = choose_panel(eligible)
    assert len(panel) == 2400
    assert audit['tier'] == [24, 100, [12, 6, 6]]
    assert panel.groupby('split').cell_line.nunique().to_dict() == {
        'train': 12, 'validation': 6, 'test_sealed': 6}
    assert audit['test_perturbation_truth_loaded'] == 0


def test_hash_sampling_keeps_lowest_priorities() -> None:
    heap = []
    for barcode in ('C', 'A', 'D', 'B'):
        push_smallest(heap, (-priority(barcode), barcode), 2)
    retained = {item[1] for item in heap}
    expected = set(sorted(('C', 'A', 'D', 'B'), key=priority)[:2])
    assert retained == expected
