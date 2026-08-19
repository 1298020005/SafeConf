from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetSpec:
    dataset_name: str
    dataset_family: str
    h5ad_path: Path
    context_col: str | None = None
    perturbation_col: str | None = None
    control_col: str | None = None
    control_values: list[str] = field(default_factory=lambda: ["control", "ctrl", "non-targeting", "nt"])
    layer: str | None = None
    n_genes: int = 5000
    min_cells: int = 6
    notes: str = ""
