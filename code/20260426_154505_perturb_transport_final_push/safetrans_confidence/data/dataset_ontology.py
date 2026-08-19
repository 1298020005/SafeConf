from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ONTOLOGY_PATH = Path(__file__).resolve().parents[1] / "config" / "datasets" / "dataset_ontology.csv"


@dataclass(frozen=True)
class DatasetOntologyRow:
    dataset_name: str
    as_run_scoring_family: str
    source_modality: str
    perturbation_unit: str
    include_in_gene_perturbation_evidence: bool
    include_in_chemical_evidence: bool
    include_in_cytokine_stimulation_evidence: bool
    chemical_exposure_contract_status: str
    ontology_status: str
    rationale: str


@lru_cache(maxsize=1)
def load_dataset_ontology() -> dict[str, DatasetOntologyRow]:
    rows: dict[str, DatasetOntologyRow] = {}
    with ONTOLOGY_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(None):
                raise ValueError(f"dataset ontology row has extra columns: {row.get(None)!r}")
            dataset_name = row["dataset_name"].strip()
            if not dataset_name:
                raise ValueError("dataset ontology contains an empty dataset_name")
            if dataset_name in rows:
                raise ValueError(f"duplicate dataset ontology row: {dataset_name}")
            item = DatasetOntologyRow(
                dataset_name=dataset_name,
                as_run_scoring_family=row["as_run_scoring_family"],
                source_modality=row["source_modality"],
                perturbation_unit=row["perturbation_unit"],
                include_in_gene_perturbation_evidence=_parse_bool(
                    row["include_in_gene_perturbation_evidence"]
                ),
                include_in_chemical_evidence=_parse_bool(row["include_in_chemical_evidence"]),
                include_in_cytokine_stimulation_evidence=_parse_bool(
                    row["include_in_cytokine_stimulation_evidence"]
                ),
                chemical_exposure_contract_status=row["chemical_exposure_contract_status"],
                ontology_status=row["ontology_status"],
                rationale=row["rationale"],
            )
            rows[item.dataset_name] = item
    return rows


def get_dataset_ontology(dataset_name: str) -> DatasetOntologyRow | None:
    return load_dataset_ontology().get(str(dataset_name))


def assign_as_run_scoring_family(dataset_name: str, default: str = "gene_main") -> str:
    row = get_dataset_ontology(dataset_name)
    return row.as_run_scoring_family if row is not None else default


def dataset_names_for_as_run_family(family: str) -> list[str]:
    return sorted(
        row.dataset_name
        for row in load_dataset_ontology().values()
        if row.as_run_scoring_family == family
    )


def build_dataset_ontology_table() -> list[dict[str, object]]:
    return [
        {
            "dataset_name": row.dataset_name,
            "as_run_scoring_family": row.as_run_scoring_family,
            "source_modality": row.source_modality,
            "perturbation_unit": row.perturbation_unit,
            "include_in_gene_perturbation_evidence": row.include_in_gene_perturbation_evidence,
            "include_in_chemical_evidence": row.include_in_chemical_evidence,
            "include_in_cytokine_stimulation_evidence": row.include_in_cytokine_stimulation_evidence,
            "chemical_exposure_contract_status": row.chemical_exposure_contract_status,
            "ontology_status": row.ontology_status,
            "rationale": row.rationale,
        }
        for row in load_dataset_ontology().values()
    ]


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value in dataset ontology: {value!r}")
