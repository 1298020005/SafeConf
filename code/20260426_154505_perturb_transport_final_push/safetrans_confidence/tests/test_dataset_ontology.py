from pathlib import Path

import yaml

from safetrans_confidence.data.dataset_ontology import (
    assign_as_run_scoring_family,
    build_dataset_ontology_table,
    dataset_names_for_as_run_family,
    get_dataset_ontology,
)
from safetrans_confidence.scoring.protocol_v0_2 import CHEM_DATASETS, assign_dataset_family

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "safetrans_confidence" / "config" / "scoring" / "protocol_v0_2.yaml"


def test_ontology_preserves_as_run_scoring_family_for_legacy_formula():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert dataset_names_for_as_run_family("chem_robust") == sorted(cfg["chem_datasets"])
    assert set(CHEM_DATASETS) == set(cfg["chem_datasets"])
    assert assign_dataset_family("SantinhaPlatt2023") == "gene_main"
    assert assign_as_run_scoring_family("unlisted_dataset", default="gene_main") == "gene_main"


def test_santinha_is_crispr_and_excluded_from_chemical_evidence():
    row = get_dataset_ontology("SantinhaPlatt2023")

    assert row is not None
    assert row.as_run_scoring_family == "gene_main"
    assert row.source_modality == "crispr_cas9"
    assert row.perturbation_unit == "gene_symbol"
    assert row.include_in_chemical_evidence is False
    assert row.chemical_exposure_contract_status == "not_chemical"


def test_cui_is_cytokine_stimulation_not_gene_perturbation_evidence():
    row = get_dataset_ontology("CuiHacohen2023")

    assert row is not None
    assert row.as_run_scoring_family == "gene_main"
    assert row.source_modality == "cytokine_stimulation"
    assert row.include_in_gene_perturbation_evidence is False
    assert row.include_in_cytokine_stimulation_evidence is True
    assert assign_dataset_family("CuiHacohen2023") == "gene_main"


def test_mcfarland_is_drug_label_but_not_a_strict_exposure_contract():
    row = get_dataset_ontology("McFarlandTsherniak2020")

    assert row is not None
    assert row.source_modality == "chemical_drug"
    assert row.include_in_chemical_evidence is True
    assert row.chemical_exposure_contract_status == "drug_only_task_key_insufficient"


def test_dataset_ontology_table_has_claim_boundary_columns():
    table = build_dataset_ontology_table()
    santinha = next(row for row in table if row["dataset_name"] == "SantinhaPlatt2023")

    assert "as_run_scoring_family" in santinha
    assert "source_modality" in santinha
    assert "include_in_chemical_evidence" in santinha
    assert santinha["include_in_chemical_evidence"] is False
    assert santinha["as_run_scoring_family"] == "gene_main"
