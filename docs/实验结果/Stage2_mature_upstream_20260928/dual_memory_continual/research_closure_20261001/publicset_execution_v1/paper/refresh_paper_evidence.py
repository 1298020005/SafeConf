#!/usr/bin/env python3
"""Copy existing, completed result tables into this paper package; no model fitting."""
from pathlib import Path
import hashlib
import json
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent.parent
OUT = PAPER / 'evidence'
OUT.mkdir(exist_ok=True)
SOURCES = {
    'core_all_seeds.csv': 'common_gene_axis/results/MATRIX_MACRO_RESULTS.csv',
    'feedback_all_seeds.csv': 'common_gene_axis/results/STRICT_FEEDBACK_MACRO.csv',
    'feedback_information.csv': 'common_gene_axis/results/STRICT_FEEDBACK_INFORMATION_LEDGER.csv',
    'support_controls.csv': 'common_gene_axis/results/SUPPORT_CONTROL_MACRO.csv',
    'source_magnitude_pairs.csv': 'source_magnitude_comparator_completion_v1/ALL6_FIXED_PAIRED_COMPARATORS.csv',
    'biology_label_pairs.csv': 'source_biology_label_control_v1/PAIRED_METRICS.csv',
    'orion_methods.csv': 'orion_preparation/actual_fixed_result_tables_v1/METHOD_METRIC_INTERVALS.csv',
    'orion_pairs.csv': 'orion_preparation/actual_fixed_result_tables_v1/ALL_PRESPECIFIED_PAIRED_COMPARISONS.csv',
    'orion_coverage.csv': 'orion_preparation/actual_fixed_result_tables_v1/COHORT_COVERAGE.csv',
    'pertema_same_information.csv': 'common_gene_axis/results/official_pertema_adaptation/MACRO.csv',
    'pertema_native_information.csv': 'common_gene_axis/results/native_control_pertema_adaptation/MACRO.csv',
    'frangieh_methods.csv': 'frangieh_cross_family_v1/MACRO_RESULTS.csv',
    'frangieh_pairs.csv': 'frangieh_cross_family_v1/PAIRED_BOOTSTRAP.csv',
    'frangieh_competence.csv': 'frangieh_cross_family_v1/UPSTREAM_COMPETENCE.csv',
    'frangieh_history_coverage.csv': 'frangieh_cross_family_v1/HISTORY_COVERAGE.csv',
    'frangieh_ridge_repair_methods.csv': 'frangieh_cross_family_v1/RidgeScoreRepair/MACRO_RESULTS.csv',
    'frangieh_ridge_repair_pairs.csv': 'frangieh_cross_family_v1/RidgeScoreRepair/PAIRED_BOOTSTRAP.csv',
    'frangieh_ridge_repair_ledger.csv': 'frangieh_cross_family_v1/RidgeScoreRepair/SCORE_REPAIR_LEDGER.csv',
    'alignment_reference_methods.csv': 'common_gene_axis/reference_estimand_diagnostic/MACRO.csv',
    'alignment_reference_pairs.csv': 'common_gene_axis/reference_estimand_diagnostic/PAIRED_COMPARISONS.csv',
    'alignment_risk_replay_methods.csv': 'common_gene_axis/reference_estimand_risk_replay/MACRO.csv',
    'alignment_risk_replay_pairs.csv': 'common_gene_axis/reference_estimand_risk_replay/PAIRED_COMPARISONS.csv',
}
PUBLICSET = 'publicset_execution_v1/'
for name in ['MACRO_METRIC_INTERVALS.csv', 'PAIRED_COMPARISONS.csv',
             'STRONG_SIMPLE_COMPARISONS.csv', 'MC_DATA_ALIGNMENT_COMPARISONS.csv',
             'COMMON_TASK_COVERAGE.csv', 'STRATUM_METRICS.csv', 'SEED_METRICS.csv',
             'PAIRED_STRATUM_DELTAS.csv', 'STRONG_SIMPLE_STRATUM_DELTAS.csv',
             'MC_DATA_ALIGNMENT_STRATUM_DELTAS.csv', 'DECISIONS.csv',
             'DECISION.json', 'RUN_MANIFEST.json', 'SEMANTIC_CHECK.json']:
    SOURCES['publicset_' + name.lower()] = PUBLICSET + 'statistics_v1/' + name
for name in ['CONFIG.json', 'RUN_STATUS.json']:
    SOURCES['strong_simple_' + name.lower()] = PUBLICSET + 'strong_simple_reference_v1/' + name
for name in ['RUN_MANIFEST.json', 'FIT_LEDGER.csv', 'MODEL_REUSE_LEDGER.csv',
             'HISTORY_SET_AUDIT.csv', 'CONSTRAINED_RECONSTRUCTION_DIAGNOSTIC.csv']:
    source = PUBLICSET + 'full_v2_resume/' + name
    if (ROOT / source).is_file():
        SOURCES['publicset_training_' + name.lower()] = source
for name in ['TEACHER_CONTEXT_PROVENANCE.csv', 'INPUT_AUDIT.json', 'REGISTRATION.json']:
    SOURCES['reader_base_' + name.lower()] = PUBLICSET + 'risk_followup_v1/' + name
# Import only a completed registered version. A running reader cannot fill a
# manuscript result table, and earlier reduced-input readers retain their scope.
registered = ROOT / PUBLICSET / 'risk_followup_v1/registered_universal_v1'
registration_status = registered / 'RESULT_MANIFEST.json'
reader_complete = (registration_status.is_file() and
                   json.loads(registration_status.read_text()).get('status') == 'COMPLETE')
if reader_complete:
    for f in sorted(registered.iterdir()):
        if f.is_file() and f.suffix in ['.csv', '.json', '.md']:
            SOURCES['registered_reader_' + f.name.lower()] = str(f.relative_to(ROOT))
manifest = []
for name, relative in SOURCES.items():
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    data = source.read_bytes()
    (OUT / name).write_bytes(data)
    rows = len(pd.read_csv(source)) if source.suffix == '.csv' else None
    manifest.append({'paper_artifact': f'evidence/{name}', 'source_relative_to_closure': relative,
                     'sha256': hashlib.sha256(data).hexdigest(), 'rows': rows,
                     'result_status': 'EXISTING_COMPLETED_RESULT_NOT_NEW_EXPERIMENT'})

core = pd.read_csv(OUT / 'core_all_seeds.csv')
methods = ['Magnitude', 'Prediction_hgb', 'Manual_WeightedHistoryDistance',
           'Learned_WeightedHistoryDistance', 'Manual_hgb', 'Learned_hgb']
core[(core.seed == 20260930) & core.method.isin(methods)].to_csv(OUT/'table2_core.csv', index=False)
core[core.method.isin(methods)].groupby(['line','method']).agg(
    utility20_mean=('utility20','mean'), utility20_seed_sd=('utility20','std'),
    training_seeds=('seed','nunique')).reset_index().to_csv(OUT/'table_s1_seed_variation.csv', index=False)
feedback = pd.read_csv(OUT/'feedback_all_seeds.csv')
feedback[(feedback.seed == 20260930) & feedback.method.isin(
    ['Shared','TargetOnly_HGB','PublicTarget_HGB','SharedTarget_HGB','ResidualHGB'])].to_csv(
    OUT/'table4_feedback.csv', index=False)
pertema = pd.read_csv(OUT/'pertema_same_information.csv')
pertema[(pertema.seed == 20260930) & (pertema.budget == 1)].to_csv(
    OUT/'table_s2_pertema_full_budget.csv', index=False)
(OUT/'RESULT_MANIFEST.json').write_text(json.dumps({
    'scope': 'Publication draft evidence snapshot; no new truth access or model fit.',
    'primary_display_seed': 20260930,
    'axes': {'Source_McFaline': 2840, 'Orion_fixed_confirmation': 3285, 'Frangieh_retrospective_stress': 512},
    'completed_publicset_training': 'full_v2_resume',
    'completed_publicset_statistics': 'statistics_v1',
    'registered_reader_imported': reader_complete,
    'source_files': manifest}, indent=2) + '\n')
print(json.dumps({'files_copied':len(manifest),'new_fits':0,'new_test_reads':0}))
