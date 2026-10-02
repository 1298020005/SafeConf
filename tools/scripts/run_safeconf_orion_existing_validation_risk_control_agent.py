#!/usr/bin/env python3
"""Separately registered SEEN validation-label controls; frozen Orion/Source intact."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC, CDF_KEYS, fit_risk, rank_labels, cluster_weights, ids_hash

BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/existing_validation_risk_control_v1'
OUT = BASE / 'orion_existing_validation_risk_control_20261002_v1'
SEAL = BASE / 'orion_public_bank_extension_20261002_v4_partitioned/expanded_risk_seal'
COMPARISON = SEAL.parent / 'expanded_comparison'
COMP = BASE / 'orion_postfit_followthrough_20261002_v4_decimal_tokens/competence'
FIT = BASE / 'orion_published_lm_20261002_v3_decimal_tokens'
SOURCE = BASE / 'orion_source_core_20261002_v1'
FORMAL = BASE / 'orion_registered_test_extendedbank_20261002_v1'
PRIMARY = FORMAL / 'fixed_risk_evaluation/primary_common_COHORT.csv'
ERROR_PATH = FORMAL / 'registered_test_truth/TEST_TASK_ERRORS.parquet'
METRIC_CODE = ROOT / 'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py'
METRIC_SHA = 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c'
SCHEMA = 'safeconf_orion_existing_validation_risk_control_v1'
SALT = 'SafeConf_Orion_validation_reuse_20261002_v1'
BUDGETS = [10, 25, 50, 75, 100]
ARMS = {'TargetOnly': P, 'PublicTarget': P + PUBLIC, 'SharedTarget': P + PUBLIC + ['frozen_shared_risk']}
KINDS = ['ridge', 'hgb']
CONTEXTS = ['HCT116', 'HEK293T']
ANCHORS = {'FrozenLearnedSourceHGB': 'Learned_hgb', 'LearnedWeightedDistance': 'Learned_WeightedHistoryDistance',
           'NegativeSourceSupport': 'NegativeSourceHistorySupport'}
QUERY = ['query_id', 'target_gene_id', 'target_gene_symbol', 'context_id', 'role']
SHARED = 'Learned_hgb'
SEED, BOOT_SEED, DRAWS = 20260930, 20260929, 5000
WALL, RSS = 1800, 2 * 1024**3
OUTPUT_CONTRACT = 'Source3285_CP4000_log1p_matched_batch_delta_v1'
CODE_FILES = [Path(__file__).resolve(), ROOT / 'tools/safeconf_continual/research.py',
              ROOT / 'tools/safeconf_continual/learners.py', ROOT / 'tools/safeconf_continual/contracts.py', METRIC_CODE]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024**2), b''): h.update(data)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(item, readonly=False):
    path = Path(item['path']).resolve()
    if binding(path) != item or (readonly and path.stat().st_mode & 0o222):
        raise RuntimeError(f'Exact immutable binding differs: {path}')
    return path


def clean(obj):
    if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    return obj


def write_json(path, value, readonly=True):
    with Path(path).open('x') as f: f.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if readonly: Path(path).chmod(0o444)


def require_finite(values, name):
    if not np.isfinite(np.asarray(values, float)).all(): raise RuntimeError(f'Nonfinite {name}; no survivor filtering')


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def model_plan():
    return [{'budget': b, 'arm': arm, 'learner': kind, 'method': f'B{b:03d}_{arm}_{kind}', 'columns': cols}
            for b in BUDGETS for arm, cols in ARMS.items() for kind in KINDS]


def contrast_plan():
    result = []
    for b in BUDGETS:
        for kind in KINDS:
            names = {arm: f'B{b:03d}_{arm}_{kind}' for arm in ARMS}
            pairs = [(names['PublicTarget'], names['TargetOnly']), (names['SharedTarget'], names['PublicTarget'])]
            pairs += [(names[arm], anchor) for arm in ARMS for anchor in ['LearnedWeightedDistance', 'FrozenLearnedSourceHGB']]
            result += [{'budget': b, 'learner': kind, 'method_a': a, 'method_b': z} for a, z in pairs]
    return result


def budget_genes(genes, budget):
    ordered = sorted(set(map(str, genes)), key=lambda g: (hashlib.sha256(f'{SALT}|{g}'.encode()).hexdigest(), g))
    return ordered[:max(1, math.ceil(len(ordered) * budget / 100))]


def mask_unsupported(features, status, shared):
    derived = features.copy()
    derived['frozen_shared_risk'] = np.asarray(shared, float)
    unsupported = np.asarray(status) != 'KNOWN_SAME_GENE_SOURCE_HISTORY'
    derived.loc[unsupported, PUBLIC + ['frozen_shared_risk']] = np.nan
    return derived, unsupported


def code_checks():
    genes = [f'ENSG_TOY_{g}' for g in range(20)]
    frame = pd.DataFrame([(g, c) for g in genes for c in CONTEXTS], columns=['gene', 'context'])
    previous = set()
    counts = []
    for budget in BUDGETS:
        selected = set(budget_genes(genes, budget))
        assert previous <= selected and len(selected) == math.ceil(len(genes) * budget / 100)
        used = frame[frame.gene.isin(selected)]
        assert set(used.gene) == selected and used.groupby('gene').context.nunique().eq(2).all()
        counts.append(len(selected)); previous = selected
    toy = pd.DataFrame(np.arange(4 * 13, dtype=float).reshape(4, 13), columns=P + PUBLIC)
    before = toy.copy()
    derived, missing = mask_unsupported(toy, ['KNOWN_SAME_GENE_SOURCE_HISTORY', 'NO_SAME_GENE_SOURCE_HISTORY'] * 2, np.arange(4.))
    assert toy.equals(before) and bits(derived[P].to_numpy(), before[P].to_numpy())
    assert derived.loc[missing, PUBLIC + ['frozen_shared_risk']].isna().all().all()
    assert bits(derived.loc[~missing, P + PUBLIC].to_numpy(), before.loc[~missing].to_numpy())
    plan, contrasts = model_plan(), contrast_plan()
    methods = [x['method'] for x in plan] + list(ANCHORS)
    assert len(plan) == len({x['method'] for x in plan}) == 30 and len(contrasts) == 80
    assert all(x['method_a'] in methods and x['method_b'] in methods for x in contrasts)
    assert all(len([x for x in plan if x['budget'] == b]) == 6 for b in BUDGETS)
    return {'status': 'PASS', 'script': binding(__file__), 'generated_only': True,
            'budget_nested_isolation_and_all_contexts': True, 'toy_budget_gene_counts': counts,
            'unsupported_mask_all7Public_and_shared_only_no_source_mutation': True,
            'all30models_and80prespecified_pairs_coverage': True,
            'actual_feature_or_error_values_read': 0, 'fits': 0}


def input_paths():
    paths = [DOC / 'PREPARATION_READINESS.json', SEAL / 'RISK_SEAL_MANIFEST.json', SEAL / 'ARTIFACT_HASHES.json',
             SEAL / 'Learned_P_PUBLIC_FEATURES.parquet', SEAL / 'P_ONLY_FEATURES.parquet',
             COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet', COMPARISON / 'COMPARISON_SPEC.json',
             COMP / 'VALIDATION_TASK_ERRORS.csv', COMP / 'COMPETENCE_RESULT.json', COMP / 'INPUT_IDENTITIES.json',
             COMP / 'ARTIFACT_HASHES.json', COMP / 'C_VALIDATION_ACCESS_LEDGER.json', PRIMARY,
             FORMAL / 'IMMUTABLE_POSTSEAL_TEST_ACCESS.json', FORMAL / 'registered_test_truth/EXTENSION_TRUTH_RECEIPT.json',
             FORMAL / 'registered_test_truth/TRUTH_READER_RECEIPT.json']
    paths += list((FORMAL / 'fixed_risk_evaluation').glob('*.json')) + list((FORMAL / 'fixed_risk_evaluation').glob('*.csv'))
    paths += [ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/ORION_METHOD_CONTRACT.json']
    paths += [FIT / 'RUN_MANIFEST.tsv']
    for ctx in CONTEXTS: paths += [FIT / ctx / 'QUERY_IDENTITIES.tsv', FIT / ctx / 'OUTPUT_GENE_AXIS.tsv', FIT / ctx / 'STATUS.tsv', FIT / ctx / 'MODEL.rds']
    pins = json.loads((SEAL / 'RISK_SEAL_MANIFEST.json').read_text())
    paths += [Path(pins['public_bank_manifest']['path']), Path(pins['public_bank_extension_contract']['path'])]
    paths += [Path(x['path']) for x in pins['source_core_bindings'] if Path(x['path']).suffix == '.joblib' or Path(x['path']).name in ('SOURCE_FEATURE_MANIFEST.json', 'MODEL_MANIFEST.json', 'GENE_MANIFEST.csv')]
    eligibility = next(x['path'] for x in json.loads((COMP / 'INPUT_IDENTITIES.json').read_text()) if x['path'].endswith('VALIDATION_ELIGIBILITY.tsv'))
    paths.append(Path(eligibility))
    return sorted(set(p.resolve() for p in paths))


def prepare():
    if OUT.exists() or (DOC / 'PROPOSED_CONTROL_SCOPE.json').exists(): raise RuntimeError('Fresh proposal/output required')
    if sha(METRIC_CODE) != METRIC_SHA: raise RuntimeError('Original frozen metric implementation changed')
    ids = pd.read_csv(COMP / 'VALIDATION_TASK_ERRORS.csv', usecols=QUERY, keep_default_na=False)
    primary = pd.read_csv(PRIMARY, usecols=QUERY, keep_default_na=False)
    if len(ids) != 2790 or ids.target_gene_id.nunique() != 1661 or len(primary) != 232 or primary.target_gene_id.nunique() != 144:
        raise RuntimeError('Fixed validation/primary identity budgets differ')
    if set(ids.target_gene_id) & set(primary.target_gene_id): raise RuntimeError('Validation/TEST gene overlap')
    receipt = json.loads((FORMAL / 'registered_test_truth/EXTENSION_TRUTH_RECEIPT.json').read_text())
    error_binding = receipt['truth_errors']  # Recorded metadata only; ERROR_PATH is not opened before prediction seal.
    if Path(error_binding['path']).resolve() != ERROR_PATH or receipt['status'] != 'COMPLETE':
        raise RuntimeError('Exact existing completed SEEN truth receipt required')
    proof = code_checks()
    write_json(DOC / 'CONTROL_CODE_CHECKS.json', proof)
    proposal = {'schema': SCHEMA + '_proposal', 'status': 'PROPOSED_NOT_EXECUTED',
        'analysis_role': 'retrospective_SEEN_existing_validation_risk_control_supplement_not_new_confirmation',
        'runtime_output': str(OUT), 'report_output': str(DOC), 'information_arms': ARMS,
        'public_reference': 'Learned', 'frozen_shared_score': SHARED, 'anchor_columns': ANCHORS,
        'budgets_percent': BUDGETS, 'budget_gene_order': 'sha256(salt|canonical_Ensembl), lexical gene tiebreak', 'budget_salt': SALT,
        'allowed_validation_tasks': 2790, 'allowed_validation_genes': 1661, 'all_three_arms_same_training_pool': True,
        'unsupported_VAL_tasks': 2566, 'unsupported_mask': 'all7PUBLIC plus frozen_shared_risk NaN in derived copy only; no shared fallback',
        'imputation_and_missing_flags': 'existing NumericPreprocessor fit only allowed budget rows',
        'CDF': {'helper': 'existing research.rank_labels/FrozenErrorCDF', 'keys': CDF_KEYS, 'fit_budget_VAL_only': True, 'TEST_fit': False},
        'models': model_plan(), 'max_target_risk_fits': 30,
        'Ridge': {'alpha': 10}, 'HGB': {'max_iter': 200, 'learning_rate': .05, 'max_depth': 3, 'min_samples_leaf': 20, 'l2_regularization': 10.},
        'fit_seed': SEED, 'weights': 'existing inverse records per gene normalized mean1 within allowed budget',
        'new_upstream_attempts': 0, 'Source_refits': 0, 'Public_fits': 0, 'new_raw_expression_reads': 0,
        'primary_test_tasks': 232, 'primary_test_genes': 144, 'primary_contexts': CONTEXTS,
        'primary_cohort_binding': binding(PRIMARY), 'validation_identity_hash': ids_hash(ids.query_id),
        'all_predictions_models_CDFs_budgetIDs_and_mask_sealed_before_cachedTESTerrors_open': True,
        'evaluation_only_error_binding_recorded_not_opened': error_binding,
        'bootstrap': {'replicates': DRAWS, 'seed': BOOT_SEED, 'gene_order': 'canonical sorted TEST target_gene_id',
                      'unit': 'complete gene blocks jointly bothcontexts andall33scores; original queryIDs/ties retained',
                      'contexts_minimum_tasks': 20, 'macro': 'equal both predefined contexts; undefined staysNA',
                      'original_metric_code_sha256': METRIC_SHA, 'contrasts': contrast_plan()},
        'budget_accounting': '2790labels already used for competence; downstream reuse amounts, not newly revealed or zero-cost labels',
        'new_scope_overlay': 'Separate permitted existing-SEEN C-label risk/CDF use; original competence-only receipts/6b8/c94/primary13 unmodified',
        'model_or_winner_or_reference_selection': False, 'OOF_calibration_or_hyperparameter_search': False,
        'resources': {'wall_seconds': WALL, 'RSS_bytes': RSS, 'CPU_threads': 4, 'GPU_hours': 0, 'download_bytes': 0},
        'input_bindings': [binding(p) for p in input_paths()], 'code_bindings': [binding(p) for p in CODE_FILES],
        'generated_checks': binding(DOC / 'CONTROL_CODE_CHECKS.json')}
    write_json(DOC / 'PROPOSED_CONTROL_SCOPE.json', proposal)
    write_json(DOC / 'CONTROL_ROOT_APPROVAL_TEMPLATE.json', {'schema': SCHEMA + '_root_approval', 'status': 'DRAFT_NOT_AUTHORIZED',
        'authorize_existing_SEEN_validation_risk_control_reuse': False, 'authorize_cached_SEEN_TEST_evaluation_after_prediction_seal': False,
        'proposal': binding(DOC / 'PROPOSED_CONTROL_SCOPE.json'), 'independent_review': {'path': 'ROOT_TO_BIND_ACTUAL_PASS', 'bytes': None, 'sha256': None}})
    print(json.dumps({'proposal': binding(DOC / 'PROPOSED_CONTROL_SCOPE.json'), 'code': binding(__file__),
                      'checks': binding(DOC / 'CONTROL_CODE_CHECKS.json'), 'actual_run_started': False}, indent=2), flush=True)


def authorize(path):
    path = Path(path).resolve()
    if path.stat().st_mode & 0o222: raise RuntimeError('Readonly actual Root approval required')
    approval = json.loads(path.read_text())
    if (approval.get('schema') != SCHEMA + '_root_approval' or approval.get('status') != 'APPROVED'
        or approval.get('authorize_existing_SEEN_validation_risk_control_reuse') is not True
        or approval.get('authorize_cached_SEEN_TEST_evaluation_after_prediction_seal') is not True):
        raise RuntimeError('Separate Source-preserving label-reuse/evaluation approval required')
    proposal = json.loads(checked(approval['proposal'], readonly=True).read_text())
    review = json.loads(checked(approval['independent_review'], readonly=True).read_text())
    if review.get('status') != 'PASS' or review.get('proposal') != approval['proposal']: raise RuntimeError('Exact independent PASS required')
    if proposal.get('schema') != SCHEMA + '_proposal' or proposal.get('models') != model_plan() or proposal.get('bootstrap', {}).get('contrasts') != contrast_plan():
        raise RuntimeError('Fixed30models/80contrasts proposal changed')
    if proposal.get('runtime_output') != str(OUT) or proposal.get('report_output') != str(DOC) or proposal.get('budget_salt') != SALT:
        raise RuntimeError('Fixed isolated output/budget registration changed')
    if proposal.get('code_bindings') != [binding(p) for p in CODE_FILES]: raise RuntimeError('Actual/dependency code changed')
    for item in proposal['input_bindings']: checked(item)
    if json.loads(checked(proposal['generated_checks'], readonly=True).read_text()).get('status') != 'PASS': raise RuntimeError('Generated checks missing')
    if OUT.exists() or (DOC / 'CONTROL_RESULT_MANIFEST.json').exists(): raise RuntimeError('Actual output must be fresh')
    return approval, proposal


def limits():
    def abort(_sig, _frame): raise RuntimeError('Registered30min/2GiB resource bound exceeded')
    signal.signal(signal.SIGALRM, abort); signal.signal(signal.SIGUSR1, abort); signal.alarm(WALL)
    stop = threading.Event()
    def watch():
        while not stop.wait(.1):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RSS:
                os.kill(os.getpid(), signal.SIGUSR1); return
    threading.Thread(target=watch, daemon=True).start()
    return stop


def metric_module():
    spec = importlib.util.spec_from_file_location('_frozen_orion_control_metrics', METRIC_CODE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def bootstrap(frame, names, module):
    point = module.metric_values(frame, names)
    genes = sorted(frame.target_gene_id.astype(str).unique())
    blocks = [np.flatnonzero(frame.target_gene_id.astype(str).eq(g)) for g in genes]
    rng = np.random.default_rng(BOOT_SEED)
    selections = rng.integers(0, len(genes), size=(DRAWS, len(genes)), dtype=np.int64)
    np.save(OUT / 'BOOTSTRAP_GENE_INDICES.npy', selections, allow_pickle=False)
    draws = np.full((DRAWS, 3, len(names), len(module.METRICS)), np.nan)
    original_valid = [sum(frame.context_id.eq(c)) >= 20 for c in CONTEXTS]
    original_valid.append(all(original_valid))
    for number, selected in enumerate(selections):
        sampled = frame.iloc[np.concatenate([blocks[g] for g in selected])].reset_index(drop=True)
        values = module.metric_values(sampled, names)
        for ci, context in enumerate(CONTEXTS + ['macro']):
            if original_valid[ci]:
                for mi, method in enumerate(names): draws[number, ci, mi] = [values[context][method][m] for m in module.METRICS]
        if number % 500 == 0: print(json.dumps({'phase': 'shared_gene_bootstrap', 'completed_draws': number, 'total': DRAWS}), flush=True)
    np.save(OUT / 'SHARED_BOOTSTRAP_METRIC_DRAWS.npy', draws, allow_pickle=False)
    rows, pairs = [], []
    for ci, context in enumerate(CONTEXTS + ['macro']):
        for mi, name in enumerate(names):
            for ki, metric in enumerate(module.METRICS):
                data = draws[:, ci, mi, ki]; valid = data[np.isfinite(data)]
                lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) >= 2 else (np.nan, np.nan)
                rows.append({'context': context, 'method': name, 'metric': metric, 'point': point[context][name][metric],
                             'ci95_lower': lo, 'ci95_upper': hi, 'valid_draws': len(valid), 'bootstrap_replicates': DRAWS,
                             'original_context_valid': original_valid[ci], 'SEEN_supplement': True})
        for pair in contrast_plan():
            ai, bi = names.index(pair['method_a']), names.index(pair['method_b'])
            for ki, metric in enumerate(module.METRICS):
                data = draws[:, ci, ai, ki] - draws[:, ci, bi, ki]; valid = data[np.isfinite(data)]
                lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) >= 2 else (np.nan, np.nan)
                pairs.append(pair | {'context': context, 'metric': metric,
                    'difference_a_minus_b': point[context][pair['method_a']][metric] - point[context][pair['method_b']][metric],
                    'ci95_lower': lo, 'ci95_upper': hi, 'valid_draws': len(valid), 'bootstrap_replicates': DRAWS,
                    'original_context_valid': original_valid[ci], 'SEEN_supplement': True})
    return pd.DataFrame(rows), pd.DataFrame(pairs), genes


def run(approval_path):
    started, cpu = time.monotonic(), time.process_time(); stop = limits()
    approval, proposal = authorize(approval_path)
    OUT.mkdir(); (OUT / 'models').mkdir(); (OUT / 'CDFs').mkdir()
    fitted_count, fits_started = 0, 0
    write_json(OUT / 'EXECUTION_REGISTRATION.json', {'schema': SCHEMA, 'pid': os.getpid(), 'approval': binding(approval_path),
        'proposal': approval['proposal'], 'script': binding(__file__), 'start_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'cached_TEST_errors_opened': False, 'label_role': 'already_seen_C_validation_downstream_reuse'})
    try:
        meta = pd.read_parquet(COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet', columns=QUERY + ['history_status'] + list(set(ANCHORS.values())), use_threads=False)
        feats = pd.read_parquet(SEAL / 'Learned_P_PUBLIC_FEATURES.parquet', use_threads=False)
        pbase = pd.read_parquet(SEAL / 'P_ONLY_FEATURES.parquet', use_threads=False)
        if len(meta) != 14716 or not meta.query_id.is_unique or list(feats.columns) != P + PUBLIC or not bits(feats[P].to_numpy(float), pbase[P].to_numpy(float)):
            raise RuntimeError('Exact frozen Learned13/P6 cached contract differs')
        require_finite(feats[P], 'all frozen P6')
        derived, unsupported = mask_unsupported(feats, meta.history_status, meta[SHARED])
        require_finite(derived.loc[~unsupported, P + PUBLIC + ['frozen_shared_risk']], 'known-history inputs')
        errors = pd.read_csv(COMP / 'VALIDATION_TASK_ERRORS.csv', usecols=QUERY + ['model_rmse'], keep_default_na=False, float_precision='round_trip')
        require_finite(errors.model_rmse, 'allowed C validation errors')
        if (errors.model_rmse < 0).any() or errors.query_id.duplicated().any(): raise RuntimeError('Invalid C validation identities/errors')
        meta['global_query_row'] = np.arange(len(meta))
        validation = errors.merge(meta[QUERY + ['global_query_row', 'history_status']], on=QUERY, validate='one_to_one')
        primary = pd.read_csv(PRIMARY, usecols=QUERY, keep_default_na=False)
        primary = primary.merge(meta[QUERY + ['global_query_row', 'history_status']], on=QUERY, validate='one_to_one')
        if len(validation) != 2790 or validation.target_gene_id.nunique() != 1661 or len(primary) != 232 or primary.target_gene_id.nunique() != 144:
            raise RuntimeError('No training/evaluation attrition allowed')
        if set(validation.target_gene_id) & set(meta[meta.role.eq('TEST')].target_gene_id): raise RuntimeError('Any validation/TEST gene overlap forbidden')
        if not validation.role.eq('VALIDATION').all() or not primary.role.eq('TEST').all() or not primary.history_status.eq('KNOWN_SAME_GENE_SOURCE_HISTORY').all():
            raise RuntimeError('Fixed role/public-supported primary identities differ')
        vrows, trows = validation.global_query_row.to_numpy(int), primary.global_query_row.to_numpy(int)
        if unsupported[vrows].sum() != 2566: raise RuntimeError('Explicit validation unsupported mask differs')
        train = derived.iloc[vrows].reset_index(drop=True)
        query = derived.iloc[trows].reset_index(drop=True)
        model_versions = {x['context']: x['sha256'] for x in json.loads((SEAL / 'RISK_SEAL_MANIFEST.json').read_text())['LM_prediction_and_control_bindings'] if x['role'] == 'upstream_model'}
        for field, values in {'task_id': validation.query_id, 'gene': validation.target_gene_id, 'target': validation.context_id,
                              'true_error_rmse': validation.model_rmse}.items(): train[field] = values.to_numpy()
        train['dataset_id'] = 'Huang2025_XAtlasOrion'; train['upstream'] = 'AhlmannEltze2025_PublishedLM_PCA10_Ridge0.1_seed1'
        train['model_version'] = train.target.map(model_versions); train['output_contract_id'] = OUTPUT_CONTRACT
        mask_table = validation[QUERY + ['global_query_row', 'history_status']].copy()
        mask_table['unsupported_derived_Public_and_shared_NaN'] = unsupported[vrows]
        mask_table.to_csv(OUT / 'VALIDATION_DERIVED_MASK_IDENTITIES.csv', index=False)
        train.to_parquet(OUT / 'DERIVED_VALIDATION_TRAIN_FEATURES.parquet', index=False)
        scores = {name: meta.iloc[trows][column].to_numpy(float) for name, column in ANCHORS.items()}
        require_finite(np.column_stack(list(scores.values())), 'three fixed primary anchors')
        budget_records, cdf_audits, fit_ledger = [], [], []
        with threadpool_limits(limits=4):
            for budget in BUDGETS:
                genes = budget_genes(train.gene, budget)
                allowed = train[train.gene.isin(genes)].copy().reset_index(drop=True)
                labels, audit = rank_labels(allowed, f'{SCHEMA}/budget{budget}', budget / 100)
                require_finite(labels, 'budget-only CDF labels')
                if len(audit) != 2 or any(a['status'] not in ('OK', 'CONSTANT_LABEL') for a in audit): raise RuntimeError('Both context budget CDFs required')
                np.save(OUT / 'CDFs' / f'B{budget:03d}_TRAIN_RANK_LABELS.npy', labels, allow_pickle=False)
                cdfs = []
                for key, group in allowed.groupby(CDF_KEYS, dropna=False, sort=True):
                    cdfs.append(dict(zip(CDF_KEYS, key)) | {'sorted_allowed_validation_errors': np.sort(group.true_error_rmse.to_numpy(float)).tolist(),
                        'training_records_hash': ids_hash(group.task_id), 'training_genes_hash': ids_hash(group.gene.unique())})
                write_json(OUT / 'CDFs' / f'B{budget:03d}_CDF.json', {'formula': '(left_count+right_count)/(2*n_budget_context)',
                           'fit_scope': 'allowed_budget_VAL_errors_only', 'groups': cdfs, 'TEST_CDF_fit': False})
                for g in genes: budget_records.append({'budget': budget, 'target_gene_id': g})
                cdf_audits += audit
                for item in [x for x in model_plan() if x['budget'] == budget]:
                    if fits_started >= 30: raise RuntimeError('30final_target_fit cap reached')
                    fits_started += 1
                    model = fit_risk(allowed, labels, item['columns'], item['learner'], SEED)
                    fitted_count += 1
                    params = model.model.get_params()
                    expected = {'alpha': 10} if item['learner'] == 'ridge' else {'max_iter': 200, 'learning_rate': .05, 'max_depth': 3, 'min_samples_leaf': 20, 'l2_regularization': 10., 'random_state': SEED}
                    if any(params.get(k) != v for k, v in expected.items()): raise RuntimeError('Fixed learner params differ')
                    path = OUT / 'models' / f"{item['method']}.joblib"
                    joblib.dump(model, path, compress=3)
                    value = np.asarray(model.predict(query), float)
                    restored = joblib.load(path)
                    if not bits(value, restored.predict(query)): raise RuntimeError('Reloaded target risk prediction bytes differ')
                    require_finite(value, item['method'])
                    scores[item['method']] = value
                    fit_ledger.append(item | {'fit_rows': len(allowed), 'fit_gene_clusters': len(genes),
                        'fit_records_hash': ids_hash(allowed.task_id), 'fit_genes_hash': ids_hash(genes),
                        'unsupported_fit_rows': int(allowed[PUBLIC + ['frozen_shared_risk']].isna().all(axis=1).sum()),
                        'weight_sum': float(cluster_weights(allowed).sum()), 'model': binding(path),
                        'parameters': params, 'imputation_fit_only_budget': True, 'Source_model_changed': False,
                        'labels_previously_used_for_competence': True, 'new_labels_revealed': 0})
                    print(json.dumps({'phase': 'fixed_target_risk_fit', 'method': item['method'], 'completed_fits': fitted_count}), flush=True)
            names = [x['method'] for x in model_plan()] + list(ANCHORS)
            if fitted_count != 30 or set(scores) != set(names): raise RuntimeError('All30fixedmodels/3anchors must be preserved')
            matrix = np.column_stack([scores[n] for n in names])
            require_finite(matrix, 'all fixed primary prediction scores')
            np.save(OUT / 'ALL_PRIMARY_SCORES_FLOAT64.npy', matrix, allow_pickle=False)
            pd.concat([primary[QUERY].reset_index(drop=True), pd.DataFrame(matrix, columns=names)], axis=1).to_parquet(OUT / 'ALL_PRIMARY_PREDICTIONS.parquet', index=False)
            pd.DataFrame(budget_records).to_csv(OUT / 'BUDGET_GENE_IDENTITIES.csv', index=False)
            pd.DataFrame(cdf_audits).to_csv(OUT / 'TRAIN_ONLY_CDF_AUDIT.csv', index=False)
            write_json(OUT / 'FIT_MODEL_BUDGET_LEDGER.json', {'schema': SCHEMA, 'models': fit_ledger,
                'validation_labels_already_revealed_for_competence': 2790, 'validation_gene_clusters_already_seen': 1661,
                'budget_meaning': 'downstream label uses; not newly revealed labels and not free0label', 'new_fits': fitted_count})
            for p in OUT.rglob('*'):
                if p.is_file(): p.chmod(0o444)
            seal = {'schema': SCHEMA + '_prediction_seal', 'status': 'SEALED_ALL30_BEFORE_CACHED_TEST_ERROR_ACCESS',
                'approval': binding(approval_path), 'proposal': approval['proposal'], 'score_columns': names,
                'n_primary_tasks': 232, 'n_primary_gene_clusters': 144, 'new_target_risk_fits': fitted_count,
                'new_upstream_attempts': 0, 'Source_or_Public_refits': 0, 'TEST_error_file_opened': False,
                'artifact_bindings': [binding(p) for p in sorted(OUT.rglob('*')) if p.is_file()],
                'SEEN_supplement_not_confirmation': True, 'elapsed_seconds': time.monotonic() - started}
            write_json(OUT / 'ALL_PREDICTIONS_SEAL.json', seal)
            seal_binding = binding(OUT / 'ALL_PREDICTIONS_SEAL.json')
            for item in seal['artifact_bindings']: checked(item, readonly=True)
            print(json.dumps({'phase': 'all_predictions_sealed', 'seal': seal_binding, 'TEST_error_file_opened': False}), flush=True)
            # First opening of existing TEST errors is after all predictions/models have sealed.
            checked(proposal['evaluation_only_error_binding_recorded_not_opened'])
            write_json(OUT / 'CACHED_SEEN_TEST_ERROR_ACCESS_STARTED.json', {'pre_error_prediction_seal': seal_binding,
                       'error_binding': proposal['evaluation_only_error_binding_recorded_not_opened'], 'already_SEEN': True})
            truth = pd.read_parquet(ERROR_PATH, columns=QUERY + ['true_error_rmse'],
                                    filters=[('query_id', 'in', primary.query_id.tolist())], use_threads=False)
            if len(truth) != 232: raise RuntimeError('Only the fixed232cached error rows may enter evaluation')
            evaluation = primary[QUERY].merge(truth, on=QUERY, validate='one_to_one')
            if len(evaluation) != 232 or not evaluation.query_id.equals(primary.query_id): raise RuntimeError('Fixed primary truth row identity/order differs')
            require_finite(evaluation.true_error_rmse, 'fixed primary cached errors')
            if (evaluation.true_error_rmse < 0).any(): raise RuntimeError('Negative cached errors')
            for name in names: evaluation[name] = scores[name]
            module = metric_module()
            measured, pairs, bootstrap_genes = bootstrap(evaluation, names, module)
        if len(measured) != 33 * 3 * len(module.METRICS) or len(pairs) != 80 * 3 * len(module.METRICS): raise RuntimeError('All method/metric/contrast rows required')
        measured.to_csv(DOC / 'CONTROL_ALL_METHOD_METRIC_INTERVALS.csv', index=False)
        pairs.to_csv(DOC / 'CONTROL_ALL_PRESPECIFIED_PAIRED_INTERVALS.csv', index=False)
        for item in proposal['input_bindings'] + proposal['code_bindings']: checked(item)
        checked(proposal['evaluation_only_error_binding_recorded_not_opened'])
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if peak > RSS or time.monotonic() - started > WALL: raise RuntimeError('Measured resource bounds exceeded')
        result = {'schema': SCHEMA, 'status': 'COMPLETE', 'approval': binding(approval_path), 'proposal': approval['proposal'],
            'prediction_seal': seal_binding, 'new_target_risk_fits': fitted_count, 'new_upstream_attempts': 0,
            'Source_refits': 0, 'Public_refits': 0, 'new_raw_expression_reads': 0, 'prior_vector_loads': 0,
            'cached_C_validation_reused': 2790, 'cached_C_validation_gene_clusters': 1661, 'new_labels_revealed': 0,
            'TEST_CDF_fit': False, 'new_models_used_TESTlabels_before_seal': False, 'original_artifact_and_Source_parameter_hashes_unchanged': True,
            'n_primary_tasks': 232, 'n_primary_genes': 144, 'actual_contexts': 2, 'all33method_coverage': True,
            'all80prespecified_pairs_coverage': True, 'bootstrap_replicates': DRAWS, 'bootstrap_seed': BOOT_SEED,
            'bootstrap_gene_order_hash': ids_hash(bootstrap_genes), 'undefined_metrics_preserved': True,
            'winner_or_hyperparameter_or_reference_selection': False, 'SEEN_supplement_not_new_confirmation': True,
            'elapsed_seconds': time.monotonic() - started, 'CPU_seconds': time.process_time() - cpu,
            'peak_RSS_bytes': peak, 'max_CPU_threads': 4, 'GPU_hours': 0, 'download_bytes': 0,
            'output_bindings': [binding(p) for p in sorted(OUT.rglob('*')) if p.is_file()],
            'report_bindings': [binding(DOC / n) for n in ['CONTROL_ALL_METHOD_METRIC_INTERVALS.csv', 'CONTROL_ALL_PRESPECIFIED_PAIRED_INTERVALS.csv']]}
        write_json(DOC / 'CONTROL_RESULT_MANIFEST.json', result)
        for p in OUT.rglob('*'):
            if p.is_file(): p.chmod(0o444)
        print(json.dumps({'phase': 'COMPLETE', 'result': binding(DOC / 'CONTROL_RESULT_MANIFEST.json'),
                          'elapsed_seconds': result['elapsed_seconds'], 'peak_RSS_bytes': peak}), flush=True)
    except BaseException as error:
        write_json(OUT / 'ABORT.json', {'schema': SCHEMA, 'status': 'ABORT', 'type': type(error).__name__, 'message': str(error),
                   'actual_target_fits_started': fits_started, 'actual_target_fits_completed': fitted_count, 'elapsed_seconds': time.monotonic() - started,
                   'all_existing_objects_preserved': True})
        raise
    finally:
        signal.alarm(0); stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    actual = commands.add_parser('run'); actual.add_argument('--root-approval', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    else: run(args.root_approval)


if __name__ == '__main__': main()
