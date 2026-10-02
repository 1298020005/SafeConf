#!/usr/bin/env python3
"""Prepare one fixed CP4000 TRAIN-control proxy on all232 SEEN query IDs."""
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
REPORT = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/target_control_expression_preparation_v1'
OUTPUT = BASE / 'orion_target_control_expression_preparation_20261002_v1'
SEAL = BASE / 'orion_public_bank_extension_20261002_v4_partitioned/expanded_risk_seal/RISK_SEAL_MANIFEST.json'
PRIMARY = BASE / 'orion_registered_test_extendedbank_20261002_v1/fixed_risk_evaluation/primary_common_COHORT.csv'
SCIENCE = REPORT.parent / 'ORION_METHOD_CONTRACT.json'
SCIENCE_SHA = '6b8a26939c8c6134dbf2e8294ee7905fccc2122b570b046f3183a3d2f5caba97'
RSCRIPT = Path('/home/yyf/.conda/envs/safeconf-orion-lm-20261002/bin/Rscript')
R_CLI = ROOT / 'tools/scripts/run_safeconf_orion_lm_blind.R'
R_CLI_SHA = '086fb5489bc88520b1854d0101db117f66f913d7ce94064a6df18772633aaee1'
QUERY = ['query_id', 'target_gene_id', 'target_gene_symbol', 'context_id', 'role']
CONTEXTS = ['HCT116', 'HEK293T']
METHOD = 'TargetControlExpression_CP4000'
R_EXPORT = '''args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
model <- readRDS(args[[1L]])
x <- model$baseline
stopifnot(is.double(x), length(x) == 38606L, !anyDuplicated(names(x)),
          all(is.finite(x)), all(x >= 0))
writeLines(names(x), args[[2L]], useBytes = TRUE)
writeBin(as.double(x), args[[3L]], size = 8L, endian = "little")
'''


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def bits(a, b):
    a, b = np.asarray(a, dtype='<f8'), np.asarray(b, dtype='<f8')
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    began, cpu = time.monotonic(), time.process_time()
    if OUTPUT.exists() or REPORT.exists():
        raise RuntimeError('Fresh isolated preparation roots required')
    if bind(SCIENCE)['sha256'] != SCIENCE_SHA or bind(R_CLI)['sha256'] != R_CLI_SHA:
        raise RuntimeError('Original science or fitted-model CLI code differs')
    science = json.loads(SCIENCE.read_text())
    if science['normalization']['formula'] != 'mean_cell(log1p(4000 * raw_gene_UMI / official_full_library_total_counts))':
        raise RuntimeError('Original CP4000 normalization definition required')
    seal = json.loads(SEAL.read_text())
    if seal.get('status') != 'SEALED_PRETRUTH' or seal.get('scientific_contract_sha256') != SCIENCE_SHA:
        raise RuntimeError('Exact original expanded risk seal required')
    contexts = {}
    inputs = [Path(__file__), SCIENCE, R_CLI, SEAL, PRIMARY, RSCRIPT]
    for context in CONTEXTS:
        entries = {x['role']: x for x in seal['LM_prediction_and_control_bindings'] if x['context'] == context}
        required = ['train_control', 'axis', 'queries', 'upstream_model']
        contexts[context] = {role: Path(entries[role]['path']) for role in required}
        for role in required:
            item = bind(contexts[context][role])
            if item['sha256'] != entries[role]['sha256']:
                raise RuntimeError('Original LM/control/identity binding differs: ' + role)
            inputs.append(contexts[context][role])
        contexts[context]['full_axis'] = contexts[context]['upstream_model'].parent / 'FULL_GENE_AXIS.tsv'
        inputs.append(contexts[context]['full_axis'])
    before = [bind(p) for p in inputs]
    query = pd.read_csv(PRIMARY, usecols=QUERY, keep_default_na=False)
    if (len(query) != 232 or not query.query_id.is_unique or query.target_gene_id.nunique() != 144
            or not query.role.eq('TEST').all()
            or query.context_id.value_counts().to_dict() != {'HEK293T': 125, 'HCT116': 107}):
        raise RuntimeError('Original complete232tasks/144genes/two-context cohort required')
    OUTPUT.mkdir(); REPORT.mkdir()
    write_json(REPORT / 'EXECUTION_REGISTRATION.json', {
        'schema': 'safeconf_orion_fixed_target_control_preparation_v1',
        'pid': os.getpid(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'method': METHOD, 'risk_formula': 'own-context TRAIN NTC mean_cell_logCP4000[target_gene_id]',
        'direction': 'larger expression means larger risk; ascending scores retain lower-risk tasks',
        'unit_correction': {'initial_root_message_unit': 'CP10000',
            'root_corrected_unit': 'mean_cell_log1p_cp4000', 'CP10000_scores_ever_generated': False,
            'root_message': 'CP10000是我在消息中写错的单位，明确以原科学合同/actualbridge/LM实际固定mean_cell_log1p_cp4000为唯一规范；不要换算已平均值，不重读raw。'},
        'all_original232query_ids_retained': True, 'readout87subset_not_selected': True,
        'input_gene_domain': 38606, 'existing_risk_readout_gene_domain': 3285,
        'extra_available_biological_input_coordinates_disclosed': True,
        'new_models_or_upstream_calls': 0, 'new_fits': 0, 'new_labels_revealed': 0,
        'raw_expression_read': False, 'cached_error_numeric_read': False,
        'original13_primary_or_Source_parameters_changed': False, 'new_confirmation': False,
        'original_input_bindings': before,
        'resources': {'wall_cap_seconds': 300, 'CPU_threads': 4, 'GPU_hours': 0}})
    print(json.dumps({'pid': os.getpid(), 'phase': 'fixed_cached_TRAIN_control_only', 'query_truth_read': False}), flush=True)
    export_path = OUTPUT / 'EXPORT_FITTED_BASELINE.R'
    export_path.write_text(R_EXPORT)
    result = query.copy()
    result[METHOD] = np.nan
    checks = []
    for context in CONTEXTS:
        paths = contexts[context]
        full_axis = pd.read_csv(paths['full_axis'], sep='\t', keep_default_na=False)
        endpoint = pd.read_csv(paths['axis'], sep='\t', keep_default_na=False)
        identities = pd.read_csv(paths['queries'], sep='\t', usecols=QUERY, keep_default_na=False)
        control = pd.read_csv(paths['train_control'], sep='\t', keep_default_na=False, float_precision='round_trip')
        if control.columns.tolist() != ['gene_id', 'mean_cell_logCP4000']:
            raise RuntimeError('Normalized CP4000 means required, not raw sums or transformed means')
        values = control.mean_cell_logCP4000.to_numpy(np.float64)
        if (len(control) != 38606 or not control.gene_id.is_unique
                or not np.array_equal(control.gene_id, full_axis.gene_id)
                or not np.isfinite(values).all() or (values < 0).any()
                or len(endpoint) != 3285 or not endpoint.gene_id.is_unique):
            raise RuntimeError('Full TRAIN control or exact measured axis invalid')
        selected = query[query.context_id.eq(context)]
        identity_rows = identities[identities.query_id.isin(selected.query_id)]
        if set(map(tuple, identity_rows[QUERY].to_numpy())) != set(map(tuple, selected[QUERY].to_numpy())):
            raise RuntimeError('Frozen query/control context identity mismatch')
        lookup = full_axis.set_index('gene_id').gene_symbol
        if not np.array_equal(lookup.loc[selected.target_gene_id], selected.target_gene_symbol):
            raise RuntimeError('Canonical target Ensembl/symbol mismatch')
        folder = OUTPUT / context; folder.mkdir()
        ids_path, binary = folder / 'MODEL_BASELINE_GENE_IDS.txt', folder / 'MODEL_BASELINE.f64le'
        run = subprocess.run([str(RSCRIPT), '--vanilla', str(export_path), str(paths['upstream_model']),
                              str(ids_path), str(binary)], capture_output=True, text=True, timeout=120)
        (folder / 'BASELINE_EXPORT.log').write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError('Existing model baseline extraction failed: ' + context)
        model_ids = ids_path.read_text().splitlines()
        model_control = np.fromfile(binary, dtype='<f8')
        if model_ids != control.gene_id.tolist() or not bits(values, model_control):
            raise RuntimeError('TRAIN sidecar must match existing fitted model baseline bitwise')
        index = pd.Series(np.arange(len(control)), index=control.gene_id)
        endpoint_rows = index.loc[endpoint.gene_id].to_numpy(int)
        projection = values[endpoint_rows]
        if not bits(projection, model_control[endpoint_rows]):
            raise RuntimeError('Original3285control projection must match fitted baseline bitwise')
        np.save(folder / 'ORIGINAL3285_CONTROL_PROJECTION.npy', projection, allow_pickle=False)
        query_rows = index.loc[selected.target_gene_id].to_numpy(int)
        result.loc[selected.index, METHOD] = values[query_rows]
        checks.append({'context': context, 'full_control_genes': 38606, 'projection_genes': 3285,
            'query_tasks': len(selected), 'full38606_model_baseline_bitwise_equal': True,
            'original3285_model_control_projection_bitwise_equal': True,
            'target_gene_symbol_Ensembl_canonical_equal': True,
            'target_outside3285_count': int((~selected.target_gene_id.isin(endpoint.gene_id)).sum()),
            'model_baseline_binary': bind(binary), 'projection': bind(folder / 'ORIGINAL3285_CONTROL_PROJECTION.npy')})
    if not np.isfinite(result[METHOD].to_numpy()).all() or result.query_id.tolist() != query.query_id.tolist():
        raise RuntimeError('All232scores must be finite and original order retained')
    scores = OUTPUT / 'FIXED_PRIMARY_TARGET_CONTROL_RISKS.parquet'
    result.to_parquet(scores, index=False)
    np.save(OUTPUT / 'RISK_FLOAT64.npy', result[METHOD].to_numpy(np.float64), allow_pickle=False)
    reloaded = pd.read_parquet(scores)
    if not bits(result[METHOD], reloaded[METHOD]) or not result[QUERY].equals(reloaded[QUERY]):
        raise RuntimeError('Sealed score reload must preserve exact scores and identities')
    pd.DataFrame(checks).drop(columns=['model_baseline_binary', 'projection']).to_csv(REPORT / 'CONTROL_IDENTITY_AND_BITWISE_PROJECTION_CHECKS.csv', index=False)
    write_json(REPORT / 'INFORMATION_SCOPE_LEDGER.json', {
        'method': METHOD, 'cohort_tasks': 232, 'gene_clusters': 144,
        'allowed_actual_information': 'existing own-context TRAIN NTC mean vectors plus target identity',
        'input_domain_genes': 38606, 'prior_P_PUBLIC_readout_domain_genes': 3285,
        'not_same_feature_information_budget': True, 'no_Public_treated_truth_or_error_supervision': True,
        'new_upstream_attempts': 0, 'new_fits': 0, 'new_model_inference_calls': 0,
        'new_labels_revealed': 0, 'target_error_numeric_reads': 0,
        'new_expression_acquisition': 0, 'existing_TRAIN_control_numeric_cache_read': True,
        'control_matched_level': 'own-context TRAIN NTC; not query-specific plate/batch',
        'source_risk_core_or_original_primary_changed': False, 'new_confirmation': False})
    if before != [bind(p) for p in inputs]:
        raise RuntimeError('Original code/models/control/identities changed')
    elapsed = time.monotonic() - began
    if elapsed > 300:
        raise RuntimeError('Preparation exceeded300second wall bound')
    receipt = {'status': 'COMPLETE_FIXED232_TRAIN_CONTROL_PROXY_PREPARATION',
        'method': METHOD, 'query_tasks': 232, 'gene_clusters': 144, 'score_dtype': 'float64',
        'rule': 'risk=own_context_TRAIN_NTC_mean_cell_logCP4000[target_gene_id]',
        'higher_score_is_higher_risk': True, 'old3285_subset_not_selected': True,
        'projection_checks': checks, 'full_context_control_and_model_baselines_bitwise_equal': True,
        'all_scores_sealed_before_any_cached_error_numeric_read': True,
        'target_error_numeric_reads': 0, 'statistics_performed': False, 'CP10000_executed': False,
        'new_fits': 0, 'new_upstream_calls': 0, 'model_prediction_calls': 0,
        'elapsed_seconds': elapsed, 'CPU_seconds': time.process_time()-cpu,
        'peak_parent_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'peak_child_RSS_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
        'original_input_bindings': before,
        'outputs': [bind(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()]}
    write_json(REPORT / 'PREPARATION_RESULT.json', receipt)
    for root in [REPORT, OUTPUT]:
        for path in root.rglob('*'):
            if path.is_file(): path.chmod(0o444)
    print(json.dumps({'status': receipt['status'], 'elapsed_seconds': elapsed}), flush=True)


if __name__ == '__main__':
    main()
