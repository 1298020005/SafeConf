"""Preserve the complete original inference branch under bank expansion."""
from pathlib import Path
import numpy as np
import pandas as pd
from tools.safeconf_continual import orion_public_bank_extension as extension
from tools.scripts import seal_safeconf_orion_source_risk_agent as base


def infer_partitioned(queries, delta, controls, core, expanded_core, old_genes):
    original = base.infer(queries, delta, controls, core)
    positions = np.flatnonzero(~queries.target_gene_symbol.astype(str).isin(old_genes).to_numpy())
    if not len(positions):
        invariant = extension.assert_legacy_invariance(queries, original, original, old_genes)
        return original, original, invariant
    branch = base.infer(queries.iloc[positions].reset_index(drop=True), np.asarray(delta)[positions], controls, expanded_core)
    old_prediction_features = original['P_features'].iloc[positions].reset_index(drop=True)
    if not extension.frame_equal_bits(old_prediction_features, branch['P_features']):
        raise RuntimeError('Partition changed prediction-only feature bits')
    for frame in branch['reference_features'].values():
        if not extension.frame_equal_bits(old_prediction_features, frame[base.P]):
            raise RuntimeError('Archived Public feature inputs differ from branch risk inputs')
    result = {'scores': original['scores'].copy(deep=True), 'P_features': original['P_features'].copy(deep=True),
        'reference_features': {name: frame.copy(deep=True) for name, frame in original['reference_features'].items()},
        'priors': {name: values.copy() for name, values in original['priors'].items()}}
    # Prediction-only output has no bank input. Preserve its full original
    # batch for ALL queries, while new history features use the new branch.
    for column in branch['scores']:
        if column in base.QUERY or column in ['Magnitude', 'P_only_ridge', 'P_only_hgb', 'P_only_ridge_status', 'P_only_hgb_status']:
            continue
        result['scores'].loc[positions, column] = branch['scores'][column].to_numpy()
    for name, frame in branch['reference_features'].items():
        result['reference_features'][name].loc[positions, base.PUBLIC] = frame[base.PUBLIC].to_numpy()
        result['priors'][name][positions] = branch['priors'][name]
    new_pairs = branch['pairs'].copy(deep=True)
    if len(new_pairs): new_pairs['query_row'] = positions[new_pairs.query_row.to_numpy(int)]
    result['pairs'] = pd.concat([original['pairs'], new_pairs], ignore_index=True).sort_values(['query_row', 'memory_row'], kind='stable').reset_index(drop=True)
    weights = pd.concat([original['weights'], branch['weights']], ignore_index=True)
    if len(weights):
        query_order = dict(zip(queries.query_id, range(len(queries))))
        memory_order = dict(zip(expanded_core['memory'].experiment_id, expanded_core['memory'].effect_vector_row))
        weights['_query_order'] = weights.query_id.map(query_order)
        weights['_reference_order'] = weights.reference.map({'Uniform': 0, 'Manual': 1, 'Learned': 2})
        weights['_memory_order'] = weights.memory_id.map(memory_order)
        weights = weights.sort_values(['_query_order', '_reference_order', '_memory_order'], kind='stable').drop(columns=['_query_order', '_reference_order', '_memory_order']).reset_index(drop=True)
    result['weights'] = weights
    invariant = extension.assert_legacy_invariance(queries, original, result, old_genes)
    invariant.update(execution_partition='complete_original_full_scope_branch_for_old580; same_original_infer_new_absent580_branch',
        prediction_only_all_query_original_batch_preserved=True, original_math_sha256=extension.BASE_RISK_SHA,
        executed_partition_helper=extension.binding(__file__), new_parameter_fits=0, tolerance_changed=False)
    return original, result, invariant


def checked_amendment(path, registration, sealer_path, chain_path=None):
    if path is None: raise RuntimeError('Explicit ROOT legacy-batch technical amendment required')
    item = extension.binding(path); value = extension.read_json(item)
    if (value.get('schema') != 'safeconf_orion_legacy_batch_partition_technical_amendment_v1'
        or value.get('status') != 'ROOT_APPROVED_TECHNICAL_CONTINUATION'
        or value.get('new_model_fits') != 0 or value.get('math_or_tolerance_changed') is not False
        or value.get('old_complete_branch_bytes_required') is not True or value.get('TEST_numeric_access_authorized') is not False):
        raise RuntimeError('Exact no-fit/no-math-change/byte-preserving technical amendment required')
    implementations = {'partition_helper': Path(__file__), 'executed_sealer': Path(sealer_path), 'base_math': Path(base.__file__)}
    if chain_path is not None: implementations['executed_chain'] = Path(chain_path)
    for name, path in implementations.items():
        if value.get(name) != extension.binding(path): raise RuntimeError('Executed partition implementation differs: ' + name)
    for name in ['extension_contract', 'bank_manifest', 'base_comparison_manifest']:
        if value.get(name) != registration[name]: raise RuntimeError('Technical partition registration differs: ' + name)
    return item
