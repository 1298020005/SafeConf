#!/usr/bin/env python3
"""Explicit expanded-history pretruth seal; original Source math and fits fixed."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual import orion_public_bank_extension as extension
from tools.scripts import seal_safeconf_orion_source_risk_agent as base
from tools.safeconf_continual import orion_legacy_partition as partition


def seal(registration_path, output, amendment_path=None):
    registration_binding = extension.binding(registration_path)
    registration = extension.read_json(registration_binding)
    if registration.get('schema') != 'safeconf_orion_public_bank_extension_seal_registration_v1':
        raise RuntimeError('Explicit new-bank pretruth registration required')
    for name, path in [('extension_sealer', Path(__file__)), ('extension_helper', Path(extension.__file__))]:
        item = registration.get('implementation_bindings', {}).get(name, {})
        if extension.checked(item, False) != path.resolve():
            raise RuntimeError('Executed expanded sealer/helper implementation not registered')
    amendment = partition.checked_amendment(amendment_path, registration, __file__)
    extension.contract(registration['extension_contract'])
    manifest_path = extension.checked(registration['base_comparison_manifest'])
    manifest = base.frozen_manifest(manifest_path)
    core = base.source_core(manifest.get('source_core_root', str(base.SOURCE)))
    if manifest.get('source_core_bindings') != core['bindings']:
        raise RuntimeError('Original Source fitted model/axis/CDF/bank pins differ')
    new_core, bank, bank_bindings, old_genes = extension.load_bank(
        registration['bank_manifest'], registration['extension_contract'], core)
    queries, delta, controls, lm_bindings = base.load_inputs(manifest, core)
    output = Path(output).resolve()
    input_roots = [Path(core['root']).resolve(), Path(registration['bank_manifest']['path']).resolve().parent]
    input_roots += [Path(item['path']).resolve().parent for item in lm_bindings]
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in input_roots):
        raise RuntimeError('Expanded seal must be isolated from frozen Source/bank/LM input directories')
    if output.exists(): raise FileExistsError('Immutable expanded risk seal already exists')
    stage = output.with_name(output.name + '.incomplete')
    stage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original, result, invariant = partition.infer_partitioned(queries, delta, controls, core, new_core, old_genes)
    coverage = extension.checked_coverage(bank['coverage_certificate'], queries, result['scores'], result['reference_features']['Manual'])
    original['scores'].to_csv(stage / 'ORIGINAL2008_FIXED_PRETRUTH_RISKS.tsv', sep='\t', index=False,
                              float_format='%.17g', na_rep='NaN')
    result['scores'].to_csv(stage / 'PRETRUTH_RISKS.tsv', sep='\t', index=False, float_format='%.17g', na_rep='NaN')
    result['P_features'].to_parquet(stage / 'P_ONLY_FEATURES.parquet', index=False)
    result['pairs'].to_parquet(stage / 'SOURCE_HISTORY_PAIR_FEATURES.parquet', index=False)
    result['weights'].to_csv(stage / 'SOURCE_PRIOR_WEIGHTS.tsv', sep='\t', index=False, float_format='%.17g')
    for reference, frame in result['reference_features'].items():
        frame.to_parquet(stage / f'{reference}_P_PUBLIC_FEATURES.parquet', index=False)
        np.save(stage / f'{reference}_PRIOR_EFFECTS.npy', result['priors'][reference], allow_pickle=False)
    report = {
        'schema': base.SCHEMA, 'status': 'SEALED_PRETRUTH',
        'comparison_manifest_sha256': base.sha(manifest_path),
        'scientific_contract_sha256': extension.BASE_METHOD_SHA,
        'risk_code_sha256': extension.BASE_RISK_SHA,
        'risk_code_role': 'delegated unchanged original numerical inference functions',
        'executed_extension_sealer_code': extension.binding(__file__),
        'executed_extension_helper_code': extension.binding(extension.__file__),
        'executed_partition_helper_code': extension.binding(partition.__file__), 'legacy_partition_technical_amendment': amendment,
        'source_core_bindings': core['bindings'], 'LM_prediction_and_control_bindings': lm_bindings,
        'retrieval_bank_bindings': bank_bindings,
        'public_bank_extension_contract': registration['extension_contract'],
        'public_bank_manifest': registration['bank_manifest'],
        'n_queries': len(queries), 'known_history_queries': int((result['scores'].source_history_n > 0).sum()),
        'source_only_risk_models': 6, 'public_biology_target': 'continuous_transfer_RMSE_predict_clip_False',
        'pair_features': base.PAIR, 'P_features': base.P, 'P_public_features': base.P + base.PUBLIC,
        'history_retrieval': 'registered expanded Source bank; only targets absent original eligible580 appended',
        'learned_prior_rule': 'unchanged Source exp/std-floor then0.5learned+0.5cellweights',
        'query_truth_input_or_dummy_labels': False, 'query_errors_used': False, 'target_CDF_fit': False,
        'new_learners_or_parameter_fits': 0, 'identity_features_used': False,
        'data_role_registry_sha256': manifest['data_role_registry_sha256'],
        'TEST_identity_metadata_status': 'SEEN_METADATA_ONLY', 'TEST_truth_status': 'CLOSED',
        'whole_method_pristine_claim': False, 'old_confirmation_inherited': False,
        'legacy_scores_archive': 'ORIGINAL2008_FIXED_PRETRUTH_RISKS.tsv',
        'legacy_scores_primary_intersection_policy': 'separate fixed archive/cohort; never shrink expanded primary by legacy unsupported rows',
        'elapsed_seconds': time.monotonic() - started}
    base.write_json(stage / 'RISK_SEAL_MANIFEST.json', report)
    base.write_json(stage / 'EXTENSION_SEAL_RECEIPT.json', {
        'schema': 'safeconf_orion_expanded_bank_pretruth_seal_receipt_v1', 'status': 'COMPLETE',
        'seal_registration': registration_binding, 'extension_contract': registration['extension_contract'],
        'bank_manifest': registration['bank_manifest'], 'new_parameter_fits': 0,
        'base_risk_math': extension.binding(base.__file__),
        'executed_sealer': extension.binding(__file__), 'executed_helper': extension.binding(extension.__file__),
        'executed_partition_helper': extension.binding(partition.__file__), 'legacy_partition_technical_amendment': amendment,
        'base_risk_seal': {'path': str(output / 'RISK_SEAL_MANIFEST.json'),
                           'sha256': base.sha(stage / 'RISK_SEAL_MANIFEST.json'),
                           'bytes': (stage / 'RISK_SEAL_MANIFEST.json').stat().st_size},
        'retrieval_bank_bindings': bank_bindings, 'original_core_bindings': core['bindings'],
        'legacy_invariance': invariant, 'coverage': coverage,
        'original_fixed_scores': {'path': str(output / 'ORIGINAL2008_FIXED_PRETRUTH_RISKS.tsv'),
                                 'sha256': base.sha(stage / 'ORIGINAL2008_FIXED_PRETRUTH_RISKS.tsv'),
                                 'bytes': (stage / 'ORIGINAL2008_FIXED_PRETRUTH_RISKS.tsv').stat().st_size}})
    artifacts = [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': base.sha(path)}
                 for path in sorted(stage.iterdir()) if path.is_file()]
    base.write_json(stage / 'ARTIFACT_HASHES.json', artifacts)
    for path in stage.iterdir(): path.chmod(0o444)
    stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--technical-amendment', type=Path, required=True)
    args = parser.parse_args()
    seal(args.registration, args.output, args.technical_amendment)


if __name__ == '__main__': main()
