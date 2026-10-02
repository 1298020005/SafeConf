#!/usr/bin/env python3
"""Interpret completed frozen summaries only; never open task errors or models."""
import argparse, csv, hashlib, json, math
from pathlib import Path

BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
DOC = Path(__file__).resolve().parents[1]
PINS = {
    'operation': (BASE/'orion_registered_test_extendedbank_20261002_v1/IMMUTABLE_POSTSEAL_TEST_ACCESS.json', 'c94bc298c95bbe4a4d6f92dac8263b9aa93be18c48297f5181232bf73d44881d'),
    'claim': (BASE/'ORION_REGISTERED_TEST_FIRST_OPEN_SCOPE.json', 'b12048618a0be885d2e66536c0fe3b0cc25d535b69a00c5c60f0634b2a1ef534'),
    'registry': (DOC/'orion_preparation/risk_evaluation/ORION_CLAIM_INTERPRETATION_REGISTRY.json', 'd227bf92e1cc4396a332b82f0ae5a3ffd90a03ce21dadac50ed8d1e0c9382bea'),
    'safety': (DOC/'orion_preparation/public_bank_extension_actual_pretruth_independent_review_v1/PRETRUTH_SAFETY_SCOPE_ADDENDUM.json', 'bc3be48f66a4f2af2f17eba20355b22e9e1df3aedbf1f7d6a100d7b735f3e85c'),
    'source_split': (BASE/'orion_source_core_20261002_v1/SPLIT_MANIFEST.csv', '350e53a18f54108eef56062b96875f5ca0ad3d1095eb155aa2303eb35422aa71')}
CONTEXTS = ('HCT116', 'HEK293T')
METRICS = ('utility20', 'spearman', 'aurc', 'error_at_10', 'error_at_20', 'error_at_50', 'high_risk_miss_rate')
CONTRASTS = {'primary': ('Learned_hgb', 'Learned_WeightedHistoryDistance'), 'manual_secondary': ('Manual_hgb', 'Manual_WeightedHistoryDistance'), 'support': ('Learned_hgb', 'NegativeSourceHistorySupport'), 'Public': ('Learned_hgb', 'P_only_hgb')}
CHAIN = '19ea756bd8fd2d9be3cc81392abb1952c648a1654401d8748a6af400de8a377d'
EVALUATOR = 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c'

def check(ok, message):
    if not ok: raise ValueError(message)

def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

def bound(item):
    actual = binding(item['path'])
    check(all(actual[k] == item[k] for k in item if k in actual), 'Binding differs: '+item['path'])
    return Path(actual['path'])

def read_json(path): return json.loads(Path(path).read_text())
def rows(path): return list(csv.DictReader(Path(path).open()))
def number(value):
    try: value = float(value)
    except (ValueError, TypeError): return None
    return value if math.isfinite(value) else None

def unique(table, **keys):
    selected = [x for x in table if all(x.get(k) == v for k, v in keys.items())]
    check(len(selected) == 1, 'Expected exactly one fixed row: '+str(keys))
    return selected[0]

def inspect(root):
    receipt_path = root/'EXTENSION_EVALUATION_RECEIPT.json'
    receipt = read_json(receipt_path)
    check(receipt.get('schema') == 'safeconf_orion_extended_bank_evaluation_receipt_v1' and receipt.get('status') == 'COMPLETE', 'Completed extension evaluation required before any CSV access')
    for path, digest in PINS.values(): check(binding(path)['sha256'] == digest, 'Pretruth pin differs: '+str(path))
    op, claim = (read_json(PINS[k][0]) for k in ('operation', 'claim'))
    registry, safety = (read_json(PINS[k][0]) for k in ('registry', 'safety'))
    section = op['public_bank_extension']; truth = read_json(bound(receipt['extension_truth_receipt']))
    check(Path(receipt['extension_truth_receipt']['path']).resolve() == Path(op['output_path']).resolve()/'EXTENSION_TRUTH_RECEIPT.json', 'Actual registered truth output path required')
    check(truth.get('schema') == 'safeconf_orion_extended_bank_truth_receipt_v1' and truth.get('status') == 'COMPLETE' and truth.get('bank_verified_before_any_TEST_reader_call') is True, 'Exact completed extension truth receipt required')
    check(truth['executed_extension_chain']['sha256'] == CHAIN and truth['delegated_unchanged_reader']['sha256'] == '852fcd764b567a63f04c4a5539a118c8a715f664bf534184018010ac269087b2', 'Completed actual truth implementation differs')
    check(truth['operation'] == binding(PINS['operation'][0]) and truth['exclusive_registered_scope'] == binding(PINS['claim'][0]), 'Exact c94 operation/b120 claim required')
    check(claim['operation'] == truth['operation'] and claim['scientific_contract'] == op['scientific_contract'] and claim['forty_object_manifest'] == op['forty_object_manifest'] and claim['gene_split_sha256'] == op['metadata_bindings']['GENE_SPLIT.csv'] and claim['output_path'] == op['output_path'], 'Claim scope differs')
    check(claim.get('automatic_second_registration_allowed') is False and claim.get('failure_does_not_release_scope') is True, 'Once-only semantics differ')
    check(truth['extension_operation'] == section and receipt['comparison_receipt'] == section['comparison_receipt'] and receipt['bank_manifest'] == section['bank_manifest'] and receipt['extension_contract'] == section['extension_contract'], 'Actual extension lineage differs')
    check(receipt['executed_chain']['sha256'] == CHAIN and receipt['delegated_evaluator']['sha256'] == EVALUATOR and receipt['new_fits'] == truth['new_fits'] == 0 and truth['target_CDF_fit'] is False and receipt['legacy_scores_remain_separate_archive'] is True, 'Frozen evaluator/code/fit policy differs')
    bound(receipt['executed_chain']); bound(receipt['delegated_evaluator'])
    comparison = read_json(bound(receipt['comparison_receipt'])); spec = read_json(bound(comparison['base_comparison']['spec']))
    check(spec['methods'][-1] == 'NegativeSourceHistorySupport' and len(spec['methods']) == 13 and spec['primary_pair'] == list(CONTRASTS['primary']) and spec['prespecified_manual_pair'] == list(CONTRASTS['manual_secondary']) and spec['bootstrap_replicates'] == 5000 and spec['bootstrap_seed'] == 20260929, 'Fixed method/contrast/bootstrap differs')
    manifest = read_json(root/'EVALUATION_MANIFEST.json'); artifacts = read_json(root/'ARTIFACT_HASHES.json')
    check(manifest.get('schema') == 'safeconf_orion_frozen_risk_evaluation_v1' and manifest.get('status') == 'COMPLETE_FIXED_COMPARISON' and manifest['comparison_spec_sha256'] == comparison['base_comparison']['spec']['sha256'] and manifest['truth_reader_receipt_sha256'] == truth['base_truth_receipt']['sha256'] and manifest['truth_errors_sha256'] == truth['truth_errors']['sha256'], 'Completed evaluator provenance differs')
    check(manifest['candidate_selection_after_truth'] is False and manifest['new_fits'] == 0 and manifest['target_CDF_fit'] is False and manifest['bootstrap_replicates'] == 5000 and manifest['bootstrap_seed'] == 20260929, 'No-selection/fits/statistics policy differs')
    check(manifest['primary_pair'] == spec['primary_pair'] and manifest['prespecified_manual_pair'] == spec['prespecified_manual_pair'] and manifest['all_nominal_CIs_reported'] is True, 'Completed fixed contrast reporting differs')
    allowed = ['EVALUATION_MANIFEST.json', 'COHORT_COVERAGE.csv', 'METHOD_METRIC_INTERVALS.csv', 'ALL_PRESPECIFIED_PAIRED_COMPARISONS.csv', 'primary_common_COHORT.csv']
    for name in allowed:
        entry = unique(artifacts, path=name); check(binding(root/name)['sha256'] == entry['sha256'], 'Completed output hash differs: '+name)
    points, pairs, coverage, cohort = (rows(root/name) for name in allowed[2:4]+[allowed[1], allowed[4]])
    source = {x['gene'] for x in rows(PINS['source_split'][0])}; check(len(source) == 575, 'Source575 pretruth scope differs')
    for context in CONTEXTS:
        row = unique(coverage, cohort='primary_common', context=context)
        check(int(row['n_tasks']) == sum(x['context_id'] == context for x in cohort), 'Primary cohort count differs')
    results = {}
    for label, (a, b) in CONTRASTS.items():
        evidence, guards, finite = {}, {}, True
        for context in CONTEXTS+('macro',):
            direct = [x for x in pairs if x['cohort'] == 'primary_common' and x['context'] == context and x['metric'] == 'utility20' and (x['method_a'], x['method_b']) in [(a,b),(b,a)]]
            check(len(direct) == 1, 'Fixed contrast missing/duplicated: '+label+'/'+context)
            row = direct[0]; delta, lo, hi = (number(row[k]) for k in ('difference', 'ci95_lower', 'ci95_upper'))
            if row['method_a'] != a: delta, lo, hi = (-delta if delta is not None else None, -hi if hi is not None else None, -lo if lo is not None else None)
            finite = finite and all(x is not None for x in (delta, lo, hi))
            evidence[context] = {'delta_U20': delta, 'CI95': [lo,hi], 'valid_draws': int(row['valid_draws'])}
            metric_points = {method: {metric: number(unique(points, cohort='primary_common', context=context, method=method, metric=metric)['point']) for metric in METRICS} for method in (a,b)}
            finite = finite and all(x is not None for values in metric_points.values() for x in values.values())
            guards[context] = {}
            for metric in METRICS[2:]:
                av, bv = metric_points[a][metric], metric_points[b][metric]
                change = None if av is None or bv is None else (av-bv if metric == 'high_risk_miss_rate' else (av/bv-1 if bv > 0 else None))
                change = number(change); limit = 0.02 if metric == 'high_risk_miss_rate' else 0.05
                guards[context][metric] = {'a':av, 'b':bv, 'worsening':change, 'limit':limit, 'pass':None if change is None else change <= limit}
        valid_contexts = all(int(unique(coverage, cohort='primary_common', context=ctx)['n_tasks']) >= 20 for ctx in CONTEXTS)
        nonnegative_contexts = all(evidence[ctx]['delta_U20'] is not None and evidence[ctx]['delta_U20'] >= 0 for ctx in CONTEXTS)
        macro = evidence['macro']; practical = macro['delta_U20'] is not None and macro['delta_U20'] >= registry['practical_margin_U20']
        noninferiority = macro['CI95'][0] is not None and macro['CI95'][0] >= registry['CI_negative_effect_tolerance']
        positive = macro['CI95'][0] is not None and macro['CI95'][0] > 0
        safety_pass = all(x['pass'] is True for scope in guards.values() for x in scope.values())
        required = label in ('primary', 'support')
        supported = finite and valid_contexts and practical and positive and (not required or (nonnegative_contexts and safety_pass))
        results[label] = {'methods':[a,b], 'status':'PASS' if supported else 'NOT_ESTABLISHED', 'signed_U20':evidence, 'practical_point_margin_met':practical, 'CI_noninferiority_style_guard_met':noninferiority, 'positive_CI_met':positive, 'all_required_points_intervals_finite':finite, 'both_contexts_valid':valid_contexts, 'both_context_deltas_nonnegative':nonnegative_contexts, 'safety_required_by_registry':required, 'macro_and_both_context_safety':guards, 'safety_pass':safety_pass, 'secondary_never_replaces_primary':label == 'manual_secondary'}
    seen = {x['target_gene_id'] for x in cohort if x['target_gene_symbol'] in source}; unseen = {x['target_gene_id'] for x in cohort if x['target_gene_symbol'] not in source}
    return {'status':'COMPLETE_FIXED_INTERPRETATION', 'evaluation_receipt':binding(receipt_path), 'interpretation_registry':binding(PINS['registry'][0]), 'safety_scope':binding(PINS['safety'][0]), 'operation':binding(PINS['operation'][0]), 'claim':binding(PINS['claim'][0]), 'fixed_contrasts':results, 'primary_source_transfer':results['primary']['status'], 'beyond_strong_history_rules':'PASS' if results['primary']['status'] == results['support']['status'] == 'PASS' else 'NOT_ESTABLISHED', 'actual_primary_gene_clusters':len(seen|unseen), 'Source_seen_unseen_gene_clusters_descriptive_only':[len(seen),len(unseen)], 'no_champion_or_subgroup_selection':True, 'negative_results_retained':True, 'uncomputed_or_invalid_guards_never_pass':True, 'limits':['Nominal paired CIs conditional on fixed fits; no rerun/bootstrap/refit.', 'Public package is not isolated effect-content/Quality benefit.', 'Available2790 C-validation errors were not used for risk training; no comparison establishes label/call efficiency against an affordable C-validation learner.', 'No full-task/no-history fallback or pristine confirmation claim.']}

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--evaluation-root', type=Path, required=True); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    check(not args.output.exists(), 'New immutable interpretation output required')
    check(args.output.resolve().is_relative_to(Path(__file__).resolve().parent), 'Write only to this independent interpretation folder')
    try: result = inspect(args.evaluation_root.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error: result = {'status':'UNCOMPUTED_CONTRACT_OR_COMPLETION_UNAVAILABLE', 'reason':str(error), 'no_performance_claim':True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    args.output.chmod(0o444)

if __name__ == '__main__': main()
