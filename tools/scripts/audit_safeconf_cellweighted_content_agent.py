#!/usr/bin/env python3
"""Fixed cell-weighted Public content audit on already seen frozen assets.

Reuse all five existing matched-support donor orders. No raw H5AD access,
upstream call, learner fitting, CDF fitting, truth change or winner selection.
Only this agent's output directory is written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual import PublicMemoryStore
from tools.safeconf_continual.research import SEEDS, bootstrap_u20, ids_hash, summarize
from tools.scripts import run_safeconf_common_axis_closure as common
from tools.scripts.run_safeconf_public_mechanisms import permutation

OUT = common.closure.OUT / 'evidence_review_agent' / 'cellweighted_content_followup'
BANK = common.COMMON / 'public_mcfaline_trainval'
CELL_EFFECT = common.COMMON / 'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.perf_counter()
    if (OUT / 'STATUS.json').exists():
        raise FileExistsError('completed diagnostic cannot be overwritten')
    OUT.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(__file__), BANK / 'public_memory.parquet', BANK / 'effect_vectors.npy',
        BANK / 'control_vectors.npy', CELL_EFFECT, common.COMMON / 'risk_cache/external_Manual.parquet',
        common.COMMON / 'TEST_CALIBRATED_EFFECTS.npy', common.COMMON / 'TEST_CONTROLS.npy',
        common.COMMON / 'TEST_TRUE_EFFECTS.npy', common.COMMON / 'TEST_TASKS.csv',
        common.DOC / 'results/MATRIX_TASK_PREDICTIONS.csv.gz',
        common.COMMON / 'reference_estimand_diagnostic/TASK_PREDICTIONS.csv.gz',
        ROOT / 'tools/scripts/run_safeconf_public_mechanisms.py']
    inputs = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in input_paths]
    pd.DataFrame(inputs).to_csv(OUT / 'INPUT_HASHES.csv', index=False)
    donor_seeds = [SEEDS[0] + i for i in range(5)]
    registered = {
        'role': 'SEEN_POST_CONFIRMATION_DIAGNOSTIC',
        'reference': 'frozen cell-weighted within-experiment effects',
        'donor_implementation': 'existing public_mechanisms.permutation(memory, seed, McFaline)',
        'donor_seeds': donor_seeds, 'all_five_orders_reported': True,
        'support_matching': 'same study/contract/gene space/context/condition and n_cells quartile',
        'weights': 'unchanged n_cells / sum(n_cells) for the original eligible historical rows',
        'query_cohort': '543 original tasks, 380 gene clusters',
        'truth': 'exact canonical CSV error; recompute arrays only to verify it',
        'bootstrap_replicates': 5000, 'bootstrap_seed': SEEDS[0],
        'raw_h5ad_reads': 0, 'new_model_fits': 0, 'new_upstream_calls': 0,
        'method_selection_allowed': False, 'input_hashes': inputs,
    }
    (OUT / 'REGISTERED_DIAGNOSTIC.json').write_text(json.dumps(registered, indent=2) + '\n')
    memory, guide_effects, controls, _ = PublicMemoryStore(BANK).load()
    cell_effects = np.load(CELL_EFFECT, mmap_mode='r')
    if cell_effects.shape != guide_effects.shape or not np.isfinite(cell_effects).all():
        raise RuntimeError('cell-weighted bank differs in shape or has nonfinite values')
    if not np.array_equal(memory.index.to_numpy(), np.arange(len(memory))):
        raise RuntimeError('existing donor implementation requires canonical row indices')
    base = pd.read_parquet(common.COMMON / 'risk_cache/external_Manual.parquet')
    tasks = pd.read_csv(common.COMMON / 'TEST_TASKS.csv')
    prediction = np.load(common.COMMON / 'TEST_CALIBRATED_EFFECTS.npy')
    truth = np.load(common.COMMON / 'TEST_TRUE_EFFECTS.npy')
    test_controls = np.load(common.COMMON / 'TEST_CONTROLS.npy')
    if len(base) != 543 or base.gene.nunique() != 380 or list(base.task_id) != list(tasks.task_id):
        raise RuntimeError('frozen query identity or cohort changed')
    primary = pd.read_csv(common.DOC / 'results/MATRIX_TASK_PREDICTIONS.csv.gz')
    primary = primary[(primary.line == 'TxPert_to_McFaline') & (primary.seed == SEEDS[0]) &
        (primary.method == 'Manual_WeightedHistoryDistance')].set_index('task_id').loc[base.task_id]
    if not np.allclose(primary.true_error_rmse, np.sqrt(np.mean((prediction - truth) ** 2, axis=1)),
                       rtol=1e-6, atol=1e-8):
        raise RuntimeError('unchanged canonical truth could not be reproduced')
    base['true_error_rmse'] = primary.true_error_rmse.to_numpy(float)
    pairs = common.closure.mc.build_pairs(base, test_controls, memory, guide_effects, controls, None)
    pairs.to_csv(OUT / 'FROZEN_PAIR_IDENTITIES.csv.gz', index=False,
        compression={'method': 'gzip', 'mtime': 0})
    groups = []
    for q, group in pairs.groupby('task_row', sort=True):
        rows = group.memory_row.to_numpy(int)
        counts = np.expm1(group.log_source_cells.to_numpy(float))
        groups.append((q, rows, counts / counts.sum()))
    if sorted(q for q, _, _ in groups) != list(range(543)):
        raise RuntimeError('content audit changed eligible history coverage')

    def distance(effects, donor=None):
        score = np.full(len(base), np.nan)
        for q, rows, weights in groups:
            donor_rows = rows if donor is None else donor[rows]
            vectors = np.asarray(effects[donor_rows], float)
            score[q] = np.sqrt(weights @ np.mean((vectors - prediction[q]) ** 2, axis=1))
        if not np.isfinite(score).all():
            raise RuntimeError('content permutation changed query coverage')
        return score

    scores = {'FrozenGuideEqualReference': distance(guide_effects),
        'CellWeightedReference': distance(cell_effects),
        'NegativeHistorySupport': -base.log_history_support.to_numpy(float)}
    if not np.allclose(scores['FrozenGuideEqualReference'], primary.risk, rtol=1e-6, atol=1e-8):
        raise RuntimeError('frozen guide-equal reference did not reproduce')
    previous = pd.read_csv(common.COMMON / 'reference_estimand_diagnostic/TASK_PREDICTIONS.csv.gz')
    previous = previous[previous.method.eq('CellWeightedReference')].set_index('task_id').loc[base.task_id]
    if not np.allclose(scores['CellWeightedReference'], previous.risk, rtol=1e-12, atol=1e-14):
        raise RuntimeError('frozen true cell-weighted reference did not reproduce exactly')

    audits, effectiveness, identities = [], [], []
    used_rows = np.concatenate([rows for _, rows, _ in groups])
    for order, seed in enumerate(donor_seeds):
        donor, audit = permutation(memory, seed, 'McFaline')
        method = f'CellWeightedContentNull_{order}'
        scores[method] = distance(cell_effects, donor)
        identity = pd.DataFrame({'order': order, 'donor_seed': seed,
            'recipient_row': np.arange(len(memory)), 'donor_row': donor,
            'recipient_experiment_id': memory.experiment_id.to_numpy(str),
            'donor_experiment_id': memory.iloc[donor].experiment_id.to_numpy(str)})
        identities.append(identity)
        for entry in audit:
            entry['order'] = order
        audits.extend(audit)
        ratio = memory.iloc[donor].n_cells.to_numpy(float) / memory.n_cells.to_numpy(float)
        effectiveness.append({'order': order, 'donor_seed': seed, 'method': method,
            'n_public_rows': len(memory), 'n_moved_rows': int(np.sum(donor != np.arange(len(memory)))),
            'moved_fraction': float(np.mean(donor != np.arange(len(memory)))),
            'query_pair_count': len(used_rows), 'n_moved_query_pairs': int(np.sum(donor[used_rows] != used_rows)),
            'moved_query_pair_fraction': float(np.mean(donor[used_rows] != used_rows)),
            'donor_array_sha256': hashlib.sha256(donor.astype('<i8').tobytes()).hexdigest(),
            'donor_identity_hash': ids_hash(identity.recipient_experiment_id + ' -> ' + identity.donor_experiment_id),
            'donor_support_ratio_min': float(ratio.min()), 'donor_support_ratio_median': float(np.median(ratio)),
            'donor_support_ratio_max': float(ratio.max()),
            'matches_counts_exactly': False, 'matches_existing_support_quartile': True})
        print(f'Content order {order}, seed {seed}: all 543 distances calculated', flush=True)
    pd.concat(identities, ignore_index=True).to_csv(OUT / 'DONOR_IDENTITIES.csv.gz', index=False,
        compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(audits).to_csv(OUT / 'PERMUTATION_GROUP_AUDIT.csv', index=False)
    pd.DataFrame(effectiveness).to_csv(OUT / 'PERMUTATION_EFFECTIVENESS.csv', index=False)
    records = []
    for method, risk in scores.items():
        frame = base[['task_id', 'target', 'gene', 'true_error_rmse']].copy()
        frame['method'] = method; frame['risk'] = risk
        frame['line'] = 'TxPert_to_McFaline'; frame['seed'] = SEEDS[0]
        records.append(frame)
    predictions = pd.concat(records, ignore_index=True)
    predictions.to_csv(OUT / 'TASK_PREDICTIONS.csv.gz', index=False,
        compression={'method': 'gzip', 'mtime': 0})
    strata, macro = summarize(predictions)
    strata.to_csv(OUT / 'STRATA.csv', index=False); macro.to_csv(OUT / 'MACRO.csv', index=False)
    comparisons = []
    for method_b in ['NegativeHistorySupport', 'FrozenGuideEqualReference']:
        comparisons.append({'method_a': 'CellWeightedReference', 'method_b': method_b,
            **bootstrap_u20(base, scores['CellWeightedReference'], scores[method_b], 5000)})
    for order, seed in enumerate(donor_seeds):
        method = f'CellWeightedContentNull_{order}'
        for method_a, method_b in [('CellWeightedReference', method), (method, 'NegativeHistorySupport')]:
            comparisons.append({'order': order, 'donor_seed': seed, 'method_a': method_a, 'method_b': method_b,
                **bootstrap_u20(base, scores[method_a], scores[method_b], 5000)})
        pd.DataFrame(comparisons).to_csv(OUT / 'PAIRED_BOOTSTRAP.csv', index=False)
        print(f'Content order {order}: two 5000-draw paired cluster intervals completed', flush=True)
    # Verify that every frozen input still has its registration hash.
    for entry in inputs:
        if digest(Path(entry['path'])) != entry['sha256']:
            raise RuntimeError(f"registered input changed during read-only audit: {entry['path']}")
    status = {'status': 'COMPLETE', 'role': 'SEEN_POST_CONFIRMATION_DIAGNOSTIC',
        'n_tasks': len(base), 'n_gene_clusters': base.gene.nunique(), 'n_public_rows': len(memory),
        'n_query_pairs': len(pairs), 'n_donor_orders': 5, 'bootstrap_replicates': 5000,
        'raw_h5ad_reads': 0, 'new_model_fits': 0, 'new_upstream_calls': 0,
        'all_frozen_inputs_unchanged': True, 'canonical_truth_unchanged': True,
        'original_guide_distance_reproduced': True, 'true_cell_distance_reproduced': True,
        'elapsed_seconds': time.perf_counter() - started,
        'pair_identity_sha256': digest(OUT / 'FROZEN_PAIR_IDENTITIES.csv.gz')}
    (OUT / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(macro[['method', 'utility20', 'spearman']].to_string(index=False), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
