#!/usr/bin/env python3
"""Requested fixed contrasts from existing joint bootstrap draws; no new fits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis/results/source_row_weight_diagnostic'
RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/source_row_weight_diagnostic')
PAIRS = [('Full_pool_standard', 'GAT_duplicate2'),
         ('Full_pool_standard', 'Exphormer_duplicate2'),
         ('Full_pool_single_source_weight', 'GAT_unique'),
         ('Full_pool_single_source_weight', 'Exphormer_unique')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs = [DOC / 'SUPPLEMENT_REGISTRATION.json', DOC / 'REQUESTED_PAIRED_GENE_U20.csv',
               DOC / 'SUPPLEMENT_STATUS.json']
    if any(path.exists() for path in outputs):
        raise RuntimeError('Existing supplementary outputs cannot be overwritten')
    status_path, macro_path = DOC / 'STATUS.json', DOC / 'MACRO.csv'
    status = json.loads(status_path.read_text())
    if status.get('status') != 'COMPLETE' or status.get('fitted_table_models') != 12:
        raise RuntimeError('Completed original fixed diagnostic required')
    # Recheck existing bootstrap artifact bindings before any calculation.
    bindings = {Path(entry['path']): entry['sha256'] for entry in status['output_bindings']}
    draws_paths = [RUNTIME / f'{ref}_JOINT_GENE_BOOTSTRAP.npz' for ref in ('Manual', 'Learned')]
    for path in draws_paths:
        if sha(path) != bindings.get(path):
            raise RuntimeError('Previously frozen joint bootstrap draw hash differs')
    registration = {'role': status['role'], 'requested_fixed_pairs': [list(pair) for pair in PAIRS],
                    'new_fits': 0, 'new_CDFs': 0, 'new_bootstrap_draws': 0, 'Orion_access': False,
                    'frozen_base_script_sha256': status['script_sha256'],
                    'code_sha256': sha(__file__),
                    'input_bindings': [{'path': str(path), 'sha256': sha(path)}
                                       for path in [status_path, macro_path] + draws_paths]}
    outputs[0].write_text(json.dumps(registration, indent=2) + '\n')
    macro = pd.read_csv(macro_path)
    records = []
    for ref, path in zip(('Manual', 'Learned'), draws_paths):
        with np.load(path, allow_pickle=False) as archive:
            names = archive['scenario'].tolist()
            draws = archive['macro_utility20']
            seed = int(archive['seed'])
        if draws.shape != (5000, 6) or seed != 20260930 or not np.isfinite(draws).all():
            raise RuntimeError('Original frozen finite joint-gene draws required')
        observed = macro[macro.reference.eq(ref)].set_index('scenario').utility20
        for a, b in PAIRS:
            delta = draws[:, names.index(a)] - draws[:, names.index(b)]
            lower, upper = np.quantile(delta, [.025, .975], method='linear')
            records.append({'reference': ref, 'method_a': a, 'method_b': b,
                            'delta_utility20_a_minus_b': float(observed[a] - observed[b]),
                            'bootstrap_mean_delta': float(delta.mean()),
                            'ci95_lower': float(lower), 'ci95_upper': float(upper),
                            'bootstrap_replicates': 5000, 'valid_draws': len(delta),
                            'bootstrap_seed': seed, 'n_tasks': 543, 'n_gene_clusters': 380,
                            'n_contexts': 3, 'macro': 'equal context mean',
                            'direction': 'positive favors method_a',
                            'new_fits': 0, 'new_draws': 0})
    pd.DataFrame(records).to_csv(outputs[1], index=False, lineterminator='\n')
    completed = {'status': 'COMPLETE', 'fixed_comparisons': len(records), 'new_fits': 0,
                 'new_CDFs': 0, 'new_bootstrap_draws': 0, 'original_outputs_preserved': True,
                 'Orion_access': False, 'parameter_or_winner_selection': False,
                 'output_bindings': [{'path': str(path), 'sha256': sha(path)} for path in outputs[:2]]}
    outputs[2].write_text(json.dumps(completed, indent=2) + '\n')
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
