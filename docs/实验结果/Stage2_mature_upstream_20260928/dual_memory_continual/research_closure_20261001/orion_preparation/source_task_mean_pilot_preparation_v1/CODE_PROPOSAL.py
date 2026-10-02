"""Static code proposal only; not a CLI or experiment authorization.
Caller must first validate a new immutable Root-approved Source-only scope
contract, ordered metadata/axis/array SHA bindings, attempt accounting and fresh
output root. Do not import/call this module to generate predictions yet.
"""

def propose_five_gene_fold_task_mean(tasks, true_delta, matched_controls):
    # No I/O, model fitting, risk/CDF, Public retrieval or Orion input.
    # Arrays come from the original frozen Source1808x3285 cache only.
    import numpy as np
    if len(tasks) != 1808 or true_delta.shape != (1808, 3285) or matched_controls.shape != true_delta.shape:
        raise ValueError('Exact Source task/axis shape required')
    if len({r['task_id'] for r in tasks}) != 1808:
        raise ValueError('Physical tasks must occur once, not once per predictor')
    genes = np.asarray([r['gene'] for r in tasks], dtype=object)
    contexts = np.asarray([r['target'] for r in tasks], dtype=object)
    folds = np.asarray([int(r['fold']) for r in tasks], dtype=np.int64)
    if set(folds) != set(range(5)) or len(set(genes)) != 575:
        raise ValueError('Original five Source gene folds required')
    for gene in set(genes):
        if len(set(folds[genes == gene])) != 1:
            raise ValueError('Same gene differs across contexts/folds')
    if set(contexts) != {'K562', 'RPE1', 'hepg2', 'jurkat'}:
        raise ValueError('Four frozen Source contexts required')
    predicted_delta = np.full((1808, 3285), np.nan, dtype=np.float64)
    assigned = np.zeros(1808, dtype=np.uint8)
    centroids = []
    for heldout_fold in range(5):
        for context in ('K562', 'RPE1', 'hepg2', 'jurkat'):
            train = (folds != heldout_fold) & (contexts == context)
            query = (folds == heldout_fold) & (contexts == context)
            if not train.any() or not query.any() or set(genes[train]) & set(genes[query]):
                raise ValueError('Nonempty gene-disjoint training/query required')
            # Fitted mean sees training perturbation truth only. Equal TASK
            # weights; no cell count, upstream-record duplication or gene IDs.
            training_absolute = np.asarray(true_delta[train], dtype=np.float64) + matched_controls[train]
            query_controls = np.asarray(matched_controls[query], dtype=np.float64)
            if not np.isfinite(training_absolute).all() or not np.isfinite(query_controls).all():
                raise ValueError('Finite frozen training effects/controls required')
            fitted_absolute_mean = training_absolute.mean(axis=0, dtype=np.float64)
            predicted_delta[query] = fitted_absolute_mean - query_controls
            assigned[query] += 1
            centroids.append((heldout_fold, context, fitted_absolute_mean))
    if not np.all(assigned == 1) or not np.isfinite(predicted_delta).all():
        raise ValueError('Every Source heldout query must be predicted once')
    # First stage/seal predicted_delta +20 centroids +row/fit hashes. Only after
    # that seal may evaluation read heldout true_delta to compute RMSE and
    # cached GAT/Exph errors/magnitudes for fixed per-context diagnostics.
    return predicted_delta, centroids

if __name__ == '__main__':
    raise SystemExit('Proposal only: execution awaits the Root Source-only scope contract')
