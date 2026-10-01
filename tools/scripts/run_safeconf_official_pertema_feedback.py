#!/usr/bin/env python3
"""Fixed-commit PertEMA model factory on SafeConf's strict feedback contract.

This is explicitly an adaptation, not a reproduction of PertEMA's CD4 study,
64-dimensional reference features, Pearson-error target, or bundled weights.
The scientific tree parameters come from the actual official gbt() function.
Raw and inner-OOF isotonic scores are both saved; holdout cannot select them.
All target error uses, including calibration/CDFs, stay within the same budget.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import (
    P, PUBLIC, SEEDS, bootstrap_u20, cluster_weights, ids_hash, rank_labels, summarize,
)
from tools.scripts import run_safeconf_research_closure as closure
from tools.safeconf_continual.learners import NumericPreprocessor

COMMIT = '43c09a32e23d0ee2ae5dfbab21b2deeab27f1803'
OFFICIAL = Path('/home/yyf/runtime_artifacts/official_pertema_43c09a')
OUT = closure.OUT / 'official_pertema_adaptation'
FEATURE_SETS = {
    'PertEMA_P_adapted': P,
    'PertEMA_Public_adapted': P + PUBLIC,
    'PertEMA_Shared_adapted': P + PUBLIC + ['shared_risk'],
}


def factory():
    import subprocess
    head = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise RuntimeError('official source commit changed')
    path = OFFICIAL / 'src/pertema/run_estimator.py'
    spec = importlib.util.spec_from_file_location('official_pertema_estimator_fixed', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gbt, hashlib.sha256(path.read_bytes()).hexdigest()


class NativePredictor:
    """Fold-local native training similarity and optional HGB preprocessing."""

    def __init__(self, model, columns, kind):
        self.model, self.columns, self.kind = model, columns, kind
        self.embedding_positions = [i for i, c in enumerate(columns) if c.startswith('native_embedding_')]
        self.similarity_position = columns.index('native_training_similarity')

    def design(self, x):
        x = np.asarray(x, float).copy()
        embeddings = x[:, self.embedding_positions]
        valid = np.isfinite(embeddings).all(axis=1)
        x[:, self.similarity_position] = np.nan
        if valid.any():
            x[valid, self.similarity_position] = self.neighbors.kneighbors(embeddings[valid])[0].ravel()
        return x

    def fit(self, x, y, sample_weight):
        embeddings = np.asarray(x, float)[:, self.embedding_positions]
        valid = np.isfinite(embeddings).all(axis=1)
        prototypes = np.unique(embeddings[valid], axis=0)
        if not len(prototypes):
            raise RuntimeError('no train-feedback genes with a native control embedding')
        self.neighbors = NearestNeighbors(n_neighbors=1, algorithm='brute', n_jobs=4).fit(prototypes)
        values = self.design(x)
        self.preprocessor = NumericPreprocessor().fit(values) if self.kind == 'hgb' else None
        self.model.fit(self.preprocessor.transform(values) if self.preprocessor else values, y,
                       sample_weight=sample_weight)
        return self

    def predict(self, x):
        values = self.design(x)
        return self.model.predict(self.preprocessor.transform(values) if self.preprocessor else values)


def train(gbt, fit, labels, columns, seed, kind='xgb'):
    # Preserve native NaN handling; no data-dependent feature/model selection.
    model = gbt().set_params(n_jobs=4, random_state=seed) if kind == 'xgb' else HistGradientBoostingRegressor(
        max_iter=200, learning_rate=.05, max_depth=3, min_samples_leaf=20,
        l2_regularization=10, random_state=seed)
    if 'native_training_similarity' in columns:
        return NativePredictor(model, columns, kind).fit(fit[columns].to_numpy(float), labels,
                                                       cluster_weights(fit))
    model.fit(fit[columns].to_numpy(float), labels, sample_weight=cluster_weights(fit))
    return model


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--native-controls', action='store_true')
    args = parser.parse_args()
    feature_sets, kinds = FEATURE_SETS, {}
    native_audit = None
    if args.native_controls:
        native_root = closure.RUNTIME / 'native_control_reference'
        native_audit = json.loads((native_root / 'AUDIT.json').read_text())
        native_columns = native_audit['feature_names']
        OUT = closure.OUT / 'native_control_pertema_adaptation'
        feature_sets = {'NativeControl_HGB': native_columns,
                        'PertEMA_Native_adapted': native_columns,
                        'PertEMA_Native_Public_adapted': native_columns + PUBLIC,
                        'PertEMA_Native_Shared_adapted': native_columns + PUBLIC + ['shared_risk']}
        kinds = {'NativeControl_HGB': 'hgb'}
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'RUN_STATUS.json').exists():
        raise FileExistsError('completed official adaptation is immutable; create a new run ID')
    gbt, official_hash = factory()
    import xgboost
    config = {
        'official_repository': 'https://github.com/OfficialBishal/PertEMA',
        'official_commit': COMMIT, 'official_factory': 'src/pertema/run_estimator.py::gbt',
        'official_source_sha256': official_hash, 'xgboost_version': xgboost.__version__,
        'official_parameters': gbt().get_params(), 'feature_sets': feature_sets,
        'adaptations': ['SafeConf prediction/public inputs replace CD4 reference features',
            'training-budget context CDF of task RMSE replaces Pearson-error target',
            'biological-cluster equal training weights', 'registered SafeConf seeds',
            '4 CPU threads', 'optional group-OOF isotonic calibration'],
        'role': 'SEEN_POST_CONFIRMATION', 'primary_output': 'raw official-factory score',
        'validation_errors_used_for_target_CDF': 0,
        'no_split_conformal_claim': 'no separate conformal set is allocated; no coverage guarantee claimed',
        'bootstrap_replicates': args.bootstrap,
        'own_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_control_audit': native_audit,
    }
    closure.tx.atomic_json(OUT / 'EXECUTION_CONFIG.json', config)
    frame = pd.read_parquet(closure.RUNTIME / 'native_control_reference/TASK_FEATURES.parquet'
                            if args.native_controls else closure.RUNTIME / 'external_Learned.parquet')
    base = pd.read_csv(closure.OUT / 'STRICT_FEEDBACK_TASK_PREDICTIONS.csv.gz')
    baseline = base[base.method.eq('Shared') & base.seed.eq(SEEDS[0]) & base.budget.eq(0)]
    holdout_ids = set(baseline.task_id)
    query = frame[frame.task_id.isin(holdout_ids)].reset_index(drop=True)
    pool = frame[~frame.task_id.isin(holdout_ids)].reset_index(drop=True)
    if set(pool.gene) & set(query.gene):
        raise RuntimeError('feedback and permanent evaluation overlap')
    shared = pd.read_csv(closure.OUT / 'MATRIX_TASK_PREDICTIONS.csv.gz')
    shared = shared[shared.line.eq('TxPert_to_McFaline') & shared.method.eq('Learned_hgb')
                    & shared.seed.eq(SEEDS[0])].set_index('task_id').risk
    for part in (pool, query):
        part['shared_risk'] = part.task_id.map(shared)
    ordered = sorted(pool.gene.unique(), key=lambda g: hashlib.sha256(
        f'SafeConf-McFaline-feedback-v1\0{g}'.encode()).hexdigest())
    records, ledger, cdf_audit, costs, calibrations = [], [], [], [], []
    for budget in (.1, .25, .5, .75, 1.):
        n = int(np.ceil(budget * len(ordered)))
        fit = pool[pool.gene.isin(ordered[:n])].reset_index(drop=True)
        labels, audits = rank_labels(fit, f'official-pertema/{budget}', budget)
        cdf_audit.extend(audits)
        original_ledger = pd.read_csv(closure.OUT / 'STRICT_FEEDBACK_INFORMATION_LEDGER.csv')
        allowed = original_ledger[(original_ledger.budget.eq(budget)) &
                                  original_ledger.method.eq('TargetOnly_HGB')].iloc[0]
        if ids_hash(fit.task_id) != allowed.allowed_feedback_records_hash:
            raise RuntimeError('official adaptation changed the feedback budget records')
        for method, columns in feature_sets.items():
            for seed in SEEDS:
                started = time.monotonic()
                oof_raw = np.full(len(fit), np.nan)
                oof_target = np.full(len(fit), np.nan)
                for inner, (i, j) in enumerate(GroupKFold(3).split(fit, groups=fit.gene)):
                    train_frame = fit.iloc[i].reset_index(drop=True)
                    val_frame = fit.iloc[j].reset_index(drop=True)
                    if set(train_frame.gene) & set(val_frame.gene):
                        raise RuntimeError('isotonic OOF clusters overlap')
                    inner_labels, audits = rank_labels(train_frame,
                        f'official-pertema/{budget}/{method}/{seed}/inner{inner}', budget)
                    cdf_audit.extend(audits)
                    model = train(gbt, train_frame, inner_labels, columns, seed, kinds.get(method, 'xgb'))
                    oof_raw[j] = model.predict(val_frame[columns].to_numpy(float))
                    oof_target[j] = closure.mapped_query_labels(train_frame, val_frame)
                iso = IsotonicRegression(out_of_bounds='clip').fit(oof_raw, oof_target)
                model = train(gbt, fit, labels, columns, seed, kinds.get(method, 'xgb'))
                fit_seconds = time.monotonic() - started
                started = time.monotonic()
                raw = model.predict(query[columns].to_numpy(float))
                calibrated = iso.predict(raw)
                predict_seconds = time.monotonic() - started
                for label, scores in [(method, np.clip(raw, 0, 1)),
                                      (method + '_isotonic', np.clip(calibrated, 0, 1))]:
                    closure.add_predictions(records, query, scores, 'McFaline_feedback', label,
                                            seed, budget=budget, n_feedback_clusters=n, n_feedback_rows=len(fit))
                costs.append({'budget': budget, 'method': method, 'seed': seed,
                    'fit_seconds_including_OOF_isotonic': fit_seconds,
                    'prediction_seconds': predict_seconds, 'inner_fits': 3, 'full_fits': 1,
                    'n_feedback_clusters': n, 'n_feedback_rows': len(fit), 'n_holdout_rows': len(query)})
                calibrations.append({'budget': budget, 'method': method, 'seed': seed,
                    'n_calibration_rows': len(fit), 'calibration_mode': 'group-OOF isotonic',
                    'n_output_knots': len(iso.y_thresholds_),
                    'fit_feedback_records_hash': ids_hash(fit.task_id)})
            ledger.append({'budget': budget, 'method': method, 'n_feedback_clusters': n,
                'n_unique_target_errors_used': len(fit), 'n_extra_validation_errors': 0,
                'error_uses': 'full-budget CDF, inner training-only CDF, risk fitting, OOF isotonic',
                'allowed_feedback_records_hash': ids_hash(fit.task_id),
                'holdout_task_hash': ids_hash(query.task_id), 'new_upstream_model_calls': 0,
                'source_errors_used': 'via fixed shared score' if 'shared_risk' in columns else 'NONE',
                'public_effect_truth_used': bool(set(columns) & set(PUBLIC)),
                'native_unperturbed_control_features': args.native_controls,
                'native_control_feature_error_labels': 0,
                'conditions': 'same precomputed prediction/history arrays; no new wet experiments'})
        predictions = pd.concat(records, ignore_index=True)
        closure.tx.atomic_csv(OUT / 'TASK_PREDICTIONS.csv.gz', predictions)
        print(json.dumps({'phase': 'official_pertema_feedback', 'budget': budget,
                          'feedback_clusters': n, 'completed_model_fits': len(costs)}), flush=True)
    predictions = pd.concat(records, ignore_index=True)
    strata, macro = summarize(predictions)
    for name, data in [('STRATA.csv', strata), ('MACRO.csv', macro), ('CDF_AUDIT.csv', pd.DataFrame(cdf_audit)),
                       ('INFORMATION_LEDGER.csv', pd.DataFrame(ledger)), ('COSTS.csv', pd.DataFrame(costs)),
                       ('CALIBRATION_AUDIT.csv', pd.DataFrame(calibrations))]:
        closure.tx.atomic_csv(OUT / name, data)
    combined = pd.concat([base, predictions], ignore_index=True)
    intervals = []
    comparisons = [('PertEMA_P_adapted', 'TargetOnly_HGB'),
                   ('PertEMA_Public_adapted', 'PublicTarget_HGB'),
                   ('PertEMA_Shared_adapted', 'SharedTarget_HGB'),
                   ('ResidualHGB', 'PertEMA_Public_adapted'),
                   ('PertEMA_Public_adapted', 'Shared'),
                   ('PertEMA_Public_adapted', 'PertEMA_P_adapted')]
    if args.native_controls:
        comparisons = [('PertEMA_Native_adapted', 'NativeControl_HGB'),
                       ('PertEMA_Native_Public_adapted', 'PertEMA_Native_adapted'),
                       ('PertEMA_Native_Shared_adapted', 'PertEMA_Native_Public_adapted'),
                       ('ResidualHGB', 'PertEMA_Native_Public_adapted'),
                       ('PertEMA_Native_Public_adapted', 'Shared'),
                       ('PertEMA_Native_adapted', 'TargetOnly_HGB')]
    for budget, part in combined[combined.seed.eq(SEEDS[0]) & combined.budget.gt(0)].groupby('budget'):
        meta = ['task_id', 'target', 'gene', 'fold', 'upstream', 'true_error_rmse']
        wide = part.pivot(index=meta, columns='method', values='risk').reset_index()
        for a, b in comparisons:
            intervals.append({'budget': budget, 'method_a': a, 'method_b': b,
                **bootstrap_u20(wide, wide[a].to_numpy(), wide[b].to_numpy(), args.bootstrap)})
        closure.tx.atomic_csv(OUT / 'PAIRED_BOOTSTRAP.csv', pd.DataFrame(intervals))
        print(json.dumps({'phase': 'official_pertema_bootstrap', 'budget': budget,
                          'completed_comparisons': len(intervals)}), flush=True)
    closure.tx.atomic_json(OUT / 'RUN_STATUS.json', {'status': 'COMPLETE', 'official_commit': COMMIT,
        'role': 'SEEN_POST_CONFIRMATION', 'n_fits': len(costs), 'n_error_records_by_budget': ledger,
        'n_holdout_tasks': len(query), 'n_holdout_clusters': query.gene.nunique(),
        'primary_output': 'raw official-factory adaptation; isotonic sensitivity separately reported'})
    print(macro[macro.seed.eq(SEEDS[0])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
