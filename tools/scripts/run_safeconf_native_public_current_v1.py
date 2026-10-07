#!/usr/bin/env python3
"""Fixed same-budget Native61 versus Native61+Public, current McFaline truth.

Official PertEMA factory is used; control-feature, RMSE/CDF, cluster-weight
and context adaptations are enumerated. This is a SEEN comparison, never
automatic selection on holdout. Raw score is primary, OOF isotonic secondary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from tools.safeconf_continual.research import (
    PUBLIC, bootstrap_u20, cluster_weights, ids_hash, metrics, rank_labels,
)
from tools.scripts.run_safeconf_pertema_current_truth_v1 import (
    BUDGETS, SEEDS, NATIVE_FEATURES, OFFICIAL_ROOT, OFFICIAL_COMMIT,
    POOL, HOLD, NATIVE, fixed_gene_order, load_frames, mapped_labels, sha,
    write_json, write_csv,
)
from tools.scripts.run_safeconf_official_pertema_feedback import factory, NativePredictor


def fit_model(gbt, frame, labels, columns, seed):
    valid = np.isfinite(labels)
    train = frame.loc[valid].reset_index(drop=True)
    if len(train) < 2:
        raise ValueError('no finite training labels')
    model = NativePredictor(gbt().set_params(n_jobs=4, random_state=seed), columns, 'xgb')
    return model.fit(train[columns].to_numpy(float), labels[valid], cluster_weights(train))


def validate_inputs(pool, hold):
    if set(pool.gene) & set(hold.gene):
        raise ValueError('feedback/evaluation genes overlap')
    for frame in (pool, hold):
        if frame.task_id.duplicated().any():
            raise ValueError('duplicate task IDs')
        if not np.isfinite(frame[PUBLIC].to_numpy(float)).all():
            raise ValueError('public feature nonfinite')
        if not np.isfinite(frame.true_error_rmse.to_numpy(float)).all():
            raise ValueError('current truth nonfinite')
    # The old native cache used an earlier prediction axis. Controls remain
    # fixed; the sole prediction-derived native column must be current.
    for frame in (pool, hold):
        frame['native_prediction_abs_mean'] = frame.prediction_abs_mean.to_numpy(float)


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'RUN_STATUS.json'
    if status.exists() and json.loads(status.read_text()).get('status') == 'COMPLETE':
        raise FileExistsError('completed outputs are immutable; choose another output')
    started = time.monotonic()
    cpu_started = time.process_time()
    pool, hold, audit = load_frames()
    old_pred_difference = float(np.max(np.abs(pool.native_prediction_abs_mean - pool.prediction_abs_mean)))
    validate_inputs(pool, hold)
    gbt, factory_hash = factory()
    feature_sets = {
        'Native61': NATIVE_FEATURES,
        'Native61_Public': NATIVE_FEATURES + PUBLIC,
    }
    config = {
        'role': 'SEEN_CURRENT_TRUTH_FIXED_COMPARISON',
        'official_commit': OFFICIAL_COMMIT, 'official_source_sha256': factory_hash,
        'official_factory': 'src/pertema/run_estimator.py::gbt',
        'parameters': gbt().set_params(n_jobs=4, random_state=SEEDS[0]).get_params(),
        'budgets': list(BUDGETS), 'seeds': list(SEEDS),
        'feature_sets': feature_sets, 'pool_sha256': sha(POOL), 'holdout_sha256': sha(HOLD),
        'native_sha256': sha(NATIVE), 'own_script_sha256': sha(Path(__file__)),
        'pool_ids_hash': ids_hash(pool.task_id), 'holdout_ids_hash': ids_hash(hold.task_id),
        'labels': 'budget-only midrank error CDF per compatible context/model group',
        'weights': 'same gene-cluster-equal weights for both methods',
        'similarity': 'nearest finite control embedding in each fitting partition only',
        'prediction_feature': 'current prediction_abs_mean replaces older native_prediction_abs_mean',
        'previous_prediction_feature_max_abs_difference': old_pred_difference,
        'calibration': 'gene-OOF scores plus labels mapped through each inner training CDF',
        'primary': 'unclipped raw official-factory score; risk direction fixed higher=more error',
        'secondary': 'OOF isotonic, not selected using holdout',
        'main_default': 'PublicRule unchanged; no holdout-based automatic adoption',
        'adaptations': ['current McFaline control-only features and states',
            'technical plate variance proxy instead of donor variance',
            'RMSE error CDF target instead of official Pearson target',
            'cluster-equal weights; fixed SafeConf seeds; 4 CPU threads'],
        'complete_official_conformal_pipeline': False,
        'permanent_test_truth_opened': False, 'new_gpu_hours': 0.0, 'new_download_bytes': 0,
        **audit,
    }
    write_json(out / 'EXECUTION_CONFIG.json', config)
    write_json(status, {'status': 'RUNNING', 'pid': os.getpid(), 'output': str(out)})
    records, ledger, cdf_audits, calibration_audits = [], [], [], []
    order = fixed_gene_order(pool)
    for budget in BUDGETS:
        n_genes = int(np.ceil(budget * len(order)))
        fit = pool[pool.gene.isin(order[:n_genes])].reset_index(drop=True)
        labels, audits = rank_labels(fit, f'native-public-current/{budget}', budget)
        cdf_audits.extend(audits)
        splits = list(GroupKFold(3).split(fit, groups=fit.gene))
        for feature_set, columns in feature_sets.items():
            for seed in SEEDS:
                t0 = time.monotonic()
                raw_oof = np.full(len(fit), np.nan)
                target_oof = np.full(len(fit), np.nan)
                for inner, (i, j) in enumerate(splits):
                    tr, va = fit.iloc[i].reset_index(drop=True), fit.iloc[j].reset_index(drop=True)
                    if set(tr.gene) & set(va.gene):
                        raise ValueError('inner fold gene overlap')
                    inner_labels, audits = rank_labels(tr, f'native-public-current/{budget}/{feature_set}/{seed}/inner{inner}', budget)
                    cdf_audits.extend(audits)
                    model = fit_model(gbt, tr, inner_labels, columns, seed)
                    raw_oof[j] = model.predict(va[columns].to_numpy(float))
                    target_oof[j] = mapped_labels(tr, va)
                    calibration_audits.append({'budget': budget, 'feature_set': feature_set, 'seed': seed,
                        'inner_fold': inner, 'fit_records_hash': ids_hash(tr.task_id),
                        'held_records_hash': ids_hash(va.task_id), 'fit_genes': tr.gene.nunique(),
                        'held_genes': va.gene.nunique(), 'similarity_prototypes': model.neighbors.n_samples_fit_})
                valid = np.isfinite(raw_oof) & np.isfinite(target_oof)
                iso = None
                if valid.sum() >= 5 and np.ptp(raw_oof[valid]) > 0:
                    iso = IsotonicRegression(out_of_bounds='clip').fit(raw_oof[valid], target_oof[valid])
                model = fit_model(gbt, fit, labels, columns, seed)
                raw = model.predict(hold[columns].to_numpy(float))
                if not np.isfinite(raw).all():
                    raise ValueError('nonfinite scores')
                for output_type, risk in [('raw', raw), ('isotonic', iso.predict(raw) if iso else np.full(len(raw),np.nan))]:
                    part = hold[['task_id','target','gene','true_error_rmse']].copy()
                    part['budget'], part['feature_set'], part['seed'] = budget, feature_set, seed
                    part['output_type'], part['risk'] = output_type, risk
                    records.append(part)
                # Save the native Booster; this avoids an sklearn/XGBoost
                # wrapper metadata mismatch without changing fit or scores.
                model.model.get_booster().save_model(out / f'model_{feature_set}_b{budget:.2f}_s{seed}.json')
                np.savez_compressed(out / f'similarity_{feature_set}_b{budget:.2f}_s{seed}.npz',
                    prototypes=model.neighbors._fit_X)
                ledger.append({'budget': budget, 'feature_set': feature_set, 'seed': seed,
                    'n_fit_rows': len(fit), 'n_fit_genes': n_genes, 'inner_fits': 3, 'full_fits': 1,
                    'seconds': time.monotonic()-t0, 'feedback_records_hash': ids_hash(fit.task_id),
                    'error_uses': 'CDF/inner risk/full risk/OOF isotonic',
                    'extra_error_records': 0, 'source_errors': 0,
                    'public_features': feature_set == 'Native61_Public', 'weights': 'gene-cluster-equal',
                    'similarity_prototypes': model.neighbors.n_samples_fit_})
                pd.concat(records).to_parquet(out / 'TASK_PREDICTIONS.parquet', index=False)
                write_json(status, {'status':'RUNNING','pid':os.getpid(),'completed_configs':len(ledger),
                    'total_configs':30,'wall_seconds':time.monotonic()-started})
                print(json.dumps({'budget':budget,'feature':feature_set,'seed':seed,'completed':len(ledger)}),flush=True)
    pred = pd.concat(records,ignore_index=True)
    summary = []
    for (budget, feature, seed, output_type), part in pred.groupby(['budget','feature_set','seed','output_type'],sort=True):
        for target, context in part.groupby('target',sort=True):
            summary.append({'budget':budget,'feature_set':feature,'seed':seed,'output_type':output_type,
                'target':target,**metrics(context,context.risk.to_numpy(float))})
    strata = pd.DataFrame(summary)
    columns = ['utility20','aurc','spearman','high_risk_miss_rate']
    macro = strata.groupby(['budget','feature_set','seed','output_type'],as_index=False)[columns].mean()
    write_csv(out/'STRATA.csv',strata);write_csv(out/'MACRO.csv',macro)
    write_csv(out/'SEED_SUMMARY.csv',macro.groupby(['budget','feature_set','output_type'],as_index=False)[columns].agg(['mean','min','max']).reset_index())
    write_csv(out/'FIT_AND_INFORMATION_LEDGER.csv',pd.DataFrame(ledger))
    write_csv(out/'ERROR_LABEL_CDF_AUDIT.csv',pd.DataFrame(cdf_audits))
    write_csv(out/'OOF_CALIBRATION_AUDIT.csv',pd.DataFrame(calibration_audits))
    pairs=[]
    for budget in BUDGETS:
        base=pred[(pred.budget==budget)&(pred.seed==SEEDS[0])&(pred.output_type=='raw')]
        wide=hold[['task_id','target','gene','true_error_rmse','simple_history_risk']].copy()
        for feature in feature_sets:
            scores=base[base.feature_set==feature].set_index('task_id').risk
            wide[feature]=wide.task_id.map(scores)
        for a,b in [('Native61_Public','Native61'),('Native61_Public','simple_history_risk'),('Native61','simple_history_risk')]:
            pairs.append({'budget':budget,'seed':SEEDS[0],'method_a':a,'method_b':b,
                **bootstrap_u20(wide,wide[a].to_numpy(float),wide[b].to_numpy(float),5000,SEEDS[0])})
        write_csv(out/'PAIRED_BOOTSTRAP.csv',pd.DataFrame(pairs))
        print(json.dumps({'statistics_budget':budget,'comparisons':len(pairs)}),flush=True)
    write_json(status,{'status':'COMPLETE','configs':len(ledger),'xgb_fits':len(ledger)*4,
        'budgets':list(BUDGETS),'seeds':list(SEEDS),'bootstrap_replicates':5000,
        'pool_rows':len(pool),'holdout_rows':len(hold),'holdout_genes':int(hold.gene.nunique()),
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu_started,
        'new_gpu_hours':0,'new_download_bytes':0,'permanent_test_truth_opened':False,
        'complete_official_conformal_pipeline':False,'default_unchanged':'PublicRule'})
    print(macro[macro.output_type.eq('raw')].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
