#!/usr/bin/env python3
"""One isolated Source DEV PublicBiology refit lifecycle; three fixed arms."""
from pathlib import Path
import json
import os
import resource
import signal
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.scripts import run_safeconf_continual_runtime_replay_agent as old
from tools.scripts import run_dual_memory_txpert_public_biology as tx
from tools.scripts.diagnose_safeconf_continual_components_agent import ImmutableErrorView, counter_metrics, METRICS
from tools.safeconf_continual import PublicMemoryStore
from tools.safeconf_continual.learners import PublicBiologyLearner
from tools.safeconf_continual.research import P, PUBLIC, ids_hash, metrics
from tools.safeconf_continual.versioned_runtime import ServingModelRegistry

BASE = old.COMMON.parent
FORMAL = BASE / 'continual_runtime_replay_20261002_v2'
FORMAL_REPORT = old.DOC_BASE / 'continual_runtime_replay/actual_v2'
OUT = BASE / 'continual_public_biology_refit_20261002_v1'
DOC = old.DOC_BASE / 'continual_runtime_replay/public_biology_refit_v1'
BOOT = BASE / 'continual_component_diagnostic_20261002_v1/JOINT_GENE_BOOTSTRAP_COUNTS.npz'
ARMS = ['A_initial_learned', 'B_updated_bank_errors_old_Bio', 'C_updated_bank_errors_refit_Bio']
PAIRS = tx.PAIR_FEATURES
ROLES = ['OLD_ANCHOR', 'NEW_TASK_GATE']
BIO_SEED = 20260929


def dump(path, value):
    joblib.dump(value, path)
    Path(path).chmod(0o444)
    return old.bind(path)


def numeric_pair_inputs(tasks, bank, controls):
    """Original nine pair formulas, no query treated vector or transfer labels."""
    memory, effects, _, manifest = bank
    eligibility = pd.read_parquet(Path(manifest['_root']) / 'eligibility.parquet')
    eligible = eligibility.groupby(['target_context', 'condition']).public_experiment_id.apply(list).to_dict()
    byid = memory.set_index('experiment_id')
    source_controls = bank[2]
    rows = []
    for q, task in enumerate(tasks.itertuples(index=False)):
        ids = eligible.get((str(task.target), str(task.condition)), [])
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError('Frozen initial-known-history cohort lost legal history')
        items = byid.loc[ids]
        if (items.context.eq(task.target).any() or not items.condition.eq(task.condition).all()
                or not items.perturbation_target.eq(task.gene).all()):
            raise RuntimeError('Current task/context history or different target entered retrieval')
        indices = items.effect_vector_row.to_numpy(int)
        vectors = np.asarray(effects[indices], float)
        ctrls = np.asarray(source_controls[indices], float)
        conflict = tx.rmse_rows(vectors, vectors.mean(axis=0, keepdims=True))
        for j, (memory_id, memory_row) in enumerate(zip(ids, indices)):
            x = items.iloc[j]
            rows.append({'task_row': q, 'task_id': task.task_id, 'target': task.target,
                'gene': task.gene, 'fold': int(task.fold), 'memory_id': memory_id, 'memory_row': int(memory_row),
                'log_source_cells': float(np.log1p(x.n_cells)), 'log_source_batches': float(np.log1p(x.n_batches)),
                'control_rmse': float(tx.rmse_rows(ctrls[j:j+1], controls[q:q+1])[0]),
                'control_cosine': float(tx.cosine_rows(ctrls[j:j+1], controls[q:q+1])[0]),
                'source_effect_magnitude': float(np.sqrt(np.mean(vectors[j]**2))),
                'source_effect_abs_mean': float(np.mean(np.abs(vectors[j]))),
                'source_conflict': float(conflict[j]), 'support_fraction': float(x.n_cells / max(items.n_cells.sum(), 1)),
                'quality_missing': 1.0})
    pair = pd.DataFrame(rows)
    if not np.isfinite(pair[PAIRS].to_numpy(float)).all():
        raise RuntimeError('Pair inputs must be finite')
    return pair


def fit_bio(tasks, pair, bank, folds, tag, source_rows, truth, audits, fits):
    selected = tasks[tasks.fold.isin(folds)].copy().reset_index(drop=True)
    if selected.empty or set(selected.fold) & {0, 1}:
        raise RuntimeError('Forbidden Public biological query-label role')
    original = tx.build_pairs(selected, np.asarray(truth[source_rows.loc[selected.task_id].to_numpy(int)], float),
        CONTROL_LOOKUP.loc[selected.task_id].to_numpy(), *bank[:3],
        pd.read_parquet(Path(bank[3]['_root']) / 'eligibility.parquet'))
    expected = pair[pair.task_id.isin(selected.task_id)].reset_index(drop=True)
    if (original[['task_id','memory_id']].to_numpy().tolist() != expected[['task_id','memory_id']].to_numpy().tolist()
            or not np.array_equal(original[PAIRS].to_numpy(), expected[PAIRS].to_numpy())):
        raise RuntimeError('Unlabelled pair formulas differ from original tx.build_pairs')
    started = time.monotonic()
    model = PublicBiologyLearner(kind='hgb', seed=BIO_SEED).fit(original[PAIRS].to_numpy(float), original.transfer_rmse.to_numpy(float))
    artifact = dump(OUT / f'{tag}.joblib', model)
    audits.append({'tag': tag, 'query_label_folds': '|'.join(map(str, folds)), 'training_pairs': len(original),
        'training_tasks': selected.task_id.nunique(), 'training_gene_clusters': selected.gene.nunique(),
        'training_gene_hash': ids_hash(selected.gene.unique()), 'training_pair_ids_hash': ids_hash(original.task_id+'|'+original.memory_id),
        'target': 'biological transfer_rmse; no upstream prediction error', 'bank_version': bank[3]['version'],
        'anchor_or_newgate_query_labels_used': 0, 'model_sha256': artifact['sha256']})
    fits.append({'component':'PublicBiology', 'tag':tag, 'fits':1, 'fit_save_seconds':time.monotonic()-started,
        'seed': BIO_SEED, 'training_rows':len(original), 'training_gene_clusters':selected.gene.nunique()})
    return model, artifact


def features_for(bundle, base, tasks, bank, controls, predictions, source_rows, save_prefix=None):
    pair = numeric_pair_inputs(tasks, bank, controls)
    score = np.full(len(pair), np.nan)
    for fold, group in pair.groupby('fold', sort=True):
        model = bundle['bio_oof'].get(int(fold), bundle['bio_final'])
        score[group.index] = model.predict_transfer_error(group[PAIRS].to_numpy(float))
    if not np.isfinite(score).all(): raise RuntimeError('Public scores incomplete')
    pair['transfer_score'] = score
    priors = np.empty((len(tasks), 2840), float)
    summaries, weights = [], []
    for q, group in pair.groupby('task_row', sort=True):
        prior, w, uncertainty, effective = tx.prior_from_scores(group, bank[1], 'transfer_score', 'learned_regularized')
        priors[int(q)] = prior
        summaries.append({'task_id': tasks.iloc[int(q)].task_id, 'prior_uncertainty':uncertainty,
            'effective_sources':effective, 'history_conflict':float(group.source_conflict.mean()),
            'log_history_support':float(np.log1p(np.expm1(group.log_source_cells).sum()))})
        for row, weight in zip(group.itertuples(), w):
            weights.append({'task_id':row.task_id,'memory_id':row.memory_id,'weight':float(weight)})
    summaries = pd.DataFrame(summaries).set_index('task_id')
    task_index = pd.Series(np.arange(len(tasks)), index=tasks.task_id)
    full_priors = np.zeros((len(source_rows), 2840), float)
    full_priors[source_rows.loc[tasks.task_id].to_numpy(int)] = priors
    frames = []
    for upstream in old.MODELS:
        frame = base[base.upstream.eq(upstream)].copy().reset_index(drop=True)
        q = task_index.loc[frame.task_id].to_numpy(int)
        rows = source_rows.loc[frame.task_id].to_numpy(int)
        p = predictions[upstream][rows]
        frame['prior_magnitude'] = np.sqrt(np.mean(priors[q]**2, axis=1))
        frame['prediction_prior_rmse'] = tx.rmse_rows(p, priors[q])
        frame['prediction_prior_cosine'] = tx.cosine_rows(predictions[upstream], full_priors)[rows]
        for field in ['prior_uncertainty','effective_sources','history_conflict','log_history_support']:
            frame[field] = frame.task_id.map(summaries[field])
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result[P+PUBLIC].to_numpy(float)).all(): raise RuntimeError('Learned feature frame incomplete')
    if save_prefix:
        result.to_parquet(OUT / f'{save_prefix}_FEATURES.parquet', index=False)
        np.save(OUT / f'{save_prefix}_PRIORS.npy', priors, allow_pickle=False)
        pd.DataFrame(weights).to_parquet(OUT / f'{save_prefix}_WEIGHTS.parquet', index=False)
    return result, priors, pd.DataFrame(weights)


def main():
    global CONTROL_LOOKUP
    began = time.monotonic(); signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1800second bound'))); signal.alarm(1800)
    if OUT.exists() or DOC.exists(): raise RuntimeError('Fresh isolated replay roots required')
    OUT.mkdir(); DOC.mkdir()
    state = {'pid':os.getpid(),'status':'REGISTERING','started_utc':old.utc(),'Orion_access':False,'McFaline_access':False}
    def progress(phase, **extra):
        state.update(status=phase, **extra); (OUT/'STATUS.json').write_text(json.dumps(state,indent=2)); print(json.dumps(state),flush=True)
    progress('REGISTERING')
    fit_costs=[]
    try:
        registration = json.loads((FORMAL_REPORT/'REGISTRATION.json').read_text())
        original_pins = registration['input_bindings']+registration['code_bindings']
        for item in original_pins:
            if old.sha(item['path']) != item['sha256']: raise RuntimeError('Original replay dependency changed')
        paths=[Path(__file__),Path(old.__file__),Path(tx.__file__),old.ROOT/'tools/safeconf_continual/learners.py',
            old.ROOT/'tools/safeconf_continual/research.py',old.ROOT/'tools/safeconf_continual/versioned_runtime.py',
            old.ROOT/'tools/scripts/diagnose_safeconf_continual_components_agent.py',
            old.ROOT/'tools/scripts/run_safeconf_source_scaling_uncertainty_agent.py',
            FORMAL_REPORT/'REGISTRATION.json',FORMAL_REPORT/'REAL_INGESTION_TRIGGER_RECEIPT.json',
            FORMAL_REPORT/'FROZEN_TASK_ROLE_SCOPE.csv',FORMAL/'FEATURES_PUBLIC_v1.parquet',BOOT,
            old.COMMON/'SOURCE_TRUE_EFFECTS.npy',old.COMMON/'risk_cache/TX_TASK_SPLIT.csv']
        for version in [1,2]: paths += [p for p in (FORMAL/f'public_memory/versions/v{version:04d}').rglob('*') if p.is_file()]
        for upstream in old.MODELS: paths.append(old.COMMON/f'SOURCE_{upstream}_PREDICTED_EFFECTS.npy')
        paths.append(tx.VECTOR_ROOT/'e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy')
        pins=[old.bind(p) for p in paths]
        old.write_json(DOC/'REGISTRATION.json',{'schema':'safeconf_Source_DEV_PublicBiology_actual_refit_v1',
            'pid':os.getpid(),'arms':ARMS,'only_release_candidate':ARMS[2], 'diagnostic_only_arm':ARMS[1],
            'gene_axis':2840,'fixed_tasks':1699,'gene_clusters':563,'predictor_task_records':3398,
            'Public_parameters':{'kind':'hgb','seed':BIO_SEED,'iterations':200,'learning_rate':.05,'depth':3,'leaf':20,'L2':10},
            'risk_seed':old.SEED,'Public_features':PAIRS,'risk_features':P+PUBLIC,'learned_support_blend':.5,
            'initial_Bio_OOF':'held2fit3;held3fit2;finalfit2,3',
            'updated_Bio_OOF':'held2fit3,4;held3fit2,4;held4fit2,3;finalfit2,3,4',
            'B_OOF':'reuseoldheld2/3learners onnewbank;fold4oldfinalfit2,3',
            'fixed_gate':'original finite_release_gate CvsA; Bnoteligible; no winner selection',
            'planned_fits':{'PublicBiology':7,'risk':3,'total':10},'ErrorMemory':'existingimmutableexactversion realised_error only; historicalshared/residual notused',
            'initial_Error_folds':[2,3],'updated_Error_folds':[2,3,4],
            'gate_labels_only_after_all_models_priors_scores_sealed':True,
            'retrospective_study_provenance':'initialbiological/errortraining alreadycoversbothSource studies; notchronologicalstudytransfer',
            'existing_source_predictors':2,'existing_source_families':1,'new_upstream_calls':0,
            'bootstrap':'reusefixed5000joint227gene countarray; noRNG; finiteonlyCIwithNApropagation',
            'resources':{'CPU_threads':4,'wall_cap_seconds':1800,'RSS_cap_bytes':2147483648,'GPU_hours':0},
            'input_code_bindings':pins,'original_replay_bindings':original_pins},immutable=True)
        scope=pd.read_csv(FORMAL_REPORT/'FROZEN_TASK_ROLE_SCOPE.csv'); all_tasks=pd.read_csv(old.COMMON/'risk_cache/TX_TASK_SPLIT.csv')
        tasks=scope[scope.initial_known_history].copy().reset_index(drop=True)
        if len(tasks)!=1699 or tasks.gene.nunique()!=563 or tasks.groupby('gene').fold.nunique().max()!=1: raise RuntimeError('Originalcohort changed')
        source_rows=pd.Series(np.arange(len(all_tasks)),index=all_tasks.task_id)
        identity=['upstream','model_version','dataset_id','output_contract_id','task_id','condition','gene','target','fold','replay_role']
        base=pd.read_parquet(FORMAL/'FEATURES_PUBLIC_v1.parquet',columns=identity+P)
        if len(base)!=3398 or base.task_id.nunique()!=1699 or base.groupby('gene').fold.nunique().max()!=1: raise RuntimeError('Baseidentity changed')
        for upstream in old.MODELS:
            part=base[base.upstream.eq(upstream)]; aligned=tasks.set_index('task_id').loc[part.task_id]
            if (set(part.task_id)!=set(tasks.task_id) or part.model_version.nunique()!=1
                or not np.array_equal(part[['gene','target','condition','fold']].to_numpy(),aligned[['gene','target','condition','fold']].to_numpy())): raise RuntimeError('Sourceidentity mismatch')
        genes=json.loads((old.COMMON/'GENE_IDS.json').read_text()); native=json.loads((tx.STORE_ROOT/'gene_ids.json').read_text())['gene_ids']
        columns=np.array([native.index(g) for g in genes]); native_control=np.load(tx.VECTOR_ROOT/'e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy',mmap_mode='r')
        controls=np.asarray(native_control[all_tasks.source_mean_delta_row.to_numpy(int)][:,columns])
        CONTROL_LOOKUP=pd.DataFrame(controls,index=all_tasks.task_id)
        current_controls=CONTROL_LOOKUP.loc[tasks.task_id].to_numpy()
        predictions={u:np.load(old.COMMON/f'SOURCE_{u}_PREDICTED_EFFECTS.npy',mmap_mode='r') for u in old.MODELS}
        truth=np.load(old.COMMON/'SOURCE_TRUE_EFFECTS.npy',mmap_mode='r')
        banks={}
        for version in [1,2]:
            root=FORMAL/f'public_memory/versions/v{version:04d}'
            memory,effect,ctrl,manifest=PublicMemoryStore(root).load(); manifest=manifest|{'_root':str(root)}
            if len(memory)!={1:1047,2:2008}[version] or effect.shape!=(len(memory),2840): raise RuntimeError('Bankversion changed')
            banks[version]=(memory,effect,ctrl,manifest)
        pair1=numeric_pair_inputs(tasks,banks[1],current_controls); pair2=numeric_pair_inputs(tasks,banks[2],current_controls)
        audits=[]; old_oof={}; new_oof={}; bio_bindings=[]
        for held in [2,3]:
            model,artifact=fit_bio(tasks,pair1,banks[1],[x for x in [2,3] if x!=held],f'BIO_v1_OOF_hold{held}',source_rows,truth,audits,fit_costs)
            old_oof[held]=model; bio_bindings.append(artifact)
        bio1,binding=fit_bio(tasks,pair1,banks[1],[2,3],'BIO_v1_FINAL',source_rows,truth,audits,fit_costs);bio_bindings.append(binding)
        for held in [2,3,4]:
            model,artifact=fit_bio(tasks,pair2,banks[2],[x for x in [2,3,4] if x!=held],f'BIO_v2_OOF_hold{held}',source_rows,truth,audits,fit_costs)
            new_oof[held]=model; bio_bindings.append(artifact)
        bio2,binding=fit_bio(tasks,pair2,banks[2],[2,3,4],'BIO_v2_FINAL',source_rows,truth,audits,fit_costs);bio_bindings.append(binding)
        old.write_csv(DOC/'PUBLIC_BIOLOGY_TRAINING_SCOPE.csv',pd.DataFrame(audits),immutable=True)
        progress('SEVEN_REAL_PUBLIC_BIOLOGY_FITS_PERSISTED')
        ingestion=json.loads((FORMAL_REPORT/'REAL_INGESTION_TRIGGER_RECEIPT.json').read_text())
        views={1:ImmutableErrorView(ingestion['Error_initial_manifests']),2:ImmutableErrorView(ingestion['Error_updated_manifests'])}
        scores={}; feature_frames={}; bundles={}; model_bindings=[]; cdf_audits=[]
        for arm,bank_version,error_version,bio_final,bio_oof,allowed in [
            (ARMS[0],1,1,bio1,old_oof,{'TRAIN_INITIAL'}),
            (ARMS[1],2,2,bio1,old_oof,{'TRAIN_INITIAL','TRAIN_ADDED'}),
            (ARMS[2],2,2,bio2,new_oof,{'TRAIN_INITIAL','TRAIN_ADDED'})]:
            bundle={'schema':'SourceDEV_joint_PublicBio_Risk_bank_v1','arm':arm,'retrieval_bank_version':bank_version,
                'biological_training_bank_version':1 if arm!=ARMS[2] else 2,
                'bank_manifest_binding':old.bind(Path(banks[bank_version][3]['_root'])/'manifest.json'),
                'bio_final':bio_final,'bio_oof':bio_oof,'pair_columns':PAIRS,'risk_columns':P+PUBLIC}
            frame,prior,weights=features_for(bundle,base,tasks,banks[bank_version],current_controls,predictions,source_rows,arm)
            started=time.monotonic();risk,fit,labels,audit=old.fit_from_error_memory(frame,views[error_version],allowed)
            fit_costs.append({'component':'Risk','tag':arm,'fits':1,'fit_save_seconds':time.monotonic()-started,'seed':old.SEED,
                'training_rows':len(fit),'training_gene_clusters':fit.gene.nunique()})
            bundle.update(risk=risk,cdfs=old.cdf_objects(fit),training_fold_scope=sorted(map(int,fit.fold.unique())),
                training_records_hash=ids_hash(old.frame_records(fit)),training_genes_hash=ids_hash(fit.gene.unique()))
            model_bindings.append(dump(OUT/f'{arm}_BUNDLE.joblib',bundle))
            score=risk.predict(frame)
            if score.shape!=(3398,) or not np.isfinite(score).all(): raise RuntimeError('Allfixedscoresmustfinite')
            loaded=joblib.load(OUT/f'{arm}_BUNDLE.joblib')
            reload_frame,reload_prior,reload_weights=features_for(loaded,base,tasks,banks[bank_version],current_controls,predictions,source_rows)
            if (frame[P+PUBLIC].to_numpy().tobytes()!=reload_frame[P+PUBLIC].to_numpy().tobytes()
                or prior.tobytes()!=reload_prior.tobytes() or weights.weight.to_numpy().tobytes()!=reload_weights.weight.to_numpy().tobytes()
                or score.tobytes()!=loaded['risk'].predict(reload_frame).tobytes()): raise RuntimeError('JointBioRiskbankreloadnotbitexact')
            scores[arm]=score;feature_frames[arm]=frame;bundles[arm]=bundle
            cdf_audits.extend([{'arm':arm,**x} for x in audit])
        if len(fit_costs)!=10: raise RuntimeError('Exactly7Bio+3riskfitsrequired')
        frozen=base[identity].copy()
        for arm in ARMS:frozen[arm]=scores[arm]
        frozen.to_parquet(OUT/'ALL_FROZEN_RISK_SCORES.parquet',index=False)
        for p in OUT.iterdir():
            if p.is_file() and p.name!='STATUS.json':p.chmod(0o444)
        old.write_json(DOC/'PRE_GATE_MODEL_PRIOR_PREDICTION_FREEZE.json',{'models':model_bindings,'biological_models':bio_bindings,
            'runtime_artifacts':[old.bind(p) for p in OUT.iterdir() if p.is_file() and p.name!='STATUS.json'],
            'all10fits_and3fullscorearms_sealed_before_query_gate_outcomes':True,
            'joint_bio_risk_bank_reload_bitwise_equal_allarms':True,'all_fixed_queries':1699,'all_predictor_query_records':3398},immutable=True)
        serving=ServingModelRegistry(OUT/'serving_models');component='SourceDEV_Learned_PublicBiology_and_Risk'
        split_hash=old.sha(FORMAL_REPORT/'FROZEN_TASK_ROLE_SCOPE.csv');schema_hash=ids_hash(P+PUBLIC+PAIRS)
        serving.register(component,'learned-initial',OUT/f'{ARMS[0]}_BUNDLE.joblib',split_hash,schema_hash,status='RELEASED')
        serving.publish(component,'learned-initial','initial isolated Source DEV joint bundle; no formal deployment')
        progress('ALL_ARMS_PRIORS_MODELS_SCORES_SEALED_GATE_TRUTH_NOT_YET_DECODED')
        gate_ids=tasks[tasks.replay_role.isin(ROLES)].task_id.tolist()
        raw=pd.read_parquet(old.SOURCE,columns=['upstream','task_id','true_error_rmse'],filters=[('task_id','in',gate_ids)])
        if raw.duplicated(['upstream','task_id']).any():raise RuntimeError('DEVoutcomeidentityduplicate')
        lookup=raw.set_index(old.frame_records(raw)).true_error_rmse
        def measure(first,second):
            rows=[]
            for role in ROLES:
                for upstream in old.MODELS:
                    for context in old.CONTEXTS:
                        mask=base.replay_role.eq(role)&base.upstream.eq(upstream)&base.target.eq(context)
                        frame=base[mask].copy();frame['true_error_rmse']=old.frame_records(frame).map(lookup)
                        if len(frame)<20 or not np.isfinite(frame.true_error_rmse).all():raise RuntimeError('DEVstratum invalid')
                        a,b=metrics(frame,scores[first][mask]),metrics(frame,scores[second][mask])
                        row={'role':role,'upstream':upstream,'context':context,'n_tasks':len(frame),'n_gene_clusters':frame.gene.nunique()}
                        row.update({f'{k}_v1':v for k,v in a.items() if k not in ['n_tasks','n_planned']})
                        row.update({f'{k}_v2':v for k,v in b.items() if k not in ['n_tasks','n_planned']});rows.append(row)
            return pd.DataFrame(rows)
        ac=measure(ARMS[0],ARMS[2]);gate=old.finite_release_gate(ac[ac.role.eq(ROLES[0])],ac[ac.role.eq(ROLES[1])])
        old.write_csv(DOC/'C_VS_A_ACTUAL_DEV_GATE_METRICS.csv',ac,immutable=True)
        decision='RELEASED' if gate['passes'] else 'REJECTED'
        serving.register(component,'learned-updated-refit',OUT/f'{ARMS[2]}_BUNDLE.joblib',split_hash,schema_hash,parent_version='learned-initial',status=decision)
        if gate['passes']:serving.publish(component,'learned-updated-refit','only fixed C candidate passed original finite DEV gate')
        artifact=serving.current(component);loaded=joblib.load(artifact)
        selected=ARMS[2] if gate['passes'] else ARMS[0];bank=banks[loaded['retrieval_bank_version']]
        served,_,_=features_for(loaded,base,tasks,bank,current_controls,predictions,source_rows)
        if loaded['risk'].predict(served).tobytes()!=scores[selected].tobytes():raise RuntimeError('Actualservingjointreloadfailed')
        old.write_json(DOC/'ACTUAL_RELEASE_OR_RETAIN_DECISION.json',gate|{'candidate_status':decision,'serving_arm':selected,
            'serving_bundle':old.bind(artifact),'serving_retrieval_bank_version':loaded['retrieval_bank_version'],
            'serving_Bio_training_bank_version':loaded['biological_training_bank_version'],
            'joint_Bio_risk_bank_recomputed_predictions_bitwise_equal':True,
            'B_diagnostic_never_eligible_for_release':True,'old_immutable_public_versions_and_formal_serving_untouched':True,
            'new_data_remain_available_even_if_rejected':True,'no_newfit_after_gate':True},immutable=True)
        progress('ORIGINAL_DEV_GATE_APPLIED',candidate_status=decision)
        evalmask=base.replay_role.isin(ROLES);evaluation=base[evalmask].copy().reset_index(drop=True)
        evaluation['true_error_rmse']=old.frame_records(evaluation).map(lookup)
        score_matrix=np.column_stack([scores[a][evalmask] for a in ARMS])
        saved=np.load(BOOT,allow_pickle=False);eg=sorted(evaluation.gene.unique())
        if saved['genes'].tolist()!=eg or saved['counts'].shape!=(5000,227):raise RuntimeError('Originaljoint227gene bootstrap mismatch')
        counts=saved['counts'];gindex=evaluation.gene.map({g:i for i,g in enumerate(eg)}).to_numpy(int)
        all_draws=np.empty((2,5000,3,len(METRICS)));all_points=np.empty((2,3,len(METRICS)));strata=[];macros=[];paired=[]
        for ri,role in enumerate(ROLES):
            pts=[];ds=[]
            for upstream in old.MODELS:
                for context in old.CONTEXTS:
                    ix=np.flatnonzero((evaluation.replay_role.eq(role)&evaluation.upstream.eq(upstream)&evaluation.target.eq(context)).to_numpy())
                    part=evaluation.iloc[ix];w=counts[:,gindex[ix]].astype(np.int32);point=[];draw=[]
                    for ai,arm in enumerate(ARMS):
                        measured=metrics(part,score_matrix[ix,ai]);v=np.array([measured[k] for k in METRICS])
                        test=counter_metrics(part.true_error_rmse.to_numpy(),part.task_id.to_numpy(str),score_matrix[ix,ai],np.ones((1,len(ix)),int))[0]
                        if not np.allclose(v,test,rtol=0,atol=1e-12,equal_nan=True):raise RuntimeError('Savedcountmetric pointvalidator mismatch')
                        point.append(v);draw.append(counter_metrics(part.true_error_rmse.to_numpy(),part.task_id.to_numpy(str),score_matrix[ix,ai],w))
                        strata.append({'role':role,'upstream':upstream,'context':context,'arm':arm,'n_tasks':len(ix),**measured})
                    pts.append(point);ds.append(np.stack(draw,axis=1))
            all_points[ri]=np.mean(pts,axis=0);all_draws[ri]=np.mean(ds,axis=0)
            for ai,arm in enumerate(ARMS):macros.append({'role':role,'arm':arm,'n_strata':8,'upstream_predictors':2,'upstream_families':1,**dict(zip(METRICS,all_points[ri,ai]))})
            for first,second in [(2,0),(1,0),(2,1)]:
                delta=all_draws[ri,:,first]-all_draws[ri,:,second]
                for ki,key in enumerate(METRICS):
                    finite=delta[:,ki][np.isfinite(delta[:,ki])];lo,hi=np.quantile(finite,[.025,.975]) if len(finite)>=2 else [np.nan,np.nan]
                    paired.append({'role':role,'arm_a':ARMS[first],'arm_b':ARMS[second],'metric':key,
                        'point_difference':all_points[ri,first,ki]-all_points[ri,second,ki],
                        'ci95_lower':lo,'ci95_upper':hi,'valid_draws':len(finite),'saved_draws':5000,'new_RNG_draws':0})
            progress('SAVED_GENE_BOOTSTRAP_METRICS_DONE',role=role)
        np.savez_compressed(OUT/'THREE_ARM_SAVED_GENE_METRIC_DRAWS.npz',macro=all_draws,points=all_points,arms=np.asarray(ARMS),roles=np.asarray(ROLES),metrics=np.asarray(METRICS))
        for filename,frame in [('ALL48_DEV_STRATA.csv',strata),('ALL6_DEV_MACRO.csv',macros),('ALL42_FIXED_PAIRED_INTERVALS.csv',paired),
                               ('FIT_COSTS.csv',fit_costs),('CDF_ALLOWED_BUDGET_AUDIT.csv',cdf_audits)]:old.write_csv(DOC/filename,pd.DataFrame(frame),immutable=True)
        for item in pins+original_pins:
            if old.sha(item['path'])!=item['sha256']:raise RuntimeError('Originalbank/formalassets/codechanged')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2147483648:raise RuntimeError('2GiB RSSboundexceeded')
        old.write_json(DOC/'RESULT_MANIFEST.json',{'status':'COMPLETE_ACTUAL_SOURCE_DEV_PUBLIC_BIOLOGY_REFIT',
            'candidate_status':decision,'serving_arm':selected,'actual_PublicBiology_fits':7,'actual_risk_fits':3,'actual_total_fits':10,
            'used_existing_immutable_ErrorMemory_realised_error_only':True,'new_Error_OOF_fits':0,
            'tasks':1699,'gene_clusters':563,'predictor_task_records':3398,'evaluation_gene_clusters':227,
            'all_three_arms_retained':True,'strata_rows':48,'macro_rows':6,'paired_rows':42,
            'new_bootstrap_draws':0,'saved_gene_draws':5000,'new_upstream_calls':0,'GPU_hours':0,
            'Orion_access':False,'McFaline_access':False,'formal_assets_before_after_unchanged':True,
            'elapsed_seconds':time.monotonic()-began,'CPU_seconds':time.process_time(),
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'output_bindings':[old.bind(p) for p in DOC.iterdir() if p.is_file()]},immutable=True)
        progress('COMPLETE_ACTUAL_SOURCE_DEV_PUBLIC_BIOLOGY_REFIT',candidate_status=decision,actual_fits=10)
    except Exception as exc:
        old.write_json(DOC/'ABORT.json',{'status':'ABORT','error':repr(exc),'elapsed_seconds':time.monotonic()-began,'completed_fit_rows':fit_costs},immutable=True)
        progress('ABORT',error=repr(exc));raise


if __name__=='__main__':main()
