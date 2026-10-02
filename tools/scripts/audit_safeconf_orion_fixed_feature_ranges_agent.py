"""Fixed descriptive feature/risk-tie audit; no fit, raw X or error-label input."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math
import sys
import time
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.scripts import seal_safeconf_orion_source_risk_agent as base

BASE=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
SOURCE=BASE/'orion_source_core_20261002_v1'
SEAL=BASE/'orion_public_bank_extension_20261002_v4_partitioned/expanded_risk_seal'
EVALUATION=BASE/'orion_registered_test_extendedbank_20261002_v1/fixed_risk_evaluation'
DEFAULT=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/actual_fixed_failure_diagnosis_v1'
CONTEXTS=('HCT116','HEK293T')
META=['task_id','gene','target','upstream','model_version','output_contract_id']
QUANTILES=[0,.01,.05,.25,.5,.75,.95,.99,1]
QNAMES=['min','q01','q05','q25','median','q75','q95','q99','max']


def sha(path):return base.sha(path)

def binding(path):
 path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}

def identity_hash(values):return hashlib.sha256('\n'.join(sorted(map(str,values))).encode()).hexdigest()

def summary(values):
 values=np.asarray(values,float);finite=values[np.isfinite(values)]
 out={'n_rows':len(values),'n_finite':len(finite),'missing_nonfinite_n':len(values)-len(finite),'missing_nonfinite_rate':float((~np.isfinite(values)).mean())}
 out.update(dict(zip(QNAMES,np.quantile(finite,QUANTILES).tolist())) if len(finite) else dict.fromkeys(QNAMES,None))
 return out

def run(output):
 started=time.monotonic();output=Path(output)
 if output.exists():raise RuntimeError('New bounded diagnostic output required')
 output.mkdir(parents=True)
 reads=[];bindings=[]
 feature_manifest=SOURCE/'SOURCE_FEATURE_MANIFEST.json'
 if sha(feature_manifest)!='b74b9dfbc64f5acc82b7b0d34023aa6a523bd11d1904ced7fd638c5bec4ef2d1':raise RuntimeError('Frozen Source feature contract differs')
 manifest=json.loads(feature_manifest.read_text());columns=manifest['risk_P_columns']+manifest['risk_public_columns']
 if columns!=base.P+base.PUBLIC or len(columns)!=13:raise RuntimeError('Exact original13 features required')
 bindings.append(binding(feature_manifest))
 registry={x['path']:x for x in json.loads((SOURCE/'ARTIFACT_HASHES.json').read_text())}
 cohort_path=EVALUATION/'primary_common_COHORT.csv';cohort=pd.read_csv(cohort_path,usecols=base.QUERY,keep_default_na=False)
 reads.append({'path':str(cohort_path),'columns':base.QUERY,'purpose':'fixed completed primary identity only'})
 bindings.append(binding(cohort_path))
 if len(cohort)!=232 or not cohort.query_id.is_unique or set(cohort.context_id)!=set(CONTEXTS) or not cohort.role.eq('TEST').all():raise RuntimeError('Fixed232 primary identities differ')
 score_path=SEAL/'PRETRUTH_RISKS.tsv'
 risks=['Magnitude','P_only_ridge','P_only_hgb','Manual_ridge','Manual_hgb','Learned_ridge','Learned_hgb','Manual_WeightedHistoryDistance','Learned_WeightedHistoryDistance','Uniform_DirectRMSE','Manual_DirectRMSE','Learned_DirectRMSE']
 score=pd.read_csv(score_path,sep='\t',usecols=base.QUERY+risks,keep_default_na=False,float_precision='round_trip')
 reads.append({'path':str(score_path),'columns':base.QUERY+risks,'purpose':'already frozen prediction-time risk ties only'})
 bindings.append(binding(score_path))
 indices=score.set_index('query_id').index.get_indexer(cohort.query_id)
 if (indices<0).any() or score.iloc[indices][base.QUERY].reset_index(drop=True).to_dict('records')!=cohort.to_dict('records'):raise RuntimeError('Frozen target score/cohort identity axis differs')
 target_scores=score.iloc[indices].reset_index(drop=True)
 source_genes_path=SOURCE/'SOURCE_TASKS.parquet';source_genes=set(pd.read_parquet(source_genes_path,columns=['gene']).gene.astype(str))
 reads.append({'path':str(source_genes_path),'columns':['gene'],'purpose':'oldSource575 identity scope only'})
 bindings.append(binding(source_genes_path))
 if len(source_genes)!=575:raise RuntimeError('Original Source575 gene scope differs')
 ranges=[];comparisons=[];coef=[];scopes=[];source_quantiles={}
 for reference in ['Manual','Learned']:
  source_path=SOURCE/f'source_{reference}.parquet'
  expected=registry[source_path.name]
  if sha(source_path)!=expected['sha256'] or source_path.stat().st_size!=expected['bytes']:raise RuntimeError('Actual saved Source OOF features differ')
  source=pd.read_parquet(source_path,columns=META+columns)
  reads.append({'path':str(source_path),'columns':META+columns,'purpose':'exact saved risk-fit OOF feature inputs, no labels'})
  bindings.append(binding(source_path))
  if len(source)!=3616 or set(source.gene.astype(str))!=source_genes or source.duplicated(['upstream','task_id']).any():raise RuntimeError('Actual fixed risk training record identities differ')
  source_features=np.load(SOURCE/f'SOURCE_{reference}_RISK_FEATURES.npy',allow_pickle=False)
  reads.append({'path':str(SOURCE/f'SOURCE_{reference}_RISK_FEATURES.npy'),'purpose':'feature-only model-input equality; no effect/label vector'})
  if source_features.shape!=(3616,13) or not np.array_equal(source[columns].to_numpy(float),source_features,equal_nan=True):raise RuntimeError('Parquet features do not equal original model input array')
  bindings.append(binding(SOURCE/f'SOURCE_{reference}_RISK_FEATURES.npy'))
  target_path=SEAL/f'{reference}_P_PUBLIC_FEATURES.parquet';target=pd.read_parquet(target_path,columns=columns).iloc[indices].reset_index(drop=True)
  reads.append({'path':str(target_path),'columns':columns,'purpose':'frozen target features aligned to fixed primary IDs'})
  bindings.append(binding(target_path))
  if reference=='Manual':target_scores['NegativeSourceHistorySupport']=-target.log_history_support.to_numpy(float)
  source_quantiles[reference]={name:summary(source[name]) for name in columns}
  for group,part in [('Source_pool',source)]+[(f'Source_{upstream}',frame) for upstream,frame in source.groupby('upstream',sort=True)]:
   for name in columns:ranges.append({'reference':reference,'domain':group,'feature':name,**summary(part[name])})
  for context in CONTEXTS:
   part=target.loc[cohort.context_id.eq(context)]
   for name in columns:
    source_range=source_quantiles[reference][name];value=part[name].to_numpy(float);finite=np.isfinite(value);n=int(finite.sum())
    for domain_stats in [summary(value)]:ranges.append({'reference':reference,'domain':context,'feature':name,**domain_stats})
    low=finite&(value<source_range['min']);high=finite&(value>source_range['max']);tail_low=finite&(value<source_range['q01']);tail_high=finite&(value>source_range['q99'])
    comparisons.append({'reference':reference,'context':context,'feature':name,'n_rows':len(value),'n_finite':n,
     'Source_min':source_range['min'],'Source_max':source_range['max'],'Source_q01':source_range['q01'],'Source_q99':source_range['q99'],
     'below_Source_min_n':int(low.sum()),'above_Source_max_n':int(high.sum()),'outside_Source_minmax_rate_finite':float((low|high).sum()/n) if n else None,
     'outside_Source_q01_q99_rate_finite':float((tail_low|tail_high).sum()/n) if n else None,'target_median':float(np.median(value[finite])) if n else None})
  scopes.append({'reference':reference,'Source_records':len(source),'Source_tasks':source.task_id.nunique(),'Source_genes':source.gene.nunique(),
   'Source_record_ids_sha256':identity_hash(source.upstream.astype(str)+'::'+source.task_id.astype(str)),'Source_gene_ids_sha256':identity_hash(source_genes),
   'target_fixed_query_ids_sha256':identity_hash(cohort.query_id),'target_rows':len(target)})
  model_path=SOURCE/f'SOURCE_RISK_{reference}_ridge.joblib';model=joblib.load(model_path)
  reads.append({'path':str(model_path),'purpose':'frozen model/preprocessor metadata only; predict not called'})
  bindings.append(binding(model_path))
  if model.columns!=columns or sha(model_path)!=registry[model_path.name]['sha256']:raise RuntimeError('Frozen Ridge metadata/column contract differs')
  for i,name in enumerate(columns):coef.append({'reference':reference,'feature':name,'standardized_feature_coefficient':float(model.model.coef_[i]),
    'coefficient_sign':int(np.sign(model.model.coef_[i])),'missing_flag_coefficient':float(model.model.coef_[len(columns)+i]),
    'Source_frozen_preprocessor_center':float(model.preprocessor.center_[i]),'Source_frozen_preprocessor_scale':float(model.preprocessor.scale_[i]),
    'interpretation':'descriptive historical Source rank model coefficient; not causal or a target calibration'})
 ties=[]
 for context in CONTEXTS:
  part=target_scores[cohort.context_id.eq(context)].copy()
  for name in risks+['NegativeSourceHistorySupport']:
   values=pd.to_numeric(part[name],errors='raise').to_numpy(float)
   if not np.isfinite(values).all():raise RuntimeError('No filtering of fixed primary risk rows')
   counts=pd.Series(values).value_counts();n=len(values);top=math.ceil(.2*n);order=np.lexsort((part.query_id.to_numpy(str),-values));boundary=values[order[top-1]]
   ties.append({'context':context,'method':name,'n_rows':n,'unique_exact_scores':len(counts),'rows_in_exact_ties':int(counts[counts>1].sum()),
    'fraction_rows_in_exact_ties':float(counts[counts>1].sum()/n),'largest_exact_tie_group':int(counts.max()),
    'fraction_equal_unordered_pairs':float(np.sum(counts.to_numpy()*(counts.to_numpy()-1))/(n*(n-1))),
    'coverage20_selected_rows':top,'coverage20_boundary_tie_size':int(np.sum(values==boundary)),
    'coverage20_boundary_selected_rows':int(np.sum(values[order[:top]]==boundary)),
    'ties_not_broken_with_truth':'frozen query_id lexicographic only'})
 metric_path=EVALUATION/'METHOD_METRIC_INTERVALS.csv';metrics=pd.read_csv(metric_path)
 reads.append({'path':str(metric_path),'purpose':'already completed aggregate7metric summaries only, no error vector'})
 bindings.append(binding(metric_path));metrics=metrics[metrics.cohort.eq('primary_common')].copy()
 metrics.to_csv(output/'EXISTING_PRIMARY_METRIC_SUMMARIES.csv',index=False)
 pd.DataFrame(ranges).to_csv(output/'FEATURE_DISTRIBUTIONS.csv',index=False)
 pd.DataFrame(comparisons).to_csv(output/'TARGET_OUTSIDE_SOURCE_RANGES.csv',index=False)
 pd.DataFrame(ties).to_csv(output/'FIXED_RISK_TIES.csv',index=False)
 pd.DataFrame(coef).to_csv(output/'FROZEN_RIDGE_FEATURE_COEFFICIENTS.csv',index=False)
 pd.DataFrame(scopes).to_csv(output/'INPUT_IDENTITY_SCOPES.csv',index=False)
 report={'schema':'safeconf_orion_fixed_feature_range_diagnosis_v1','status':'COMPLETE_DESCRIPTIVE_AUDIT_NO_NEW_FIT_OR_SCORE_SELECTION',
  'created_utc':datetime.now(timezone.utc).isoformat(),'code':binding(__file__),'input_bindings':bindings,'column_projection_access_audit':reads,
  'Source_feature_contract_sha256':sha(feature_manifest),'fixed_primary_rows':len(cohort),'fixed_primary_gene_clusters':cohort.target_gene_id.nunique(),
  'features':columns,'Source_quantile_reference':'all3616 original risk-fit OOF records, unweighted descriptive distribution; separate2upstream summaries also provided',
  'missing_definition':'all nonfinite values; summaries/ranges use finite rows and report missing separately',
  'out_of_range_definition':'strict below pooled Source observed min or above max; q01/q99 tails separate; rate denominator finite target rows',
  'risk_tie_definition':'exact float64 score equality; no rounded or error-derived ties; frozenquery_id top20 boundary',
  'target_raw_or_expression_or_error_parquet_read':False,'Source_raw_or_effect_or_transfer_error_label_read':False,
  'model_predict_called':False,'new_fits':0,'new_bootstrap_draws':0,'candidate_or_parameter_or_gate_change':False,
  'interpretation_limit':'OOD range/tie descriptions do not prove cause of the observed failure; no automatic significant shift claim, correction, calibration or gate reinterpretation',
  'elapsed_seconds':time.monotonic()-started}
 base.write_json(output/'DIAGNOSIS_MANIFEST.json',report)
 for p in output.iterdir():p.chmod(0o444)
 print(json.dumps({'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],'output':str(output),'ranges':len(ranges),'comparisons':len(comparisons),'ties':len(ties)},indent=2))


if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else DEFAULT)
