"""Read-only frozen pretruth legacy inference comparison; no fits/truth reads."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual import orion_public_bank_extension as ext
from tools.scripts import seal_safeconf_orion_source_risk_agent as base


def compare(left,right):
    a=np.asarray(left);b=np.asarray(right)
    if a.dtype!=b.dtype or a.shape!=b.shape:return {'shape_dtype_equal':False,'bits_equal':False}
    numeric=np.issubdtype(a.dtype,np.number)
    if not numeric:return {'bits_equal':bool(np.array_equal(a,b))}
    finite=np.isfinite(a)&np.isfinite(b);numeric_equal=bool(np.array_equal(a,b,equal_nan=True))
    bits=(np.ascontiguousarray(a).view(np.uint8).reshape(a.size,a.dtype.itemsize)!=np.ascontiguousarray(b).view(np.uint8).reshape(b.size,b.dtype.itemsize)).any(axis=1)
    return {'shape_dtype_equal':True,'numeric_equal':numeric_equal,'bits_equal':not bool(bits.any()),'different_values':int(bits.sum()),
      'maximum_finite_abs_gap':float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.}


def run(output):
    started=time.monotonic();root=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
    registration=root/'orion_public_bank_extension_20261002_v3_decimal_tokens/SEAL_REGISTRATION.json'
    registered=ext.read_json(ext.binding(registration));manifest=base.frozen_manifest(ext.checked(registered['base_comparison_manifest']))
    core=base.source_core();new,bank,bindings,oldgenes=ext.load_bank(registered['bank_manifest'],registered['extension_contract'],core)
    queries,delta,controls,lm=base.load_inputs(manifest,core)
    original=base.infer(queries,delta,controls,core);expanded=base.infer(queries,delta,controls,new)
    use=queries.target_gene_symbol.astype(str).isin(oldgenes).to_numpy();legacy_queries=queries.loc[use].reset_index(drop=True)
    scores={};mismatches=[]
    for name in original['scores'].select_dtypes(include=[np.number]).columns:
      left=original['scores'].loc[use,name].to_numpy();right=expanded['scores'].loc[use,name].to_numpy();scores[name]=compare(left,right)
      if not scores[name]['bits_equal']:
       for i,(a,b) in enumerate(zip(left,right)):
        if np.asarray(a).tobytes()!=np.asarray(b).tobytes():
         mismatches.append({**legacy_queries.iloc[i].to_dict(),'column':name,'original':float(a),'expanded':float(b),'abs_gap':float(abs(a-b))})
    features={ref:compare(original['reference_features'][ref].loc[use].to_numpy(float),expanded['reference_features'][ref].loc[use].to_numpy(float)) for ref in ['Uniform','Manual','Learned']}
    priors={ref:compare(original['priors'][ref][use],expanded['priors'][ref][use]) for ref in ['Uniform','Manual','Learned']}
    pair_use=original['pairs'].query_id.isin(legacy_queries.query_id);new_pair_use=expanded['pairs'].query_id.isin(legacy_queries.query_id)
    pairs_old=original['pairs'].loc[pair_use].reset_index(drop=True);pairs_new=expanded['pairs'].loc[new_pair_use].reset_index(drop=True)
    weights_old=original['weights'][original['weights'].query_id.isin(legacy_queries.query_id)].reset_index(drop=True)
    weights_new=expanded['weights'][expanded['weights'].query_id.isin(legacy_queries.query_id)].reset_index(drop=True)
    pair_check=ext.frame_equal_bits(pairs_old,pairs_new);weight_check=ext.frame_equal_bits(weights_old,weights_new)
    # Complete old-bank branch is identical when used as frozen original scope.
    # Replaying identical feature rows within the original prediction batch
    # isolates the batch-shape hypothesis without overwriting any scores.
    replay={}
    valid=original['scores'].source_history_n.to_numpy()>0
    new_valid=expanded['scores'].source_history_n.to_numpy()>0
    for ref in ['Manual','Learned']:
      model=core['models'][(ref,'ridge')]
      old_x=original['reference_features'][ref].loc[valid].reset_index(drop=True)
      new_x=expanded['reference_features'][ref].loc[new_valid].reset_index(drop=True)
      old_prediction=model.predict(old_x)
      matched=expanded['reference_features'][ref].loc[valid].reset_index(drop=True)
      original_batch_prediction=model.predict(matched)
      replay[ref]={'original_batch_rows':len(old_x),'expanded_batch_rows':len(new_x),
        'same_feature_rows_at_original_batch_shape':compare(old_prediction,original_batch_prediction),
        'preprocessor_numeric_feature_bits_equal':compare(model.preprocessor.transform(old_x[model.columns].to_numpy(float)),model.preprocessor.transform(matched[model.columns].to_numpy(float)))}
    output=Path(output);output.mkdir(exist_ok=False)
    pd.DataFrame(mismatches).to_csv(output/'LEGACY_SCORE_MISMATCHES.csv',index=False,float_format='%.17g')
    report={'schema':'safeconf_orion_actual_legacy_batch_bit_diagnosis_v1','status':'DIAGNOSIS_COMPLETE_NO_FITS_OR_TEST',
      'registration':ext.binding(registration),'original_scientific_contract_sha256':ext.BASE_METHOD_SHA,'bank_manifest':registered['bank_manifest'],
      'source_core_bindings':core['bindings'],'LM_pretruth_bindings':lm,'legacy_query_count':int(use.sum()),'score_checks':scores,
      'feature_checks':features,'prior_checks':priors,'all_legacy_pairs_numeric_bits_and_identity_equal':pair_check,
      'all_legacy_weights_numeric_bits_and_identity_equal':weight_check,'same_original_batch_replay':replay,
      'new_fits':0,'Source_raw_X_read':False,'target_truth_or_labels_read':False,'actual_TEST_numeric_access':False,'elapsed_seconds':time.monotonic()-started}
    base.write_json(output/'DIAGNOSIS.json',report)
    for p in output.iterdir():p.chmod(0o444)
    print(json.dumps({'elapsed_seconds':report['elapsed_seconds'],'score_checks':scores,'features':features,'priors':priors,'pairs_bits_equal':pair_check,'weights_bits_equal':weight_check,'replay':replay},indent=2))


if __name__=='__main__':run(sys.argv[1])
