"""Synthetic integration and frozen-baseline guards; no real expression."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
import numpy as np
import pandas as pd
REPO=Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921');sys.path.insert(0,str(REPO))
from tools.scripts import bridge_safeconf_orion_lm_inputs_agent as bridge
from tools.scripts import evaluate_safeconf_orion_competence_agent as evaluator
RSCRIPT=Path('/home/yyf/.conda/envs/safeconf-orion-lm-20261002/bin/Rscript')
SOURCE=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_lm_bridge_20261002_v1/synthetic_full_v1')

def immutable(path,value):bridge.write_json(path,value);path.chmod(0o444)
def run_r(args,log,check=True):
    with log.open('w') as f:return subprocess.run([str(RSCRIPT),'--vanilla',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},check=check)
def main(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);bio=root/'biology';bio.mkdir()
    for p in (SOURCE/'biology').iterdir():
        if p.name not in ['BIOLOGY_MANIFEST.json','ARTIFACT_HASHES.json']:shutil.copyfile(p,bio/p.name)
    contract=json.loads((SOURCE/'SYNTHETIC_CONTRACT.json').read_text())
    contract['competence']={'relative_macro_error_gap_max':.02,'minimum_noninferior_strata_fraction':.6,'bootstrap_replicates':5000,'bootstrap_seed':20260929}
    contract_path=root/'SYNTHETIC_CONTRACT.json';immutable(contract_path,contract)
    permit=json.loads((SOURCE/'SYNTHETIC_PERMIT.json').read_text());permit['frozen_method_contract_path']=str(contract_path);permit['frozen_method_contract_sha256']=bridge.sha(contract_path)
    permit_path=root/'SYNTHETIC_PERMIT.json';immutable(permit_path,permit)
    units=bridge.read_csv(bio/'QUERY_METADATA_ONLY.csv')
    effects=np.lib.format.open_memmap(bio/'ENDPOINT3285_EFFECTS.npy',mode='w+',dtype=np.float64,shape=(len(units),3285))
    effects[:]=915255254.25 # Forbidden TRAIN/NTC effect rows for this evaluator.
    expected={}
    for context in bridge.CONTEXTS:
        pred=pd.read_csv(SOURCE/'fit_binary'/context/'PREDICTIONS_DELTA.tsv.gz',sep='\t',float_precision='round_trip')
        for i,r in enumerate(units[(units.context==context)&(units.role=='VALIDATION')].itertuples(index=False)):
            error=.002*(i+1);effects[r.matrix_row]=pred[r.biological_unit].to_numpy()+error;expected[r.biological_unit]=error
    effects.flush();del effects
    manifest=json.loads((SOURCE/'biology/BIOLOGY_MANIFEST.json').read_text());manifest['scientific_contract_sha256']=bridge.sha(contract_path);manifest['permit_sha256']=bridge.sha(permit_path)
    manifest['fixture_origin']='invented validation effects with known constant errors; non-VALIDATION effect rows are forbidden sentinels; no actual raw-loader execution'
    bridge.write_json(bio/'BIOLOGY_MANIFEST.json',manifest)
    bridge.write_json(bio/'ARTIFACT_HASHES.json',[{'path':p.name,'bytes':p.stat().st_size,'sha256':bridge.sha(p)}for p in sorted(bio.iterdir())])
    br=root/'bridge';bridge.bridge(bio,SOURCE/'metadata',contract_path,permit_path,SOURCE/'query_scope/METADATA_ONLY_QUERY_SCOPE.tsv',br)
    fit=root/'fit';run_r([bridge.R_CLI,'--mode','fit','--manifest',br/'FIT_MANIFEST.tsv','--output-dir',fit],root/'FIT_SYNTHETIC.log')
    result=evaluator.evaluate(fit,bio,br,permit_path,contract_path,SOURCE/'query_scope/VALIDATION_ELIGIBILITY.tsv',root/'competence',RSCRIPT)
    errors=pd.read_csv(root/'competence/VALIDATION_TASK_ERRORS.csv',float_precision='round_trip')
    for r in errors.itertuples(index=False):assert abs(r.model_rmse-expected[r.query_id])<1e-12
    ledger=json.loads((root/'competence/C_VALIDATION_ACCESS_LEDGER.json').read_text())
    assert all(r['role']=='VALIDATION'for r in ledger['numeric_effect_rows_read'])
    assert ledger['TEST_truth_rows_read']==0 and ledger['TEST_prediction_columns_converted']==0
    assert result['published_family_attempts']==1 and result['context_models']==2
    guards=[]
    # Even a newly hashed but changed baseline-selection CSV cannot override
    # the choice stored in the fitted model/selected TRAIN baseline object.
    changed=root/'changed_baseline';shutil.copytree(fit/'HCT116',changed)
    selection=pd.read_csv(changed/'BASELINE_SELECTION.tsv',sep='\t');selection['selected']=~selection['selected'];bridge.tsv(changed/'BASELINE_SELECTION.tsv',selection)
    call=run_r([evaluator.BASELINE_HELPER,changed,changed/'OUTPUT_GENE_AXIS.tsv',root/'must_not_extract_changed_baseline',br/'HCT116/INPUT_RECEIPT.tsv'],root/'CHANGED_BASELINE_REJECT.log',check=False)
    assert call.returncode!=0 and not (root/'must_not_extract_changed_baseline.f64le').exists();guards.append('changed_frozen_baseline_selection_rejected')
    changed_receipt=root/'CHANGED_RECEIPT.tsv';shutil.copyfile(br/'HCT116/INPUT_RECEIPT.tsv',changed_receipt)
    with changed_receipt.open('a')as f:f.write('extra_key\textra_value\n')
    call=run_r([evaluator.BASELINE_HELPER,fit/'HCT116',fit/'HCT116/OUTPUT_GENE_AXIS.tsv',root/'must_not_extract_rebound_model',changed_receipt],root/'REBOUND_RECEIPT_REJECT.log',check=False)
    assert call.returncode!=0 and not (root/'must_not_extract_rebound_model.f64le').exists();guards.append('fitted_model_receipt_rebinding_rejected')
    report={'status':'PASS','runtime_root':str(root),'known_validation_RMSE_checks':len(errors),'guard_cases':guards,
        'selected_TRAIN_baseline_never_reselected_on_VAL':True,'TRAIN_NTC_effect_sentinels_not_read':True,'TEST_truth_or_prediction_values_not_read':True,
        'paired_bootstrap_replicates':5000,'seed':20260929,'two_context_models_one_family_attempt':True,
        'real_expression_accessed':False,'real_upstream_attempt_started':False,'synthetic_models_fitted':2,
        'evaluator_sha256':bridge.sha(evaluator.__file__),'baseline_helper_sha256':bridge.sha(evaluator.BASELINE_HELPER),
        'result':result}
    bridge.write_json(root/'VERIFY_COMPETENCE_PIPELINE.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
