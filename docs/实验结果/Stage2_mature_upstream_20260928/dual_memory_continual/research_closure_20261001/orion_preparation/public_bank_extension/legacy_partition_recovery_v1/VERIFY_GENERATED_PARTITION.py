"""Generated execution-layout proof, no actual data/model/truth reads."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
REPO=Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921');sys.path.insert(0,str(REPO))
from tools.safeconf_continual import orion_public_bank_extension as e
from tools.safeconf_continual import orion_legacy_partition as p
from tools.scripts import seal_safeconf_orion_source_risk_agent as b


class Fixed:
 def __init__(self,cols):self.columns=cols
 def predict(self,frame,clip=True):
  a=.2+.001*frame[self.columns].to_numpy(float).sum(axis=1)
  return np.clip(a,0,1) if clip else a


class BatchSensitive(Fixed):
 def predict(self,frame,clip=True):
  a=super().predict(frame,clip)
  if len(frame)>2:a[0]=np.nextafter(a[0],np.inf)
  return a


def run():
 n=3285
 core={'axis':pd.DataFrame({'gene_name':[f'M{i}' for i in range(n)]}),
 'memory':pd.DataFrame({'experiment_id':['OLD1','OLD2'],'perturbation_target':['OLD','OLD'],
 'perturbation_type':['genetic_single_gene']*2,'effect_vector_row':[0,1],'n_cells':[30,60],'n_batches':[1,1],'eligibility':[True,True]}),
 'effects':np.full((2,n),.1),'controls':np.full((2,n),.2),
 'models':{key:Fixed(b.P if key[0]=='P_only' else b.P+b.PUBLIC) for key in b.METHODS},'public':Fixed(b.PAIR)}
 core['models'][('Manual','ridge')]=BatchSensitive(b.P+b.PUBLIC)
 core['models'][('P_only','ridge')]=BatchSensitive(b.P)
 new=dict(core,memory=pd.concat([core['memory'],pd.DataFrame({'experiment_id':['NEW1'],'perturbation_target':['NEW'],
 'perturbation_type':['genetic_single_gene'],'effect_vector_row':[2],'n_cells':[50],'n_batches':[1],'eligibility':[True]})],ignore_index=True),
 effects=np.full((3,n),.1),controls=np.full((3,n),.2))
 queries=pd.DataFrame([['O1','EO','OLD','HCT116','TEST'],['N1','EN','NEW','HCT116','TEST'],['O2','EO','OLD','HEK293T','VALIDATION'],
 ['N2','EN','NEW','HEK293T','VALIDATION'],['U1','EU','UNKNOWN','HCT116','TEST']],columns=b.QUERY)
 delta=np.full((len(queries),n),.5);controls={'HCT116':np.full(n,.2),'HEK293T':np.full(n,.2)}
 old=b.infer(queries,delta,controls,core);unpartitioned=b.infer(queries,delta,controls,new)
 assert old['scores'].loc[0,'Manual_ridge']!=unpartitioned['scores'].loc[0,'Manual_ridge']
 original,result,proof=p.infer_partitioned(queries,delta,controls,core,new,{'OLD'})
 e.assert_legacy_invariance(queries,old,result,{'OLD'})
 for col in ['Magnitude','P_only_ridge','P_only_hgb']:
  assert old['scores'][col].to_numpy().tobytes()==result['scores'][col].to_numpy().tobytes()
 assert result['scores'].source_history_n.tolist()==[2,1,2,1,0]
 assert np.isfinite(result['scores'].loc[1,'Manual_hgb']) and np.isnan(result['scores'].loc[4,'Manual_hgb'])
 assert sorted(result['pairs'].query_row.unique())==[0,1,2,3]
 assert result['weights'].query_id.drop_duplicates().tolist()==['O1','N1','O2','N2']
 onlyold=queries[queries.target_gene_symbol.eq('OLD')].reset_index(drop=True)
 p.infer_partitioned(onlyold,delta[[0,2]],controls,core,new,{'OLD'})
 report={'schema':'safeconf_orion_generated_partition_invariant_proof_v1','status':'PASS',
 'partition_helper_sha256':e.sha(p.__file__),'new_fits':0,'actual_Source_or_Orion_data_or_models_read':False,'TEST_truth_access':False,
 'checks':['emulated_batch_layout_change_detected_before_fix','complete_original6scores_features_priors_pairs_weights_byte_preserved',
 'ALLquery_P_only_Magnitude_original_fullbatch_bytes_preserved','new_supported_and_unknown_history_status_correct',
 'original_query_row_axis_and_prior_weight_order_restored','all_legacy_only_scope_supported'],
 'execution_formula':proof['execution_partition'],'no_numeric_tolerance_or_math_change':True}
 path=Path(__file__).parent/'GENERATED_PARTITION_PROOF.json';b.write_json(path,report);path.chmod(0o444);print(json.dumps(report,indent=2))


if __name__=='__main__':run()
