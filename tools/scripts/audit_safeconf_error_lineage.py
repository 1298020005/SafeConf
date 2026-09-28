#!/usr/bin/env python3
"""Read existing validation records only; do not infer provenance from names."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from run_safeconf_model_decision import ROOT,OUT,SOURCE,allowed_frame,META
x=allowed_frame(SOURCE,META+['run_dir','true_error_rmse'])
rows=[]
for ds,g in x.groupby('dataset_name'):
 run=Path('/home/yyf/safeconf_runtime')/g.run_dir.iloc[0]
 record=run/'tables/PREDICTION_RECORDS.csv'
 rec=allowed_frame(record,META+['true_error_rmse'])
 for pred,part in g.groupby('predictor_name'):
  base=rec[rec.predictor_name==pred]
  joined=part.merge(base,on=META,suffixes=('_matrix','_original'),how='left',validate='one_to_one')
  matched=np.isclose(joined.true_error_rmse_matrix,joined.true_error_rmse_original,atol=1e-10,rtol=1e-8)
  rows.append(dict(dataset=ds,predictor=pred,n_candidates=len(part),original_record_rows_matched=int(matched.sum()),original_run_dir_exists=run.exists(),original_split_manifest_exists=(run/'tables/HELDOUT_PAIR_SPLITS.csv').exists(),original_prediction_archive_exists=(run/'input/predicted_effects.npz').exists(),original_runtime_script_exists=(run/'scripts/run_confidence_mvp_v2_1.py').exists(),code_intends_train_only=True,full_frozen_model_state_and_train_exclusion_link_verified=False,accepted_formal_error_memory_rows=0,reason='record agreement and current code intent do not prove original frozen-state training exclusion; PertMean is later derived'))
  if pred!='PertMeanPredictor' and not matched.all(): raise AssertionError((ds,pred,'original mismatch'))
result=pd.DataFrame(rows);result.to_csv(OUT/'ERROR_LINEAGE_AUDIT.csv',index=False)
print(result.to_string(index=False))
