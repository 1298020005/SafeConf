#!/usr/bin/env python3
"""Recompute summary tables and package deterministic compressed audit records."""
from datetime import datetime
import gzip
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import scipy
import sklearn
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_e225_raw_evidence_residual_ranker as a
import run_e226_local_history_calibration as b
import run_e227_local_nonlinear_ranker as c
ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'docs/实验结果'
NAMES=['E225_raw_evidence_residual_ranker_20260922','E226_local_history_calibration_20260922',
       'E227_local_nonlinear_ranker_20260922']

def csv(path):
    return pd.read_csv(path if path.exists() else path.with_suffix(path.suffix+'.gz'))

def close(actual,expected,keys):
    actual=actual.sort_values(keys).reset_index(drop=True)
    expected=expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-10,atol=1e-12)

def main():
    reports=[]; studies=[]
    for name,fn in zip(NAMES,(a.summarize,b.summarize,c.summarize)):
        directory=RESULTS/name
        status=json.loads((directory/'RUN_STATUS.json').read_text())
        blocks=csv(directory/'BLOCK_RESULTS.csv')
        study,intervals=fn(blocks)
        studykeys=['dataset','method','budget']+(['history_fraction'] if name.startswith('E226') else [])
        intkeys=['method','baseline','budget','metric']+(['history_fraction'] if name.startswith('E226') else [])
        close(study,pd.read_csv(directory/'STUDY_RESULTS.csv'),studykeys)
        close(intervals,pd.read_csv(directory/'INTERVALS.csv'),intkeys)
        studies.append(study)
        files=[]
        for path in sorted(directory.glob('*.csv')):
            if path.name.startswith('BLOCK_RESULTS') or path.name.startswith('PREDICTIONS_'):
                archived=path.with_suffix('.csv.gz')
                archived.write_bytes(gzip.compress(path.read_bytes(),compresslevel=9,mtime=0))
                files.append({'raw_file':path.name,'raw_bytes':path.stat().st_size,
                    'raw_sha256':a.sha(path),'git_archive':archived.name,
                    'archive_bytes':archived.stat().st_size,'archive_sha256':a.sha(archived)})
        # On a clean clone, raw files can be absent; verify existing archives instead.
        if not files and (directory/'ARCHIVE_MANIFEST.json').exists():
            files=json.loads((directory/'ARCHIVE_MANIFEST.json').read_text())['files']
        for f in files:
            archived=directory/f['git_archive']
            if a.sha(archived)!=f['archive_sha256']: raise ValueError('Archive checksum changed')
            import hashlib
            if hashlib.sha256(gzip.decompress(archived.read_bytes())).hexdigest()!=f['raw_sha256']:
                raise ValueError('Decompressed checksum changed')
        (directory/'ARCHIVE_MANIFEST.json').write_text(json.dumps({'files':files},ensure_ascii=False,indent=2)+'\n')
        reports.append({'experiment':name,'summary_recomputed':True,'block_rows':len(blocks),
            'undefined_utility_rows':int(blocks.utility.isna().sum()),
            'run_script_sha256':status['script_sha256'],'development_gate':status['development_gate'],
            'archive_bytes':sum(f['archive_bytes'] for f in files)})
    local=studies[1].loc[studies[1].history_fraction.eq(.5)].drop(columns='history_fraction').copy()
    local['method']='ridge_'+local.method
    local=local[local.method.ne('ridge_magnitude')]
    nonlinear=studies[2][studies[2].method.str.startswith('ridge_')].copy()
    close(local,nonlinear,['dataset','method','budget'])
    splits=json.loads((RESULTS/NAMES[1]/'GROUP_SPLITS.json').read_text())
    for row in splits:
        if set(row['calibration_perturbations'])&set(row['test_perturbations']):
            raise ValueError('Group split overlap')
    for pred in json.loads((RESULTS/NAMES[0]/'RUN_STATUS.json').read_text())['prediction_files']:
        path=RESULTS/NAMES[0]/pred['file']
        if path.exists() and a.sha(path)!=pred['sha256']: raise ValueError('Prediction checksum changed')
        table=csv(path)
        if a.ERROR in table: raise ValueError('Truth field stored in prediction output')
    sample=csv(RESULTS/NAMES[1]/'BLOCK_RESULTS.csv')
    sample=sample[(sample.method=='magnitude')&(sample.history_fraction==.5)&
                  (sample.history_seed==11)&(sample.budget==.2)]
    result={'checked_at':datetime.now().astimezone().isoformat(),'reports':reports,
        'e226_e227_ridge_replication_agrees':True,'all_calibration_test_groups_disjoint':True,
        'n_split_records':len(splits),'e225_frozen_prediction_checks_pass':True,
        'e226_blocks_per_fixed_method_seed_fraction_budget':len(sample),
        'e226_singleton_blocks_per_fixed_slice':int((sample.n_tasks==1).sum()),
        'note':'Repeated budget/method/seed rows are not independent experiments. Undefined utility is retained, not changed to zero.',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,
                       'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__}}
    output=ROOT/'docs/方法设计/20260922_条件风险模型/RESULT_AUDIT.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
