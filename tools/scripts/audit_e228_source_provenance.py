#!/usr/bin/env python3
"""Locate outer-target expression reuse in fixed historical scoring features."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--e201-docs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    feature=a.e201_docs/'tables/E201_PRETRUTH_RISK_FEATURES.csv'
    support=a.e201_docs/'tables/E201_SOURCE_CONTEXT_SUPPORT.csv'
    f=pd.read_csv(feature); s=pd.read_csv(support); rows=[]
    assert not (s.target==s.source_context).any(), 'Original per-target isolation failed'
    for held in sorted(f.target.unique()):
        test=f[f.target==held]; train=f[f.target!=held]
        cross=s[(s.target!=held)&(s.source_context==held)].copy()
        cross['task_id']=cross.target+'::'+cross.condition
        cross=cross[cross.task_id.isin(train.task_id)]
        rows.append({'outer_target':held,'n_scoring_training_tasks':len(train),
            'training_tasks_with_outer_target_expression_in_source':cross.task_id.nunique(),
            'training_tasks_with_outer_test_condition_as_source':cross[cross.condition.isin(test.condition)].task_id.nunique(),
            'outer_test_conditions_seen_in_training_task_source':cross[cross.condition.isin(test.condition)].condition.nunique(),
            'original_per_target_self_source_rows':0})
    a.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output/'E228_SOURCE_REUSE_AUDIT.csv',index=False)
    (a.output/'PROVENANCE_AUDIT.json').write_text(json.dumps({'status':'OUTER_PIPELINE_ISOLATION_NOT_ESTABLISHED',
        'feature_sha256':sha(feature),'support_sha256':sha(support),'rows':rows,
        'interpretation':'E228 removed outer error labels only; other-target feature source includes outer-target experimental expression. Original E201 per-target self-source isolation remains intact.'},ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
