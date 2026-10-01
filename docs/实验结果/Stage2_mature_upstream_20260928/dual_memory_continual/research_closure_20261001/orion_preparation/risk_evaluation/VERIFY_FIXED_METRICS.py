"""Small invented 40-task/context rank fixture; no target artifacts read."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
sys.path.insert(0,str(ROOT))
from tools.scripts import evaluate_safeconf_orion_frozen_risk_agent as e


def run():
    rows=[{'query_id':f'{c}|G{i:02d}','target_gene_id':f'E{i:02d}',
           'target_gene_symbol':f'G{i:02d}','context_id':c,'role':'TEST','n_cells':40,
           'true_error_rmse':float(i+1),'perfect':float(i+1),'reverse':float(40-i)}
          for c in e.CONTEXTS for i in range(40)]
    frame=pd.DataFrame(rows)
    point=e.metric_values(frame,['perfect','reverse'])
    assert point['macro']['perfect']['utility20']==1
    assert point['macro']['perfect']['spearman']==1
    assert point['macro']['reverse']['spearman']==-1
    assert point['macro']['perfect']['aurc']==10.75
    assert point['macro']['perfect']['error_at_20']==4.5
    assert point['macro']['perfect']['high_risk_miss_rate']==0
    assert point['macro']['reverse']['high_risk_miss_rate']==1
    small=pd.concat([frame[frame.context_id.eq('HCT116')].iloc[:19],
        frame[frame.context_id.eq('HEK293T')].iloc[:30]],ignore_index=True)
    intervals,pairs=e.cohort_statistics(small,['perfect','reverse'],replicates=40,seed=e.SEED)
    invalid=intervals.context.isin(['HCT116','macro'])
    assert intervals.loc[invalid,['point','ci95_lower','ci95_upper']].isna().all().all()
    assert intervals.loc[invalid,'valid_draws'].eq(0).all()
    assert pairs.loc[pairs.context.isin(['HCT116','macro']),'valid_draws'].eq(0).all()
    # Reproduce a shared gene-block draw whose duplicate HCT116 blocks reach20;
    # this must not change original-cohort validity in the evaluator.
    genes=small.target_gene_id.unique();rng=np.random.default_rng(e.SEED)
    reached20=False
    for _ in range(40):
        sample=rng.choice(genes,len(genes),replace=True)
        hct_count=sum(int(((small.target_gene_id==g)&small.context_id.eq('HCT116')).sum()) for g in sample)
        reached20|=hct_count>=20
    assert reached20
    intervals.to_csv(e.DOC/'ORIGINAL_N19_N30_INTERVALS.csv',index=False)
    pairs.to_csv(e.DOC/'ORIGINAL_N19_N30_PAIRED.csv',index=False)
    report={'status':'FIXED_HAND_METRICS_AND_ORIGINAL_CONTEXT_VALIDITY_PASS',
        'evaluator_sha256':e.risk.sha(e.__file__),'synthetic_tasks_per_context':40,
        'perfect_macro_U20':1,'perfect_macro_Spearman':1,'reverse_macro_Spearman':-1,
        'perfect_macro_AURC':10.75,'perfect_macro_error_at20':4.5,
        'perfect_high_risk_miss_rate':0,'reverse_high_risk_miss_rate':1,
        'original_context_sizes':{'HCT116':19,'HEK293T':30},
        'bootstrap_duplicate_draw_reaches20_for_original_invalid_context':True,
        'original_invalid_context_and_macro_points_CIs_remain_NA':True,
        'original_invalid_context_and_macro_valid_draws':0,
        'fixture_bootstrap_draws':40,'production_bootstrap_draws_fixed':5000,
        'actual_Orion_expression_truth_or_error_artifact_read':False,'new_fits':0,'target_CDF_fit':False}
    e.write_json(e.DOC/'FIXED_METRIC_PROOF.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
