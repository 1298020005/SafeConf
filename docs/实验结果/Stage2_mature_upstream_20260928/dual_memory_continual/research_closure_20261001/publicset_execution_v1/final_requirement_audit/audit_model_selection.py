#!/usr/bin/env python3
"""Audit saved Source bootstrap and frozen feedback identity; no fits/test arrays."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

AUDIT=Path(__file__).resolve().parent
PUBLIC=AUDIT.parent
CLOSURE=PUBLIC.parent
REAL=PUBLIC/'risk_followup_v1/registered_universal_v1'
RT=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')

def bound(p):
    p=Path(p)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def receipt(p):
    p=Path(p)
    return {'binding':bound(p),'value':json.loads(p.read_text())} if p.exists() else {'path':str(p),'status':'PENDING_FILE_ABSENT'}

gates=pd.read_csv(REAL/'FIXED_ADOPTION_GATES.csv')
means=pd.read_csv(REAL/'SEED_MEAN_MACRO_RESULTS.csv')
checks=[];old_counts=None
for path in sorted((RT/'publicset_execution_v1/risk_followup_v1/registered_universal_v1').glob('*JOINT_GENE_BOOTSTRAP.npz')):
    direction=path.name.removesuffix('_JOINT_GENE_BOOTSTRAP.npz')
    source,target=direction.split('_to_')
    with np.load(path,allow_pickle=False) as z:
        methods=z['methods'].tolist(); observed=z['observed']; draws=z['draws']; counts=z['gene_multiplicities']
        points=observed.mean(axis=(0,1))
        saved=means[(means.source_upstream==source)&(means.upstream==target)].set_index('method')
        model_diff=float(np.max(np.abs(points-saved.loc[methods].utility20.to_numpy())))
        point_diff=0.;ci_diff=0.
        for row in gates[(gates.source_upstream==source)&(gates.upstream==target)].itertuples():
            ai,bi=methods.index(row.candidate),methods.index(row.comparator)
            difference=(draws[:,:,:,ai]-draws[:,:,:,bi]).mean(axis=(1,2))
            low,high=np.quantile(difference,[.025,.975])
            point_diff=max(point_diff,abs(points[ai]-points[bi]-row.delta_utility20))
            ci_diff=max(ci_diff,abs(low-row.ci95_lower),abs(high-row.ci95_upper))
        checks.append({'direction':direction,'bootstrap_binding':bound(path),
            'point_vs_saved_macro_max_difference':model_diff,'paired_gate_point_max_difference':float(point_diff),
            'paired_CI_max_difference':float(ci_diff),'bootstrap_shape':list(draws.shape),
            'all_draws_have_575_genes':bool(np.all(counts.sum(1)==575)),
            'same_draws_as_previous_direction':None if old_counts is None else bool(np.array_equal(old_counts,counts)),
            'actual_expansion_check_max_difference':float(z['expansion_equivalence_max_difference'])})
        old_counts=counts.copy()
cache=RT/'common_gene_axis/risk_cache/external_Learned.parquet'
feedback_identity=pd.read_parquet(cache,columns=['task_id','gene','target','model_version','output_contract_id'])
info=CLOSURE/'common_gene_axis/results/STRICT_FEEDBACK_INFORMATION_LEDGER.csv'
ledger=pd.read_csv(info)
report={
    'audit_time_utc':datetime.now(timezone.utc).isoformat(),
    'scope':'Completed Source risk model selection and frozen feedback contracts only; no manuscript/PDF work, fits or MC test arrays',
    'main_result':receipt(REAL/'RESULT_MANIFEST.json'),
    'main_statistics':receipt(REAL/'STATISTICS_RECEIPT.json'),
    'neural_prior_integrity':receipt(REAL/'NEURAL_PRIOR_CACHE_INTEGRITY_AUDIT.json'),
    'independent_saved_bootstrap_checks':checks,
    'checks_pass':all(max(c['point_vs_saved_macro_max_difference'],c['paired_gate_point_max_difference'],c['paired_CI_max_difference'])<1e-12 and c['all_draws_have_575_genes'] for c in checks),
    'gate_source':bound(REAL/'FIXED_ADOPTION_GATES.csv'),
    'gates':gates.to_dict('records'),
    'fixed_reference_decision':receipt(PUBLIC/'statistics_v1/DECISION.json'),
    'method_policy':'Retain incumbent version; do not adopt B2 global replacement or expand DeepSets. This is an adoption-policy result, not proof that B1 is statistically optimal.',
    'mechanism_status':'PENDING_PHYSICAL_CONTENT_NULL_COMPLETION',
    'null_result':receipt(REAL/'content_null_v1/RESULT_MANIFEST.json'),
    'feedback_cache':bound(cache),
    'feedback_metadata':{'n_tasks':len(feedback_identity),'n_genes':int(feedback_identity.gene.nunique()),
        'model_versions':feedback_identity.model_version.unique().tolist(),
        'output_contracts':feedback_identity.output_contract_id.unique().tolist()},
    'feedback_split':receipt(CLOSURE/'common_gene_axis/results/FEEDBACK_SPLIT_CONTRACT.json'),
    'feedback_information_ledger':bound(info),
    'feedback_budget_rows':ledger[ledger.method.eq('PublicTarget_HGB')][['budget','c_feedback_error_rows',
        'c_feedback_error_clusters','c_validation_risk_error_rows','c_validation_upstream_calibration_rows',
        'c_validation_biology_rows','allowed_feedback_records_hash']].to_dict('records'),
    'guide_bank_manifest':receipt(RT/'common_gene_axis/public_mcfaline_trainval/manifest.json'),
    'feedback_reuse_condition':'Exact same P, Public features, Shared scores, feedback/error labels, split, budget and output version; current main Source-only reader does not itself modify the old external feedback cache.',
    'new_cell_prior_changes':'Rebuild content-dependent Public features for the identical frozen 543-task cohort; rerun PublicTarget/SharedTarget/Residual and Public/Shared PertEMA combinations. Reuse unchanged prediction-only/native-control-only models. Reevaluate affected zero-feedback rules and any changed Shared scores on exactly the 212-task holdout.',
    'zero_feedback_caveat':'Budget0 means zero feedback-pool errors; fixed upstream calibration and public-biology validation information remain disclosed separately.',
}
(AUDIT/'MODEL_SELECTION_FEEDBACK_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
gates.to_csv(AUDIT/'ACTUAL_MAIN_RISK_GATES.csv',index=False)
print(json.dumps({'checks_pass':report['checks_pass'],'completed_main_fits':report['main_result']['value']['actual_risk_fits'],
    'feedback_tasks':len(feedback_identity),'method_policy':report['method_policy'],'mechanism_status':report['mechanism_status']},ensure_ascii=False))
