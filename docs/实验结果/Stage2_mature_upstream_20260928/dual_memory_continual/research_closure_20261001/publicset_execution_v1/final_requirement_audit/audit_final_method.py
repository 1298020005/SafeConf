#!/usr/bin/env python3
"""Final settings/result consistency only; no models, new data or paper work."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import re
import numpy as np
import pandas as pd

AUDIT=Path(__file__).resolve().parent
PUBLIC=AUDIT.parent
REPO=PUBLIC.parents[5]
RUNTIME=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/publicset_execution_v1/risk_followup_v1/registered_universal_v1')
checks=[]
def bound(p):
    p=Path(p)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def check(name,passed,**details):checks.append({'name':name,'passed':bool(passed),**details})
def value(name,display,actual):
    displayed=float(display)
    check(name,abs(displayed-float(actual))<=.500001e-6,displayed=displayed,actual=float(actual))
def receipt(p):return json.loads(Path(p).read_text())

decision=PUBLIC/'FINAL_EXPERIMENTAL_DECISION.md'
settingfile=PUBLIC/'FINAL_MODEL_SETTINGS.json'
text=decision.read_text();settings=receipt(settingfile)
for entry in settings['evidence_bindings']:
    q=REPO/entry['path']
    check('settings SHA '+q.name,bound(q)['sha256']==entry['sha256'],path=str(q))
repair=REPO/settings['release_gate_technical_repair']['receipt']
check('release repair SHA',bound(repair)['sha256']==settings['release_gate_technical_repair']['sha256'])

main=PUBLIC/'risk_followup_v1/registered_universal_v1'
null=main/'content_null_v1'
feedback=PUBLIC/'feedback_alignment_v1'
for name,p,fields in [
    ('main',main/'RESULT_MANIFEST.json',{'status':'COMPLETE','actual_risk_fits':400,'task_prediction_rows':65088}),
    ('null',null/'RESULT_MANIFEST.json',{'status':'COMPLETE','actual_risk_fits':1200,'null_task_prediction_rows':108480}),
    ('feedback',feedback/'RUN_STATUS.json',{'status':'COMPLETE','actual_fits':45})]:
    got=receipt(p)
    for key,want in fields.items():check(name+' '+key,got[key]==want,actual=got[key],expected=want)
    check(name+' actual ledger fits',len(pd.read_csv(p.parent/'RISK_FIT_LEDGER.csv' if name in ('main','null') else p.parent/'FIT_LEDGER.csv'))==fields['actual_risk_fits' if name in ('main','null') else 'actual_fits'])
for name,p in [('main',main),('null',null),('feedback',feedback)]:
    r=receipt(p/'STATISTICS_RECEIPT.json')
    check(name+'5000 statistics',r['status']=='COMPLETE' and r['bootstrap_replicates']==5000)
check('nested neural240',len(pd.read_csv(PUBLIC/'risk_followup_v1/NN_FIT_LEDGER.csv'))==240)
fa=receipt(feedback/'INPUT_AUDIT.json')
for key,want in [('frame_tasks',543),('frame_genes',380),('feedback_tasks',331),('feedback_genes',228),('evaluation_tasks',212),('evaluation_genes',152)]:
    check('feedback cohort '+key,fa[key]==want,actual=fa[key],expected=want)
ma=pd.read_csv(main/'SEED_MEAN_MACRO_RESULTS.csv')
mf=pd.read_csv(feedback/'SEED_MEAN_METRICS.csv')
gn=pd.read_csv(null/'FIXED_ADOPTION_GATES.csv')
gm=pd.read_csv(main/'FIXED_ADOPTION_GATES.csv')
alignment=pd.read_csv(PUBLIC/'statistics_v1/MC_DATA_ALIGNMENT_COMPARISONS.csv')
aligned=alignment[(alignment.readout=='risk')&(alignment.metric=='utility20')].iloc[0]
for col,display in [('comparator_value','.689895'),('candidate_value','.849059'),('delta','.159165'),('ci95_lower','.041750'),('ci95_upper','.214427')]:
    value('decision aligned MC '+col,display,aligned[col])
    check('alignment number appears '+display,display.removeprefix('.') in text)
fixed=pd.read_csv(PUBLIC/'statistics_v1/MACRO_METRIC_INTERVALS.csv')
fixed=fixed[(fixed.scope=='McFaline/DecoderOnly')&(fixed.readout=='risk')&(fixed.metric=='utility20')].set_index('builder')
for method,display in [('SameContextSupport','.872175'),('B1_HGB','.827009'),('B2_Pointwise','.869870'),('B3_DeepSets','.864788')]:
    value('decision fixed MC '+method,display,fixed.loc[method].value)
head=receipt(PUBLIC/'reference_headroom_v1/RESULT_MANIFEST.json')
check('headroom2350 zero fit GPU',head['status']=='COMPLETE' and head['queries_completed']==2350 and head['new_model_fits']==0 and head['GPU_hours']==0 and head['solver_failures']==0)
check('headroom reported seconds',round(head['wall_seconds'],2)==11.42)
feedback_run=receipt(feedback/'RUN_STATUS.json')
check('feedback reported times',round(feedback_run['fit_seconds'],2)==6.08 and round(feedback_run['elapsed_seconds'],2)==50.86)
for line in text.splitlines():
    cells=[x.strip() for x in line.strip().strip('|').split('|')]
    if len(cells)==7 and cells[0] in ('GAT→Exphormer','Exphormer→GAT'):
        source,target=('TxPert_GAT','TxPert_Exphormer') if cells[0]=='GAT→Exphormer' else ('TxPert_Exphormer','TxPert_GAT')
        sub=ma[(ma.source_upstream==source)&(ma.upstream==target)].set_index('method')
        for display,method in zip(cells[1:],['P_only','Support_only','B0_SupportMean','B1_HGB','B2_Pointwise','B3_DeepSets']):
            value('decision main '+cells[0]+'/'+method,display,sub.loc[method].utility20)
    if len(cells)==4 and re.fullmatch(r'\d+%',cells[0]):
        budget=float(cells[0][:-1])/100
        for display,method in zip(cells[1:],['TargetOnly_HGB','PublicTarget_HGB','SharedTarget_HGB']):
            value('decision feedback '+cells[0]+'/'+method,display,mf[(mf.method==method)&(mf.budget==budget)].iloc[0].utility20)

def verify_inline(prefix,comparisons,table):
    line=next(line for line in text.splitlines() if line.startswith(prefix))
    all_values=re.findall(r'[+−-]?\d*\.\d+',line)
    for values,(source,target,a,b) in zip([all_values[i:i+3] for i in range(0,len(all_values),3)],comparisons):
        row=table[(table.source_upstream==source)&(table.upstream==target)&(table.candidate==a)&(table.comparator==b)].iloc[0]
        for display,col in zip(values,['delta_utility20','ci95_lower','ci95_upper']):value(prefix+col+source,display.replace('−','-'),row[col])
directions=[('TxPert_GAT','TxPert_Exphormer'),('TxPert_Exphormer','TxPert_GAT')]
verify_inline('Pointwise相对原参照',[(s,t,a,b) for a,b in [('B2_Pointwise','B1_HGB'),('B3_DeepSets','B2_Pointwise')] for s,t in directions],gm)
verify_inline('Pointwise真实内容减置乱内容',[(s,t,'B2_Pointwise','B2_Pointwise_ContentNull') for s,t in directions],gn)
for method,display in [('SameContext_R_history','.837668'),('NegativeHistorySupport','.795264')]:
    value('feedback zero rule '+method,display,mf[(mf.method==method)&(mf.budget==0)].iloc[0].utility20)
support=gn[gn.candidate.str.endswith('_ContentNull')&gn.comparator.eq('Support_only')]
check('null vs support intervals include zero',bool(((support.ci95_lower<=0)&(support.ci95_upper>=0)).all()))
check('B2 replacement gates fail both directions',not gm[(gm.candidate=='B2_Pointwise')&(gm.comparator=='B1_HGB')].fixed_gate_pass.any())
check('B3 structure gates fail both directions',not gm[(gm.candidate=='B3_DeepSets')&(gm.comparator=='B2_Pointwise')].fixed_gate_pass.any())
check('P plus quantity label correct','Quantity-only' not in text and '数量-only' not in text and 'Prediction+quantity' in text)
check('Source profile exact version',settings['source_reader'].get('profile_version')=='registered_universal_v1')
check('neural not promoted','not promoted' in settings['public']['neural_decision'])
check('cross study not auto enabled',settings['source_reader']['default_cross_study_override'] is False)
check('family remains stress','stress' in settings['source_reader']['foreign_family_status'].lower())
check('Shared remains legacy control','legacy' in settings['target_updates']['old_shared_score'].lower())
check('feedback trigger does not enable',settings['target_updates']['automatic_enable_by_feedback_count'] is False)
check('no untouched confirmation claim','not a new untouched confirmation' in settings['status'])
check('paper deferred',settings['paper_work']=='DEFERRED_BY_USER')

release=receipt(repair)
thresholds={'delta_u20_new_tasks_min':-.005,'relative_aurc_degradation_anchor_max':.05,'miss_rate_degradation_max':.02,'nonnegative_strata_fraction_min':.60}
check('original release thresholds',release['scientific_thresholds_unchanged']==thresholds)
check('tested release code still exact',bound(REPO/'tools/safeconf_continual/update.py')['sha256']==release['updated_code']['sha256'])
check('finite release tests passed',release['validation']['exit_code']==0 and release['validation']['test_methods_passed']==6 and release['validation']['each_field_nonfinite_cases_passed']==12)
check('triggers unchanged',release['update_triggers_unchanged'] is True)
tiny=pd.read_csv(feedback/'TINY_BUDGET_TREE_DIAGNOSTIC.csv')
check('six10pct root-only diagnostics',len(tiny)==6 and tiny.all_trees_root_only.all() and (tiny.actual_trees==200).all() and (tiny.n_train==37).all())

# Reproduce final null point/CI from the saved joint draws, no new sampling.
null_checks=[]
for p in sorted((RUNTIME/'content_null_v1').glob('*JOINT_GENE_BOOTSTRAP.npz')):
    source,target=p.name.removesuffix('_JOINT_GENE_BOOTSTRAP.npz').split('_to_')
    with np.load(p,allow_pickle=False) as z:
        names=z['methods'].tolist();points=z['observed'];draws=z['draws'];maximum=0.
        for row in gn[(gn.source_upstream==source)&(gn.upstream==target)].itertuples():
            ai,bi=names.index(row.candidate),names.index(row.comparator)
            point=float((points[:,:,ai]-points[:,:,bi]).mean());boot=(draws[:,:,:,ai]-draws[:,:,:,bi]).mean(axis=(1,2))
            low,high=np.quantile(boot,[.025,.975])
            maximum=max(maximum,abs(point-row.delta_utility20),abs(low-row.ci95_lower),abs(high-row.ci95_upper))
        check('final null saved bootstrap '+source,maximum<=1e-12,maximum_difference=float(maximum))
        null_checks.append(bound(p))

final={'status':'PASS' if all(c['passed'] for c in checks) else 'REQUIRES_CORRECTION',
    'audit_utc':datetime.now(timezone.utc).isoformat(),'checked_items':len(checks),
    'checked_files':[bound(decision),bound(settingfile),bound(main/'FIXED_ADOPTION_GATES.csv'),bound(null/'FIXED_ADOPTION_GATES.csv'),
        bound(feedback/'SEED_MEAN_METRICS.csv'),bound(feedback/'INPUT_AUDIT.json'),bound(repair)],
    'null_bootstrap_bindings':null_checks,'checks':checks,'failures':[c for c in checks if not c['passed']],
    'scope':'Read-only consistency of final experimental decision/configuration and saved result statistics; no training, fresh data, MC TEST arrays, paper/PDF or Git.',
    'conclusion':'Final settings bind completed experimental results; old Shared stays legacy, native512 stays SEEN stress, neural replacements are rejected, and update release still requires finite metrics and unchanged original gates.'}
(AUDIT/'FINAL_METHOD_AUDIT.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':final['status'],'checked_items':len(checks),'failures':final['failures']},ensure_ascii=False))
