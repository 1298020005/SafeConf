#!/usr/bin/env python3
"""Read existing summaries and code only; write exclusively to this audit folder.

No model invocation, raw expression read, sealed/test read or Git operation.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd

AUDIT = Path(__file__).resolve().parent
PUBLIC = AUDIT.parent
CLOSURE = PUBLIC.parent

def file_binding(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}

def record(path):
    path = Path(path)
    if not path.exists():
        return {'path': str(path), 'status': 'PENDING_FILE_ABSENT'}
    return {'binding': file_binding(path), 'value': json.loads(path.read_text())}

def main():
    rows = []
    fit_frames = []
    for run in ('prototype_v2_cpu_environment_repair', 'full_v1', 'full_v2_resume'):
        f = pd.read_csv(PUBLIC/run/'FIT_LEDGER.csv')
        f['run_id'] = run
        fit_frames.append(f)
        rows.append({'run_id': run, 'neural_stages': len(f),
                     'stage_seconds': float(f.elapsed_seconds.sum()),
                     'ledger': file_binding(PUBLIC/run/'FIT_LEDGER.csv')})
    fitted = pd.concat(fit_frames, ignore_index=True)
    unique = fitted.drop_duplicates(['domain','fold','builder','seed','stage'])
    rt = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/publicset_execution_v1')
    b1_audits = []
    for run in ('prototype_v1', 'prototype_v2_cpu_environment_repair', 'full_v1', 'full_v2_resume'):
        for p in sorted((rt/run).glob('*/outer*/B1_HGB/FIT_AUDIT.json')):
            b1_audits.append({'run_id': run, 'binding': file_binding(p)})
    f = pd.read_csv(PUBLIC/'full_v2_resume/TASK_PREDICTIONS.csv.gz')
    arm_rows = []
    for (domain, upstream, builder, seed), part in f.groupby(['domain','upstream','builder','seed']):
        arm_rows.append({'domain':domain,'upstream':upstream,'builder':builder,'seed':int(seed),
                         'n_tasks':len(part),'n_genes':part.gene.nunique(),
                         'n_unique_tasks':part.task_id.nunique()})
    b = pd.read_csv(CLOSURE/'frangieh_cross_family_v1/FIT_LEDGER.csv')
    competence = pd.read_csv(CLOSURE/'frangieh_cross_family_v1/UPSTREAM_COMPETENCE.csv')
    counts = b.groupby('method').size().to_dict()
    manifest = json.loads((PUBLIC/'paper/evidence/RESULT_MANIFEST.json').read_text())
    paper_checks = []
    for entry in manifest['source_files']:
        source = CLOSURE/entry['source_relative_to_closure']
        copied = PUBLIC/'paper'/entry['paper_artifact']
        paper_checks.append({'source':str(source), 'match':
            hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256'] ==
            hashlib.sha256(copied.read_bytes()).hexdigest()})
    snapshot = {
        'audit_time_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Existing registered summaries, saved score tables and code; no model invocation or new truth access',
        'overall_status':'PENDING_FINAL_REQUIREMENTS',
        'neural_stage_runs':rows,
        'unique_main_neural_stages':len(unique),
        'actual_main_neural_stages':len(fitted),
        'prototype_is_subset_of_full':True,
        'main_neural_stages_by_type':fitted.stage.value_counts().to_dict(),
        'actual_main_neural_stage_seconds':float(fitted.elapsed_seconds.sum()),
        'actual_main_B1_CPU_saved_fit_audits':len(b1_audits),
        'main_B1_CPU_unique_fold_scopes':10,
        'main_B1_CPU_audits':b1_audits,
        'B0_and_strong_simple_reference_new_fits':0,
        'full_arm_counts':arm_rows,
        'first_stage_statistics':record(PUBLIC/'statistics_v1/RUN_STATUS.json'),
        'first_stage_scope':'Biology reconstruction plus fixed R_hist/direct readouts only; final joint-reader decision remains pending',
        'nested_registration':record(PUBLIC/'risk_followup_v1/registered_universal_v1/REGISTRATION.json'),
        'nested_registered_result':record(PUBLIC/'risk_followup_v1/registered_universal_v1/RESULT_MANIFEST.json'),
        'nested_registered_statistics':record(PUBLIC/'risk_followup_v1/registered_universal_v1/STATISTICS_RECEIPT.json'),
        'nested_physical_content_null':record(PUBLIC/'risk_followup_v1/registered_universal_v1/content_null_v1/RESULT_MANIFEST.json'),
        'nested_physical_content_null_statistics':record(PUBLIC/'risk_followup_v1/registered_universal_v1/content_null_v1/STATISTICS_RECEIPT.json'),
        'nested_B1_donor_legality':record(PUBLIC/'risk_followup_v1/registered_universal_v1/content_null_v1/ALL_B1_DONOR_LEGALITY_AUDIT.json'),
        'native512_total_small_risk_fits':len(b),
        'native512_fits_by_method':counts,
        'native512_HGB_fits':sum(n for method,n in counts.items() if method.endswith('HGB')),
        'native512_Ridge_fits':sum(n for method,n in counts.items() if method.endswith('Ridge')),
        'native512_competence_passes':int(competence.competence_pass.sum()),
        'native512_competence_models':len(competence),
        'native512_scope':'Completed SEEN retrospective stress experiment; failed competence prevents primary-model qualification',
        'native512_score_repair':record(CLOSURE/'frangieh_cross_family_v1/RidgeScoreRepair/RUN_STATUS.json'),
        'paper_source_bindings_all_match':all(x['match'] for x in paper_checks),
        'paper_source_bindings':paper_checks,
        'paper_numeric_status':record(PUBLIC/'paper/build/NUMERIC_CHECKS.json'),
        'paper_build_status':record(PUBLIC/'paper/build/BUILD_STATUS.json'),
        'paper_pending_results':manifest.get('pending_results'),
        'final_git_state':'PENDING_ROOT_VERIFICATION; reviewer did not run Git',
        'feedback_reuse_rule':'Existing curves reusable only if final serving Public, P and shared-score version hashes are unchanged; changed combinations require same-split/same-budget reevaluation',
    }
    (AUDIT/'ACTUAL_AUDIT.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:snapshot[k] for k in ('overall_status','actual_main_neural_stages',
        'unique_main_neural_stages','actual_main_B1_CPU_saved_fit_audits','native512_HGB_fits',
        'native512_Ridge_fits','native512_competence_passes','paper_source_bindings_all_match')},ensure_ascii=False))

if __name__ == '__main__':
    main()
