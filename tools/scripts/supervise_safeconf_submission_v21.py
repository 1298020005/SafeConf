#!/usr/bin/env python3
"""Durable supervision of the already-authorized external pipeline.

Watches only the owned PID. Restarts at most three times on explicit transport
failures using immutable caches. Scientific failures and budget limits remain
visible decisions; this supervisor never weakens a gate or opens truth itself.
"""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.submission_evidence import write_json
RUN=Path('/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21');OUT=RUN/'external'
PY='/home/miniconda/bin/python';SCRIPT=ROOT/'tools/scripts/run_safeconf_gladstone_v21.py'
FALLBACK=ROOT/'tools/scripts/run_safeconf_kolf_knn_fallback_v1.py'
LOG=RUN/'gladstone_preconfirmation.log'


def alive_owned(pid,script=SCRIPT):
    try:
        fields=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        target=Path(script).resolve()
        for field in fields:
            if not field:continue
            try:
                if Path(field.decode()).resolve()==target:return True
            except (OSError,UnicodeDecodeError):
                pass
        return False
    except FileNotFoundError:return False


def load(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def pipeline_command(script,outroot):
    return [PY,str(script),'--result-root',str(outroot)] if script==FALLBACK else [PY,str(script),'--phase','complete']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);ap.add_argument('--poll-seconds',type=int,default=30)
    ap.add_argument('--result-root',type=Path,default=OUT)
    ap.add_argument('--pipeline-script',type=Path,default=SCRIPT)
    a=ap.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    allowed=[SCRIPT,ROOT/'tools/scripts/repair_safeconf_gladstone_training_v21.py',ROOT/'tools/scripts/run_safeconf_kolf_panel_v21.py',FALLBACK]
    if a.pipeline_script.resolve() not in [p.resolve() for p in allowed]:raise RuntimeError('unowned pipeline script')
    outroot=a.result_root.resolve()
    if outroot.parent!=RUN.resolve():raise RuntimeError('unowned pipeline result directory')
    script=a.pipeline_script.resolve()
    active_script=script
    lock=(RUN/'SUPERVISOR.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=load(RUN/'SUPERVISOR_STATE.json',{})
    pid=a.pid;retries=int(previous.get('technical_restarts',0));start=time.monotonic()
    while True:
        tail=LOG.read_bytes()[-5000:].decode(errors='replace') if LOG.exists() else ''
        shared=load(outroot/'SHARED_RESOURCE_CACHE.json',{})
        ledger=load(Path(shared.get('ledger',outroot/'DOWNLOAD_RESOURCE_LEDGER.json')), {})
        state={'supervisor_pid':os.getpid(),'pipeline_pid':pid,'technical_restarts':retries,
            'status':'RUNNING' if alive_owned(pid,active_script) else 'PROCESS_ENDED','last_update_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'pipeline_script':str(active_script),'result_root':str(outroot),
            'wall_seconds_supervised':time.monotonic()-start,'log':str(LOG),
            'new_payload_bytes':ledger.get('this_run_network_payload_bytes'),
            'confirmation_truth_opened':(outroot/'CONFIRMATION_EVALUATION_OPEN_EVENT.json').exists(),
            'cohort_frozen':(outroot/'ROLE_FREEZE.json').exists(),
            'predictors_frozen':(outroot/'PREDICTOR_FREEZE.json').exists(),
            'risk_scores_frozen':(outroot/'RISK_FREEZE.json').exists(),
            'next_action':'Complete registered extraction → predictor gate → risk freeze → conditional confirmation',
            'protected_processes_touched':False}
        handoff=load(RUN/'ENGINEERING_STAGE_HANDOFF.json',{})
        if active_script==SCRIPT and handoff.get('status')=='WAIT_ALLOWED_EXTRACTION' and (outroot/'TRAIN_FEEDBACK_READ_RECEIPT.json').exists() and alive_owned(pid,active_script):
            # Reload the validated shared scoring entrypoint at a saved stage
            # boundary. No numerical target rows need to be fetched again.
            os.kill(pid,signal.SIGTERM)
            for _ in range(20):
                if not alive_owned(pid,active_script):break
                time.sleep(.25)
            with LOG.open('a') as out:
                out.write('\nSAVED-STAGE RESUME: validated unified scoring entrypoint; extraction cache preserved\n');out.flush()
                child=subprocess.Popen([PY,str(script),'--phase','complete'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            pid=child.pid;handoff.update(status='RELOADED_AT_SAVED_STAGE',pipeline_pid=pid)
            write_json(RUN/'ENGINEERING_STAGE_HANDOFF.json',handoff)
            state.update(status='SAVED_STAGE_RESUME',pipeline_pid=pid)
        if (outroot/'CONFIRMATION_COMPLETE.json').exists():
            state.update(status='EXTERNAL_CONFIRMATION_COMPLETE',next_action='Integrate frozen confirmation statistics and component decision',
                result=load(outroot/'CONFIRMATION_COMPLETE.json'));write_json(RUN/'SUPERVISOR_STATE.json',state)
            subprocess.run([PY,str(ROOT/'tools/scripts/collect_safeconf_submission_results_v21.py')],cwd=ROOT,check=True)
            break
        if not alive_owned(pid,active_script):
            decision=load(outroot/'EXTERNAL_DECISION.json',{})
            if decision and not decision.get('passed_predictors'):
                # A bounded alternate upstream predictor is allowed only after
                # both registered predictors fail.  It uses the same frozen
                # KOLF root and cannot open confirmation until its own risk
                # freeze exists.  The original failed package remains intact.
                fallback_decision=outroot/'KNN_FALLBACK_DECISION.json'
                fallback_failure=outroot/'KNN_FALLBACK_FAILURE_RECEIPT.json'
                event=outroot/'CONFIRMATION_EVALUATION_OPEN_EVENT.json'
                if outroot.name=='external_kolf_panel1400_v1' and not fallback_decision.exists() and not fallback_failure.exists() and not event.exists():
                    with LOG.open('a') as out:
                        out.write('\nSUPERVISOR starting bounded KNN upstream-predictor fallback\n');out.flush()
                        child=subprocess.Popen([PY,str(FALLBACK),'--result-root',str(outroot)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                    pid=child.pid;active_script=FALLBACK
                    state.update(status='BOUNDED_KNN_FALLBACK',pipeline_pid=pid,
                        next_action='Run one upstream-only KNN fallback; preserve original failures')
                    write_json(RUN/'SUPERVISOR_STATE.json',state)
                    continue
                state.update(status='COMPETENCE_FAILED_CONFIRMATION_SEALED',next_action='Retain development stress-test receipt; prepare independent asset decision')
                write_json(RUN/'SUPERVISOR_STATE.json',state)
                subprocess.run([PY,str(ROOT/'tools/scripts/collect_safeconf_submission_results_v21.py')],cwd=ROOT,check=True)
                break
            receipt=load(outroot/('KNN_FALLBACK_FAILURE_RECEIPT.json' if active_script==FALLBACK else 'FAILURE_RECEIPT.json'),{})
            latest_type=receipt.get('error_type','')
            # Inspect the latest failure receipt, never an older traceback in
            # the shared append-only log.  A stale SSL error must not cause a
            # new KeyError or data-contract fault to be retried as transport.
            transport=latest_type in ['ReadTimeout','ConnectTimeout','ConnectionError','RemoteDisconnected','HTTPError','URLError','SSLError','IncompleteRead']
            budget=any(t in receipt.get('error','') for t in ['cumulative download allowance reached','budget limit'])
            if transport and not budget and retries<3:
                with LOG.open('a') as out:
                    out.write(f'\nSUPERVISOR technical resume {retries+1}, same roles/cache/config\n');out.flush()
                    child=subprocess.Popen(pipeline_command(active_script,outroot),cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                pid=child.pid;retries+=1;state.update(status='TECHNICAL_RESUME',pipeline_pid=pid,technical_restarts=retries)
            else:
                state.update(status='BUDGET_LIMIT' if budget else 'IMPLEMENTATION_FAILURE_REQUIRES_REPAIR',failure_tail=tail,
                    next_action='Repair recorded implementation fault before resuming identical protocol; no new truth or weakened gate')
                write_json(RUN/'EXTERNAL_FAILURE_RECEIPT.json',state);write_json(RUN/'SUPERVISOR_STATE.json',state);break
        write_json(RUN/'SUPERVISOR_STATE.json',state)
        time.sleep(a.poll_seconds)

if __name__=='__main__':main()
