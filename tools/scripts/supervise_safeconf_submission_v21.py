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
LOG=RUN/'gladstone_preconfirmation.log'


def alive_owned(pid):
    try:
        fields=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return any(b'run_safeconf_gladstone_v21.py' in x for x in fields)
    except FileNotFoundError:return False


def load(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);ap.add_argument('--poll-seconds',type=int,default=30)
    a=ap.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'SUPERVISOR.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=load(RUN/'SUPERVISOR_STATE.json',{})
    pid=a.pid;retries=int(previous.get('technical_restarts',0));start=time.monotonic()
    while True:
        tail=LOG.read_bytes()[-5000:].decode(errors='replace') if LOG.exists() else ''
        ledger=load(OUT/'DOWNLOAD_RESOURCE_LEDGER.json',{})
        state={'supervisor_pid':os.getpid(),'pipeline_pid':pid,'technical_restarts':retries,
            'status':'RUNNING' if alive_owned(pid) else 'PROCESS_ENDED','last_update_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'wall_seconds_supervised':time.monotonic()-start,'log':str(LOG),
            'new_payload_bytes':ledger.get('this_run_network_payload_bytes'),
            'confirmation_truth_opened':(OUT/'CONFIRMATION_EVALUATION_OPEN_EVENT.json').exists(),
            'cohort_frozen':(OUT/'ROLE_FREEZE.json').exists(),
            'predictors_frozen':(OUT/'PREDICTOR_FREEZE.json').exists(),
            'risk_scores_frozen':(OUT/'RISK_FREEZE.json').exists(),
            'next_action':'Complete registered extraction → predictor gate → risk freeze → conditional confirmation',
            'protected_processes_touched':False}
        handoff=load(RUN/'ENGINEERING_STAGE_HANDOFF.json',{})
        if handoff.get('status')=='WAIT_ALLOWED_EXTRACTION' and (OUT/'TRAIN_FEEDBACK_READ_RECEIPT.json').exists() and alive_owned(pid):
            # Reload the validated shared scoring entrypoint at a saved stage
            # boundary. No numerical target rows need to be fetched again.
            os.kill(pid,signal.SIGTERM)
            for _ in range(20):
                if not alive_owned(pid):break
                time.sleep(.25)
            with LOG.open('a') as out:
                out.write('\nSAVED-STAGE RESUME: validated unified scoring entrypoint; extraction cache preserved\n');out.flush()
                child=subprocess.Popen([PY,str(SCRIPT),'--phase','complete'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            pid=child.pid;handoff.update(status='RELOADED_AT_SAVED_STAGE',pipeline_pid=pid)
            write_json(RUN/'ENGINEERING_STAGE_HANDOFF.json',handoff)
            state.update(status='SAVED_STAGE_RESUME',pipeline_pid=pid)
        if (OUT/'CONFIRMATION_COMPLETE.json').exists():
            state.update(status='EXTERNAL_CONFIRMATION_COMPLETE',next_action='Integrate frozen confirmation statistics and component decision',
                result=load(OUT/'CONFIRMATION_COMPLETE.json'));write_json(RUN/'SUPERVISOR_STATE.json',state)
            subprocess.run([PY,str(ROOT/'tools/scripts/collect_safeconf_submission_results_v21.py')],cwd=ROOT,check=True)
            break
        if not alive_owned(pid):
            decision=load(OUT/'EXTERNAL_DECISION.json',{})
            if decision and not decision.get('passed_predictors'):
                state.update(status='COMPETENCE_FAILED_CONFIRMATION_SEALED',next_action='Execute qualified backup asset; retain development stress-test receipt')
                write_json(RUN/'SUPERVISOR_STATE.json',state)
                subprocess.run([PY,str(ROOT/'tools/scripts/collect_safeconf_submission_results_v21.py')],cwd=ROOT,check=True)
                break
            transport=any(t in tail for t in ['ReadTimeout','ConnectTimeout','ConnectionError','RemoteDisconnected','HTTPError','URLError','SSLError','IncompleteRead'])
            budget=any(t in tail for t in ['cumulative download allowance reached','budget limit'])
            if transport and not budget and retries<3:
                with LOG.open('a') as out:
                    out.write(f'\nSUPERVISOR technical resume {retries+1}, same roles/cache/config\n');out.flush()
                    child=subprocess.Popen([PY,str(SCRIPT),'--phase','complete'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                pid=child.pid;retries+=1;state.update(status='TECHNICAL_RESUME',pipeline_pid=pid,technical_restarts=retries)
            else:
                state.update(status='BUDGET_LIMIT' if budget else 'IMPLEMENTATION_FAILURE_REQUIRES_REPAIR',failure_tail=tail,
                    next_action='Repair recorded implementation fault before resuming identical protocol; no new truth or weakened gate')
                write_json(RUN/'EXTERNAL_FAILURE_RECEIPT.json',state);write_json(RUN/'SUPERVISOR_STATE.json',state);break
        write_json(RUN/'SUPERVISOR_STATE.json',state)
        time.sleep(a.poll_seconds)

if __name__=='__main__':main()
