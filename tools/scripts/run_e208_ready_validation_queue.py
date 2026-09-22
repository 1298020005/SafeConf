#!/usr/bin/env python3
"""Prefetch registered validation jobs as training finishes, then resume old queue."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_e208_posttraining_pretruth_queue as queue


def ready_checkpoint(root,architecture,seed):
    path=root/architecture/f'seed_{seed}'/'E208_RUN_STATUS.json'
    if not path.is_file(): return None
    row=queue.read_json(path)
    if row.get('status')!='COMPLETE': return None
    checkpoint=Path(str(row.get('best_checkpoint','')))
    if (row.get('architecture')!=architecture or row.get('seed')!=seed
        or row.get('test_perturbed_expression_rows_read')!=0 or not checkpoint.is_file()
        or checkpoint.stat().st_size<=0):
        raise queue.PostTrainingFailure('Completed job has invalid checkpoint contract')
    return checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','perturbench-repo','python','training-root','validation-cache-dir',
                 'test-control-cache-dir','tasks','manifest-status','output-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cuda-devices',default='0,1'); p.add_argument('--poll-seconds',type=int,default=30)
    args=p.parse_args(); args.devices=args.cuda_devices.split(',')
    status_path=args.output_root/'E208_READY_VALIDATION_STATUS.json'
    status={'status':'RUNNING','started_at':queue.now(),'pid':os.getpid(),
            'policy':'same validation jobs, completed models only; reserve all active training GPUs',
            'test_perturbed_expression_rows_read':0,'attempts':{}}
    active={}; attempts={job:0 for job in queue.JOBS}
    try:
        while True:
            for device,(process,job) in list(active.items()):
                if process.poll() is None: continue
                del active[device]
                if process.returncode!=0 or not queue.completed_prediction(args.output_root,'validation',*job):
                    if attempts[job]>=2: raise queue.PostTrainingFailure(f'Validation failed twice: {job}')
            completed=[job for job in queue.JOBS if queue.completed_prediction(args.output_root,'validation',*job)]
            if len(completed)==5:
                if queue.checkpoints(args.training_root) is None: raise queue.PostTrainingFailure('All validation but training queue incomplete')
                status.update(status='COMPLETE_RESUMING_ORIGINAL_QUEUE',finished_at=queue.now())
                queue.atomic_json(status_path,status)
                script=args.repo/'tools/scripts/run_e208_posttraining_pretruth_queue.py'
                os.execv(str(args.python),[str(args.python),str(script),*sys.argv[1:]])
            training=queue.read_json(args.training_root/'E208_FORMAL_QUEUE_STATUS.json')
            reserved={str(r['device']) for r in training.get('active',[])}
            inflight={job for process,job in active.values()}
            ready=[]
            for job in queue.JOBS:
                checkpoint=ready_checkpoint(args.training_root,*job)
                if checkpoint is not None and job not in completed and job not in inflight:
                    ready.append((job,checkpoint))
            for device in args.devices:
                if device in active or device in reserved or not ready: continue
                job,checkpoint=ready.pop(0); attempts[job]+=1
                log=args.output_root/'_logs'/f'early_validation_{job[0]}_seed{job[1]}_attempt{attempts[job]}.log'
                process=queue.launch_prediction(args=args,stage='validation',architecture=job[0],seed=job[1],
                    checkpoint=checkpoint,device=device,log_path=log)
                active[device]=(process,job)
            status.update(updated_at=queue.now(),completed=len(completed),
                active=[{'device':d,'architecture':j[0],'seed':j[1],'pid':p.pid} for d,(p,j) in active.items()],
                training_gpus_reserved=sorted(reserved),attempts={f'{a}/{s}':n for (a,s),n in attempts.items()})
            queue.atomic_json(status_path,status)
            time.sleep(args.poll_seconds)
    except BaseException as e:
        status.update(status='FAILED',reason=repr(e),updated_at=queue.now()); queue.atomic_json(status_path,status); raise


if __name__=='__main__': main()
