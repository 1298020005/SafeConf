#!/usr/bin/env python3
"""Metered official CD4 aggregate download. No expression/truth is read.

Historical unmetered downloads receive a conservative reservation; this run
never treats a new contract as a new 100-GB allowance. Range resumes validate
Content-Range; an ignored Range cannot append a second complete file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
import concurrent.futures
import threading
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.submission_evidence import write_json,sha

RUN=Path('/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21')
OUT=RUN/'external/raw'
BASE='https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/'
FILES={'GWCD4i.DE_stats.h5ad':16_786_240_107,'GWCD4i.pseudobulk_merged.h5ad':44_566_657_140}
CAP=100_000_000_000
HISTORICAL_RESERVATION=38_000_000_000


def segmented(key, expected, meter, meter_path, started):
    """Four Range workers; preserve a previously downloaded contiguous prefix."""
    lock=threading.Lock();dest=OUT/key;prefix=OUT/(key+'.prefix')
    plan_path=OUT/(key+'.segments.json');segment_dir=OUT/(key+'.segments')
    segment_dir.mkdir(exist_ok=True)
    if not plan_path.exists():
        old=OUT/(key+'.part')
        if old.exists():os.replace(old,prefix)
        size=prefix.stat().st_size if prefix.exists() else 0
        chunks=[(s,min(s+(64<<20),expected)-1) for s in range(size,expected,64<<20)]
        write_json(plan_path,{'prefix_bytes':size,'chunks':chunks,'chunk_size_bytes':64<<20})
    plan=json.loads(plan_path.read_text());chunks=plan['chunks']
    def charge(n):
        with lock:
            if HISTORICAL_RESERVATION+meter['this_run_network_payload_bytes']+n>CAP:
                raise RuntimeError('metered cumulative download budget exhausted')
            meter['this_run_network_payload_bytes']+=n
    def fetch(pair):
        begin,end=pair;path=segment_dir/f'{begin:012d}-{end:012d}'
        expected_piece=end-begin+1
        for attempt in range(5):
            have=path.stat().st_size if path.exists() else 0
            if have==expected_piece:return path
            if have>expected_piece:raise RuntimeError('segment larger than permitted Range')
            req=urllib.request.Request(BASE+key,headers={'Range':f'bytes={begin+have}-{end}'})
            try:
                with urllib.request.urlopen(req,timeout=45) as response:
                    if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {begin+have}-{end}/'):
                        raise RuntimeError('server ignored exact segment Range')
                    with path.open('ab' if have else 'wb') as stream:
                        while True:
                            b=response.read(4<<20)
                            if not b:break
                            charge(len(b));stream.write(b)
                if path.stat().st_size==expected_piece:return path
            except RuntimeError:raise
            except Exception:
                if attempt==4:raise
                time.sleep(min(3*(attempt+1),10))
        raise RuntimeError('incomplete Range segment')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(fetch,c) for c in chunks]
        completed=0;last=time.monotonic()
        for future in concurrent.futures.as_completed(futures):
            future.result();completed+=1
            if time.monotonic()-last>10 or completed==len(futures):
                with lock:
                    write_json(meter_path,meter)
                    unique=plan['prefix_bytes']+sum(p.stat().st_size for p in segment_dir.iterdir())
                    write_json(RUN/'external/DOWNLOAD_STATUS.json',{'status':'RUNNING','pid':os.getpid(),
                        'file':key,'received_bytes':unique,'expected_bytes':expected,
                        'workers':4,'wall_seconds':time.monotonic()-started,'target_truth_read':False})
                print(json.dumps({'file':key,'received_GB':round(unique/1e9,3),'workers':4}),flush=True)
                last=time.monotonic()
    assembled=OUT/(key+'.assembled')
    with assembled.open('wb') as output:
        if prefix.exists():
            with prefix.open('rb') as f:shutil.copyfileobj(f,output,8<<20)
        for begin,end in chunks:
            with (segment_dir/f'{begin:012d}-{end:012d}').open('rb') as f:shutil.copyfileobj(f,output,8<<20)
    if assembled.stat().st_size!=expected:raise RuntimeError('assembled official aggregate size mismatch')
    os.replace(assembled,dest)
    return dest


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meter_path=RUN/'external/DOWNLOAD_RESOURCE_LEDGER.json'
    meter=json.loads(meter_path.read_text()) if meter_path.exists() else {
        'original_cumulative_cap_bytes':CAP,'historical_download_reservation_bytes':HISTORICAL_RESERVATION,
        'historical_is_reservation_not_measured':True,
        'reservation_basis':'Oct03 ledger reported no new dataset download; reserve additionally for Orion 18.138GB and unmetered earlier receipts; no budget reset',
        'this_run_network_payload_bytes':0,'files':{},'target_truth_read':False}
    if HISTORICAL_RESERVATION+sum(FILES.values())>CAP:raise RuntimeError('cumulative download cap')
    started=time.monotonic()
    write_json(RUN/'external/DOWNLOAD_STATUS.json',{'status':'RUNNING','pid':os.getpid(),'output':str(OUT)})
    for key,expected in FILES.items():
        dest=OUT/key;partial=OUT/(key+'.part');url=BASE+key
        if dest.exists():
            if dest.stat().st_size!=expected:raise RuntimeError('completed download size changed')
            continue
        dest=segmented(key,expected,meter,meter_path,started)
        meter['files'][key]={'bytes':expected,'sha256':sha(dest),'source':url,'status':'COMPLETE'}
        write_json(meter_path,meter)
        continue
        for attempt in range(5):
            have=partial.stat().st_size if partial.exists() else 0
            if have>expected:raise RuntimeError('unexpected partial size; preserve file')
            if have==expected:break
            req=urllib.request.Request(url,headers={'Range':f'bytes={have}-'} if have else {})
            try:
                with urllib.request.urlopen(req,timeout=45) as response:
                    if have and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {have}-')):
                        raise RuntimeError('server ignored/mismatched Range; original partial preserved')
                    with partial.open('ab' if have else 'wb') as output:
                        last=time.monotonic()
                        while True:
                            chunk=response.read(8<<20)
                            if not chunk:break
                            if HISTORICAL_RESERVATION+meter['this_run_network_payload_bytes']+len(chunk)>CAP:
                                raise RuntimeError('metered cumulative budget exhausted')
                            output.write(chunk);have+=len(chunk);meter['this_run_network_payload_bytes']+=len(chunk)
                            if time.monotonic()-last>10:
                                meter['files'][key]={'received_bytes':have,'expected_bytes':expected,'source':url}
                                write_json(meter_path,meter)
                                write_json(RUN/'external/DOWNLOAD_STATUS.json',{'status':'RUNNING','pid':os.getpid(),
                                    'file':key,'received_bytes':have,'expected_bytes':expected,
                                    'wall_seconds':time.monotonic()-started,'target_truth_read':False})
                                print(json.dumps({'file':key,'received_GB':round(have/1e9,3),
                                    'expected_GB':expected/1e9}),flush=True);last=time.monotonic()
                if have==expected:break
            except Exception as exc:
                write_json(meter_path,meter)
                print(json.dumps({'file':key,'attempt':attempt,'error':type(exc).__name__+': '+str(exc)}),flush=True)
                if isinstance(exc,RuntimeError):raise
                if attempt==4:raise
                time.sleep(min(5*(attempt+1),20))
        if not partial.exists() or partial.stat().st_size!=expected:raise RuntimeError('incomplete download')
        os.replace(partial,dest)
        meter['files'][key]={'bytes':expected,'sha256':sha(dest),'source':url,'status':'COMPLETE'}
        write_json(meter_path,meter)
    write_json(RUN/'external/DOWNLOAD_STATUS.json',{'status':'COMPLETE','files':meter['files'],
        'wall_seconds':time.monotonic()-started,'target_truth_read':False,'new_large_upstream':0})
    print(json.dumps({'status':'COMPLETE','downloaded_bytes':meter['this_run_network_payload_bytes']}),flush=True)


if __name__=='__main__':main()
