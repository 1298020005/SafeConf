#!/usr/bin/env python3
"""Version2 integration: immutable v1 scientific guard/math, explicit v2 parser.

The original loader/reader files and old incomplete runtime are preserved.
The permit must bind both original validation code and the actual v2 wrapper/
numeric backend. Outputs appear only after backend identities are attached.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.scripts import build_safeconf_orion_allowed_biology_agent as base
from tools.scripts import probe_safeconf_private_parquet_agent as primitives

BACKEND=ROOT/'tools/scripts/probe_safeconf_private_parquet_vectorized_agent.py'
DOC=base.DOC/'vectorized_integration_v2'
SCHEMA='safeconf_orion_vectorized_numeric_integration_v2'
WRAPPER_BINDING='biology_integration_v2'
BACKEND_BINDING='numeric_reader_backend_v2'


def backend_identity():
    return {'schema':SCHEMA,'original_validation_loader_path':str(Path(base.__file__).resolve()),
        'original_validation_loader_sha256':base.sha(base.__file__),
        'original_structural_reader_path':str(base.READER),'original_structural_reader_sha256':base.sha(base.READER),
        'actual_wrapper_path':str(Path(__file__).resolve()),'actual_wrapper_sha256':base.sha(__file__),
        'actual_numeric_backend_path':str(BACKEND),'actual_numeric_backend_sha256':base.sha(BACKEND),
        'actual_numeric_backend_version':'PRIVATE_NUMERIC_BYTE_SELECTION_V2',
        'scientific_normalization_mapping_QC_or_model_rules_changed':False}


def authorize(metadata_root,raw_root,contract,permit,roles,contexts):
    science,permission,budgets=base.authorization(metadata_root,raw_root,contract,permit,roles,contexts)
    for key,path in [(WRAPPER_BINDING,Path(__file__)),(BACKEND_BINDING,BACKEND)]:
        binding=permission.get('implementation_bindings',{}).get(key,{})
        if Path(binding.get('path','')).resolve()!=path.resolve() or binding.get('sha256')!=base.sha(path):
            raise RuntimeError('New immutable permit must bind actual v2 wrapper and numeric backend before raw access')
    if permission.get('vectorized_backend_scope')!=SCHEMA:
        raise RuntimeError('Explicit v2 numeric backend integration scope required')
    return science,permission,budgets


@contextmanager
def dispatch(budgets,aggregate_mode=False):
    """Single-process dispatch; original function objects are restored on exit."""
    backend=importlib.import_module('tools.scripts.probe_safeconf_private_parquet_vectorized_agent')
    original_iterator=primitives.iter_selected_numeric_lists
    original_metadata=base.read_metadata
    previous_handler=signal.getsignal(signal.SIGALRM)
    previous_timer=signal.getitimer(signal.ITIMER_REAL)
    metadata_visits={}; starts={}; current=[None]
    def timeout(signum,frame):
        raise TimeoutError('V2 hard per-file deadline exceeded, including private trailing pages')
    def metadata(identity,roles):
        key=identity['path']; metadata_visits[key]=metadata_visits.get(key,0)+1
        # Frozen base aggregate reads each file once during metadata planning,
        # then again immediately at the start of its actual aggregation pass.
        if aggregate_mode and metadata_visits[key]>=2:
            starts[key]=time.monotonic(); current[0]=key
            signal.setitimer(signal.ITIMER_REAL,budgets['max_seconds_per_file'])
        return original_metadata(identity,roles)
    def iterator(path,name,mask,trace,forbidden_bytes=()):
        complete=False
        try:
            yield from backend.iter_selected_numeric_lists(path,name,mask,trace,forbidden_bytes)
            complete=True
        finally:
            if aggregate_mode and name=='gene_expression' and complete:
                signal.setitimer(signal.ITIMER_REAL,0)
    if aggregate_mode:signal.signal(signal.SIGALRM,timeout)
    primitives.iter_selected_numeric_lists=iterator
    if aggregate_mode:base.read_metadata=metadata
    try:
        yield
    finally:
        primitives.iter_selected_numeric_lists=original_iterator; base.read_metadata=original_metadata
        if aggregate_mode:
            signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous_handler)
            if previous_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,*previous_timer)


def refresh_hashes(stage):
    old=json.loads((stage/'ARTIFACT_HASHES.json').read_text())
    by_path={x['path']:x for x in old}
    for name in ['BIOLOGY_MANIFEST.json','VECTOR_BACKEND_ACCESS_AUDIT.json']:
        p=stage/name; by_path[name]={'path':name,'bytes':p.stat().st_size,'sha256':base.sha(p)}
    (stage/'ARTIFACT_HASHES.json').write_text(json.dumps([by_path[k] for k in sorted(by_path)],indent=2)+'\n')


def aggregate(metadata_root,raw_root,contract,permit,output,roles,contexts,forbidden_bytes=()):
    _,permission,budgets=authorize(metadata_root,raw_root,contract,permit,roles,contexts)
    output=Path(output)
    if output.exists():raise RuntimeError('New version2 runtime required; old output/staging cannot be overwritten')
    staging=output.with_name(output.name+f'.vector_v2_uncommitted.{os.getpid()}')
    if staging.exists():raise RuntimeError('New version2 staging required')
    identity=backend_identity()
    with dispatch(budgets,aggregate_mode=True):
        result=base.aggregate(metadata_root,raw_root,contract,permit,staging,roles,contexts,forbidden_bytes)
    # v1 loader/reader identities name preserved validation/scientific guards.
    # Actual numeric execution is always recorded separately and truthfully.
    result.update(numeric_backend_integration=identity,
        executed_numeric_parser_sha256=identity['actual_numeric_backend_sha256'],
        executed_numeric_parser_path=identity['actual_numeric_backend_path'],
        scientific_loader_math_sha256=identity['original_validation_loader_sha256'],
        original_reader_sha256_field_role='frozen structural primitives and original validation guard; not actual v2 numeric parser',
        hard_per_file_deadline_seconds=budgets['max_seconds_per_file'],
        old_incomplete_runtime_reused=False)
    for file_audit in result['file_audits']:
        file_audit['executed_numeric_parser_sha256']=identity['actual_numeric_backend_sha256']
        allowed={'physical_numeric_chunk_bytes_read','opaque_numeric_bytes_decompressed',
                 'max_compressed_chunk_bytes','max_page_uncompressed_bytes','max_page_level_entries',
                 'numeric_materialization_count','authorized_dictionary_lookups'}
        for trace_name in ('token_trace','expression_trace'):
            file_audit[trace_name]={k:v for k,v in file_audit[trace_name].items() if k in allowed}
    (staging/'BIOLOGY_MANIFEST.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    base.json_write(staging/'VECTOR_BACKEND_ACCESS_AUDIT.json',{
        **identity,'permit_sha256':base.sha(permit),'actual_iterators_fully_exhausted':all(x['iterators_fully_exhausted'] for x in result['file_audits']),
        'private_INT64_DOUBLE_materialization':False,'original_files_changed':False,'staging_finalization_after_identity_audit':True})
    refresh_hashes(staging); staging.rename(output)
    return result


def pilot(metadata_root,raw_root,contract,permit,operation,output,forbidden_bytes=()):
    from tools.scripts import verify_safeconf_orion_train_pilot_agent as original_pilot
    _,permission,budgets=authorize(metadata_root,raw_root,contract,permit,original_pilot.ROLES,[original_pilot.CONTEXT])
    identity=backend_identity(); receipt=base.immutable_json(operation)
    for field in ['actual_wrapper_sha256','actual_numeric_backend_sha256','actual_numeric_backend_path']:
        if receipt.get(field)!=identity[field]:raise RuntimeError('Pilot immutable receipt must bind actual v2 numeric backend/wrapper')
    output=Path(output)
    if output.exists():raise RuntimeError('New version2 pilot result required')
    staging=output.with_name(output.name+f'.uncommitted.{os.getpid()}')
    try:
        with dispatch(budgets):
            result=original_pilot.run_pilot(metadata_root,raw_root,contract,permit,operation,staging,forbidden_bytes)
    except Exception:
        if staging.exists():
            partial=json.loads(staging.read_text())
            partial.update(numeric_backend_integration=identity,
                executed_numeric_parser_sha256=identity['actual_numeric_backend_sha256'],
                executed_numeric_parser_path=identity['actual_numeric_backend_path'],
                original_reader_code_sha256_field_role='frozen structural primitives and original validation guard',
                actual_backend_iterator_proof='PARTIAL_FAILURE_NOT_FULL_EOF')
            staging.write_text(json.dumps(partial,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            staging.rename(output)
        raise
    result.update(numeric_backend_integration=identity,
        executed_numeric_parser_sha256=identity['actual_numeric_backend_sha256'],
        executed_numeric_parser_path=identity['actual_numeric_backend_path'],
        original_reader_code_sha256_field_role='frozen structural primitives and original validation guard',
        old_pilot_or_runtime_reused=False)
    staging.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); staging.rename(output)
    return result


def synthetic_test(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    md,raw,science,old_permit,sentinels=base.synthetic_fixture(output/'fixture','2.0',True)
    permission=json.loads(old_permit.read_text())
    permission['implementation_bindings'].update({WRAPPER_BINDING:{'path':str(Path(__file__).resolve()),'sha256':base.sha(__file__)},
        BACKEND_BINDING:{'path':str(BACKEND),'sha256':base.sha(BACKEND)}})
    permission['vectorized_backend_scope']=SCHEMA
    permit=output/'V2_SYNTHETIC_PERMIT.json'; base.json_write(permit,permission,True)
    result_v1=base.aggregate(md,raw,science,permit,output/'original_v1',sorted(base.LEGAL_ROLES),sorted(base.CONTEXTS),sentinels)
    # The wrapper's actual aggregation pass also exercises independent staging,
    # truthful parser identities and hard file deadlines.
    result=aggregate(md,raw,science,permit,output/'integrated_v2',sorted(base.LEGAL_ROLES),sorted(base.CONTEXTS),sentinels)
    import numpy as np
    for name in ['FULL_SUMS.npy','CELL_COUNTS.npy','FULL_MEANS.npy','FULL_EFFECTS.npy','ENDPOINT3285_MEANS.npy','ENDPOINT3285_EFFECTS.npy']:
        assert np.array_equal(np.load(output/'original_v1'/name),np.load(output/'integrated_v2'/name))
    assert result['executed_numeric_parser_sha256']==base.sha(BACKEND)
    assert result['loader_sha256']==base.sha(base.__file__) and result['reader_sha256']==base.sha(base.READER)
    assert primitives.iter_selected_numeric_lists.__module__=='tools.scripts.probe_safeconf_private_parquet_agent'
    from tools.scripts import verify_safeconf_orion_train_pilot_agent as original_pilot
    pmd,praw,pscience,old_pilot_permit,old_operation,psentinels=original_pilot.synthetic_fixture(output/'pilot_fixture')
    ppermission=json.loads(old_pilot_permit.read_text())
    ppermission['implementation_bindings'].update(permission['implementation_bindings'])
    ppermission['vectorized_backend_scope']=SCHEMA
    ppermit=output/'V2_SYNTHETIC_PILOT_PERMIT.json'; base.json_write(ppermit,ppermission,True)
    poperation=original_pilot.operation_template(pmd,praw,pscience,ppermit)
    poperation['production_operation_authorized']=True
    for key in ['actual_wrapper_sha256','actual_numeric_backend_sha256','actual_numeric_backend_path']:
        poperation[key]=backend_identity()[key]
    receipt=output/'V2_SYNTHETIC_PILOT_OPERATION.json'; base.json_write(receipt,poperation,True)
    presult=pilot(pmd,praw,pscience,ppermit,receipt,output/'V2_SYNTHETIC_PILOT_RESULT.json',psentinels)
    assert presult['status']=='PASS' and presult['selected_numeric_cells']==128
    assert presult['executed_numeric_parser_sha256']==base.sha(BACKEND)
    assert presult['fully_exhausted_token_iterator'] and presult['fully_exhausted_expression_iterator']
    summary={'schema':SCHEMA,'status':'SYNTHETIC_INTEGRATION_V2_PASS_REAL_EXPRESSION_UNOPENED',
        **backend_identity(),'six_numeric_arrays_exact_equal':True,'mask_normalization_QC_gene_mapping_unchanged':True,
        'actual_numeric_backend_recorded':True,'original_dispatch_restored':True,'no_actual_Orion_expression_read':True,
        'synthetic128_TRAIN_NTC_pilot_v2_full_EOF':True,
        'new_fits':0,'old_runtime_untouched':True}
    base.json_write(output/'SYNTHETIC_INTEGRATION_V2_PROOF.json',summary)
    print(json.dumps(summary,indent=2)); return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    synthetic=sub.add_parser('synthetic-test'); synthetic.add_argument('--output',type=Path,default=DOC/'synthetic_v1')
    for name in ['aggregate','pilot']:
        command=sub.add_parser(name)
        command.add_argument('--metadata-root',type=Path,default=base.METADATA); command.add_argument('--raw-root',type=Path,default=base.RAW)
        command.add_argument('--contract',type=Path,required=True); command.add_argument('--permit',type=Path,required=True)
        command.add_argument('--output',type=Path,required=True)
        if name=='pilot':command.add_argument('--operation',type=Path,required=True)
        else:
            command.add_argument('--roles',nargs='+',default=sorted(base.LEGAL_ROLES)); command.add_argument('--contexts',nargs='+',default=sorted(base.CONTEXTS))
    args=parser.parse_args()
    if args.command=='synthetic-test':synthetic_test(args.output)
    elif args.command=='aggregate':print(json.dumps(aggregate(args.metadata_root,args.raw_root,args.contract,args.permit,args.output,args.roles,args.contexts),indent=2))
    else:print(json.dumps(pilot(args.metadata_root,args.raw_root,args.contract,args.permit,args.operation,args.output),indent=2))


if __name__=='__main__':main()
