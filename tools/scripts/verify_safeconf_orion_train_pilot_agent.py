#!/usr/bin/env python3
"""Exactly 64 TRAIN + 64 TRAIN NTC cells; semantic pilot without model fitting.

Production uses the existing immutable access permit and an additional exact
pilot operation receipt. Synthetic testing never reads the real data root.
Only TRAIN/control aggregate count and sum-gap diagnostics are persisted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import resource
import signal
import struct
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import build_safeconf_orion_allowed_biology_agent as biology

DOC = biology.DOC
SCHEMA = 'safeconf_orion_train_semantic_pilot_v1'
FILE = 'data/HCT116_Batch1.parquet'
CONTEXT = 'HCT116'
N_PER_ROLE = 64
ROLES = ('TRAIN', biology.CONTROL_ROLE)
MAX_SECONDS = 600
MAX_MEMORY = 6*1024**3
SELECTION = 'first64_each_exact_frozen_TRAIN_and_TRAIN_CONTROL_SOURCE_SCOPE_in_physical_row_order'


def selected_metadata(metadata_root):
    identities = json.loads((Path(metadata_root)/'FILE_METADATA_IDENTITY.json').read_text())
    matches = [row for row in identities if row['path']==FILE and row['context']==CONTEXT]
    if len(matches)!=1:
        raise RuntimeError('Exactly pinned HCT116_Batch1 identity required; no replacement shard')
    identity = matches[0]
    md, _ = biology.read_metadata(identity, set(ROLES))
    rows = {}
    for role in ROLES:
        rows[role] = [i for i,r in enumerate(md['row_role']) if r==role][:N_PER_ROLE]
        if len(rows[role])!=N_PER_ROLE:
            raise RuntimeError('Exactly64 cells per role required; no adaptive pilot selection')
    allowed = set(rows[ROLES[0]]+rows[ROLES[1]])
    mask = [i in allowed for i in range(identity['rows'])]
    for i in allowed:
        if md['row_role'][i] not in ROLES:
            raise RuntimeError('Pilot role selection includes a private or validation row')
    return identity, md, mask, rows


def operation_template(metadata_root, raw_root, contract_path, permit_path):
    identity, md, mask, rows = selected_metadata(metadata_root)
    return {'schema':SCHEMA,'purpose':'TRAIN_RAW_UMI_SEMANTIC_PILOT_ONLY',
       'metadata_root':str(Path(metadata_root).resolve()),'raw_root':str(Path(raw_root).resolve()),
       'access_permit_path':str(Path(permit_path).resolve()),'access_permit_sha256':biology.sha(permit_path),
       'scientific_contract_path':str(Path(contract_path).resolve()),'scientific_contract_sha256':biology.sha(contract_path),
       'pilot_code_path':str(Path(__file__).resolve()),'pilot_code_sha256':biology.sha(__file__),
       'loader_code_sha256':biology.sha(biology.__file__),'reader_code_sha256':biology.sha(biology.READER),
       'file':FILE,'context':CONTEXT,'roles':list(ROLES),'cells_per_role':N_PER_ROLE,'selection_rule':SELECTION,
       'metadata_parquet_sha256':identity['metadata_parquet_sha256'],
       'authorized_physical_row_mask_sha256':__import__('hashlib').sha256(np.asarray(mask,dtype=np.uint8).tobytes()).hexdigest(),
       'selected_original_row_indices':rows,'publisher_whole_file_bytes':identity['publisher_whole_file_bytes'],
       'publisher_LFS_sha256':identity['publisher_LFS_sha256'],'max_seconds':MAX_SECONDS,'max_memory_bytes':MAX_MEMORY,
       'model_fits_permitted':False,'prediction_calls_permitted':False,'cell_vector_persistence_permitted':False,
       'test_validation_numeric_materialization_permitted':False,'production_operation_authorized':False}


def operation_check(operation_path, expected):
    operation=biology.immutable_json(operation_path)
    for key,value in expected.items():
        if key=='production_operation_authorized':
            continue
        if operation.get(key)!=value:
            raise RuntimeError(f'Exact immutable pilot operation differs: {key}')
    if operation.get('production_operation_authorized') is not True:
        raise RuntimeError('Separate exact pilot operation receipt is required')
    return operation


def sanitized_traces(result):
    """Persist byte/budget provenance and selected numeric counters only."""
    allowed={'physical_numeric_chunk_bytes_read','opaque_numeric_bytes_decompressed',
             'max_compressed_chunk_bytes','max_page_uncompressed_bytes','max_page_level_entries',
             'numeric_materialization_count','authorized_dictionary_lookups'}
    for name in ('token_trace','expression_trace'):
        if name in result:
            result[name]={key:value for key,value in result[name].items() if key in allowed}


@contextmanager
def limits():
    """Six-GiB address-space ceiling and ten-minute wall deadline on Linux."""
    previous_limit=resource.getrlimit(resource.RLIMIT_AS)
    cap=min(MAX_MEMORY,previous_limit[1]) if previous_limit[1]!=resource.RLIM_INFINITY else MAX_MEMORY
    previous_handler=signal.getsignal(signal.SIGALRM)
    previous_timer=signal.getitimer(signal.ITIMER_REAL)
    def alarm(signum, frame):
        raise TimeoutError('Pilot600second deadline exceeded; incomplete iterator proof')
    signal.signal(signal.SIGALRM,alarm)
    resource.setrlimit(resource.RLIMIT_AS,(cap,previous_limit[1]))
    signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        resource.setrlimit(resource.RLIMIT_AS,previous_limit)
        signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0:
            signal.setitimer(signal.ITIMER_REAL,*previous_timer)


def run_pilot(metadata_root,raw_root,contract_path,permit_path,operation_path,output,forbidden_bytes=()):
    output=Path(output)
    if output.exists():
        raise RuntimeError('Pilot result cannot overwrite an existing artifact')
    output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    result={'schema':SCHEMA,'status':'RUNNING','real_expression_accessed':False,
            'selected_numeric_cells':0,'fully_exhausted_token_iterator':False,
            'fully_exhausted_expression_iterator':False,'explicit_partial_reader_proof':True,
            'model_fits':0,'prediction_calls':0,'cell_vectors_saved':False,
            'validation_test_DEV_numeric_materialization':False,
            'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with limits():
            contract,permit,budgets=biology.authorization(metadata_root,raw_root,contract_path,permit_path,ROLES,[CONTEXT])
            expected=operation_template(metadata_root,raw_root,contract_path,permit_path)
            operation=operation_check(operation_path,expected)
            identity,md,mask,selected=selected_metadata(metadata_root)
            full,endpoint,columns,symbol_map=biology.gene_axes(metadata_root,bool(permit.get('synthetic_only')))
            raw_path=biology.checked_child(raw_root,identity['path'])
            if not raw_path.is_file() or raw_path.stat().st_size!=identity['publisher_whole_file_bytes']:
                raise RuntimeError('Pilot raw file incomplete or size differs')
            if biology.sha(raw_path)!=identity['publisher_LFS_sha256']:
                raise RuntimeError('Full raw SHA differs; no numeric expression conversion permitted')
            from tools.scripts import probe_safeconf_private_parquet_agent as reader
            footer=reader.fastparquet.ParquetFile(raw_path)
            if len(footer.row_groups)!=1 or footer.row_groups[0].num_rows!=identity['rows']:
                raise RuntimeError('Pinned physical rowgroup/row identity differs')
            for name,kind in [('gene_token_id',reader.pt.Type.INT64),('gene_expression',reader.pt.Type.DOUBLE)]:
                leaves=[c.meta_data for c in footer.row_groups[0].columns if c.meta_data.path_in_schema[0]==name]
                if len(leaves)!=1 or leaves[0].type!=kind:
                    raise RuntimeError('Pilot requires exact INT64tokens and DOUBLEexpressions')
            token_trace,value_trace=biology.trace_new(budgets),biology.trace_new(budgets)
            result.update(access_permit_sha256=biology.sha(permit_path),operation_receipt_sha256=biology.sha(operation_path),
               scientific_contract_sha256=biology.sha(contract_path),pilot_code_sha256=biology.sha(__file__),
               loader_code_sha256=biology.sha(biology.__file__),reader_code_sha256=biology.sha(biology.READER),
               file=FILE,context=CONTEXT,selected_roles=list(ROLES),selection_rule=SELECTION,
               selected_original_row_indices=selected,authorized_physical_row_mask_sha256=expected['authorized_physical_row_mask_sha256'],
               raw_sha256=identity['publisher_LFS_sha256'],metadata_parquet_sha256=identity['metadata_parquet_sha256'],
               synthetic_only=bool(permit.get('synthetic_only')),full_measured_gene_count=len(full),
               resource_limits={'wall_seconds':MAX_SECONDS,'address_space_bytes':MAX_MEMORY,
                  'numeric_column_chunk_bytes_each':budgets['max_compressed_chunk_bytes'],
                  'page_uncompressed_bytes_each':budgets['max_page_uncompressed_bytes'],
                  'paired_numeric_column_buffers':2})
            token_forbidden,value_forbidden=forbidden_bytes if forbidden_bytes else ((),())
            tokens=reader.iter_selected_numeric_lists(raw_path,'gene_token_id',mask,token_trace,token_forbidden)
            values=reader.iter_selected_numeric_lists(raw_path,'gene_expression',mask,value_trace,value_forbidden)
            sentinel=object()
            rows_seen=[]
            diagnostics={role:{'n_cells':0,'numeric_UMI_values_checked':0,'max_integer_absolute_residual':0.,
                      'max_rawsum_absolute_gap':0.,'max_rawsum_relative_gap':0.,'rawsum_mismatch_cells':0} for role in ROLES}
            result['count_sum_gap_diagnostics']=diagnostics
            result['token_trace']=token_trace; result['expression_trace']=value_trace
            result['real_expression_accessed']=not bool(permit.get('synthetic_only'))
            for token_row,value_row in itertools.zip_longest(tokens,values,fillvalue=sentinel):
                if token_row is sentinel or value_row is sentinel:
                    raise RuntimeError('Paired numeric iterators differ in authorized row coverage')
                ti,ids=token_row; vi,raw=value_row
                if ti!=vi or not mask[ti] or md['row_role'][ti] not in ROLES:
                    raise RuntimeError('Exact token/expression permitted physical row pairing failed')
                if ids is None or raw is None or len(ids)!=len(raw) or not len(ids):
                    raise RuntimeError('Null/empty/truncated numeric lists cannot represent a positive measured library')
                if len(ids)>len(full) or any(type(t) is not int for t in ids):
                    raise RuntimeError('Token list primitive type/cardinality differs')
                indices=np.asarray(ids,dtype=np.int64)
                if (indices<0).any() or (indices>=len(full)).any() or len(np.unique(indices))!=len(indices):
                    raise RuntimeError('Unknown/duplicate token; no gene zero imputation')
                if any(x is None for x in raw):
                    raise RuntimeError('Null numeric measurement')
                counts=np.asarray(raw,dtype=float)
                if not np.isfinite(counts).all() or (counts<0).any():
                    raise RuntimeError('Raw counts must be finite nonnegative UMI')
                integer_gap=float(np.max(np.abs(counts-np.rint(counts))))
                if integer_gap>1e-6:
                    raise RuntimeError('Raw counts are not integer UMI within1e-6')
                total=float(md['total_counts'][ti]); raw_sum=float(counts.sum())
                if not np.isfinite(raw_sum) or raw_sum<=0:
                    raise RuntimeError('Positive full library requires positive complete measured counts')
                absolute_gap=abs(raw_sum-total); relative_gap=absolute_gap/total
                diagnostic=diagnostics[md['row_role'][ti]]
                diagnostic['n_cells']+=1; diagnostic['numeric_UMI_values_checked']+=len(counts)
                diagnostic['max_integer_absolute_residual']=max(diagnostic['max_integer_absolute_residual'],integer_gap)
                diagnostic['max_rawsum_absolute_gap']=max(diagnostic['max_rawsum_absolute_gap'],absolute_gap)
                diagnostic['max_rawsum_relative_gap']=max(diagnostic['max_rawsum_relative_gap'],relative_gap)
                consistent=bool(np.isclose(raw_sum,total,rtol=1e-6,atol=1e-6))
                diagnostic['rawsum_mismatch_cells']+=int(not consistent)
                rows_seen.append(ti); result['selected_numeric_cells']=len(rows_seen)
                if not consistent:
                    raise RuntimeError('TRAIN/NTC rawsum differs from official full library; denominator unchanged')
                biology.check_memory({'max_resident_bytes':MAX_MEMORY})
            # zip_longest returned only after both generators reached EOF,
            # including opaque/private trailing pages after the selected rows.
            if rows_seen!=np.flatnonzero(mask).tolist() or any(d['n_cells']!=N_PER_ROLE for d in diagnostics.values()):
                raise RuntimeError('Pilot does not cover exactly frozen64+64 rows')
            result.update(status='PASS',fully_exhausted_token_iterator=True,
               fully_exhausted_expression_iterator=True,explicit_partial_reader_proof=False,
               whole_file_opaque_sha_verified=True,all_private_trailing_pages_structurally_skipped=True,
               numeric_rows_materialized_only_selected_TRAIN_or_CONTROL=True,
               raw_UMI_semantics='finite nonnegative integer; sparse known genes measured zero; fullrawsum matches officialtotal',
               rawsum_check_atol=1e-6,rawsum_check_rtol=1e-6,
               normalization_performed=False,denominator_changed=False)
    except Exception as error:
        result.update(status='FAILED',error_type=type(error).__name__,error=str(error),
           explicit_partial_reader_proof=True,fully_exhausted_token_iterator=False,
           fully_exhausted_expression_iterator=False)
        result['elapsed_seconds']=round(time.monotonic()-started,3)
        result['peak_resident_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        sanitized_traces(result)
        biology.json_write(output,result)
        raise
    result['elapsed_seconds']=round(time.monotonic()-started,3)
    result['peak_resident_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    sanitized_traces(result)
    biology.json_write(output,result)
    return result


def synthetic_fixture(base,bad_counts=False):
    metadata=base/'metadata'; raw_root=base/'raw'; metadata.mkdir(parents=True); (raw_root/'data').mkdir(parents=True)
    full=pd.DataFrame({'ensembl_id':['E0','E1','E2'],'gene_name':['G1','G2','G3'],'gene_token_id':[0,1,2]})
    pq.write_table(pa.Table.from_pandas(full,preserve_index=False),metadata/'gene_metadata.parquet')
    pd.DataFrame({'axis_index':[0,1],'source_index':[0,2],'gene_name':['G1','G3'],
                'orion_gene_token_id':[0,2],'orion_ensembl_id':['E0','E2']}).to_csv(metadata/'GENE_MANIFEST.csv',index=False)
    biology.json_write(metadata/'PREPARATION_POLICY.json',{'synthetic_only':True},True)
    biology.json_write(metadata/'SYNTHETIC_FIXTURE_ONLY.json',{'synthetic_only':True},True)
    roles=[]; ids=[]; values=[]; totals=[]; genes=[]
    forbidden_token=918273645; forbidden_value=918273645.125
    for i in range(70):
        scale=float(i%5+1)
        for role in [biology.CONTROL_ROLE,'TRAIN','VALIDATION','TEST','DEV_EXPOSED_CELL']:
            roles.append(role)
            genes.append('Non-Targeting' if role==biology.CONTROL_ROLE else 'G1')
            if role==biology.CONTROL_ROLE:
                ids.append([1]); values.append([scale]); totals.append(scale)
            elif role=='TRAIN':
                ids.append([0,2]); values.append([3.*scale,scale]); totals.append(4.*scale)
            else:
                ids.append([forbidden_token]); values.append([forbidden_value]); totals.append(1.)
    if bad_counts:
        values[1]=[2.5,1.5]
    # Even same-role cells after the first64 contain forbidden sentinels: the
    # proof binds exact selected rows rather than a broad role-only mask.
    for row in range(64*5,len(roles)):
        ids[row]=[forbidden_token]; values[row]=[forbidden_value]
    raw=raw_root/FILE
    pq.write_table(pa.table({'gene_token_id':pa.array(ids,type=pa.list_(pa.int64())),
                           'gene_expression':pa.array(values,type=pa.list_(pa.float64()))}),raw,
                   compression='zstd',use_dictionary=True,data_page_version='2.0',row_group_size=len(roles),
                   data_page_size=128,write_batch_size=2)
    mdpath=metadata/'row_metadata.parquet'
    pq.write_table(pa.table({'original_row_index':list(range(len(roles))),'original_row_group':[0]*len(roles),
       'row_role':roles,'gene_target':genes,'sample':['HCT116_Batch1']*len(roles),
       'cell_barcode':[f'SYNTHETIC:{i}' for i in range(len(roles))],'total_counts':totals,
       'biological_unit':[f'SyntheticStudy|{gene}|HCT116' for gene in genes],
       'fixed_metadata_QC_pass':[True]*len(roles)}),mdpath)
    mdpath.chmod(0o444)
    identity={'path':FILE,'context':CONTEXT,'sample':'HCT116_Batch1','rows':len(roles),'row_groups':1,
              'publisher_whole_file_bytes':raw.stat().st_size,'publisher_LFS_sha256':biology.sha(raw),
              'metadata_parquet_path':str(mdpath),'metadata_parquet_sha256':biology.sha(mdpath)}
    biology.json_write(metadata/'FILE_METADATA_IDENTITY.json',[identity],True)
    contract=base/'SYNTHETIC_CONTRACT.json'
    biology.json_write(contract,{'synthetic_only':True,
      'immutable_dependencies':[{'path':str(metadata/name),'sha256':biology.sha(metadata/name)} for name in ['FILE_METADATA_IDENTITY.json','GENE_MANIFEST.csv','gene_metadata.parquet','PREPARATION_POLICY.json']],
      'privacy':{'current_permitted_roles':list(ROLES),'private_numeric_materialization':False},
      'normalization':{'formula':'mean_cell(log1p(4000 * raw_gene_UMI / official_full_library_total_counts))','unknown_token_policy':'fail_closed'}},True)
    permit=base/'SYNTHETIC_ACCESS_PERMIT.json'
    biology.json_write(permit,{'synthetic_only':True,'expression_row_materialization_permitted':True,
      'test_numeric_materialization_permitted':False,'authorized_roles':list(ROLES),'authorized_contexts':[CONTEXT],
      'reader_guarantee':biology.GUARANTEE,'frozen_method_contract_path':str(contract),'frozen_method_contract_sha256':biology.sha(contract),
      'metadata_root':str(metadata),'raw_root':str(raw_root),
      'metadata_bindings':{name:biology.sha(metadata/name) for name in ['FILE_METADATA_IDENTITY.json','GENE_MANIFEST.csv','gene_metadata.parquet','PREPARATION_POLICY.json']},
      'implementation_bindings':{'loader':{'path':str(Path(biology.__file__).resolve()),'sha256':biology.sha(biology.__file__)},
                                 'reader':{'path':str(biology.READER),'sha256':biology.sha(biology.READER)}},
      'resource_budgets':biology.DEFAULT_BUDGETS,'library_sum_check_roles':list(ROLES)},True)
    operation=base/'SYNTHETIC_PILOT_OPERATION.json'
    expected=operation_template(metadata,raw_root,contract,permit); expected['production_operation_authorized']=True
    biology.json_write(operation,expected,True)
    return metadata,raw_root,contract,permit,operation,((struct.pack('<q',forbidden_token),),(struct.pack('<d',forbidden_value),))


def synthetic_test(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    args=synthetic_fixture(output/'valid')
    result=run_pilot(*args[:5],output/'TRAIN_PILOT_SYNTHETIC_PASS.json',args[5])
    assert result['status']=='PASS' and result['selected_numeric_cells']==128
    assert result['fully_exhausted_token_iterator'] and result['fully_exhausted_expression_iterator']
    assert result['normalization_performed'] is False and result['cell_vectors_saved'] is False
    assert all(x['n_cells']==64 and x['max_rawsum_absolute_gap']==0 for x in result['count_sum_gap_diagnostics'].values())
    bad=synthetic_fixture(output/'fractional',True)
    try:
        run_pilot(*bad[:5],output/'TRAIN_PILOT_SYNTHETIC_REJECT.json',bad[5])
        raise AssertionError('Fractional UMI unexpectedly accepted')
    except RuntimeError as error:
        assert 'integer UMI' in str(error)
    rejected=json.loads((output/'TRAIN_PILOT_SYNTHETIC_REJECT.json').read_text())
    assert rejected['status']=='FAILED' and rejected['explicit_partial_reader_proof']
    summary={'schema':SCHEMA,'status':'SYNTHETIC_PILOT_PASS_REAL_EXPRESSION_UNOPENED',
        'pilot_code_sha256':biology.sha(__file__),'loader_code_sha256':biology.sha(biology.__file__),
        'reader_code_sha256':biology.sha(biology.READER),'real_expression_accessed':False,
        'exact128cell_selection_passed':True,'private_and_unselected_same_role_sentinels_not_converted':True,
        'full_iterator_exhaustion_passed':True,'fractional_UMI_rejected_with_explicit_partial_proof':True,
        'vectors_saved':False,'model_fits':0,'prediction_calls':0}
    biology.json_write(output/'TRAIN_PILOT_SYNTHETIC_RESULT.json',summary)
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    synthetic=sub.add_parser('synthetic-test'); synthetic.add_argument('--output',type=Path,default=DOC/'train_pilot_synthetic_v1')
    for name in ['plan','run']:
        command=sub.add_parser(name)
        command.add_argument('--metadata-root',type=Path,default=biology.METADATA)
        command.add_argument('--raw-root',type=Path,default=biology.RAW)
        command.add_argument('--contract',type=Path,required=True)
        command.add_argument('--permit',type=Path,required=True)
        command.add_argument('--output',type=Path,required=True)
        if name=='run':
            command.add_argument('--operation',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='synthetic-test': synthetic_test(args.output)
    elif args.command=='plan':
        template=operation_template(args.metadata_root,args.raw_root,args.contract,args.permit)
        biology.json_write(args.output,template)
        print(json.dumps({'status':'METADATA_ONLY_SELECTION_TEMPLATE','code_sha256':template['pilot_code_sha256'],
                          'selected_cells':128,'raw_expression_accessed':False,'operation_authorized':False}))
    else:
        result=run_pilot(args.metadata_root,args.raw_root,args.contract,args.permit,args.operation,args.output)
        print(json.dumps({key:result[key] for key in ['status','selected_numeric_cells','fully_exhausted_token_iterator','fully_exhausted_expression_iterator','elapsed_seconds','peak_resident_bytes']}))


if __name__=='__main__': main()
