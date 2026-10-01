#!/usr/bin/env python3
"""Prepare hashed blind R inputs from authorized biology aggregates.

plan-queries reads frozen metadata only. bridge reads numeric TRAIN/NTC mean
rows only and never starts R fitting. Real use requires the final technical
permit binding this bridge, the R CLI and the metadata-only query scope.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import build_safeconf_orion_allowed_biology_agent as biology
R_CLI = ROOT / 'tools/scripts/run_safeconf_orion_lm_blind.R'
CONTROL = 'TRAIN_CONTROL_SOURCE_SCOPE'
CONTEXTS = ('HCT116', 'HEK293T')
QUERY_COLUMNS = ['query_id','target_gene_id','target_gene_symbol','context_id','role']

ORIGINAL_BRIDGE = ROOT / 'tools/scripts/bridge_safeconf_orion_lm_inputs_agent.py'
ORIGINAL_BRIDGE_SHA = 'dcc7d6afcd812d89ddb372e511cb9607e216da8c1f57e484317db4b3eda52a38'
AMENDMENT_SCHEMA = 'safeconf_orion_integral_token_bridge_technical_amendment_v1'


def integral_token_id(value):
    """Accept exact integral CSV decimals, never round/truncate tokens."""
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError('Boolean cannot identify a canonical gene token')
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise RuntimeError('Canonical token must be a finite exact integer') from None
    if not number.is_finite() or number != number.to_integral_value() or not 0 <= number < 38606:
        raise RuntimeError('Canonical token must be an exact integer in the frozen full axis')
    return int(number)


def technical_amendment(path, permit_path, contract_path, biology_root, query_scope):
    if path is None:
        raise RuntimeError('Explicit token-format technical amendment path required')
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_mode & 0o222:
        raise RuntimeError('Immutable explicit token-format technical amendment required')
    value = json.loads(path.read_text())
    if (value.get('schema') != AMENDMENT_SCHEMA or value.get('status') != 'ROOT_APPROVED_TECHNICAL_CONTINUATION'
        or value.get('new_model_fits') != 0 or value.get('scientific_method_changed') is not False
        or value.get('eligibility_or_axes_or_values_changed') is not False
        or value.get('TEST_numeric_access_authorized') is not False):
        raise RuntimeError('Exact no-method-change/no-TEST technical amendment required')
    expected = {'base_permit': Path(permit_path), 'scientific_contract': Path(contract_path),
        'original_bridge': ORIGINAL_BRIDGE, 'executed_bridge': Path(__file__),
        'biology_manifest': Path(biology_root) / 'BIOLOGY_MANIFEST.json',
        'query_metadata': Path(biology_root) / 'QUERY_METADATA_ONLY.csv', 'query_scope': Path(query_scope), 'lm_cli': R_CLI}
    for name, actual in expected.items():
        item = value.get(name, {})
        if Path(item.get('path', '')).resolve() != actual.resolve() or item.get('sha256') != sha(actual):
            raise RuntimeError('Exact technical amendment binding differs: ' + name)
    if sha(ORIGINAL_BRIDGE) != ORIGINAL_BRIDGE_SHA:
        raise RuntimeError('Original registered bridge must remain unchanged')
    return {'path': str(path), 'sha256': sha(path), 'original_bridge_sha256': ORIGINAL_BRIDGE_SHA,
        'actual_bridge_sha256': sha(__file__), 'scope': 'exact_integral_CSV_token_text_only_no_numeric_data_or_selection_change'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def write_json(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def tsv(path, frame):
    frame.to_csv(path,sep='\t',index=False,float_format='%.17g')

def kv(path, values):
    tsv(path,pd.DataFrame({'key':list(values),'value':[str(x) for x in values.values()]}))

def read_csv(path):
    return pd.read_csv(path,keep_default_na=False)

def axis_map(full):
    counts=Counter(full.gene_name)
    return {r.gene_name:r for r in full.itertuples(index=False) if r.gene_name and counts[r.gene_name]==1}

def check_axes(full, endpoint):
    if len(full)!=38606 or len(endpoint)!=3285 or not full.ensembl_id.is_unique or not full.gene_token_id.is_unique:
        raise RuntimeError('Exact full38606 and endpoint3285 axes required')
    if not np.array_equal(full.gene_token_id.to_numpy(),np.arange(38606)):
        raise RuntimeError('Full token order must be contiguous from zero')
    if not np.array_equal(endpoint.axis_index.to_numpy(),np.arange(3285)) or not endpoint.gene_name.is_unique:
        raise RuntimeError('Frozen endpoint order or unique symbols differ')
    mapping=axis_map(full)
    for r in endpoint.itertuples(index=False):
        m=mapping.get(r.gene_name)
        if m is None or m.ensembl_id!=r.orion_ensembl_id or m.gene_token_id!=r.orion_gene_token_id:
            raise RuntimeError('Endpoint canonical mapping differs')
    return mapping

def plan_queries(metadata_root, output, study='Huang2025_XAtlasOrion'):
    """TEST identities come solely from frozen split times predefined contexts."""
    md=Path(metadata_root).resolve();out=Path(output).resolve()
    if out.exists():raise RuntimeError('New immutable query scope version required')
    full=pq.read_table(md/'gene_metadata.parquet',columns=['ensembl_id','gene_name','gene_token_id']).to_pandas().sort_values('gene_token_id').reset_index(drop=True)
    endpoint=read_csv(md/'GENE_MANIFEST.csv');mapping=check_axes(full,endpoint)
    split=read_csv(md/'GENE_SPLIT.csv')
    if not split.gene.is_unique or not set(split.gene_role)<={'TRAIN','VALIDATION','TEST'}:
        raise RuntimeError('Frozen gene split is invalid')
    query=[];omitted=[]
    for r in split.itertuples(index=False):
        if r.gene_role not in {'VALIDATION','TEST'}:continue
        m=mapping.get(r.gene)
        if m is None:
            omitted.append({'gene':r.gene,'role':r.gene_role,'reason':'absent_or_ambiguous_static_full_axis_mapping'});continue
        for context in CONTEXTS:
            query.append({'query_id':f'{study}|{r.gene}|{context}','target_gene_id':m.ensembl_id,
                'target_gene_symbol':r.gene,'context_id':context,'role':r.gene_role})
    # Only VALIDATION metadata rows are returned. No TEST cell counts/features
    # or TEST row-presence/coverage distributions are computed.
    counts=Counter();metadata_hashes=[]
    identities=json.loads((md/'FILE_METADATA_IDENTITY.json').read_text())
    for identity in identities:
        path=Path(identity['metadata_parquet_path'])
        if sha(path)!=identity['metadata_parquet_sha256']:raise RuntimeError('Frozen row metadata hash differs')
        table=pq.read_table(path,columns=['gene_target','biological_unit','fixed_metadata_QC_pass'],
            filters=[('row_role','=','VALIDATION')],use_threads=False)
        for r in table.to_pylist():
            if r['fixed_metadata_QC_pass'] and r['gene_target'] in mapping:
                counts[(identity['context'],r['gene_target'])]+=1
                if r['biological_unit']!=f'{study}|{r["gene_target"]}|{identity["context"]}':
                    raise RuntimeError('Frozen biological query identity differs')
        metadata_hashes.append({'path':str(path),'sha256':identity['metadata_parquet_sha256']})
    q=pd.DataFrame(query,columns=QUERY_COLUMNS)
    if q.empty or not q.query_id.is_unique:raise RuntimeError('Empty or duplicate query scope')
    eligible=[]
    for r in q[q.role=='VALIDATION'].itertuples(index=False):
        n=counts[(r.context_id,r.target_gene_symbol)]
        if n>=30:eligible.append({**r._asdict(),'n_cells_metadata':n})
    out.mkdir(parents=True)
    tsv(out/'METADATA_ONLY_QUERY_SCOPE.tsv',q)
    tsv(out/'VALIDATION_ELIGIBILITY.tsv',pd.DataFrame(eligible,columns=QUERY_COLUMNS+['n_cells_metadata']))
    write_json(out/'STATIC_MAPPING_OMISSIONS.json',omitted)
    report={'schema':'safeconf_orion_metadata_only_lm_query_scope_v1','metadata_root':str(md),
       'gene_split_sha256':sha(md/'GENE_SPLIT.csv'),'full_gene_metadata_sha256':sha(md/'gene_metadata.parquet'),
       'endpoint_gene_manifest_sha256':sha(md/'GENE_MANIFEST.csv'),'file_metadata_identity_sha256':sha(md/'FILE_METADATA_IDENTITY.json'),
       'query_scope_sha256':sha(out/'METADATA_ONLY_QUERY_SCOPE.tsv'),
       'validation_eligibility_sha256':sha(out/'VALIDATION_ELIGIBILITY.tsv'),
       'query_rule':'ALL uniquely mapped VALIDATION and TEST split targets times HCT116/HEK293T; no TEST presence/count eligibility preselection',
       'validation_rule':'only allowed frozen VALIDATION QC metadata count>=30, separate evaluation eligibility list',
       'test_cell_counts_or_distributions_computed':False,'test_numerical_metadata_features_created':False,
       'expression_read':False,'row_metadata_identity_bindings':metadata_hashes,
       'implementation_sha256':sha(__file__),'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(out/'QUERY_SCOPE_MANIFEST.json',report)
    for p in out.iterdir():p.chmod(0o444)
    return report

def verify_artifacts(root):
    records=json.loads((root/'ARTIFACT_HASHES.json').read_text());lookup={}
    for r in records:
        p=root/r['path']
        if Path(r['path']).name!=r['path'] or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
            raise RuntimeError('Biology artifact hash differs')
        lookup[r['path']]=r['sha256']
    for name in ['BIOLOGY_MANIFEST.json','FULL_MEANS.npy','FULL_GENE_AXIS.csv','ENDPOINT_GENE_MANIFEST.csv','QUERY_METADATA_ONLY.csv','CELL_COUNTS.npy']:
        if name not in lookup:raise RuntimeError('Required biology artifact missing from hash manifest')
    return lookup

def verify_numeric_backend_bindings(bio,permit,root,artifact_hashes):
    """Original scientific guards and an executed v2 backend have distinct IDs."""
    identity=bio.get('numeric_backend_integration')
    if identity is None:
        if 'executed_numeric_parser_sha256' in bio or 'executed_numeric_parser_path' in bio:
            raise RuntimeError('Executed parser requires complete integration identity')
        return None
    schema='safeconf_orion_vectorized_numeric_integration_v2'
    if identity.get('schema')!=schema or permit.get('vectorized_backend_scope')!=schema:
        raise RuntimeError('V2 integration scope differs')
    if identity.get('original_validation_loader_sha256')!=bio['loader_sha256'] or identity.get('original_structural_reader_sha256')!=bio['reader_sha256']:
        raise RuntimeError('V2 identity must preserve actual original scientific guard hashes')
    for binding_key,path_key,hash_key in [('biology_integration_v2','actual_wrapper_path','actual_wrapper_sha256'),
        ('numeric_reader_backend_v2','actual_numeric_backend_path','actual_numeric_backend_sha256')]:
        binding=permit.get('implementation_bindings',{}).get(binding_key,{})
        path=Path(identity.get(path_key,''))
        if Path(binding.get('path','')).resolve()!=path.resolve() or binding.get('sha256')!=identity.get(hash_key) or sha(path)!=identity.get(hash_key):
            raise RuntimeError('Exact actual V2 wrapper/backend binding missing or changed')
    if bio.get('executed_numeric_parser_sha256')!=identity['actual_numeric_backend_sha256'] or bio.get('executed_numeric_parser_path')!=identity['actual_numeric_backend_path']:
        raise RuntimeError('Executed V2 numeric parser identity differs')
    name='VECTOR_BACKEND_ACCESS_AUDIT.json'
    if name not in artifact_hashes:raise RuntimeError('V2 backend access audit missing from artifact hashes')
    audit=json.loads((root/name).read_text())
    for key in ['actual_wrapper_path','actual_wrapper_sha256','actual_numeric_backend_path','actual_numeric_backend_sha256']:
        if audit.get(key)!=identity[key]:raise RuntimeError('V2 access audit execution binding differs')
    if audit.get('permit_sha256')!=bio['permit_sha256'] or audit.get('actual_iterators_fully_exhausted') is not True or audit.get('private_INT64_DOUBLE_materialization') is not False:
        raise RuntimeError('V2 access audit permission/private/EOF proof differs')
    if any(x.get('executed_numeric_parser_sha256')!=identity['actual_numeric_backend_sha256'] for x in bio.get('file_audits',[])):
        raise RuntimeError('Per-file actual V2 numeric parser identity differs')
    return {'executed_numeric_parser_sha256':identity['actual_numeric_backend_sha256'],
        'numeric_backend_access_audit_sha256':artifact_hashes[name]}

def bridge(biology_root,metadata_root,contract_path,permit_path,query_scope,output,attempt_id='',max_output_bytes=8*1024**3,amendment_path=None):
    started=time.monotonic();src=Path(biology_root).resolve();md=Path(metadata_root).resolve();out=Path(output).resolve()
    if out.exists():raise RuntimeError('New bridge output version required')
    bio=json.loads((src/'BIOLOGY_MANIFEST.json').read_text());permit_path=Path(permit_path).resolve();contract_path=Path(contract_path).resolve()
    permit=json.loads(permit_path.read_text())
    contract,permit,_=biology.authorization(md,Path(permit['raw_root']),contract_path,permit_path,['TRAIN',CONTROL],CONTEXTS)
    if bio.get('status')!='COMPLETE' or bio.get('scientific_contract_sha256')!=sha(contract_path) or bio.get('permit_sha256')!=sha(permit_path):
        raise RuntimeError('Complete biology identity does not bind scientific contract/permit')
    synthetic=bool(bio.get('synthetic_only'))
    if synthetic!=bool(permit.get('synthetic_only')) or not set(bio.get('roles',[]))<={'TRAIN','VALIDATION',CONTROL}:
        raise RuntimeError('Biology contains forbidden roles or synthetic mismatch')
    if bio.get('private_test_numeric_materialization') is not False or bio.get('missing_unknown_gene_imputation') is not False:
        raise RuntimeError('Biology private/imputation guarantee differs')
    if bio.get('loader_sha256')!=sha(biology.__file__) or bio.get('reader_sha256')!=sha(biology.READER):
        raise RuntimeError('Biology implementation bindings are stale')
    amendment = None
    if not synthetic:
        amendment = technical_amendment(amendment_path,permit_path,contract_path,src,query_scope)
        for name,path in [('bridge',ORIGINAL_BRIDGE),('lm_cli',R_CLI)]:
            binding=permit.get('implementation_bindings',{}).get(name,{})
            if Path(binding.get('path','')).resolve()!=path.resolve() or binding.get('sha256')!=sha(path):
                raise RuntimeError(f'Final technical permit must bind exact {name} implementation')
        scope_binding=permit.get('lm_query_scope',{})
        if Path(scope_binding.get('path','')).resolve()!=Path(query_scope).resolve() or scope_binding.get('sha256')!=sha(query_scope):
            raise RuntimeError('Final technical permit must bind frozen metadata-only query scope')
        if not attempt_id:raise RuntimeError('Prospective upstream attempt ID required; bridge does not start it')
        if not bio.get('file_audits') or not all(a.get('iterators_fully_exhausted') for a in bio['file_audits']):
            raise RuntimeError('Actual reader access/exhaustion audit required')
    artifact_hashes=verify_artifacts(src)
    backend_proof=verify_numeric_backend_bindings(bio,permit,src,artifact_hashes)
    full=read_csv(src/'FULL_GENE_AXIS.csv');endpoint=read_csv(src/'ENDPOINT_GENE_MANIFEST.csv');mapping=check_axes(full,endpoint)
    frozen_full,frozen_endpoint,_,_=biology.gene_axes(md,synthetic)
    if not full.equals(frozen_full) or not endpoint.equals(frozen_endpoint):raise RuntimeError('Biology axes differ from exact frozen metadata')
    units=read_csv(src/'QUERY_METADATA_ONLY.csv')
    if not np.array_equal(units.matrix_row.to_numpy(),np.arange(len(units))) or units.duplicated(['context','gene','role']).any():
        raise RuntimeError('Biology matrix row identities differ')
    if set(units.role)-{'TRAIN','VALIDATION',CONTROL}:raise RuntimeError('Forbidden numeric biology role')
    for role in set(units.role):
        role_path=src/f'{role}_MATRIX_ROWS.npy'
        if role_path.name not in artifact_hashes or not np.array_equal(np.load(role_path,allow_pickle=False),units.loc[units.role==role,'matrix_row'].to_numpy()):
            raise RuntimeError('Role matrix row mask differs')
    scope=pd.read_csv(query_scope,sep='\t',keep_default_na=False)
    if list(scope.columns)!=QUERY_COLUMNS or not scope.query_id.is_unique or not set(scope.role)<={'VALIDATION','TEST'}:
        raise RuntimeError('Query scope must contain unique held-out identities only')
    split=read_csv(md/'GENE_SPLIT.csv').set_index('gene').gene_role.to_dict()
    for r in scope.itertuples(index=False):
        m=mapping.get(r.target_gene_symbol)
        if r.context_id not in CONTEXTS or m is None or m.ensembl_id!=r.target_gene_id or split.get(r.target_gene_symbol)!=r.role:
            raise RuntimeError('Query scope canonical mapping or frozen role differs')
    selected={};trace={}
    for context in CONTEXTS:
        controls=units[(units.context==context)&(units.role==CONTROL)]
        train=units[(units.context==context)&(units.role=='TRAIN')&(units.n_cells_metadata>=30)].copy()
        if len(controls)!=1 or len(train)<=10:raise RuntimeError('Own-context NTC and more than ten eligible TRAIN targets required')
        control=controls.iloc[0]
        if control.gene!='Non-Targeting' or int(control.matrix_row)!=int(bio['control_matrix_rows'][context]):
            raise RuntimeError('Own-context control row differs')
        for r in train.itertuples(index=False):
            m=mapping.get(r.gene)
            if m is None or m.ensembl_id!=r.target_ensembl_id or int(m.gene_token_id)!=integral_token_id(r.target_token_id) or split.get(r.gene)!='TRAIN':
                raise RuntimeError('TRAIN canonical mapping or frozen gene role differs')
        if not train.target_ensembl_id.is_unique:raise RuntimeError('Duplicate TRAIN Ensembl condition')
        selected[context]=pd.concat([train,controls],ignore_index=True)
        trace[context]=selected[context].matrix_row.astype(int).tolist()
    expected_bytes=sum(len(x)*38606*8 for x in selected.values())
    if expected_bytes>max_output_bytes:raise RuntimeError('Bridge binary output byte budget exceeded before numeric input')
    means=np.load(src/'FULL_MEANS.npy',mmap_mode='r',allow_pickle=False)
    counts=np.load(src/'CELL_COUNTS.npy',mmap_mode='r',allow_pickle=False)
    if means.dtype!=np.float64 or tuple(means.shape)!=(len(units),38606) or list(means.shape)!=bio['matrix_shape']:
        raise RuntimeError('Full means dtype/shape differs')
    stage=out.with_name(out.name+f'.incomplete.{os.getpid()}');stage.mkdir(parents=True)
    full_out=pd.DataFrame({'gene_id':full.ensembl_id,'gene_token_id':full.gene_token_id,'gene_symbol':full.gene_name})
    endpoint_out=pd.DataFrame({'gene_id':endpoint.orion_ensembl_id,'gene_symbol':endpoint.gene_name,'source_index':endpoint.source_index})
    tsv(stage/'FULL_GENE_AXIS.tsv',full_out);tsv(stage/'OUTPUT_GENE_AXIS.tsv',endpoint_out)
    audit={'schema':'safeconf_orion_lm_bridge_access_v1','dataset_kind':'SYNTHETIC' if synthetic else 'AUTHORIZED_TRAIN',
       'biology_manifest_path':str(src/'BIOLOGY_MANIFEST.json'),'biology_manifest_sha256':sha(src/'BIOLOGY_MANIFEST.json'),
       'source_file_audits':bio.get('file_audits',[]),'numeric_rows_read_by_context':trace,
       'numeric_roles_read':['TRAIN',CONTROL],'VALIDATION_numeric_rows_read':0,'TEST_numeric_rows_read':0,
       'private_expression_values_decoded':False,'test_counts_or_coverage_computed':False,
       'npymap_page_bytes_may_include_neighbor_rows_but_no_neighbor_numeric_values_indexed':True,
       'query_scope_path':str(Path(query_scope).resolve()),'query_scope_sha256':sha(query_scope),
       'bridge_sha256':sha(__file__),'lm_cli_sha256':sha(R_CLI),'formal_upstream_training_started':False}
    if backend_proof:audit.update(backend_proof)
    if amendment:audit['integral_token_technical_amendment'] = amendment
    write_json(stage/'BRIDGE_ACCESS_AUDIT.json',audit)
    manifests=[]
    for context,rows in selected.items():
        dest=stage/context;dest.mkdir();final=out/context
        condition_ids=[r.target_ensembl_id if r.role=='TRAIN' else 'ctrl' for r in rows.itertuples(index=False)]
        conditions=pd.DataFrame({'condition_id':condition_ids,'target_gene_symbol':rows.gene,
            'context_id':context,'role':rows.role,'n_cells':rows.n_cells_metadata.astype(int)})
        tsv(dest/'TRAIN_CONDITIONS.tsv',conditions)
        q=scope[scope.context_id==context].copy()
        if q.empty or q.target_gene_id.isin(condition_ids).any():raise RuntimeError('Missing query identities or TRAIN/query leakage')
        tsv(dest/'QUERY_METADATA_ONLY.tsv',q)
        binary=dest/'TRAIN_X.f64le'
        with binary.open('xb') as f:
            for j,r in enumerate(rows.itertuples(index=False)):
                if int(counts[int(r.matrix_row)])!=int(r.n_cells_metadata):raise RuntimeError('Measured TRAIN/NTC cell counts differ')
                values=np.asarray(means[int(r.matrix_row)],dtype='<f8')
                if not np.isfinite(values).all() or (values<0).any():raise RuntimeError('TRAIN/NTC mean values invalid')
                f.write(values.tobytes(order='C'))
                if (j+1)%32==0 and hasattr(means._mmap,'madvise'):means._mmap.madvise(mmap.MADV_DONTNEED)
        control_row=int(rows.loc[rows.role==CONTROL,'matrix_row'].iloc[0])
        tsv(dest/'OWN_CONTEXT_NTC_MEAN.tsv',pd.DataFrame({'gene_id':full.ensembl_id,
            'mean_cell_logCP4000':np.asarray(means[control_row],dtype=np.float64)}))
        sidecar={'schema':'safeconf_orion_float64_le_matrix_v1','dtype':'float64','byte_order':'little',
            'disk_layout':'condition_major_gene_minor','n_genes':38606,'n_conditions':len(rows),
            'binary_bytes':binary.stat().st_size,'binary_sha256':sha(binary),
            'gene_axis_path':str(out/'FULL_GENE_AXIS.tsv'),'gene_axis_sha256':sha(stage/'FULL_GENE_AXIS.tsv'),
            'conditions_path':str(final/'TRAIN_CONDITIONS.tsv'),'conditions_sha256':sha(dest/'TRAIN_CONDITIONS.tsv')}
        kv(dest/'TRAIN_X.f64le.meta.tsv',sidecar)
        row={'context_id':context,'train_x':str(final/'TRAIN_X.f64le'),'baseline':str(final/'OWN_CONTEXT_NTC_MEAN.tsv'),
            'conditions':str(final/'TRAIN_CONDITIONS.tsv'),'queries':str(final/'QUERY_METADATA_ONLY.tsv'),
            'full_gene_axis':str(out/'FULL_GENE_AXIS.tsv'),'output_gene_axis':str(out/'OUTPUT_GENE_AXIS.tsv'),
            'receipt':str(final/'INPUT_RECEIPT.tsv')}
        receipt={'schema':'safeconf_orion_lm_blind_inputs_v1','dataset_kind':'SYNTHETIC' if synthetic else 'AUTHORIZED_TRAIN',
            'context_id':context,'estimand_id':'mean_cell_log1p_cp4000_v1','allowed_expression_roles':f'TRAIN,{CONTROL}',
            'contains_only_authorized_training_rows':'TRUE','query_metadata_only':'TRUE',
            'private_expression_values_decoded':'FALSE','upstream_attempt_id':attempt_id or 'synthetic_fixture_only',
            'expression_permit_path':str(permit_path),'expression_permit_sha256':sha(permit_path),
            'method_contract_path':str(contract_path),'method_contract_sha256':sha(contract_path),
            'access_audit_path':str(out/'BRIDGE_ACCESS_AUDIT.json'),'access_audit_sha256':sha(stage/'BRIDGE_ACCESS_AUDIT.json'),
            'biology_manifest_sha256':sha(src/'BIOLOGY_MANIFEST.json'),'train_x_sidecar_sha256':sha(dest/'TRAIN_X.f64le.meta.tsv')}
        if backend_proof:receipt.update(backend_proof)
        if amendment:
            receipt['technical_bridge_amendment_path'] = amendment['path']
            receipt['technical_bridge_amendment_sha256'] = amendment['sha256']
            receipt['actual_bridge_sha256'] = amendment['actual_bridge_sha256']
        for name in ['train_x','baseline','conditions','queries','full_gene_axis','output_gene_axis']:
            receipt[name+'_sha256']=sha(stage/Path(row[name]).relative_to(out))
        kv(dest/'INPUT_RECEIPT.tsv',receipt);manifests.append(row)
    tsv(stage/'FIT_MANIFEST.tsv',pd.DataFrame(manifests))
    result={'schema':'safeconf_orion_lm_authorized_bridge_v1','status':'COMPLETE','synthetic_only':synthetic,
       'scientific_contract_sha256':sha(contract_path),'permit_sha256':sha(permit_path),
       'biology_manifest_sha256':sha(src/'BIOLOGY_MANIFEST.json'),'query_scope_sha256':sha(query_scope),
       'bridge_sha256':sha(__file__),'lm_cli_sha256':sha(R_CLI),'numeric_layout':'little-endian float64 condition-major/gene-minor',
       'binary_output_bytes':expected_bytes,'full_gene_count':38606,'endpoint_gene_count':3285,
       'selected_TRAIN_NTC_matrix_rows':trace,'validation_numeric_rows_read':0,'test_numeric_rows_read':0,
       'new_TEST_counts_or_distributions_computed':False,'real_upstream_attempt_started':False,
       'elapsed_seconds':time.monotonic()-started,'finished_utc':datetime.now(timezone.utc).isoformat()}
    if backend_proof:result.update(backend_proof)
    if amendment:result['integral_token_technical_amendment'] = amendment
    write_json(stage/'BRIDGE_MANIFEST.json',result)
    hashes=[{'path':str(p.relative_to(stage)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
    write_json(stage/'ARTIFACT_HASHES.json',hashes);stage.rename(out)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('plan-queries');q.add_argument('--metadata-root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    b=s.add_parser('bridge')
    for name in ['biology-root','metadata-root','contract','permit','query-scope','output']:b.add_argument('--'+name,type=Path,required=True)
    b.add_argument('--attempt-id',default='');b.add_argument('--max-output-bytes',type=int,default=8*1024**3)
    b.add_argument('--technical-amendment',type=Path)
    a=p.parse_args()
    if a.command=='plan-queries':r=plan_queries(a.metadata_root,a.output)
    else:r=bridge(a.biology_root,a.metadata_root,a.contract,a.permit,a.query_scope,a.output,a.attempt_id,a.max_output_bytes,a.technical_amendment)
    print(json.dumps({k:v for k,v in r.items() if k!='row_metadata_identity_bindings'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
