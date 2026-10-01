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

def bridge(biology_root,metadata_root,contract_path,permit_path,query_scope,output,attempt_id='',max_output_bytes=8*1024**3):
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
    if not synthetic:
        for name,path in [('bridge',Path(__file__)),('lm_cli',R_CLI)]:
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
            if m is None or m.ensembl_id!=r.target_ensembl_id or int(m.gene_token_id)!=int(r.target_token_id) or split.get(r.gene)!='TRAIN':
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
    a=p.parse_args()
    if a.command=='plan-queries':r=plan_queries(a.metadata_root,a.output)
    else:r=bridge(a.biology_root,a.metadata_root,a.contract,a.permit,a.query_scope,a.output,a.attempt_id,a.max_output_bytes)
    print(json.dumps({k:v for k,v in r.items() if k!='row_metadata_identity_bindings'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
