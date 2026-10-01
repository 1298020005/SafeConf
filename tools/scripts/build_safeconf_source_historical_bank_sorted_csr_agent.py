#!/usr/bin/env python3
"""Append independent published Source biology without fitting or Orion expression."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
BASE_DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation'
DOC = BASE_DOC / 'historical_bank_v1/registered_v3_sorted_csr'
SOURCE = Path('/home/yyf/data/txpert_official_20260802/cache/K562_cross_cell_lines/de_adata_test.h5ad')
CORE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_source_core_20261002_v1')
OUTPUT = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/public_source_history_expanded_20261002_v2')
MD = Path('/home/yyf/data/safeconf_orion_frozen40_20261002/metadata_preparation_20261002_v1')
STUDY = {'K562':'Replogle_2022','RPE1':'Replogle_2022','hepg2':'Nadig_2025','jurkat':'Nadig_2025'}
SOURCE_SHA = '1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8'
SCHEMA = 'safeconf_source_public_history_expansion_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def binding(path):
    p=Path(path).resolve()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}


def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def checked(b):
    p=Path(b['path'])
    if not p.is_file() or p.stat().st_size!=b['bytes'] or sha(p)!=b['sha256']:
        raise RuntimeError(f'Immutable input differs: {p}')
    return p


def h5_column(g):
    if isinstance(g,h5py.Dataset): return g.asstr()[:] if g.dtype.kind in 'OSU' else g[:]
    if g.attrs.get('encoding-type')!='categorical': raise RuntimeError('Unsupported Source metadata encoding')
    categories=h5_column(g['categories']);codes=g['codes'][:]
    if (codes<0).any() or (codes>=len(categories)).any(): raise RuntimeError('Missing Source identity metadata')
    return categories[codes]


def metadata(path):
    # Only obs/var and structural X attributes are opened here; never X payload.
    with h5py.File(path,'r') as f:
        obs=pd.DataFrame({k:h5_column(f['obs'][k]) for k in ['_index','cell_line','condition','gene_name','control','batch']})
        genes=list(map(str,h5_column(f['var']['_index'])))
        shape=tuple(map(int,f['X'].attrs['shape']))
        if f['X'].attrs.get('encoding-type')!='csr_matrix' or shape!=(len(obs),len(genes)):
            raise RuntimeError('Expected complete processed Source CSR axis')
    if len(set(genes))!=len(genes): raise RuntimeError('Source measured gene axis is not unique')
    obs['original_row_index']=np.arange(len(obs),dtype=np.int64)
    return obs,genes


def single_gene(condition):
    parts=str(condition).split('+')
    return next((p for p in parts if p!='ctrl'),None) if len(parts)==2 and parts.count('ctrl')==1 else None


def make_plan(obs,legacy):
    old=set(legacy.loc[legacy.eligibility.eq(True),'perturbation_target'].astype(str))
    if len(legacy)!=2008 or len(old)!=580 or not legacy.experiment_id.is_unique:
        raise RuntimeError('Original 2008/580 bank identity differs')
    allowed=obs.cell_line.isin(STUDY)
    if obs.loc[allowed].duplicated(['cell_line','_index']).any():
        raise RuntimeError('Duplicate physical cells within an original Source context')
    ctrl=obs.control.astype(bool)
    if not np.array_equal(ctrl[allowed],obs.condition.eq('ctrl')[allowed]):
        raise RuntimeError('Original Source control flag disagrees with condition')
    genes=obs.condition.map(single_gene)
    selected=allowed & ~ctrl & genes.notna()
    # Literal target identity must agree with published metadata, without guesses.
    match=obs.gene_name.eq(genes)|obs.gene_name.eq(obs.condition)
    if not match[selected].all(): raise RuntimeError('Single-gene condition/target metadata conflict')
    units=obs.loc[selected,['cell_line','condition','batch','original_row_index']].copy()
    units['target_gene']=genes[selected].to_numpy()
    table=units.groupby(['cell_line','condition','target_gene'],sort=True).agg(n_cells=('original_row_index','size'),n_batches=('batch','nunique')).reset_index()
    if table.duplicated(['cell_line','target_gene']).any():
        raise RuntimeError('Multiple condition encodings for one physical historical unit')
    table['study_id']=table.cell_line.map(STUDY)
    table['append']=~table.target_gene.isin(old)&table.n_cells.ge(30)
    # All legacy units are recomputed only for compatibility diagnostics.
    keys=set(zip(legacy.context.astype(str),legacy.condition.astype(str)))
    table['legacy_diagnostic']=[(r.cell_line,r.condition) in keys for r in table.itertuples(index=False)]
    if table.legacy_diagnostic.sum()!=2008: raise RuntimeError('Not all legacy units exist in full Source cache')
    table=table[table['append']|table.legacy_diagnostic].reset_index(drop=True)
    table['group_row']=np.arange(len(table),dtype=np.int64)
    key_to_group={(r.cell_line,r.condition):int(r.group_row) for r in table.itertuples(index=False)}
    unit_index=pd.MultiIndex.from_frame(table[['cell_line','condition']])
    groups=unit_index.get_indexer(pd.MultiIndex.from_frame(obs[['cell_line','condition']])).astype(np.int64)
    controls=obs.loc[allowed&ctrl,['cell_line','batch']].drop_duplicates().sort_values(['cell_line','batch']).reset_index(drop=True)
    controls['group_row']=np.arange(len(controls),dtype=np.int64)+len(table)
    control_index=pd.MultiIndex.from_frame(controls[['cell_line','batch']])
    control_groups=control_index.get_indexer(pd.MultiIndex.from_frame(obs[['cell_line','batch']]))
    control_mask=allowed.to_numpy()&ctrl.to_numpy()
    if (control_groups[control_mask]<0).any(): raise RuntimeError('Observed Source controls lost')
    groups[control_mask]=control_groups[control_mask]+len(table)
    batch_counts=obs.loc[groups>=0,['cell_line','condition','batch']].groupby(['cell_line','condition','batch'],sort=True).size().rename('n_cells').reset_index()
    for r in batch_counts[~batch_counts.condition.eq('ctrl')].itertuples(index=False):
        if not ((controls.cell_line.eq(r.cell_line))&controls.batch.eq(r.batch)).any():
            raise RuntimeError('Own-study/context/batch observed control missing')
    return table,controls,groups,batch_counts,old


def coverage(metadata_root,genes):
    # This is metadata-SEEN coverage; no expression, predictions or errors.
    identity=json.loads((metadata_root/'FILE_METADATA_IDENTITY.json').read_text())
    files=identity['files'] if isinstance(identity,dict) else identity
    counts={}
    binds=[]
    for entry in files:
        p=Path(entry['metadata_parquet_path']);context=entry['context']
        if sha(p)!=entry['metadata_parquet_sha256']: raise RuntimeError('Frozen Orion metadata binding differs')
        frame=pd.read_parquet(p,columns=['row_role','fixed_metadata_QC_pass','gene_target','biological_unit'])
        rows=frame.loc[frame.row_role.eq('TEST')&frame.fixed_metadata_QC_pass.eq(True)]
        if any(r.biological_unit!=f'Huang2025_XAtlasOrion|{r.gene_target}|{context}' for r in rows.itertuples(index=False)):
            raise RuntimeError('Frozen metadata query identity differs')
        c=frame.loc[frame.row_role.eq('TEST')&frame.fixed_metadata_QC_pass.eq(True),'gene_target'].astype(str).value_counts()
        for gene,n in c.items():counts[(context,gene)]=counts.get((context,gene),0)+int(n)
        binds.append(binding(p))
    eligible={ctx:{g for (c,g),n in counts.items() if c==ctx and n>=30} for ctx in sorted({c for c,g in counts})}
    supported={ctx:v&set(genes) for ctx,v in eligible.items()}
    union=set().union(*supported.values());both=set.intersection(*supported.values())
    full_gene_meta=pd.read_parquet(metadata_root/'gene_metadata.parquet',columns=['gene_name','ensembl_id'])
    unique=full_gene_meta[~full_gene_meta.gene_name.duplicated(keep=False)].set_index('gene_name').ensembl_id.to_dict()
    split=pd.read_csv(metadata_root/'GENE_SPLIT.csv',usecols=['gene','gene_role'])
    test_genes=set(split.loc[split.gene_role.eq('TEST'),'gene'].astype(str))
    if any(not v<=test_genes for v in eligible.values()): raise RuntimeError('Metadata TEST gene identity differs')
    study='Huang2025_XAtlasOrion'
    query_ids={ctx:[{'query_id':f'{study}|{g}|{ctx}','target_gene_id':unique[g],'target_gene_symbol':g,'context_id':ctx,'role':'TEST'} for g in sorted(v) if g in unique] for ctx,v in supported.items()}
    return {'schema':'safeconf_public_bank_metadata_coverage_certificate_v1','TEST_truth_read':False,'risk_scores_used':False,
            'role':'ORION_TEST_METADATA_SEEN_ONLY_TRUTH_CLOSED','target_truth_or_expression_used':False,
            'selection_based_on_safeconf_outcomes':False,'fixed_metadata_min_cells':30,
            'eligible_context_counts':{c:len(v) for c,v in eligible.items()},
            'supported_context_counts':{c:len(v) for c,v in supported.items()},
            'supported_unique_gene_clusters':len(union),'supported_both_context_gene_clusters':len(both),
            'supported_gene_symbols':sorted(union),'supported_QC_query_identities':query_ids,
            'metadata_bindings':binds,'gene_metadata_binding':binding(metadata_root/'gene_metadata.parquet'),'gene_split_binding':binding(metadata_root/'GENE_SPLIT.csv')}


def save_coverage(directory,value,final_directory=None):
    rows=[{'query_id':r['query_id'],'context_id':r['context_id']} for v in value['supported_QC_query_identities'].values() for r in v]
    p=directory/'METADATA_QUALIFIED_TEST_QUERY_IDS.tsv'
    pd.DataFrame(rows,columns=['query_id','context_id']).to_csv(p,sep='\t',index=False)
    value['qualified_query_ids']=binding(p)
    if final_directory is not None:value['qualified_query_ids']['path']=str(final_directory/p.name)
    return value


def prepare(args):
    doc=args.doc.resolve();doc.mkdir(parents=True,exist_ok=True)
    if args.source.resolve()!=SOURCE or args.core.resolve()!=CORE: raise RuntimeError('Only explicitly authorized historical Source asset/core')
    obs,genes=metadata(args.source)
    legacy=pd.read_parquet(args.core/'SOURCE_PUBLIC_MEMORY_METADATA.parquet')
    table,controls,groups,batches,old=make_plan(obs,legacy)
    axis=pd.read_csv(args.core/'GENE_MANIFEST.csv')
    if len(axis)!=3285 or not np.array_equal(axis.axis_index,np.arange(3285)) or not axis.gene_name.is_unique or [genes[i] for i in axis.source_index]!=axis.gene_name.tolist():
        raise RuntimeError('Exact measured Source3285 projection differs')
    training_genes=set(pd.read_parquet(args.core/'SOURCE_TASKS.parquet',columns=['gene']).gene.astype(str))
    if len(training_genes)!=575 or not training_genes<=old or training_genes&set(table.loc[table['append'],'target_gene']):
        raise RuntimeError('Source575 training history cannot change')
    role=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/DATA_ROLE_REGISTRY.csv'
    r=pd.read_csv(role);selected=r[r.iloc[:,0].isin(['E201_TxPert_GAT','E205_TxPert_Exphormer'])]
    if len(selected)!=2 or not selected.iloc[:,7].eq('DEV').all(): raise RuntimeError('Original Source role is not DEV')
    provenance=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/history_provenance_check/TXPERT_CONTEXT_STUDY_PROVENANCE.csv'
    prov=pd.read_csv(provenance)
    if dict(zip(prov.iloc[:,0],prov.iloc[:,2]))!=STUDY: raise RuntimeError('Published original-study sidecar differs')
    names=['SOURCE_FEATURE_MANIFEST.json','MODEL_MANIFEST.json','ARTIFACT_HASHES.json']+[r['path'] for r in json.loads((args.core/'MODEL_MANIFEST.json').read_text())['models']]
    training_tasks=binding(args.core/'SOURCE_TASKS.parquet')
    parent={k:binding(args.core/n) for k,n in {'metadata':'SOURCE_PUBLIC_MEMORY_METADATA.parquet','effects':'SOURCE_PUBLIC_EFFECTS.npy','controls':'SOURCE_PUBLIC_CONTROLS.npy','axis':'GENE_MANIFEST.csv','gene_ids':'GENE_IDS.json'}.items()}
    registry={x['path']:x for x in json.loads((args.core/'ARTIFACT_HASHES.json').read_text())}
    for b in [binding(args.core/n) for n in names if n!='ARTIFACT_HASHES.json']+list(parent.values())+[training_tasks]:
        expected=registry.get(Path(b['path']).name,{})
        if expected.get('sha256')!=b['sha256'] or expected.get('bytes')!=b['bytes']: raise RuntimeError('Original Source core artifact registry differs')
    scientific=binding(BASE_DOC/'ORION_METHOD_CONTRACT.json')
    if scientific['sha256']!='6b8a26939c8c6134dbf2e8294ee7905fccc2122b570b046f3183a3d2f5caba97':raise RuntimeError('Original scientific method contract differs')
    # Source SHA is an independently completed opaque-byte attestation, checked again before X.
    source={'path':str(SOURCE),'bytes':7767053064,'sha256':SOURCE_SHA}
    table.to_csv(doc/'PLANNED_HISTORICAL_UNITS.csv',index=False)
    controls.to_csv(doc/'PLANNED_CONTROL_BATCHES.csv',index=False)
    coverage_result=save_coverage(doc,coverage(args.metadata,old|set(table.loc[table['append'],'target_gene'])))
    write_json(doc/'METADATA_COVERAGE_PREFLIGHT.json',coverage_result)
    plan={'n_legacy_items':len(legacy),'n_old_eligible_genes':len(old),'n_append_units':int(table['append'].sum()),
          'n_appended_unique_genes':int(table.loc[table['append'],'target_gene'].nunique()),
          'append_context_counts':table[table['append']].groupby('cell_line').size().to_dict(),
          'n_historical_X_rows_authorized':int((groups>=0).sum()),'n_control_cells':int(obs.loc[groups>=0,'control'].astype(bool).sum()),
          'n_control_batches':len(controls),'duplicate_context_cell_ids':0,'all_3285_endpoint_genes_measured':True,
          'Source_training_gene_count':575,'Source575_subset_old580':True,'Source575_training_gene_history_unchanged':True,
          'expression_values_read':0,'target_orion_truth_expression_read':0}
    write_json(doc/'METADATA_AUDIT_COUNTS_PREFLIGHT.json',plan)
    note={'role':'SOURCE_PUBLIC_HISTORICAL_DEV','original_source_predictors_role':'DEV','source_filename_test_is_not_Orion_TEST':True,
          'source_is_published_historical_Replogle2022_Nadig2025':True,'source_has_no_SEALED_role':True,
          'source_new_targets_are_newly_authorized_public_history':True,'independent_Orion_truth_confirmation':False,
          'Orion_TEST_metadata_role':'SEEN','Orion_TEST_truth_status':'CLOSED','Orion_expression_truth_reads_authorized':False,
          'role_registry':binding(role),'source_context_study_provenance':binding(provenance)}
    write_json(doc/'SOURCE_PUBLIC_HISTORICAL_DATA_ROLE_NOTE.json',note)
    contract={'schema':'safeconf_source_public_historical_read_contract_v1','status':'AUTHORIZED_SOURCE_PUBLIC_HISTORY_ONLY',
              'authorization':'Root explicitly authorized existing published historical Source X on 2026-10-02; no Orion expression/truth.',
              'source':source,'role_note':binding(doc/'SOURCE_PUBLIC_HISTORICAL_DATA_ROLE_NOTE.json'),
              'builder':binding(Path(__file__)),'parent_bank':parent,'base_source_core_bindings':[binding(args.core/n) for n in names],
              'source_training_tasks_binding':training_tasks,
              'base_scientific_contract':scientific,
              'normalization_evidence':binding(BASE_DOC/'SOURCE_NORMALIZATION_AUDIT.json'),
              'source_context_study_provenance':binding(provenance),'allowed_context_study':STUDY,'excluded_contexts':['K562_adamson'],
              'axis':parent['axis'],'processed_X_coordinate':'cellwise log1p(CP4000), normalized before 3352-gene restriction; no rescaling or relogging',
              'aggregation':'cell-equal arithmetic mean processed X; own-study/context/batch observed CTRL weighted by target-cell batch frequencies',
              'append_policy':{'minimum_historical_cells':30,'target_absent_old_eligible_gene_set':True,'old_eligible_gene_count':580},
              'legacy_prefix_n_items':2008,'legacy_prefix_metadata_and_vectors_must_remain_exact':True,
              'missing_readout_gene_policy':'FAIL_NO_ZERO_IMPUTATION','missing_quality_support_fields':'retain_null_no_fabrication',
              'unit_identity':'source-file SHA + context + single-gene condition; each physical cell original row is used once',
              'limitations':['guide and plate IDs/replicate-quality metadata not retained in this processed cache','batch is an original Source batch label, not a manufactured biological replicate','study/context effects are historical references, not newly trained Public biology models','official Source axis was selected upstream using held-context HVGs'],
              'runtime_budget':{'max_chunk_rows':2048,'max_chunk_nnz':8000000,'max_rss_bytes':4*1024**3,'max_seconds':1200},
              'CSR_technical_order_policy':'sort paired indices/values within bounded rows; reject duplicate measured gene entries; mathematical cell profiles unchanged',
              'prior_failed_v1_builder':binding(ROOT/'tools/scripts/build_safeconf_source_historical_bank_agent.py'),
              'metadata_only_Orion_coverage':binding(doc/'METADATA_COVERAGE_PREFLIGHT.json'),'metadata_root':str(args.metadata.resolve()),
              'no_model_fits':True,'no_downloads':True,'no_GPU':True,'no_Orion_expression_or_truth':True}
    path=doc/'SOURCE_PUBLIC_HISTORICAL_READ_CONTRACT.json';write_json(path,contract);path.chmod(0o444)
    expansion={'schema':'safeconf_orion_pretruth_public_bank_extension_v1','status':'PROSPECTIVE_TRUTH_BLIND_SOURCE_HISTORY_EXTENSION',
               'source_read_contract_binding':binding(path),'base_scientific_contract':scientific,'builder_code_binding':binding(Path(__file__)),
               'parent_bank':parent,'parent_source_core_bindings':contract['base_source_core_bindings'],
               'append_policy_id':'ONLY_GENES_ABSENT_FROM_ORIGINAL_ELIGIBLE_SOURCE_BANK','append_policy':contract['append_policy'],
               'canonical_axis_binding':parent['axis'],'old_legacy_prefix_n_items':2008,'Source575_subset_old580':True,
               'minimum_primary_gene_clusters_union':100,'minimum_context_tasks':20,
               'new_parameter_fits':0,'append_only_absent_original_eligible_genes':True,'legacy_query_invariance_required':True,
               'TEST_truth_used_for_bank_selection':False,'SafeConf_scores_used_for_bank_selection':False,
               'immutable_dependencies':[binding(Path(__file__)),training_tasks],
               'pretruth_metadata_coverage_used':True,'SafeConf_outcomes_used_for_asset_selection':False,
               'Orion_TEST_metadata':'SEEN','Orion_TEST_truth':'CLOSED','new_bank_is_not_original_confirmation':True,
               'new_model_fits':0,'old_Source_model_parameters_or_features_changed':False,'no_Orion_expression_truth_read':True}
    ep=doc/'PUBLIC_BANK_EXPANSION_CONTRACT.json';write_json(ep,expansion);ep.chmod(0o444)
    print(json.dumps({'contract':binding(path),'metadata_plan':plan,'coverage_union':coverage_result['supported_unique_gene_clusters']},indent=2),flush=True)


def selected_chunks(groups,indptr,max_rows,max_nnz):
    selected=np.flatnonzero(groups>=0)
    # Exact contiguous selected runs ensure excluded context/non-single rows are never decoded.
    ends=np.flatnonzero(np.diff(selected)!=1)+1
    for run in np.split(selected,ends):
        a=int(run[0]);end=int(run[-1])+1
        while a<end:
            b=min(a+max_rows,end)
            if int(indptr[b]-indptr[a])>max_nnz:
                b=a+int(np.searchsorted(indptr[a+1:b+1]-indptr[a],max_nnz,side='right'))
            if b<=a: raise RuntimeError('Single cell exceeds frozen bounded CSR budget')
            yield a,b
            a=b


def accumulate(path,groups,projection,n_groups,budget,progress=True):
    started=time.monotonic();sums=np.zeros((n_groups,len(projection)),dtype=np.float64);counts=np.zeros(n_groups,dtype=np.int64)
    seen=0;chunks=0;nnz=0;sorted_chunks=0
    with h5py.File(path,'r') as f:
        x=f['X'];indptr=x['indptr'][:].astype(np.int64)
        if indptr.shape!=(len(groups)+1,) or indptr[0]!=0 or (np.diff(indptr)<0).any() or indptr[-1]!=len(x['data']) or len(x['data'])!=len(x['indices']):
            raise RuntimeError('Malformed Source CSR structure')
        for a,b in selected_chunks(groups,indptr,budget['max_chunk_rows'],budget['max_chunk_nnz']):
            lo,hi=int(indptr[a]),int(indptr[b]);values=x['data'][lo:hi];indices=x['indices'][lo:hi]
            if not np.isfinite(values).all() or (values<0).any() or (indices<0).any() or (indices>=x.attrs['shape'][1]).any():
                raise RuntimeError('Invalid measured Source normalized expression')
            block=sparse.csr_matrix((values.astype(np.float64),indices,indptr[a:b+1]-lo),shape=(b-a,int(x.attrs['shape'][1])))
            if not block.has_sorted_indices:
                block.sort_indices();sorted_chunks+=1
            if not block.has_canonical_format: raise RuntimeError('Duplicate measured Source CSR gene entries')
            block=block[:,projection]
            unique,inverse=np.unique(groups[a:b],return_inverse=True)
            reducer=sparse.csr_matrix((np.ones(b-a),(inverse,np.arange(b-a))),shape=(len(unique),b-a))
            sums[unique]+=np.asarray((reducer@block).toarray(),dtype=np.float64)
            counts[unique]+=np.bincount(inverse,minlength=len(unique))
            seen+=b-a;chunks+=1;nnz+=hi-lo
            if time.monotonic()-started>budget['max_seconds']: raise RuntimeError('Source history wall-time budget exceeded')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>budget['max_rss_bytes']: raise RuntimeError('Source history resident memory budget exceeded')
            if progress and chunks%50==0: print(f'SOURCE_HISTORY_CSR rows={seen}/{int((groups>=0).sum())} elapsed={time.monotonic()-started:.1f}s',flush=True)
    if seen!=int((groups>=0).sum()) or (counts<=0).any(): raise RuntimeError('Not all exact permitted Source rows completed')
    return sums/counts[:,None],counts,{'all_exact_selected_Source_rows_completed':True,'selected_cells':seen,'selected_CSR_values':nnz,'chunks':chunks,'bounded_chunks_sorted_without_changing_paired_values':sorted_chunks,'elapsed_seconds':time.monotonic()-started,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'excluded_context_rows_decoded':0,'Orion_expression_truth_values_read':0}


def build(args):
    started=time.monotonic();cp=args.contract.resolve()
    if cp.stat().st_mode&0o222: raise RuntimeError('Read contract must be immutable before Source X')
    contract=json.loads(cp.read_text())
    if contract['schema']!='safeconf_source_public_historical_read_contract_v1' or not contract['no_Orion_expression_or_truth'] or not contract['no_model_fits']:
        raise RuntimeError('Not the authorized Source-only historical contract')
    if checked(contract['builder'])!=Path(__file__).resolve(): raise RuntimeError('Executed builder differs from registered builder')
    source=checked(contract['source']) # Opaque whole-file hash is checked BEFORE decoding Source X.
    for b in contract['base_source_core_bindings']:checked(b)
    checked(contract['source_training_tasks_binding'])
    checked(contract['prior_failed_v1_builder'])
    parent={k:checked(v) for k,v in contract['parent_bank'].items()}
    for k in ['role_note','normalization_evidence','source_context_study_provenance','metadata_only_Orion_coverage']:checked(contract[k])
    checked(contract['base_scientific_contract'])
    ep=cp.parent/'PUBLIC_BANK_EXPANSION_CONTRACT.json'
    if ep.stat().st_mode&0o222:raise RuntimeError('Prospective expansion contract must be immutable')
    expansion=json.loads(ep.read_text())
    if expansion['source_read_contract_binding']!=binding(cp) or expansion['base_scientific_contract']!=contract['base_scientific_contract'] or expansion['builder_code_binding']!=binding(Path(__file__)):
        raise RuntimeError('Prospective expansion identity differs')
    out=args.output.resolve()
    if out.exists() or out.parent!=OUTPUT.parent or not out.name.startswith('public_source_history_expanded_'): raise RuntimeError('New isolated bank output required')
    staging=out.with_name(out.name+f'.incomplete.{os.getpid()}');staging.mkdir()
    obs,genes=metadata(source);legacy=pd.read_parquet(parent['metadata']);axis=pd.read_csv(parent['axis'])
    table,control_table,groups,batches,old=make_plan(obs,legacy)
    projection=axis.source_index.to_numpy(int)
    if axis.gene_name.tolist()!=[genes[i] for i in projection] or len(axis)!=3285: raise RuntimeError('Source endpoint axis mismatch before X')
    if contract['allowed_context_study']!=STUDY or contract['append_policy']!={'minimum_historical_cells':30,'target_absent_old_eligible_gene_set':True,'old_eligible_gene_count':580}: raise RuntimeError('Historical scope mismatch')
    write_json(staging/'SOURCE_HISTORICAL_READ_START.json',{'contract':binding(cp),'builder':binding(Path(__file__)),'source':contract['source'],'role':'SOURCE_PUBLIC_HISTORICAL_DEV','Orion_truth_status':'CLOSED'})
    means,counts,read_audit=accumulate(source,groups,projection,len(table)+len(control_table),contract['runtime_budget'])
    if not np.array_equal(counts[:len(table)],table.n_cells.to_numpy()): raise RuntimeError('Cell-equal aggregation support changed')
    ctrl_lookup={(r.cell_line,r.batch):int(r.group_row) for r in control_table.itertuples(index=False)}
    by_key={(r.cell_line,r.condition):r for r in table.itertuples(index=False)}
    matched=np.zeros((len(table),3285),dtype=np.float64)
    for r in batches[~batches.condition.eq('ctrl')].itertuples(index=False):
        g=by_key[(r.cell_line,r.condition)];matched[g.group_row]+=means[ctrl_lookup[(r.cell_line,r.batch)]]*int(r.n_cells)/int(g.n_cells)
    effects=means[:len(table)]-matched
    old_effects=np.load(parent['effects'],mmap_mode='r',allow_pickle=False);old_controls=np.load(parent['controls'],mmap_mode='r',allow_pickle=False)
    if old_effects.shape!=(2008,3285) or old_controls.shape!=old_effects.shape: raise RuntimeError('Frozen projected bank shape differs')
    diagnostic=[]
    for row in legacy.itertuples(index=False):
        g=by_key[(row.context,row.condition)];i=int(row.effect_vector_row)
        ed=float(np.max(np.abs(effects[g.group_row]-old_effects[i])));cd=float(np.max(np.abs(matched[g.group_row]-old_controls[i])))
        diagnostic.append({'experiment_id':row.experiment_id,'context':row.context,'condition':row.condition,'n_cells_match':int(g.n_cells)==int(row.n_cells),'n_batches_match':int(g.n_batches)==int(row.n_batches),'max_abs_effect_gap':ed,'max_abs_control_gap':cd})
    diagnostic=pd.DataFrame(diagnostic)
    if not diagnostic.n_cells_match.all() or not diagnostic.n_batches_match.all() or diagnostic.max_abs_effect_gap.max()>1e-5 or diagnostic.max_abs_control_gap.max()>1e-5:
        diagnostic.to_csv(staging/'LEGACY_COMPATIBILITY_DIAGNOSTIC.csv',index=False)
        raise RuntimeError('Historical cache is incompatible with legacy Source means/controls; no bank committed')
    appended=table[table['append']].reset_index(drop=True);records=[];provenance=[]
    for j,r in enumerate(appended.itertuples(index=False)):
        i=2008+j;eid=f'SOURCE_HISTORY_V1::{r.study_id}::{r.cell_line}::{r.condition}'
        record={k:None for k in legacy.columns};record.update(experiment_id=eid,study_id=r.study_id,context=r.cell_line,
            perturbation_type='genetic_single_gene',perturbation_target=r.target_gene,condition=r.condition,effect_vector_row=i,
            effect_contract_id='E201_log1p_matched_batch_delta_v1',control_source='source_context_batch_matched_control',gene_space_id='Source_Orion_common_3285',
            n_cells=int(r.n_cells),n_batches=int(r.n_batches),provenance=str(source),eligibility=True,timestamp='2026-10-02T00:00:00+00:00')
        records.append(record)
    memory=pd.concat([legacy,pd.DataFrame(records,columns=legacy.columns)],ignore_index=True)
    old_groups=legacy.groupby('perturbation_target',sort=True).experiment_id.apply(list).to_dict()
    if not memory.iloc[:2008].equals(legacy) or set(appended.target_gene)&old or memory.loc[memory.perturbation_target.isin(old)].groupby('perturbation_target',sort=True).experiment_id.apply(list).to_dict()!=old_groups:
        raise RuntimeError('Old580/Source575 history changed')
    memory.to_parquet(staging/'SOURCE_PUBLIC_MEMORY_METADATA.parquet',index=False)
    selected=appended.group_row.to_numpy(int)
    np.save(staging/'SOURCE_PUBLIC_EFFECTS.npy',np.concatenate([old_effects,effects[selected].astype(old_effects.dtype)]),allow_pickle=False)
    np.save(staging/'SOURCE_PUBLIC_CONTROLS.npy',np.concatenate([old_controls,matched[selected].astype(old_controls.dtype)]),allow_pickle=False)
    for key,name in [('axis','GENE_MANIFEST.csv'),('gene_ids','GENE_IDS.json')]:shutil.copyfile(parent[key],staging/name)
    if not pd.read_parquet(staging/'SOURCE_PUBLIC_MEMORY_METADATA.parquet').iloc[:2008].equals(legacy): raise RuntimeError('Serialized legacy metadata prefix differs')
    ne=np.load(staging/'SOURCE_PUBLIC_EFFECTS.npy',mmap_mode='r');nc=np.load(staging/'SOURCE_PUBLIC_CONTROLS.npy',mmap_mode='r')
    if not np.array_equal(ne[:2008],old_effects) or not np.array_equal(nc[:2008],old_controls): raise RuntimeError('Legacy vectors prefix differs')
    for r in memory.itertuples(index=False):
        g=by_key[(r.context,r.condition)];unit_batches=batches[batches.cell_line.eq(r.context)&batches.condition.eq(r.condition)]
        provenance.append({'experiment_id':r.experiment_id,'original_study':STUDY[r.context],'source_context':r.context,'condition':r.condition,'target_gene':r.perturbation_target,
            'physical_unit_id':f'{SOURCE_SHA}::{r.context}::{r.condition}','source_sha256':SOURCE_SHA,'n_cells':int(g.n_cells),'observed_batch_labels':json.dumps(unit_batches.batch.astype(str).tolist()),
            'guide_ids_available':False,'plate_ids_available':False,'replicate_quality_available':False,'legacy_prefix':int(r.effect_vector_row)<2008})
    pd.DataFrame(provenance).to_csv(staging/'CANONICAL_UNIT_PROVENANCE.csv',index=False)
    diagnostic.to_csv(staging/'LEGACY_COMPATIBILITY_DIAGNOSTIC.csv',index=False)
    table.to_csv(staging/'HISTORICAL_UNITS.csv',index=False);control_table.to_csv(staging/'CONTROL_BATCH_METADATA.csv',index=False)
    genes_all=set(memory.loc[memory.eligibility.eq(True),'perturbation_target'].astype(str))
    write_json(staging/'ELIGIBLE_GENE_SET.json',{'old_eligible_gene_symbols':sorted(old),'new_eligible_gene_symbols':sorted(genes_all-old),'eligible_gene_symbols':sorted(genes_all)})
    certificate=save_coverage(staging,coverage(Path(contract['metadata_root']),genes_all),out);write_json(staging/'METADATA_COVERAGE_CERTIFICATE.json',certificate)
    if certificate['supported_unique_gene_clusters']<100: raise RuntimeError('Expanded metadata-supported historical cohort remains below100')
    audit={'schema':SCHEMA,'n_legacy_items':2008,'n_append_items':len(appended),'n_total_items':len(memory),'n_old_eligible_genes':580,'n_new_eligible_genes':len(genes_all-old),'n_total_eligible_genes':len(genes_all),
           'append_context_counts':appended.groupby('cell_line').size().to_dict(),'metadata_prefix_exact':True,'legacy_effects_bitwise_equal':True,'legacy_controls_bitwise_equal':True,'old580_retrieval_group_identity_exact':True,
           'Source575_risk_training_gene_history_unchanged':True,'physical_units_unique':len({r['physical_unit_id'] for r in provenance})==len(provenance),'all_endpoint_genes_measured':True,
           'old_compatibility_effect_max_abs_gap':float(diagnostic.max_abs_effect_gap.max()),'old_compatibility_control_max_abs_gap':float(diagnostic.max_abs_control_gap.max()),'new_fits':0,'Orion_expression_truth_read':False}
    write_json(staging/'METADATA_AUDIT_COUNTS.json',audit)
    read_audit.update(contract_sha256=sha(cp),builder_code_sha256=sha(__file__),source_sha256=SOURCE_SHA,normalization_altered=False,new_fits=0)
    write_json(staging/'SOURCE_HISTORICAL_READ_AUDIT.json',read_audit)
    extended={k:binding(staging/n) for k,n in {'metadata':'SOURCE_PUBLIC_MEMORY_METADATA.parquet','effects':'SOURCE_PUBLIC_EFFECTS.npy','controls':'SOURCE_PUBLIC_CONTROLS.npy','axis':'GENE_MANIFEST.csv','gene_ids':'GENE_IDS.json'}.items()}
    def final_binding(b):return dict(b,path=str(out/Path(b['path']).name))
    manifest={'schema':SCHEMA,'status':'COMPLETE','role':'SOURCE_PUBLIC_HISTORICAL_DEV_SEPARATE_NEW_BANK_VERSION','extension_contract':binding(ep),'builder_code_sha256':sha(__file__),
        'base_source_core_bindings':contract['base_source_core_bindings'],'parent_bank':contract['parent_bank'],'extended_bank':{k:final_binding(v) for k,v in extended.items()},
        'parent_source_core_bindings':contract['base_source_core_bindings'],'source_read_contract_binding':binding(cp),'builder_code_binding':binding(Path(__file__)),
        'source_asset_binding':contract['source'],'base_scientific_contract':contract['base_scientific_contract'],
        'source_training_tasks_binding':contract['source_training_tasks_binding'],
        'legacy_prefix':{'n_items':2008,'metadata_exact':True,'effects_bitwise_equal':True,'controls_bitwise_equal':True,'old_eligible_gene_count':580,'no_old_eligible_target_appended':True},
        'append_policy':dict(contract['append_policy'],allowed_contexts=list(STUDY)),'n_items':len(memory),'n_genes':3285,'new_fits':0,'target_orion_truth_expression_used':False,
        'append_policy_id':'ONLY_GENES_ABSENT_FROM_ORIGINAL_ELIGIBLE_SOURCE_BANK',
        'limitations':contract['limitations'],'elapsed_seconds':time.monotonic()-started}
    for k,n in {'eligible_gene_set':'ELIGIBLE_GENE_SET.json','canonical_unit_provenance':'CANONICAL_UNIT_PROVENANCE.csv','metadata_audit_counts':'METADATA_AUDIT_COUNTS.json','compatibility_diagnostic':'LEGACY_COMPATIBILITY_DIAGNOSTIC.csv','read_audit':'SOURCE_HISTORICAL_READ_AUDIT.json','coverage_certificate':'METADATA_COVERAGE_CERTIFICATE.json'}.items():manifest[k]=final_binding(binding(staging/n))
    manifest['coverage_certificate_binding']=manifest['coverage_certificate']
    # Reverify original models/bank inputs after the pass; no setters/model calls exist here.
    for b in contract['base_source_core_bindings']:checked(b)
    for b in contract['parent_bank'].values():checked(b)
    write_json(staging/'PUBLIC_BANK_MANIFEST.json',manifest)
    files=[final_binding(binding(p)) for p in sorted(staging.iterdir()) if p.is_file()];write_json(staging/'ARTIFACT_HASHES.json',files)
    for p in staging.iterdir():p.chmod(0o444)
    staging.rename(out)
    print(json.dumps({'status':'COMPLETE','bank_manifest':binding(out/'PUBLIC_BANK_MANIFEST.json'),'counts':audit,'coverage_union':certificate['supported_unique_gene_clusters'],'read_audit':read_audit},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--source',type=Path,default=SOURCE);a.add_argument('--core',type=Path,default=CORE);a.add_argument('--metadata',type=Path,default=MD);a.add_argument('--doc',type=Path,default=DOC)
    a=sub.add_parser('build');a.add_argument('--contract',type=Path,default=DOC/'SOURCE_PUBLIC_HISTORICAL_READ_CONTRACT.json');a.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args();prepare(args) if args.command=='prepare' else build(args)


if __name__=='__main__':main()
