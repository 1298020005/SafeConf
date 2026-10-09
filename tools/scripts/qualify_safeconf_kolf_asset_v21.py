#!/usr/bin/env python3
"""Bounded metadata/physical-cost check of a primary-author backup asset.

Never reads X/counts data values. Cell/perturbation metadata, gene identifiers,
CSR/CSC pointers and chunk offsets establish feasibility before a new numeric
protocol is launched. The original cumulative download ledger is shared.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import argparse,hashlib,json,sys,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,h5py
from tools.safeconf_continual.range_h5 import MeteredHTTPFile
from tools.safeconf_continual.submission_evidence import write_json,h5_column,sha,digest_ids

RUN=Path('/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21')
OUT=RUN/'backup_asset/kolf_raw_preflight_v1'
LEDGER=RUN/'external/DOWNLOAD_RESOURCE_LEDGER.json'
URL='https://ndownloader.figshare.com/files/64650261'
SOURCE_AXIS=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/GENE_IDS.json')
PANEL_SIZE=None


def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    card={'author_record':'https://api.figshare.com/v2/articles/27261219','author_file_id':64650261,
        'author_MD5':'afd30fde1e6ad32969c29868394385d1','source_url':URL,
        'only_basic_qc_not_response_strength_filter':True,'response_values_read':False,
        'metadata_budget_bytes':256*1024**2,'wall_seconds_cap':600,
        'source_output_axis':str(SOURCE_AXIS),'source_output_axis_sha256':sha(SOURCE_AXIS),
        'source_defined_panel_size':PANEL_SIZE,'panel_selection':'SHA256(SafeConf-v21-KOLF-resource-panel|gene); no target effects/errors',
        'cohort_gene_cap':1200,'gene_min_cells':20,'download_budget_not_restarted':True,
        'numeric_confirmation_not_authorized_by_this_preflight':True}
    write_json(OUT/'EXPERIMENT_CARD.json',card)
    before=json.loads(LEDGER.read_text())['this_run_network_payload_bytes']
    io=MeteredHTTPFile(URL,RUN/'backup_asset/kolf_raw_preflight_v1/range_cache',LEDGER,
        cap=min(100_000_000_000,38_000_000_000+before+card['metadata_budget_bytes']))
    def checkpoint(stage):
        used=json.loads(LEDGER.read_text())['this_run_network_payload_bytes']-before
        write_json(OUT/'STATUS.json',{'status':'RUNNING_METADATA_ONLY','stage':stage,'metadata_payload_bytes':used,
            'wall_seconds':time.monotonic()-start,'pid':os.getpid(),'response_values_read':False})
        if used>card['metadata_budget_bytes'] or time.monotonic()-start>600:
            raise RuntimeError('bounded metadata preflight budget reached')
    with h5py.File(io,'r') as h:
        checkpoint('hdf5_header')
        structure={'obs_keys':list(h['obs']),'var_keys':list(h['var']),'layer_keys':list(h.get('layers',{})),
            'X_attrs':{k:str(v) for k,v in h['X'].attrs.items()},'source_bytes':io.size,'response_values_read':False}
        write_json(OUT/'HDF5_STRUCTURE.json',structure)
        if 'gene_target' not in h['obs'] or 'gene_ids' not in h['var']:raise RuntimeError('author identity columns differ from pilot contract')
        names=h5_column(h['obs'],'gene_target').astype(str)
        target_counts=pd.Series(names).value_counts();ntc=names=='NTC'
        eligible=target_counts.index[(target_counts>=20)&(~target_counts.index.isin(['NTC','nan','None','unassigned']))].astype(str).tolist()
        eligible=sorted(eligible,key=lambda g:hashlib.sha256(f'SafeConf-v21-KOLF-cohort|{g}'.encode()).hexdigest())[:1200]
        if len(eligible)<600:raise RuntimeError('fewer than600 independent gene targets available')
        order=sorted(eligible,key=lambda g:hashlib.sha256(f'SafeConf-v21-KOLF-role|{g}'.encode()).hexdigest())
        a,b=len(order)//2,3*len(order)//4
        role={g:'predictor_train' if i<a else 'feedback' if i<b else 'confirmation' for i,g in enumerate(order)}
        table=pd.DataFrame([{'gene':g,'role':role[g],'n_cells':int(target_counts[g]),'task_id':g+'::KOLF2.1J'} for g in order])
        table.to_parquet(OUT/'CANDIDATE_TASK_ROLES.parquet',index=False)
        target_counts.rename_axis('gene').rename('n_cells').to_csv(OUT/'METADATA_TARGET_COUNTS.csv')
        actual=h5_column(h['var'],'gene_ids').astype(str)
        symbols=h5_column(h['var'],'_index').astype(str)
        axes=json.loads(SOURCE_AXIS.read_text());axes=axes.get('gene_ids',axes) if isinstance(axes,dict) else axes
        # The source axis may be symbols. Exact identities are recorded; never
        # fabricate missing genes or match perturbation labels to response columns.
        common_ids=sorted(set(map(str,axes))&set(actual));common_symbols=sorted(set(map(str,axes))&set(symbols))
        chosen=common_ids if len(common_ids)>=len(common_symbols) else common_symbols
        var_values=actual if len(common_ids)>=len(common_symbols) else symbols
        if PANEL_SIZE is not None:
            chosen=sorted(chosen,key=lambda g:hashlib.sha256(f'SafeConf-v21-KOLF-resource-panel|{g}'.encode()).hexdigest())[:PANEL_SIZE]
        if len(chosen)<500:raise RuntimeError('fixed existing output axis has insufficient overlap')
        cols=np.asarray([i for i,g in enumerate(var_values) if g in set(chosen)])
        write_json(OUT/'CANDIDATE_AXIS.json',{'labels':var_values[cols].tolist(),'ensembl_ids':actual[cols].tolist(),
            'column_indices':cols.tolist(),'axis_hash':digest_ids(var_values[cols]),'n_output_genes':len(cols),
            'selection_uses_effect_values':False,'missing_genes_zero_filled':False})
        if 'counts' not in h['layers']:raise RuntimeError('no explicit raw counts layer')
        group=h['layers/counts'];encoding=group.attrs.get('encoding-type');encoding=encoding.decode() if isinstance(encoding,bytes) else str(encoding)
        write_json(OUT/'COUNTS_STORAGE_METADATA.json',{'encoding':encoding,
            'arrays':{k:{'shape':list(group[k].shape),'dtype':str(group[k].dtype),'chunks':list(group[k].chunks or []),
                'filter_count':group[k].id.get_create_plist().get_nfilters()}
                for k in ['data','indices','indptr']},
            'response_values_read':False})
        checkpoint('identity_roles_axis_counts_structure')
        ptr=group['indptr'][:]
        allowed_genes=set(table[table.role.ne('confirmation')].gene)
        allowed_cells=int(table[table.role.ne('confirmation')].n_cells.sum())+int(ntc.sum())
        # np.isin(unicode_array, object-Series) takes the O(cells*genes)
        # fallback. Use a hash membership test only when CSR needs row IDs;
        # CSC cost estimation needs column IDs and metadata counts alone.
        selected=np.flatnonzero(pd.Series(names).isin(allowed_genes).to_numpy()|ntc) if encoding=='csr_matrix' else None
        # CSC storage requires selected output columns; CSR can restrict to
        # upstream/feedback/control rows. Both estimates are physical bytes,
        # without decoding a single count or index into an expression array.
        units=cols if encoding=='csc_matrix' else selected if encoding=='csr_matrix' else None
        if units is None:raise RuntimeError('unsupported raw-count sparse encoding')
        chunk_cost={};physical_bytes=0
        for key in ['data','indices']:
            dataset=group[key]
            if not dataset.chunks:raise RuntimeError('count arrays are not chunk-addressable')
            step=dataset.chunks[0];wanted=set()
            for i in units:
                begin,end=int(ptr[i]),int(ptr[i+1])
                if end>begin:wanted.update(range(begin//step,(end-1)//step+1))
            cost=[0,0]
            def accept(info):
                index=int(info.chunk_offset[0])//step
                if index in wanted:cost[0]+=int(info.size);cost[1]+=1
            if dataset.id.get_create_plist().get_nfilters()==0:
                # No compression/filter: allocated chunk length is bounded by
                # nominal elements*dtype bytes. This conservative upper bound
                # avoids thousands of costly legacy HDF5 index lookups.
                cost=[len(wanted)*step*dataset.dtype.itemsize,len(wanted)]
                mode='unfiltered_chunk_payload_upper_bound'
            else:
                mode='exact_compressed_chunk_offsets'
                for ordinal,index in enumerate(sorted(wanted)):
                    accept(dataset.id.get_chunk_info_by_coord((index*step,)))
                    if ordinal%8192==0:checkpoint('required_chunk_offsets_'+key)
            physical_bytes+=cost[0];chunk_cost[key]={'payload_bytes_upper_bound':cost[0],'chunks':cost[1],'estimation':mode}
            checkpoint('physical_chunk_cost_'+key)
        remaining=100_000_000_000-38_000_000_000-json.loads(LEDGER.read_text())['this_run_network_payload_bytes']
        feasible=physical_bytes*1.10<=remaining
        result={'status':'RAW_ASSET_METADATA_AND_COST_QUALIFIED' if feasible else 'RAW_ASSET_EXCEEDS_REMAINING_DOWNLOAD_ALLOWANCE',
            'role_counts':table.groupby('role').size().to_dict(),'gene_targets_total':len(target_counts),'cohort_gene_clusters':len(table),
            'control_cells':int(ntc.sum()),'allowed_numeric_candidate_cells':allowed_cells,'n_output_genes':len(cols),
            'counts_encoding':encoding,'count_chunks':chunk_cost,'physical_payload_upper_estimate':physical_bytes,
            'payload_estimate_plus10percent_margin':physical_bytes*1.10,'remaining_cumulative_payload_allowance':remaining,
            'response_values_read':False,'confirmation_truth_read':False,'has_frozen_predictor':False,
            'author_quantized_viewer_not_used_as_truth':True,'cohort_roles_sha256':sha(OUT/'CANDIDATE_TASK_ROLES.parquet'),
            'next_action':'Define role-guarded counts extraction and fixed reused-gene-embedding Ridge/MLP predictors; capability gate before confirmation' if feasible else 'Find author high-precision aggregated responses; do not shrink axis after seeing performance',
            'wall_seconds':time.monotonic()-start,'metadata_payload_bytes':json.loads(LEDGER.read_text())['this_run_network_payload_bytes']-before}
        write_json(OUT/'ASSET_QUALIFICATION.json',result);write_json(OUT/'STATUS.json',result);print(json.dumps(result),flush=True)
    io.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--panel-size',type=int);args=ap.parse_args()
    if args.panel_size is not None:
        if args.panel_size<500:raise ValueError('panel must contain at least500 registered output genes')
        PANEL_SIZE=args.panel_size;OUT=RUN/f'backup_asset/kolf_budget_panel{PANEL_SIZE}_v1'
    try:main()
    except Exception as e:
        write_json(OUT/'FAILURE_RECEIPT.json',{'status':'METADATA_ACCESS_OR_CONTRACT_FAILURE','error':str(e),
            'error_type':type(e).__name__,'response_values_read':False,'confirmation_truth_read':False})
        raise
