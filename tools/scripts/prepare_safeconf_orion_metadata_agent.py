#!/usr/bin/env python3
"""Freeze Orion metadata only. CLI never reads expression columns or trains.

Future expression readers require a separately authorized permit. A filtered
Arrow Scanner protects returned rows; it cannot promise that mixed private
Parquet pages remain unparsed in the native decoder.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import threading
import urllib.request

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001'
MANIFEST = RESEARCH / 'independent_asset_followup/ORION_CANDIDATE_FILES.json'
DOWNLOAD_ROOT = Path('/home/yyf/data/safeconf_orion_frozen40_20261002')
RUNTIME = DOWNLOAD_ROOT / 'metadata_preparation_20261002_v1'
SOURCE_IDS = Path('/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201/gene_ids.json')
PREFIX = 'SafeConf-Orion-20261002-v1|'
STUDY = 'Huang2025_XAtlasOrion'
FINAL_SHA = 'd16cea664ca508659b97dfedbe165ec285609102a0fcb429349335c093a014f4'
OLD_SHA = '997bdbf33c079746457646cd5fe433e3812063ebc1aefcb6f590376086f15900'
PREVIEW_GENES = frozenset('APOA4 CDK5RAP2 CHKA KCNK7 SBF2 SIGLEC5 SLC39A8 ST14 VSNL1'.split())
META_COLUMNS = ['cell_barcode', 'sample', 'num_features', 'guide_target', 'gene_target',
                'n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt',
                'pass_guide_filter']
EXPRESSION_COLUMNS = ['gene_token_id', 'gene_expression']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def immutable_write(path, data):
    """Exclusive creation + read-only permissions; fail rather than overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Frozen artifact differs: {path}')
        return
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def gene_role(gene):
    if gene in PREVIEW_GENES:
        return 'TRAIN'
    number = int(hashlib.sha256((PREFIX + gene).encode('utf-8')).hexdigest(), 16)
    # Exact integer thresholds, with no floating-point/platform variation.
    if number * 5 < (1 << 256) * 3:
        return 'TRAIN'
    if number * 5 < (1 << 256) * 4:
        return 'VALIDATION'
    return 'TEST'


def operational(manifest):
    return {'release_sha': manifest['release_sha'],
            'files': [{key: row[key] for key in ('path', 'bytes', 'lfs_sha256', 'download_url')}
                      for row in manifest['files']],
            'gene_metadata': {key: manifest['gene_metadata'][key]
                              for key in ('path', 'bytes', 'lfs_sha256', 'download_url')}}


def freeze_contract(runtime):
    final_bytes = MANIFEST.read_bytes()
    if sha(final_bytes) != FINAL_SHA:
        raise RuntimeError('Final candidate manifest changed; do not change active download registration')
    final = json.loads(final_bytes)
    old = json.loads(final_bytes)
    old['selection'] = ('HCT116 numeric Batch1-20 plus HEK293T numeric Batch1-20; '
                        'fixed before target-count audit; never replace shards')
    old['preview_exposure'].pop('selection_timing_clarification', None)
    old_bytes = json_bytes(old)
    if sha(old_bytes) != OLD_SHA:
        raise RuntimeError('Cannot exactly reconstruct registered original manifest')
    if operational(old) != operational(final):
        raise RuntimeError('Operational download fields changed')
    registration = json.loads((DOWNLOAD_ROOT / 'DOWNLOAD_REGISTRATION.json').read_bytes())
    if registration['manifest_sha256'] != OLD_SHA or registration['release_sha'] != final['release_sha']:
        raise RuntimeError('Active download identity differs')
    immutable_write(runtime / f'snapshots/ORION_CANDIDATE_FILES.{FINAL_SHA}.json', final_bytes)
    immutable_write(runtime / f'snapshots/ORION_CANDIDATE_FILES.{OLD_SHA}.json', old_bytes)
    immutable_write(runtime / 'snapshots/DOWNLOAD_REGISTRATION.observed.json', json_bytes(registration))
    amendment = {'kind': 'NON_OPERATIONAL_TEXT_AMENDMENT', 'registered_manifest_sha256': OLD_SHA,
                 'final_manifest_sha256': FINAL_SHA, 'operational_tuple_sha256': sha(packed(operational(final))),
                 'exact_old_snapshot_reconstructed_hash_matches_registration': True,
                 'forty_paths_sizes_LFS_and_download_urls_unchanged': True,
                 'active_download_registration_modified': False,
                 'difference': 'Selection-timing wording only; prior Batch1 labels/preview explicitly disclosed'}
    immutable_write(runtime / 'NON_OPERATIONAL_AMENDMENT.json', json_bytes(amendment))
    policy = {'schema': 'orion_metadata_preparation_v1', 'release_sha': final['release_sha'],
              'source_manifest_sha256': FINAL_SHA, 'operational_tuple_sha256': amendment['operational_tuple_sha256'],
              'gene_split_sha256_prefix_utf8_exact': PREFIX,
              'split_rule': 'SHA256 integer: [0,.6) TRAIN; [.6,.8) VALIDATION; [.8,1) TEST',
              'preview_genes_forced_train_all_contexts': sorted(PREVIEW_GENES),
              'seen_cells_DEV_only': final['preview_exposure']['cells'],
              'qc_rule_frozen_before_full_metadata': 'pass_guide_filter==1 AND num_features==2 AND finite positive total_counts; no expression threshold',
              'QC_thresholds': {'pass_guide_filter': 1, 'num_features': 2, 'total_counts_gt': 0},
              'pct_counts_mt_policy': 'Recorded proxy only; no adaptive mitochondrial cut-off',
              'prospective_estimator': 'mean(log1p(4000 * cellUMI / full_library_total)) per gene and own-context perturbation/control; no logsum; no fitted test median',
              'normalization_scale': 4000,
              'scale_source': 'Parent verified pinned TxPert08d82eea README/data/onboard_dataset.ipynb normalize_total(target_sum=4000) then log1p; fixed before Orion test expression',
              'library_total_scope': 'Official full-library total_counts metadata; never recompute denominator on Source3285 projection; consistency verification only on authorized training rows',
              'non_NTC_unit': 'originalstudy|gene|cellline; all models share gene role',
              'independent_cluster_key': 'gene; synchronize both cell lines in resampling',
              'control_label_exact': 'Non-Targeting',
              'control_scope': 'Own-context retained NTC cells are TRAIN_CONTROL_SOURCE_SCOPE, except seen cells; no cross-context pooling. Current contract is gene holdout within two contexts, not heldout-context authorization.',
              'biological_replicates_verified': False, 'Quality_role': 'proxy/biological_replicate_unknown',
              'validation_use': 'Metadata split only; competence/calibration subroles and method parameters require subsequent frozen model contract',
              'expression_materialization_permitted': False, 'test_truth_opened': False,
              'new_upstream_attempt_started': False, 'whole_study_pristine': False,
              'private_row_never_parsed_contract_gap': 'Mixed Parquet rowgroups: Scanner filters returned rows after native decode. Strict custodian guarantee requires physical pure-role rowgroups/files or must reject them.'}
    immutable_write(runtime / 'PREPARATION_POLICY.json', json_bytes(policy))
    return final, policy, amendment


class MetadataBuffer(io.RawIOBase):
    """Expose only cached metadata/footer intervals, never expression bytes."""
    def __init__(self, size, spans):
        self.size, self.spans, self.position = size, spans, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.position
        start, end = self.position, min(self.size, self.position + count)
        self.position = end
        for offset, content in self.spans:
            if offset <= start and end <= offset + len(content):
                return content[start-offset:end-offset]
        raise RuntimeError(f'Forbidden non-metadata byte range {start}:{end}')


def http_range(url, start, end, timeout):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status != 206:
            raise RuntimeError('Remote range not honored; expression download forbidden here')
        data = response.read(end-start+2)
        final_url = response.url
    if len(data) != end-start+1:
        raise RuntimeError('Wrong range length')
    return data, final_url


def verified_local(item):
    path = DOWNLOAD_ROOT / item['path']
    if not path.exists() or path.stat().st_size != item['bytes']:
        return None
    status_path = DOWNLOAD_ROOT / 'DOWNLOAD_STATUS.json'
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text())
    if any(row['path'] == item['path'] and row['sha256'] == item['lfs_sha256'] for row in status.get('files', [])):
        return path
    return None


def collect_shard(item, runtime, policy, gene_names_unique, remote, timeout):
    base = Path(item['path']).stem
    identity_path = runtime / 'file_identity' / f'{base}.json'
    output_path = runtime / 'row_metadata' / f'{base}.parquet'
    if identity_path.exists() and output_path.exists():
        identity = json.loads(identity_path.read_text())
        if identity['metadata_parquet_sha256'] != sha(output_path.read_bytes()):
            raise RuntimeError('Metadata cache hash mismatch')
        return identity
    local = verified_local(item)
    if local:
        with local.open('rb') as stream:
            stream.seek(item['bytes']-8); tail = stream.read(8)
            footer_size = struct.unpack('<I', tail[:4])[0]
            stream.seek(item['bytes']-footer_size-8); footer = stream.read(footer_size+8)
        reader_url = None
        access = 'LOCAL_OPAQUE_FILE_HASH_VERIFIED_BY_ACTIVE_DOWNLOADER'
    elif remote:
        tail, reader_url = http_range(item['download_url'], item['bytes']-8, item['bytes']-1, timeout)
        footer_size = struct.unpack('<I', tail[:4])[0]
        footer, reader_url = http_range(reader_url, item['bytes']-footer_size-8, item['bytes']-1, timeout)
        access = 'PINNED_REMOTE_METADATA_RANGES_ONLY_FULL_OBJECT_HASH_PENDING_DOWNLOAD'
    else:
        raise RuntimeError(f'Download pending; do not read .part: {item["path"]}')
    if tail[4:] != b'PAR1' or footer_size > 2_000_000:
        raise RuntimeError('Invalid/oversize Parquet footer')
    meta = pq.ParquetFile(io.BytesIO(b'PAR1'+footer)).metadata
    spans, forbidden = [], []
    for group in range(meta.num_row_groups):
        for column in range(meta.num_columns):
            col = meta.row_group(group).column(column)
            offset = col.dictionary_page_offset or col.data_page_offset
            interval = (offset, offset + col.total_compressed_size)
            if col.path_in_schema in META_COLUMNS:
                spans.append((group, col.path_in_schema, *interval))
            else:
                forbidden.append(interval)
    if len(spans) != len(META_COLUMNS)*meta.num_row_groups:
        raise RuntimeError('Expected metadata columns missing; no adaptive schema fallback')
    start, end = min(row[2] for row in spans), max(row[3] for row in spans)
    if any(a < end and b > start for a, b in forbidden):
        raise RuntimeError('Metadata envelope overlaps expression/token column; refuse coalesced request')
    if local:
        with local.open('rb') as stream:
            stream.seek(start); raw = stream.read(end-start)
    else:
        raw, _ = http_range(reader_url, start, end-1, timeout)
    buffer = MetadataBuffer(item['bytes'], [(start, raw), (item['bytes']-len(footer), footer)])
    table = pq.ParquetFile(buffer, metadata=meta).read(columns=META_COLUMNS, use_threads=False)
    values = table.to_pydict()
    context = item['context']
    expected_sample = f'{context}_Batch{item["batch_index"]}'
    if set(values['sample']) != {expected_sample}:
        raise RuntimeError('Sample labels do not match fixed shard identity')
    if len(set(values['cell_barcode'])) != table.num_rows:
        raise RuntimeError('Duplicate original cell barcodes within file')
    seen = {row['cell_barcode'] for row in policy['seen_cells_DEV_only']}
    roles, grow, units, groups, qc = [], [], [], [], []
    row_group = []
    for i in range(meta.num_row_groups):
        row_group.extend([i]*meta.row_group(i).num_rows)
    for i, gene in enumerate(values['gene_target']):
        valid = (values['pass_guide_filter'][i] == 1 and values['num_features'][i] == 2
                 and values['total_counts'][i] is not None
                 and math.isfinite(float(values['total_counts'][i])) and values['total_counts'][i] > 0)
        gr = 'CONTROL' if gene == 'Non-Targeting' else gene_role(gene) if isinstance(gene, str) else 'INVALID'
        if values['cell_barcode'][i] in seen:
            role = 'DEV_EXPOSED_CELL'
        elif not valid:
            role = 'EXCLUDED_FIXED_METADATA_QC'
        elif gene == 'Non-Targeting':
            role = 'TRAIN_CONTROL_SOURCE_SCOPE'
        elif gene not in gene_names_unique:
            role = 'EXCLUDED_MISSING_OR_AMBIGUOUS_TARGET_ID'
        else:
            role = gr
        grow.append(gr); roles.append(role); qc.append(valid)
        units.append(f'{STUDY}|{gene}|{context}'); groups.append(gene)
    for name, data in [('gene_role', grow), ('row_role', roles), ('fixed_metadata_QC_pass', qc),
                       ('biological_unit', units), ('cluster_gene', groups), ('original_row_index', range(table.num_rows)),
                       ('original_row_group', row_group)]:
        table = table.append_column(name, pa.array(list(data)))
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    logical_sha = sha(sink.getvalue().to_pybytes())
    output = io.BytesIO(); pq.write_table(table, output, compression='zstd')
    outbytes = output.getvalue()
    immutable_write(output_path, outbytes)
    rg_counts = {str(g): dict(collections.Counter(roles[i] for i, rg in enumerate(row_group) if rg == g))
                 for g in range(meta.num_row_groups)}
    identity = {'path': item['path'], 'context': context, 'sample': expected_sample,
                'pinned_release_sha': policy['release_sha'], 'publisher_whole_file_bytes': item['bytes'],
                'publisher_LFS_sha256': item['lfs_sha256'], 'whole_file_hash_verification': access,
                'rows': table.num_rows, 'row_groups': meta.num_row_groups,
                'original_footer_sha256': sha(footer), 'metadata_compressed_envelope_sha256': sha(raw),
                'metadata_envelope_start': start, 'metadata_envelope_bytes': len(raw),
                'metadata_columns_compressed_sha256': {f'{g}:{name}': sha(raw[a-start:b-start]) for g,name,a,b in spans},
                'logical_metadata_IPC_sha256': logical_sha, 'metadata_parquet_sha256': sha(outbytes),
                'metadata_parquet_path': str(output_path), 'metadata_columns_read': META_COLUMNS,
                'expression_columns_read': [], 'expression_rows_exposed': 0,
                'role_counts': dict(collections.Counter(roles)), 'row_group_role_counts': rg_counts,
                'original_cell_identity': 'study|cellline|cell_barcode; source row position/group retained',
                'Quality': 'proxy/biological_replicate_unknown'}
    immutable_write(identity_path, json_bytes(identity))
    return identity


def prepare_gene_manifest(manifest, runtime):
    item = manifest['gene_metadata']
    cache = runtime / 'gene_metadata.parquet'
    if cache.exists():
        blob = cache.read_bytes()
    else:
        local = verified_local(item)
        if local:
            blob = local.read_bytes()
        else:
            with urllib.request.urlopen(item['download_url'], timeout=25) as response:
                blob = response.read(item['bytes']+1)
    if len(blob) != item['bytes'] or sha(blob) != item['lfs_sha256']:
        raise RuntimeError('Official gene metadata identity mismatch')
    immutable_write(cache, blob)
    table = pq.read_table(io.BytesIO(blob), columns=['gene_name','ensembl_id','gene_token_id'])
    names = table['gene_name'].to_pylist(); counts = collections.Counter(names)
    unique = {name for name, count in counts.items() if count == 1}
    mapping = {row['gene_name']: row for row in table.to_pylist() if row['gene_name'] in unique}
    source_bytes = SOURCE_IDS.read_bytes(); source = json.loads(source_bytes)['gene_ids']
    if len(source) != 3352 or len(set(source)) != 3352:
        raise RuntimeError('Source ID axis changed')
    rows = [{'axis_index': axis, 'source_index': index, 'gene_name': gene,
             'orion_gene_token_id': mapping[gene]['gene_token_id'], 'orion_ensembl_id': mapping[gene]['ensembl_id']}
            for axis,(index,gene) in enumerate((i,g) for i,g in enumerate(source) if g in unique)]
    if len(rows) != 3285:
        raise RuntimeError('Expected frozen exact Source3285 axis changed')
    output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    immutable_write(runtime / 'GENE_MANIFEST.csv', output.getvalue().encode())
    immutable_write(runtime / 'GENE_AXIS_IDENTITY.json', json_bytes({'source_path': str(SOURCE_IDS),
        'source_ids_sha256': sha(source_bytes), 'official_gene_metadata_sha256': sha(blob),
        'Source3352_common_axis':3285, 'order':'Source ID order; exact string, unique Orion symbol only',
        'source_missing_ids':[g for g in source if g not in unique],
        'gene_manifest_sha256':sha(output.getvalue().encode()), 'mapping':'No synonyms, capitalization, orthology, zero-fill, or expression-based selection'}))
    return unique


def iter_authorized_expression_batches(runtime, permit_path, context=None, requested_roles=('TRAIN',), batch_size=512):
    """Future-only guarded interface. Metadata CLI never invokes this function.

    RETURNED_ROWS_BLIND allows native decoding of mixed pages but returns only
    authorized rows. PRIVATE_ROWS_NEVER_PARSED rejects any mixed rowgroup/file
    before constructing an expression scanner. No implicit guarantee downgrade.
    """
    permit = json.loads(Path(permit_path).read_text())
    if not permit.get('expression_row_materialization_permitted') or not permit.get('frozen_method_contract_sha256'):
        raise RuntimeError('A separate frozen method contract and expression permit are required')
    method_path = Path(permit.get('frozen_method_contract_path', ''))
    if not method_path.is_file() or sha(method_path.read_bytes()) != permit['frozen_method_contract_sha256']:
        raise RuntimeError('Frozen method contract file/hash is missing or differs')
    requested = set(requested_roles)
    if not requested or not requested <= {'TRAIN','VALIDATION','TRAIN_CONTROL_SOURCE_SCOPE'}:
        raise RuntimeError('This interface forbids TEST/DEV/private role requests')
    if not requested <= set(permit.get('authorized_roles', [])):
        raise RuntimeError('Roles exceed explicit permit')
    if context not in {'HCT116','HEK293T'} or context not in permit.get('authorized_contexts', []):
        raise RuntimeError('Explicit single authorized context required; never pool controls across contexts')
    guarantee = permit.get('reader_guarantee')
    if guarantee not in {'RETURNED_ROWS_BLIND','PRIVATE_ROWS_NEVER_PARSED'}:
        raise RuntimeError('Declare native parsing guarantee explicitly')
    policy = json.loads((runtime/'PREPARATION_POLICY.json').read_text())
    if permit.get('preparation_policy_sha256') != sha((runtime/'PREPARATION_POLICY.json').read_bytes()):
        raise RuntimeError('Permit does not bind the frozen preparation policy')
    identities = [row for row in json.loads((runtime/'FILE_METADATA_IDENTITY.json').read_text())
                  if row['context']==context]
    # Strict mode rejects mixed metadata before any expression scanner exists.
    if guarantee == 'PRIVATE_ROWS_NEVER_PARSED':
        mixed = [x['path'] for x in identities if set(x['role_counts'])-requested]
        if mixed:
            raise RuntimeError('Mixed private rows require an independent physical custodian; expression remains unopened')
    for identity in identities:
        metadata_path = Path(identity['metadata_parquet_path'])
        if sha(metadata_path.read_bytes()) != identity['metadata_parquet_sha256']:
            raise RuntimeError('Frozen row metadata was modified')
        metadata = pq.read_table(metadata_path, columns=['cell_barcode','gene_target','row_role'])
        md = metadata.to_pydict()
        allowed_cells = [c for c,r in zip(md['cell_barcode'], md['row_role']) if r in requested]
        if not allowed_cells:
            continue
        path = DOWNLOAD_ROOT/identity['path']
        item = {'path':identity['path'],'bytes':identity['publisher_whole_file_bytes'],'lfs_sha256':identity['publisher_LFS_sha256']}
        if verified_local(item) is None:
            raise RuntimeError('Expression source is not verified complete; never read .part')
        allowed_genes = sorted({g for g,r in zip(md['gene_target'],md['row_role']) if r in requested})
        predicate = ds.field('gene_target').isin(allowed_genes) & ds.field('cell_barcode').isin(allowed_cells)
        scanner = ds.dataset(str(path),format='parquet').scanner(
            columns=['cell_barcode','sample','gene_target',*EXPRESSION_COLUMNS], filter=predicate,
            batch_size=batch_size,batch_readahead=0,fragment_readahead=0,use_threads=False,
            fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False,use_buffered_stream=True))
        allowed_set = set(allowed_cells)
        for batch in scanner.to_batches():
            if not set(batch['cell_barcode'].to_pylist()) <= allowed_set:
                raise RuntimeError('Returned expression rows violate permitted metadata membership')
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--allow-remote-metadata', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--timeout', type=int, default=20)
    args = parser.parse_args()
    manifest, policy, amendment = freeze_contract(args.runtime)
    genes_unique = prepare_gene_manifest(manifest,args.runtime)
    identities, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(collect_shard,row,args.runtime,policy,genes_unique,args.allow_remote_metadata,args.timeout):row
                   for row in manifest['files']}
        for future in as_completed(futures):
            row = futures[future]
            try:
                identities.append(future.result())
                print(f'Metadata only {len(identities)}/40: {row["path"]}',flush=True)
            except Exception as error:
                errors.append({'path':row['path'],'error':str(error)})
    identities.sort(key=lambda x:x['path'])
    aggregate = collections.Counter(); context_counts={}; target_counts=collections.defaultdict(collections.Counter)
    seen_cell_count=0; all_cells=set(); target_set=set()
    for identity in identities:
        aggregate.update(identity['role_counts'])
        table=pq.read_table(identity['metadata_parquet_path'],columns=['cell_barcode','gene_target','row_role'])
        for cell,gene,role in zip(*(table[c].to_pylist() for c in ['cell_barcode','gene_target','row_role'])):
            key=(identity['context'],cell)
            if key in all_cells: raise RuntimeError('Duplicate original cell identity across shards')
            all_cells.add(key)
            if gene!='Non-Targeting' and isinstance(gene,str): target_set.add(gene)
            if role in {'TRAIN','VALIDATION','TEST'}:target_counts[identity['context']][gene]+=1
            if role=='DEV_EXPOSED_CELL':seen_cell_count+=1
        counter=context_counts.setdefault(identity['context'],collections.Counter());counter.update(identity['role_counts'])
    ready=len(identities)==40 and not errors
    summary={'schema':'orion_metadata_preparation_summary_v1','metadata_complete':ready,'n_files':len(identities),
             'expected_files':40,'runtime':str(args.runtime),'final_manifest_sha256':FINAL_SHA,'amendment':amendment,
             'fixed_policy_sha256':sha((args.runtime/'PREPARATION_POLICY.json').read_bytes()),
             'source_common_axis':3285,'original_cell_rows':sum(x['rows'] for x in identities),
             'distinct_nonNTC_gene_labels':len(target_set),'gene_role_counts':dict(collections.Counter(gene_role(g) for g in target_set)),
             'row_role_counts':dict(aggregate),'context_row_role_counts':{c:dict(v) for c,v in context_counts.items()},
             'seen_cells_DEV_matches':seen_cell_count,'metadata_envelope_bytes':sum(x['metadata_envelope_bytes'] for x in identities),
             'expression_columns_read':[],'expression_rows_exposed':0,'new_upstream_attempt_started':False,
             'Quality':'proxy/biological_replicate_unknown','whole_study_pristine':False,
             'strict_private_rows_never_parsed_ready':False,'errors':errors}
    if ready:
        immutable_write(args.runtime/'FILE_METADATA_IDENTITY.json',json_bytes(identities))
        output=io.StringIO();w=csv.writer(output);w.writerow(['gene','gene_role','hash_sha256','forced_preview_TRAIN'])
        for gene in sorted(target_set):w.writerow([gene,gene_role(gene),hashlib.sha256((PREFIX+gene).encode()).hexdigest(),gene in PREVIEW_GENES])
        immutable_write(args.runtime/'GENE_SPLIT.csv',output.getvalue().encode())
        summary['gene_split_sha256']=sha(output.getvalue().encode())
        common=set.intersection(*(set(v) for v in target_counts.values())) if len(target_counts)==2 else set()
        eligible={g for g in common if all(v[g]>=30 for v in target_counts.values())}
        summary['both_contexts_retained_ge30_gene_clusters']=len(eligible)
        summary['eligible_ge30_gene_role_counts']=dict(collections.Counter(gene_role(g) for g in eligible))
        immutable_write(args.runtime/'CONTROLS_SCOPE.json',json_bytes({'policy':policy['control_scope'],
            'counts':{c:v.get('TRAIN_CONTROL_SOURCE_SCOPE',0) for c,v in context_counts.items()},
            'seen_NTC_cell_excluded':next(x for x in policy['seen_cells_DEV_only'] if x['gene_target']=='Non-Targeting'),
            'cross_context_pooling_permitted':False,'public_control_expression_access_yet_permitted':False}))
        immutable_write(args.runtime/'SPLIT_SUMMARY.json',json_bytes(summary))
    # Only light summaries go to repository. Runtime holds full cell metadata.
    destination=RESEARCH/'orion_preparation';destination.mkdir(parents=True,exist_ok=True)
    (destination/'PREPARATION_SUMMARY.json').write_bytes(json_bytes(summary))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if not ready:raise SystemExit(2)


if __name__=='__main__':
    main()
