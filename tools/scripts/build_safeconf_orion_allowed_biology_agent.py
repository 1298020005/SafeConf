#!/usr/bin/env python3
"""Metadata planning and guarded TRAIN/VALIDATION pseudobulk aggregation.

No command opens real expression without an immutable scientific contract and
separate permit binding the exact loader/reader code, metadata and raw identity.
The synthetic-test command uses only generated fixtures. Numeric conversion is
delegated to the specialized reader; no Arrow numeric expression read is used.
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
import resource
import shutil
import struct
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/biology_loader'
METADATA = Path('/home/yyf/data/safeconf_orion_frozen40_20261002/metadata_preparation_20261002_v1')
RAW = METADATA.parent
RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_allowed_biology_20261002_v1')
READER = ROOT / 'tools/scripts/probe_safeconf_private_parquet_agent.py'
CONTROL_ROLE = 'TRAIN_CONTROL_SOURCE_SCOPE'
LEGAL_ROLES = frozenset(('TRAIN', 'VALIDATION', CONTROL_ROLE))
CONTEXTS = frozenset(('HCT116', 'HEK293T'))
GUARANTEE = 'PRIVATE_NUMERIC_ROWS_NEVER_MATERIALIZED'
SCHEMA = 'safeconf_orion_allowed_biology_loader_v1'
DEFAULT_BUDGETS = {'max_compressed_chunk_bytes': 1024**3,
                   'max_page_uncompressed_bytes': 128*1024**2,
                   'max_page_level_entries': 16*1024**2,
                   'max_resident_bytes': 6*1024**3,
                   'max_token_spool_bytes_per_file': 4*1024**3,
                   'max_group_matrix_bytes': 16*1024**3,
                   'max_seconds_per_file': 600,
                   'max_cells_per_file': 50000}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if immutable:
        path.chmod(0o444)


def immutable_json(path, expected_sha=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_mode & 0o222:
        raise RuntimeError(f'Immutable read-only file required: {path}')
    if expected_sha is not None and sha(path) != expected_sha:
        raise RuntimeError(f'Immutable file SHA differs: {path}')
    return json.loads(path.read_text())


def checked_child(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix == '.part':
        raise RuntimeError('Raw file must have a safe complete relative path')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise RuntimeError('Raw path escaped its pinned root')
    return result


def authorization(metadata_root, raw_root, contract_path, permit_path, roles, contexts):
    """Complete immutable identity/authorization checks before any raw access."""
    roles, contexts = set(roles), set(contexts)
    if not roles or not roles <= LEGAL_ROLES or CONTROL_ROLE not in roles:
        raise RuntimeError('Only exact TRAIN/VALIDATION plus own TRAIN NTC roles are allowed')
    if not contexts or not contexts <= CONTEXTS:
        raise RuntimeError('Exact registered contexts required')
    permit = immutable_json(permit_path)
    if not permit.get('expression_row_materialization_permitted'):
        raise RuntimeError('Explicit expression role permit required')
    if permit.get('test_numeric_materialization_permitted', False):
        raise RuntimeError('This loader cannot accept TEST numeric authorization')
    if not roles <= set(permit.get('authorized_roles', [])) or not contexts <= set(permit.get('authorized_contexts', [])):
        raise RuntimeError('Requested roles/contexts exceed explicit permit')
    if permit.get('reader_guarantee') != GUARANTEE:
        raise RuntimeError('Permit must bind specialized pre-mask numeric conversion guarantee')
    if Path(permit.get('frozen_method_contract_path', '')).resolve() != Path(contract_path).resolve():
        raise RuntimeError('Permit scientific contract path differs')
    contract = immutable_json(contract_path, permit.get('frozen_method_contract_sha256'))
    if not permit.get('frozen_method_contract_sha256'):
        raise RuntimeError('Contract SHA required')
    privacy = contract.get('privacy', {})
    if not roles <= set(privacy.get('current_permitted_roles', [])) or privacy.get('private_numeric_materialization') is not False:
        raise RuntimeError('Scientific contract role/privacy rules differ')
    normalization = contract.get('normalization', {})
    if normalization.get('formula') != 'mean_cell(log1p(4000 * raw_gene_UMI / official_full_library_total_counts))':
        raise RuntimeError('Frozen normalization formula differs')
    if normalization.get('unknown_token_policy') != 'fail_closed':
        raise RuntimeError('Missing or unknown tokens must fail closed')
    if Path(permit.get('metadata_root', '')).resolve() != Path(metadata_root).resolve() or Path(permit.get('raw_root', '')).resolve() != Path(raw_root).resolve():
        raise RuntimeError('Permit roots differ')
    # A reader implementation can receive a technical amendment in the separate
    # permit. The scientific dependencies have no override path.
    overrides = {str(Path(x['path']).resolve()): x for x in permit.get('technical_dependency_overrides', [])}
    if set(overrides) - {str(READER.resolve())}:
        raise RuntimeError('Only reader implementation may receive a technical dependency amendment')
    for dep in contract.get('immutable_dependencies', []):
        dep_path = Path(dep['path'])
        expected = dep['sha256']
        override = overrides.get(str(dep_path.resolve()))
        if override:
            if override.get('original_sha256') != expected:
                raise RuntimeError('Technical override does not bind original reader dependency')
            expected = override['sha256']
        if not dep_path.is_file() or sha(dep_path) != expected:
            raise RuntimeError(f'Contract dependency differs: {dep_path}')
    bindings = permit.get('implementation_bindings', {})
    for name, path in [('loader', Path(__file__)), ('reader', READER)]:
        entry = bindings.get(name, {})
        if Path(entry.get('path', '')).resolve() != path.resolve() or entry.get('sha256') != sha(path):
            raise RuntimeError(f'Exact {name} implementation binding missing or changed')
    for name in ('FILE_METADATA_IDENTITY.json', 'GENE_MANIFEST.csv', 'gene_metadata.parquet', 'PREPARATION_POLICY.json'):
        if permit.get('metadata_bindings', {}).get(name) != sha(Path(metadata_root) / name):
            raise RuntimeError(f'Exact frozen metadata binding missing: {name}')
    synthetic = bool(permit.get('synthetic_only'))
    if synthetic:
        marker = Path(metadata_root) / 'SYNTHETIC_FIXTURE_ONLY.json'
        if not contract.get('synthetic_only') or not marker.is_file():
            raise RuntimeError('Synthetic permits apply only to explicit generated fixture roots')
    elif not permit.get('audited_production_integration_approved'):
        raise RuntimeError('Separate audited production integration approval is required')
    budgets = dict(DEFAULT_BUDGETS)
    budgets.update(permit.get('resource_budgets', {}))
    for key, value in budgets.items():
        if key not in DEFAULT_BUDGETS or type(value) is not int or value <= 0 or value > DEFAULT_BUDGETS[key]:
            raise RuntimeError('Positive resource limits within published ceilings required')
    sum_check_roles = set(permit.get('library_sum_check_roles', ['TRAIN', CONTROL_ROLE]))
    if sum_check_roles != {'TRAIN', CONTROL_ROLE}:
        raise RuntimeError('Frozen library-sum checks and diagnostics are TRAIN/NTC only')
    return contract, permit, budgets


def gene_axes(metadata_root, synthetic=False):
    full = pq.read_table(Path(metadata_root) / 'gene_metadata.parquet', columns=['ensembl_id', 'gene_name', 'gene_token_id']).to_pandas()
    full = full.sort_values('gene_token_id').reset_index(drop=True)
    if full.ensembl_id.isna().any() or not full.ensembl_id.is_unique or not full.gene_token_id.is_unique:
        raise RuntimeError('Full axis needs unique measured Ensembl/token identities')
    if not np.array_equal(full.gene_token_id, np.arange(len(full))):
        raise RuntimeError('Full gene token axis must be ordered contiguous from zero')
    if not synthetic and len(full) != 38606:
        raise RuntimeError('Published full PCA gene count changed')
    endpoint = pd.read_csv(Path(metadata_root) / 'GENE_MANIFEST.csv')
    if not np.array_equal(endpoint.axis_index, np.arange(len(endpoint))) or not endpoint.gene_name.is_unique:
        raise RuntimeError('Frozen Source endpoint axis/order changed')
    if not synthetic and len(endpoint) != 3285:
        raise RuntimeError('Frozen Source endpoint gene count changed')
    symbol_counts = Counter(full.gene_name)
    unique_symbols = {row.gene_name: row for row in full.itertuples(index=False) if symbol_counts[row.gene_name] == 1}
    columns = []
    for row in endpoint.itertuples(index=False):
        mapped = unique_symbols.get(row.gene_name)
        if mapped is None or mapped.ensembl_id != row.orion_ensembl_id or mapped.gene_token_id != row.orion_gene_token_id:
            raise RuntimeError('Endpoint must have exact unique symbol/Ensembl/token mapping')
        columns.append(int(mapped.gene_token_id))
    return full, endpoint, np.asarray(columns, dtype=np.int64), unique_symbols


def read_metadata(identity, roles):
    path = Path(identity['metadata_parquet_path'])
    if sha(path) != identity['metadata_parquet_sha256']:
        raise RuntimeError('Frozen row metadata differs')
    names = ['original_row_index', 'original_row_group', 'row_role', 'gene_target', 'sample',
             'cell_barcode', 'total_counts', 'biological_unit', 'fixed_metadata_QC_pass']
    md = pq.read_table(path, columns=names, use_threads=False).to_pydict()
    if len(md['row_role']) != identity['rows'] or md['original_row_index'] != list(range(identity['rows'])):
        raise RuntimeError('Exact frozen physical row order required')
    if identity['row_groups'] != 1 or set(md['original_row_group']) != {0}:
        raise RuntimeError('Specialized reader supports one physical rowgroup only')
    if set(md['sample']) != {identity['sample']} or len(set(md['cell_barcode'])) != identity['rows']:
        raise RuntimeError('Frozen original cell/sample identity differs')
    mask = [role in roles for role in md['row_role']]
    for row in (i for i, allowed in enumerate(mask) if allowed):
        if not md['fixed_metadata_QC_pass'][row] or not np.isfinite(md['total_counts'][row]) or md['total_counts'][row] <= 0:
            raise RuntimeError('Authorized row violates frozen metadata QC')
        if md['row_role'][row] == CONTROL_ROLE and md['gene_target'][row] != 'Non-Targeting':
            raise RuntimeError('TRAIN NTC scope/label mismatch')
    return md, mask


def metadata_plan(metadata_root, roles, contexts, synthetic=False):
    roles, contexts = set(roles), set(contexts)
    if not roles or not roles <= LEGAL_ROLES or not contexts or not contexts <= CONTEXTS:
        raise RuntimeError('Metadata plan supports currently authorized TRAIN/VALIDATION/NTC only')
    full, endpoint, columns, symbol_map = gene_axes(metadata_root, synthetic)
    identities = [i for i in json.loads((Path(metadata_root) / 'FILE_METADATA_IDENTITY.json').read_text()) if i['context'] in contexts]
    groups, omitted = {}, []
    for identity in identities:
        md, mask = read_metadata(identity, roles)
        for row in (i for i, allowed in enumerate(mask) if allowed):
            gene, role = md['gene_target'][row], md['row_role'][row]
            if role != CONTROL_ROLE and gene not in symbol_map:
                # Ambiguous symbols are removed using only frozen metadata.
                omitted.append((identity['context'], gene, role))
                continue
            key = (identity['context'], gene, role)
            item = groups.setdefault(key, {'context': identity['context'], 'gene': gene, 'role': role,
                       'biological_unit': md['biological_unit'][row], 'n_cells_metadata': 0,
                       'target_ensembl_id': None if role == CONTROL_ROLE else symbol_map[gene].ensembl_id,
                       'target_token_id': None if role == CONTROL_ROLE else int(symbol_map[gene].gene_token_id)})
            if item['biological_unit'] != md['biological_unit'][row]:
                raise RuntimeError('Biological unit identity differs across batches')
            item['n_cells_metadata'] += 1
    records = [groups[key] for key in sorted(groups)]
    for index, item in enumerate(records):
        item['matrix_row'] = index
        item['competence_n_ge30'] = item['n_cells_metadata'] >= 30
    if not records:
        raise RuntimeError('No allowed metadata tasks')
    return full, endpoint, columns, records, identities, sorted(set(omitted))


def trace_new(budgets):
    return {'physical_numeric_chunk_bytes_read': 0, 'opaque_numeric_bytes_decompressed': 0,
            'data_pages': 0, 'numeric_materialization_rows': None, 'private_numeric_values_skipped': 0,
            **{key: budgets[key] for key in ['max_compressed_chunk_bytes', 'max_page_uncompressed_bytes', 'max_page_level_entries']}}


def check_memory(budgets):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if rss > budgets['max_resident_bytes']:
        raise RuntimeError('Resident memory budget exceeded; no outputs committed')
    return rss


def release_map(values):
    values.flush()
    if hasattr(values._mmap, 'madvise'):
        values._mmap.madvise(mmap.MADV_DONTNEED)


def aggregate(metadata_root, raw_root, contract_path, permit_path, output, roles, contexts, forbidden_bytes=()):
    # Authorization precedes footer reads, raw checksums and numeric conversion.
    contract, permit, budgets = authorization(metadata_root, raw_root, contract_path, permit_path, roles, contexts)
    full, endpoint, endpoint_columns, records, identities, omitted = metadata_plan(metadata_root, roles, contexts, bool(permit.get('synthetic_only')))
    planned_control_contexts = {x['context'] for x in records if x['role'] == CONTROL_ROLE}
    if planned_control_contexts != set(contexts):
        raise RuntimeError('Metadata must provide own-context TRAIN NTC before raw access')
    output = Path(output)
    if output.exists():
        raise RuntimeError('New output version required; existing results cannot be overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + f'.incomplete.{os.getpid()}')
    stage.mkdir()
    from tools.scripts import probe_safeconf_private_parquet_agent as reader_api
    iter_selected_numeric_lists = reader_api.iter_selected_numeric_lists
    shape = (len(records), len(full))
    if np.prod(shape, dtype=np.int64) * 8 > budgets['max_group_matrix_bytes']:
        raise RuntimeError('Aggregate matrix disk/mapping budget exceeded')
    sums = np.lib.format.open_memmap(stage / 'FULL_SUMS.npy', mode='w+', dtype=np.float64, shape=shape)
    counts = np.lib.format.open_memmap(stage / 'CELL_COUNTS.npy', mode='w+', dtype=np.uint64, shape=(len(records),))
    # Bounded initialization: the full mapped assay can be several GiB. Never
    # touch it all before dropping clean pages, because ru_maxrss is sticky.
    for first in range(0, len(records), 32):
        sums[first:first+32] = 0.
        release_map(sums)
        check_memory(budgets)
    counts[:] = 0
    groups = {(x['context'], x['gene'], x['role']): x['matrix_row'] for x in records}
    file_audits = []
    started = time.monotonic()
    forbidden_tokens, forbidden_values = (), ()
    if forbidden_bytes:
        forbidden_tokens, forbidden_values = forbidden_bytes
    for file_number, identity in enumerate(identities):
        file_started = time.monotonic()
        md, base_mask = read_metadata(identity, set(roles))
        mask = [bool(allowed and (identity['context'], md['gene_target'][i], md['row_role'][i]) in groups)
                for i, allowed in enumerate(base_mask)]
        if identity['rows'] > budgets['max_cells_per_file']:
            raise RuntimeError('Per-file row budget exceeded')
        raw_path = checked_child(raw_root, identity['path'])
        if not raw_path.is_file() or raw_path.stat().st_size != identity['publisher_whole_file_bytes']:
            raise RuntimeError('Raw file is incomplete or size differs')
        if sha(raw_path) != identity['publisher_LFS_sha256']:
            raise RuntimeError('Full raw SHA differs; expression remains unopened')
        # Verify physical column types from the footer before invoking any
        # numeric decoder. The reader supports both primitive types generally;
        # this biological schema assigns each column exactly one type.
        footer = reader_api.fastparquet.ParquetFile(raw_path)
        if len(footer.row_groups) != 1 or footer.row_groups[0].num_rows != identity['rows']:
            raise RuntimeError('Raw footer physical row identity differs')
        for name, expected_type in [('gene_token_id', reader_api.pt.Type.INT64),
                                    ('gene_expression', reader_api.pt.Type.DOUBLE)]:
            leaves = [c.meta_data for c in footer.row_groups[0].columns if c.meta_data.path_in_schema[0] == name]
            if len(leaves) != 1 or leaves[0].type != expected_type:
                raise RuntimeError('Exact INT64 token/DOUBLE expression column types required')
            if leaves[0].total_compressed_size > budgets['max_compressed_chunk_bytes']:
                raise RuntimeError('Numeric chunk memory budget exceeded before expression access')
        offsets = np.full(identity['rows'], -1, dtype=np.int64)
        lengths = np.full(identity['rows'], -1, dtype=np.int64)
        token_path = stage / f'tokens_{file_number}.uint32'
        token_trace = trace_new(budgets)
        values_written = 0
        token_rows = []
        with token_path.open('xb') as spool:
            for row, tokens in iter_selected_numeric_lists(raw_path, 'gene_token_id', mask, token_trace, forbidden_tokens):
                if tokens is None or any(type(t) is not int for t in tokens) or len(tokens) > len(full):
                    raise RuntimeError('Missing/invalid token list cannot be treated as measured zero')
                indices = np.asarray(tokens, dtype=np.int64)
                if (indices < 0).any() or (indices >= len(full)).any() or len(np.unique(indices)) != len(indices):
                    raise RuntimeError('Unknown/duplicate token cannot be imputed or remapped')
                offsets[row], lengths[row] = values_written, len(indices)
                spool.write(indices.astype('<u4', copy=False).tobytes())
                values_written += len(indices)
                if values_written*4 > budgets['max_token_spool_bytes_per_file']:
                    raise RuntimeError('Per-file token spool budget exceeded')
                token_rows.append(row)
                if time.monotonic() - file_started > budgets['max_seconds_per_file']:
                    raise RuntimeError('Per-file time budget exceeded')
                check_memory(budgets)
        expected_rows = np.flatnonzero(mask).tolist()
        if token_rows != expected_rows:
            raise RuntimeError('Token iterator did not exhaust exact permitted physical rows')
        tokens_mmap = np.memmap(token_path, dtype='<u4', mode='r', shape=(values_written,)) if values_written else None
        expression_trace = trace_new(budgets)
        expression_rows = []
        sum_check_roles = set(permit.get('library_sum_check_roles', ['TRAIN', CONTROL_ROLE]))
        library_gaps = {role: {'n_cells': 0, 'max_absolute_gap': 0., 'max_relative_gap': 0.,
                              'sum_absolute_gap': 0., 'mismatch_cells': 0} for role in sorted(sum_check_roles)}
        for row, values in iter_selected_numeric_lists(raw_path, 'gene_expression', mask, expression_trace, forbidden_values):
            if values is None or len(values) != lengths[row] or any(value is None for value in values):
                raise RuntimeError('Expression/token lengths/nulls differ; missing measurements cannot become zero')
            raw = np.asarray(values, dtype=float)
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise RuntimeError('Raw expression must be finite nonnegative measurements')
            if not len(raw) or np.max(np.abs(raw - np.rint(raw))) > 1e-6:
                raise RuntimeError('Positive library requires nonempty raw UMI counts integer within tolerance')
            total = float(md['total_counts'][row])
            role = md['row_role'][row]
            if role in sum_check_roles:
                raw_sum = float(raw.sum())
                if not np.isfinite(raw_sum) or raw_sum <= 0:
                    raise RuntimeError('Positive full library cannot be reconstructed from missing/zero counts')
                gap = abs(raw_sum - total)
                diagnostic = library_gaps[role]
                diagnostic['n_cells'] += 1
                diagnostic['max_absolute_gap'] = max(diagnostic['max_absolute_gap'], gap)
                diagnostic['max_relative_gap'] = max(diagnostic['max_relative_gap'], gap / total)
                diagnostic['sum_absolute_gap'] += gap
                consistent = bool(np.isclose(raw_sum, total, rtol=1e-6, atol=1e-6))
                diagnostic['mismatch_cells'] += int(not consistent)
                if not consistent:
                    raise RuntimeError(f'Authorized {role} raw UMI sum differs from official full library; denominator remains unchanged')
            indices = np.asarray(tokens_mmap[offsets[row]:offsets[row]+lengths[row]], dtype=np.int64) if len(values) else np.empty(0, dtype=np.int64)
            group = groups[(identity['context'], md['gene_target'][row], md['row_role'][row])]
            # Denominator remains official full-library metadata, independent of
            # endpoint projection or absent sparse entries on the full axis.
            normalized = np.log1p(4000. * raw / total)
            if not np.isfinite(normalized).all():
                raise RuntimeError('Normalized expression overflow/nonfinite measurement')
            sums[group, indices] += normalized
            counts[group] += 1
            expression_rows.append(row)
            if len(expression_rows) % 512 == 0:
                release_map(sums); counts.flush()
            if time.monotonic() - file_started > budgets['max_seconds_per_file']:
                raise RuntimeError('Per-file time budget exceeded')
            check_memory(budgets)
        if expression_rows != expected_rows:
            raise RuntimeError('Expression iterator did not exhaust exact permitted physical rows')
        if tokens_mmap is not None:
            del tokens_mmap
        token_path.unlink()
        release_map(sums); counts.flush()
        file_audits.append({'raw_path': str(raw_path), 'raw_sha256': identity['publisher_LFS_sha256'],
            'metadata_parquet_sha256': identity['metadata_parquet_sha256'],
            'authorized_physical_row_mask_sha256': hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest(),
            'authorized_row_count': len(expected_rows), 'context': identity['context'],
            'token_trace': token_trace, 'expression_trace': expression_trace,
            'authorized_library_sum_gaps': library_gaps,
            'iterators_fully_exhausted': True, 'token_spool_bytes': values_written*4,
            'elapsed_seconds': round(time.monotonic()-file_started, 3), 'peak_resident_bytes': check_memory(budgets)})
        print(json.dumps({'phase': 'allowed_file_aggregated', 'file_number': file_number,
                          'file': identity['path'], 'allowed_rows': len(expected_rows)}), flush=True)
    for item in records:
        if int(counts[item['matrix_row']]) != item['n_cells_metadata']:
            raise RuntimeError('All-file measured cell count differs from metadata plan')
    means = np.lib.format.open_memmap(stage / 'FULL_MEANS.npy', mode='w+', dtype=np.float64, shape=shape)
    effects = np.lib.format.open_memmap(stage / 'FULL_EFFECTS.npy', mode='w+', dtype=np.float64, shape=shape)
    control_rows = {x['context']: x['matrix_row'] for x in records if x['role'] == CONTROL_ROLE}
    if set(control_rows) != set(contexts):
        raise RuntimeError('Each context requires its own measured TRAIN NTC mean')
    endpoint_means = np.lib.format.open_memmap(stage / 'ENDPOINT3285_MEANS.npy', mode='w+', dtype=np.float64, shape=(len(records), len(endpoint)))
    endpoint_effects = np.lib.format.open_memmap(stage / 'ENDPOINT3285_EFFECTS.npy', mode='w+', dtype=np.float64, shape=(len(records), len(endpoint)))
    control_means = {context: np.asarray(sums[row] / float(counts[row]), dtype=float)
                     for context, row in control_rows.items()}
    for first in range(0, len(records), 32):
        for item in records[first:first+32]:
            row = item['matrix_row']
            means[row] = sums[row] / float(counts[row])
            effects[row] = means[row] - control_means[item['context']]
            endpoint_means[row] = means[row, endpoint_columns]
            endpoint_effects[row] = effects[row, endpoint_columns]
        for array in (sums, means, effects, endpoint_means, endpoint_effects):
            release_map(array)
        check_memory(budgets)
    for array in (sums, counts, means, effects, endpoint_means, endpoint_effects):
        release_map(array)
    full.to_csv(stage / 'FULL_GENE_AXIS.csv', index=False)
    endpoint.to_csv(stage / 'ENDPOINT_GENE_MANIFEST.csv', index=False)
    pd.DataFrame(records).to_csv(stage / 'QUERY_METADATA_ONLY.csv', index=False)
    for role in sorted(set(roles)):
        np.save(stage / f'{role}_MATRIX_ROWS.npy', np.asarray([x['matrix_row'] for x in records if x['role']==role], dtype=np.int64), allow_pickle=False)
    result = {'schema': SCHEMA, 'status': 'COMPLETE', 'synthetic_only': bool(permit.get('synthetic_only')),
        'scientific_contract_sha256': sha(contract_path), 'permit_sha256': sha(permit_path),
        'loader_sha256': sha(__file__), 'reader_sha256': sha(READER),
        'roles': sorted(roles), 'contexts': sorted(contexts), 'full_unique_ensembl_genes': len(full),
        'endpoint_genes': len(endpoint), 'aggregate_units': len(records), 'control_matrix_rows': control_rows,
        'matrix_shape': list(shape), 'normalization': contract['normalization'],
        'missing_unknown_gene_imputation': False, 'sparse_absent_known_gene_policy': 'measured literal zero on complete publisher full axis',
        'private_test_numeric_materialization': False, 'test_query_metadata_analysis': False,
        'expression_decoder': 'iter_selected_numeric_lists; byte buffers remain opaque until exact permitted row mask',
        'decoder_versions': {'fastparquet': reader_api.fastparquet.__version__, 'pyarrow': pa.__version__},
        'complete_cell_dataframe_created': False, 'dense_cell_by_gene_created': False,
        'ambiguous_or_unmapped_allowed_target_omissions_metadata_only': [list(x) for x in omitted],
        'full_PCA_fit_policy': 'TRAIN matrix rows only; separate context; this loader does not fit PCA',
        'raw_UMI_semantic_checks': {'finite_nonnegative': True, 'integer_absolute_tolerance': 1e-6,
            'nonempty_positive_full_library_required': True,
            'library_sum_checked_roles': sorted(set(permit.get('library_sum_check_roles', ['TRAIN', CONTROL_ROLE]))),
            'library_sum_absolute_tolerance': 1e-6, 'library_sum_relative_tolerance': 1e-6,
            'library_sum_mismatch_action_on_checked_roles': 'fail_closed_no_denominator_switch',
            'library_sum_gap_diagnostics_roles': sorted(set(permit.get('library_sum_check_roles', ['TRAIN', CONTROL_ROLE])))},
        'file_audits': file_audits, 'resource_budgets': budgets, 'peak_resident_bytes': check_memory(budgets),
        'elapsed_seconds': round(time.monotonic()-started, 3), 'finished_utc': datetime.now(timezone.utc).isoformat()}
    json_write(stage / 'BIOLOGY_MANIFEST.json', result)
    artifacts = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(stage.iterdir()) if p.is_file()]
    json_write(stage / 'ARTIFACT_HASHES.json', artifacts)
    stage.rename(output)
    return result


def synthetic_fixture(base, version, dictionary, bad_token=False):
    """Small complete three-gene assay; TEST/preview contain forbidden sentinels."""
    metadata_root = base / 'metadata'; raw_root = base / 'raw'
    metadata_root.mkdir(parents=True); raw_root.mkdir(parents=True)
    full = pd.DataFrame({'ensembl_id': ['E0', 'E1', 'E2'], 'gene_name': ['G1', 'G2', 'G3'], 'gene_token_id': [0,1,2]})
    pq.write_table(pa.Table.from_pandas(full, preserve_index=False), metadata_root / 'gene_metadata.parquet')
    pd.DataFrame({'axis_index': [0,1], 'source_index': [0,2], 'gene_name': ['G1','G3'],
                  'orion_gene_token_id': [0,2], 'orion_ensembl_id': ['E0','E2']}).to_csv(metadata_root / 'GENE_MANIFEST.csv', index=False)
    json_write(metadata_root / 'PREPARATION_POLICY.json', {'synthetic_only': True, 'normalization_scale':4000}, True)
    json_write(metadata_root / 'SYNTHETIC_FIXTURE_ONLY.json', {'synthetic_only':True}, True)
    sentinel, token_sentinel = 918273645.125, 918273645
    identities = []
    for context in ['HCT116','HEK293T']:
        if context == 'HCT116':
            roles = [CONTROL_ROLE, CONTROL_ROLE, 'TRAIN', 'VALIDATION', 'TEST', 'DEV_EXPOSED_CELL']
            genes = ['Non-Targeting','Non-Targeting','G1','G2','G3','Non-Targeting']
            tokens = [[0,1],[1],[0,2],[2],[token_sentinel],[token_sentinel]]
            values = [[1.,1.],[4.],[3.,1.],[2.],[sentinel],[sentinel]]
            totals = [2.,4.,4.,2.,1.,1.]
        else:
            roles = [CONTROL_ROLE,'TRAIN','TEST']
            genes = ['Non-Targeting','G1','G3']
            tokens = [[2],[0,2],[token_sentinel]]
            values = [[5.],[2.,2.],[sentinel]]
            totals = [5.,4.,1.]
        if bad_token and context == 'HCT116':
            tokens[2] = [0,3]
        sample = context + '_Batch1'
        raw = raw_root / (sample + '.parquet')
        table = pa.table({'gene_token_id':pa.array(tokens,type=pa.list_(pa.int64())),
                          'gene_expression':pa.array(values,type=pa.list_(pa.float64()))})
        pq.write_table(table, raw, compression='zstd', use_dictionary=dictionary,
                       data_page_version=version, row_group_size=len(roles), data_page_size=96, write_batch_size=1)
        md = pa.table({'original_row_index':list(range(len(roles))), 'original_row_group':[0]*len(roles),
                 'row_role':roles,'gene_target':genes,'sample':[sample]*len(roles),
                 'cell_barcode':[f'{sample}:{i}' for i in range(len(roles))], 'total_counts':totals,
                 'biological_unit':[f'SyntheticStudy|{gene}|{context}' for gene in genes],
                 'fixed_metadata_QC_pass':[True]*len(roles)})
        md_path = metadata_root / (sample + '.parquet')
        pq.write_table(md, md_path)
        md_path.chmod(0o444)
        identities.append({'path':raw.name,'context':context,'sample':sample,'rows':len(roles),'row_groups':1,
             'publisher_whole_file_bytes':raw.stat().st_size,'publisher_LFS_sha256':sha(raw),
             'metadata_parquet_path':str(md_path),'metadata_parquet_sha256':sha(md_path)})
    json_write(metadata_root / 'FILE_METADATA_IDENTITY.json', identities, True)
    contract = base / 'SYNTHETIC_CONTRACT.json'
    json_write(contract, {'schema':'synthetic_scientific_contract','synthetic_only':True,
      'immutable_dependencies':[{'path':str(metadata_root/name),'sha256':sha(metadata_root/name)} for name in ['FILE_METADATA_IDENTITY.json','GENE_MANIFEST.csv','gene_metadata.parquet','PREPARATION_POLICY.json']],
      'privacy':{'current_permitted_roles':sorted(LEGAL_ROLES),'private_numeric_materialization':False},
      'normalization':{'formula':'mean_cell(log1p(4000 * raw_gene_UMI / official_full_library_total_counts))',
                       'unknown_token_policy':'fail_closed','cell_weight':'equal per retained cell',
                       'control':'own-context TRAIN NTC excluding preview cells'}}, True)
    permit = base / 'SYNTHETIC_PERMIT.json'
    json_write(permit, {'schema':'explicit_role_permit','synthetic_only':True,
       'expression_row_materialization_permitted':True,'test_numeric_materialization_permitted':False,
       'authorized_roles':sorted(LEGAL_ROLES),'authorized_contexts':sorted(CONTEXTS),'reader_guarantee':GUARANTEE,
       'frozen_method_contract_path':str(contract),'frozen_method_contract_sha256':sha(contract),
       'metadata_root':str(metadata_root),'raw_root':str(raw_root),
       'metadata_bindings':{name:sha(metadata_root/name) for name in ['FILE_METADATA_IDENTITY.json','GENE_MANIFEST.csv','gene_metadata.parquet','PREPARATION_POLICY.json']},
       'implementation_bindings':{'loader':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
                                  'reader':{'path':str(READER),'sha256':sha(READER)}},
       'resource_budgets':DEFAULT_BUDGETS}, True)
    return metadata_root, raw_root, contract, permit, ((struct.pack('<q',token_sentinel),),(struct.pack('<d',sentinel),))


def synthetic_test(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    successes, negatives = [], []
    for version, dictionary in [('1.0',False), ('2.0',True)]:
        base = output / f'v{version}_dictionary{int(dictionary)}'
        inputs = synthetic_fixture(base, version, dictionary)
        result_path = base / 'aggregated'
        result = aggregate(*inputs[:4], result_path, sorted(LEGAL_ROLES), sorted(CONTEXTS), inputs[4])
        query = pd.read_csv(result_path / 'QUERY_METADATA_ONLY.csv')
        means = np.load(result_path / 'FULL_MEANS.npy')
        effects = np.load(result_path / 'FULL_EFFECTS.npy')
        expected = {('HCT116','Non-Targeting'):np.array([np.log1p(2000)/2,(np.log1p(2000)+np.log1p(4000))/2,0]),
                    ('HCT116','G1'):np.array([np.log1p(3000),0,np.log1p(1000)]),
                    ('HCT116','G2'):np.array([0,0,np.log1p(4000)]),
                    ('HEK293T','Non-Targeting'):np.array([0,0,np.log1p(4000)]),
                    ('HEK293T','G1'):np.array([np.log1p(2000),0,np.log1p(2000)])}
        for row in query.itertuples(index=False):
            assert np.allclose(means[row.matrix_row],expected[(row.context,row.gene)],rtol=0,atol=1e-15)
            control = expected[(row.context,'Non-Targeting')]
            assert np.allclose(effects[row.matrix_row],expected[(row.context,row.gene)]-control,rtol=0,atol=1e-15)
        assert np.array_equal(np.load(result_path/'ENDPOINT3285_EFFECTS.npy'),effects[:,[0,2]])
        assert set(query.role)<=LEGAL_ROLES and len(query)==5
        assert np.array_equal(np.load(result_path/'CELL_COUNTS.npy'),query.n_cells_metadata.to_numpy())
        successes.append({'case':base.name,'complete':True,'units':len(query),'full_genes':3,'endpoint_genes':2,
                          'private_sentinels_never_converted':True,'cell_equal_mean_correct':True,
                          'own_context_controls_correct':True,'literal_known_gene_zero_correct':True,
                          'reader_iterators_exhausted':True,'manifest_sha256':sha(result_path/'BIOLOGY_MANIFEST.json')})
        try:
            aggregate(*inputs[:4],base/'forbidden_TEST_output',['TEST',CONTROL_ROLE],['HCT116'])
            raise AssertionError('TEST role unexpectedly accepted')
        except RuntimeError as error:
            assert 'Only exact' in str(error)
            assert not (base/'forbidden_TEST_output').exists()
            negatives.append({'case':'TEST_role_rejected_before_raw_access','passed':True})
    base = output/'unknown_authorized_token'
    inputs = synthetic_fixture(base,'2.0',True,bad_token=True)
    try:
        aggregate(*inputs[:4],base/'must_not_commit',sorted(LEGAL_ROLES),sorted(CONTEXTS),inputs[4])
        raise AssertionError('Unknown token unexpectedly accepted')
    except RuntimeError as error:
        assert 'Unknown/duplicate token' in str(error)
        assert not (base/'must_not_commit').exists()
        negatives.append({'case':'unknown_authorized_token_fails_without_zero_imputation','passed':True})
    result = {'schema':SCHEMA,'status':'SYNTHETIC_END_TO_END_PASS_REAL_EXPRESSION_UNOPENED',
              'real_expression_accessed':False,'loader_sha256':sha(__file__),'reader_sha256':sha(READER),
              'successes':successes,'negative_cases':negatives,'budgets':DEFAULT_BUDGETS}
    json_write(output/'SYNTHETIC_END_TO_END_RESULT.json',result)
    print(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    synthetic=sub.add_parser('synthetic-test')
    synthetic.add_argument('--output',type=Path,default=DOC/'synthetic_e2e_v1')
    for name in ['plan','aggregate']:
        command=sub.add_parser(name)
        command.add_argument('--metadata-root',type=Path,default=METADATA)
        command.add_argument('--output',type=Path,required=True)
        command.add_argument('--roles',nargs='+',default=['TRAIN','VALIDATION',CONTROL_ROLE])
        command.add_argument('--contexts',nargs='+',default=sorted(CONTEXTS))
        if name=='aggregate':
            command.add_argument('--raw-root',type=Path,default=RAW)
            command.add_argument('--contract',type=Path,required=True)
            command.add_argument('--permit',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='synthetic-test':
        synthetic_test(args.output)
    elif args.command=='plan':
        full,endpoint,columns,records,identities,omitted=metadata_plan(args.metadata_root,args.roles,args.contexts)
        args.output.mkdir(parents=True,exist_ok=False)
        pd.DataFrame(records).to_csv(args.output/'QUERY_METADATA_ONLY.csv',index=False)
        json_write(args.output/'METADATA_PLAN.json',{'schema':SCHEMA,'expression_read':False,
          'full_genes':len(full),'endpoint_genes':len(endpoint),'allowed_units':len(records),
          'contexts':args.contexts,'roles':args.roles,'files':len(identities),'omissions':[list(x) for x in omitted],
          'test_metadata_coverage_or_distributions_examined':False,'resource_budgets':DEFAULT_BUDGETS})
    else:
        result=aggregate(args.metadata_root,args.raw_root,args.contract,args.permit,args.output,args.roles,args.contexts)
        print(json.dumps({key:result[key] for key in ['schema','status','roles','contexts','full_unique_ensembl_genes','endpoint_genes','aggregate_units','elapsed_seconds']}))


if __name__=='__main__':
    main()
