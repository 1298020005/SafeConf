#!/usr/bin/env python3
"""SYNTHETIC ONLY probe; this script has no option to open Orion expression data.

Specialized feasibility adapter, not an authorized production loader.  It reuses
fastparquet's footer, page header, level and codec machinery. Numeric dictionaries
and value payloads remain byte buffers; only values routed to metadata-authorized
rows are converted to int64/double. Reading/decompressing private bytes is distinct
from creating typed expression values or private row vectors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/private_reader_probe'
RUNTIME = Path('/home/yyf/data/safeconf_orion_frozen40_20261002/private_reader_probe_20261002_v1')
sys.path.insert(0, str(RUNTIME / 'python_deps'))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet
from fastparquet import core, parquet_thrift as pt
from fastparquet import cencoding as enc
from fastparquet.compression import decompress_data


def hybrid(raw, width, count, dtype=np.uint8):
    """Decode structural levels or dictionary routing IDs, never expressions."""
    output = np.empty(count, dtype=dtype)
    sink = enc.NumpyIO(output.view('uint8'))
    if count:
        enc.read_rle_bit_packed_hybrid(enc.NumpyIO(raw), width, len(raw), sink,
                                     itemsize=np.dtype(dtype).itemsize)
        if sink.tell() != output.nbytes:
            raise RuntimeError('Incomplete structural level/ID stream')
    return output


def iter_selected_numeric_lists(path, name, mask, trace, forbidden_bytes=()):
    """One rowgroup, one LIST<INT64|DOUBLE>, PLAIN/RLE_DICTIONARY, V1/V2.

    No Arrow expression read occurs here. Unsupported schema/encoding fails closed.
    Callers must bind a separately authorized, immutable metadata row mask.
    """
    fp = fastparquet.ParquetFile(path)
    if len(fp.row_groups) != 1:
        raise RuntimeError('Probe supports one rowgroup only')
    rg = fp.row_groups[0]
    if len(mask) != rg.num_rows or any(type(x) is not bool for x in mask):
        raise RuntimeError('Exact boolean metadata mask required')
    matches = [c.meta_data for c in rg.columns if c.meta_data.path_in_schema[0] == name]
    if len(matches) != 1:
        raise RuntimeError('Exactly one numeric LIST leaf required')
    cmd = matches[0]
    helper = fp.schema
    leaf = helper.schema_element(cmd.path_in_schema)
    root = helper.schema_element([name])
    maxdef = helper.max_definition_level(cmd.path_in_schema)
    if (len(cmd.path_in_schema) != 3 or helper.max_repetition_level(cmd.path_in_schema) != 1
            or root.converted_type != pt.ConvertedType.LIST
            or cmd.type not in {pt.Type.INT64, pt.Type.DOUBLE}
            or cmd.codec not in {pt.CompressionCodec.UNCOMPRESSED, pt.CompressionCodec.ZSTD,
                                 pt.CompressionCodec.SNAPPY}):
        raise RuntimeError('Unsupported schema/codec')
    item_nullable = leaf.repetition_type == pt.FieldRepetitionType.OPTIONAL
    list_nullable = root.repetition_type == pt.FieldRepetitionType.OPTIONAL
    listdef = maxdef - int(item_nullable)
    chunk_limit = trace.get('max_compressed_chunk_bytes', 1 << 30)
    page_limit = trace.get('max_page_uncompressed_bytes', 128 << 20)
    entry_limit = trace.get('max_page_level_entries', 16 << 20)
    if any(type(limit) is not int or limit <= 0 for limit in [chunk_limit, page_limit, entry_limit]):
        raise RuntimeError('Positive integer reader budgets required')
    if cmd.total_compressed_size > chunk_limit:
        raise RuntimeError('Compressed numeric chunk exceeds memory budget before read')
    if not any(mask):
        return
    offset = cmd.dictionary_page_offset or cmd.data_page_offset
    with open(path, 'rb') as stream:
        stream.seek(offset)
        binary = stream.read(cmd.total_compressed_size)
    if len(binary) != cmd.total_compressed_size:
        raise RuntimeError('Incomplete numeric column chunk')
    trace['physical_numeric_chunk_bytes_read'] += len(binary)
    pagefile = enc.NumpyIO(binary)
    dictionary = None
    row = -1
    entries = 0
    emitted = 0
    row_values = None
    while pagefile.tell() < len(binary):
        ph = enc.ThriftObject.from_buffer(pagefile, 'PageHeader')
        if (ph.uncompressed_page_size < 0 or ph.compressed_page_size < 0
                or ph.uncompressed_page_size > page_limit
                or pagefile.tell() + ph.compressed_page_size > len(binary)):
            raise RuntimeError('Numeric page size exceeds budget or chunk boundary before decompression')
        data_header = ph.data_page_header if ph.type == pt.PageType.DATA_PAGE else ph.data_page_header_v2
        if ph.type in {pt.PageType.DATA_PAGE, pt.PageType.DATA_PAGE_V2} and (
                data_header.num_values < 0 or data_header.num_values > entry_limit
                or data_header.num_values + entries > cmd.num_values):
            raise RuntimeError('Structural page entries exceed budget or footer count before decoding')
        if ph.type == pt.PageType.DICTIONARY_PAGE:
            if ph.dictionary_page_header.encoding != pt.Encoding.PLAIN or dictionary is not None:
                raise RuntimeError('Unsupported dictionary')
            dictionary = memoryview(core._read_page(pagefile, ph, cmd)).cast('B')
            trace['opaque_numeric_bytes_decompressed'] += len(dictionary)
            if len(dictionary) != 8 * ph.dictionary_page_header.num_values:
                raise RuntimeError('Dictionary byte length mismatch')
            continue
        if ph.type == pt.PageType.DATA_PAGE:
            dh = ph.data_page_header
            if dh.repetition_level_encoding != pt.Encoding.RLE or dh.definition_level_encoding != pt.Encoding.RLE:
                raise RuntimeError('Unsupported level encoding')
            raw = core._read_page(pagefile, ph, cmd)
            trace['opaque_numeric_bytes_decompressed'] += len(raw)
            levels = enc.NumpyIO(raw)
            reps = core.read_rep(levels, dh, helper, cmd)
            defs, _ = core.read_def(levels, dh, helper, cmd)
            defs = np.full(dh.num_values, maxdef, dtype='uint8') if defs is None else defs
            payload = memoryview(levels.read()).cast('B')
            coding, count = dh.encoding, dh.num_values
        elif ph.type == pt.PageType.DATA_PAGE_V2:
            dh = ph.data_page_header_v2
            rlen, dlen = dh.repetition_levels_byte_length, dh.definition_levels_byte_length
            body = memoryview(pagefile.read(ph.compressed_page_size)).cast('B')
            reps = hybrid(body[:rlen], 1, dh.num_values)
            defs = hybrid(body[rlen:rlen+dlen], maxdef.bit_length(), dh.num_values)
            payload = body[rlen+dlen:]
            if dh.is_compressed is not False:
                payload = memoryview(decompress_data(payload, ph.uncompressed_page_size-rlen-dlen,
                                                     cmd.codec)).cast('B')
            trace['opaque_numeric_bytes_decompressed'] += len(body[:rlen+dlen]) + len(payload)
            coding, count = dh.encoding, dh.num_values
        else:
            raise RuntimeError('Unsupported page type')
        trace['data_pages'] += 1
        if reps is None or len(reps) != count or len(defs) != count:
            raise RuntimeError('Malformed LIST levels')
        nvalues = int((defs == maxdef).sum())
        codes = None
        if coding == pt.Encoding.PLAIN:
            if len(payload) != 8*nvalues:
                raise RuntimeError('PLAIN byte length mismatch')
        elif coding in {pt.Encoding.PLAIN_DICTIONARY, pt.Encoding.RLE_DICTIONARY}:
            if dictionary is None or not payload:
                raise RuntimeError('Dictionary missing')
            codes = hybrid(payload[1:], payload[0], nvalues, np.uint32) if payload[0] else np.zeros(nvalues, dtype=np.uint32)
            if np.any(codes >= len(dictionary)//8):
                raise RuntimeError('Dictionary routing ID out of bounds')
        else:
            raise RuntimeError('Unsupported value encoding; no decoder fallback')
        value_index = 0
        for definition, repetition in zip(defs, reps):
            if repetition == 0:
                if row >= 0 and mask[row]:
                    emitted += 1
                    yield row, row_values
                row += 1
                if row >= len(mask):
                    raise RuntimeError('Too many physical rows')
                if mask[row]:
                    row_values = None if list_nullable and definition < listdef-1 else []
                else:
                    row_values = None
            elif repetition != 1 or row < 0:
                raise RuntimeError('Invalid repetition level')
            if definition == maxdef:
                # All private expression values bypass this conversion boundary.
                if mask[row]:
                    index = value_index if codes is None else int(codes[value_index])
                    source = payload if codes is None else dictionary
                    raw8 = source[index*8:(index+1)*8]
                    if len(raw8) != 8 or bytes(raw8) in forbidden_bytes:
                        raise RuntimeError('Forbidden synthetic numeric sentinel reached conversion')
                    if trace.get('numeric_materialization_rows') is not None:
                        trace['numeric_materialization_rows'].append(row)
                    trace['numeric_materialization_count'] = trace.get('numeric_materialization_count', 0) + 1
                    trace['authorized_dictionary_lookups'] = trace.get('authorized_dictionary_lookups', 0) + int(codes is not None)
                    row_values.append(struct.unpack('<q' if cmd.type == pt.Type.INT64 else '<d', raw8)[0])
                else:
                    trace['private_numeric_values_skipped'] += 1
                value_index += 1
            elif mask[row] and item_nullable and definition == maxdef-1:
                row_values.append(None)
        entries += count
    if row + 1 != rg.num_rows or entries != cmd.num_values:
        raise RuntimeError('Physical row/value coverage mismatch')
    if row >= 0 and mask[row]:
        emitted += 1
        yield row, row_values
    if emitted != sum(mask):
        raise RuntimeError('Authorized row coverage mismatch')


def selected_numeric_lists(path, name, mask, trace, forbidden_bytes=()):
    """Collecting wrapper for synthetic fixtures; production should use iterator."""
    return dict(iter_selected_numeric_lists(path, name, mask, trace, forbidden_bytes))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    sentinel = 918273645.125
    token_sentinel = 918273645
    roles = ['TRAIN', 'VALIDATION', 'TRAIN', 'TEST', 'TEST', 'TRAIN'] * 100
    mask = [role == 'TRAIN' for role in roles]
    expressions = []
    tokens = []
    for i, permitted in enumerate(mask):
        if permitted:
            expressions.append(None if i % 11 == 0 else [] if i % 7 == 0 else [1.25, None, float(i % 9)])
            tokens.append(None if i % 11 == 0 else [] if i % 7 == 0 else [7, None, i % 9])
        else:
            expressions.append([sentinel, sentinel, None])
            tokens.append([token_sentinel, token_sentinel, None])
    table = pa.table({'gene_token_id': pa.array(tokens, type=pa.list_(pa.int64())),
                      'gene_expression': pa.array(expressions, type=pa.list_(pa.float64())),
                      'gene_target': roles, 'sample': ['SYNTHETIC_ONLY']*len(roles)})
    cases = []
    for version in ['1.0', '2.0']:
        for dictionary in [False, True]:
            for codec in ['NONE', 'ZSTD', 'SNAPPY']:
                path = RUNTIME / f'synthetic_v{version}_dict{int(dictionary)}_{codec}.parquet'
                pq.write_table(table, path, data_page_version=version, compression=codec,
                               use_dictionary=dictionary, write_statistics=False,
                               row_group_size=len(roles), write_batch_size=12, data_page_size=256)
                # Metadata is the only Arrow read in the selective-reader proof.
                metadata = pq.read_table(path, columns=['gene_target','sample']).to_pydict()
                authorized = [r == 'TRAIN' for r in metadata['gene_target']]
                trace = {'physical_numeric_chunk_bytes_read': 0, 'opaque_numeric_bytes_decompressed': 0,
                         'data_pages': 0, 'numeric_materialization_rows': [], 'private_numeric_values_skipped': 0}
                for name, expected, forbidden in [('gene_expression', expressions, struct.pack('<d', sentinel)),
                                                    ('gene_token_id', tokens, struct.pack('<q', token_sentinel))]:
                    actual = selected_numeric_lists(path, name, authorized, trace, [forbidden])
                    assert actual == {i: v for i,v in enumerate(expected) if authorized[i]}
                assert all(authorized[row] for row in trace['numeric_materialization_rows'])
                trace['numeric_materialization_count'] = len(trace.pop('numeric_materialization_rows'))
                cases.append({'fixture': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'rows': len(roles), 'authorized_rows': sum(authorized), 'trace': trace, 'passed': True})
    negative = []
    for dictionary in [False, True]:
        path = RUNTIME / f'synthetic_v1.0_dict{int(dictionary)}_ZSTD.parquet'
        original = core.read_plain
        def forbid(raw, type_, count, *args, **kwargs):
            if type_ == pt.Type.DOUBLE and struct.pack('<d', sentinel) in memoryview(raw).cast('B').tobytes():
                raise RuntimeError('Private sentinel offered to full-page numeric conversion before mask')
            return original(raw, type_, count, *args, **kwargs)
        core.read_plain = forbid
        try:
            fastparquet.ParquetFile(path).to_pandas(columns=['gene_expression'], row_filter=np.array(mask))
            raise AssertionError('Expected pre-mask private conversion interception')
        except RuntimeError as error:
            assert 'Private sentinel' in str(error)
            negative.append({'dictionary': dictionary, 'full_page_numeric_conversion_intercepted': True,
                             'message': str(error)})
        finally:
            core.read_plain = original
    rejection = []
    for label, modified_table, codec in [('unsupported_FLOAT_leaf', table.set_column(1, 'gene_expression', pa.array(expressions, type=pa.list_(pa.float32()))), 'NONE'),
                                          ('unsupported_GZIP_codec', table, 'GZIP')]:
        path = RUNTIME / (label+'.parquet')
        pq.write_table(modified_table, path, compression=codec, use_dictionary=False,
                       write_statistics=False, row_group_size=len(roles))
        trace = {'physical_numeric_chunk_bytes_read': 0, 'opaque_numeric_bytes_decompressed': 0,
                 'data_pages': 0, 'numeric_materialization_rows': [], 'private_numeric_values_skipped': 0}
        try:
            selected_numeric_lists(path, 'gene_expression', mask, trace)
            raise AssertionError('Unsupported input accepted')
        except RuntimeError as error:
            assert str(error) == 'Unsupported schema/codec'
            assert trace['physical_numeric_chunk_bytes_read'] == 0
            rejection.append({'case': label, 'rejected_before_numeric_chunk_read': True})
    original_path = RUNTIME / 'synthetic_v1.0_dict0_NONE.parquet'
    malformed = bytearray(original_path.read_bytes())
    fp = fastparquet.ParquetFile(original_path)
    offset = next(c.meta_data.data_page_offset for c in fp.row_groups[0].columns
                  if c.meta_data.path_in_schema[0] == 'gene_expression')
    pagebuf = enc.NumpyIO(bytes(malformed[offset:offset+4096]))
    header = enc.ThriftObject.from_buffer(pagebuf, 'PageHeader')
    header_size = pagebuf.tell()
    header.compressed_page_size -= 1
    header.uncompressed_page_size -= 1
    changed = header.to_bytes()
    assert len(changed) == header_size
    malformed[offset:offset+header_size] = changed
    path = RUNTIME / 'malformed_first_plain_page.parquet'
    path.write_bytes(malformed)
    trace = {'physical_numeric_chunk_bytes_read': 0, 'opaque_numeric_bytes_decompressed': 0,
             'data_pages': 0, 'numeric_materialization_rows': [], 'private_numeric_values_skipped': 0}
    try:
        selected_numeric_lists(path, 'gene_expression', mask, trace)
        raise AssertionError('Malformed payload accepted')
    except RuntimeError as error:
        assert str(error) == 'PLAIN byte length mismatch'
        assert not trace['numeric_materialization_rows']
        rejection.append({'case': 'malformed_first_PLAIN_payload_length', 'rejected_before_numeric_conversion': True})
    result = {'status': 'SYNTHETIC_SPECIALIZED_ADAPTER_PASSED_NOT_PRODUCTION_AUTHORIZED',
              'real_orion_expression_or_tokens_accessed': False,
              'fastparquet': fastparquet.__version__, 'pyarrow': pa.__version__,
              'private_numeric_row_materialization': False,
              'physical_private_bytes_read_and_decompressed': True,
              'cases': cases, 'fastparquet_row_filter_negative_proof': negative,
              'fail_closed_rejection_proof': rejection,
              'limitations': ['No production expression permit is created by this test.',
                              'One LIST leaf and one rowgroup only; unknown encoding/schema/codec is rejected.',
                              'Full numeric buffers and dictionaries remain opaque uint8/byte views.',
                              'Structural def/repetition levels and dictionary routing IDs are parsed for all rows.',
                              'Audited production integration, immutable role-mask binding and streaming budgets remain required.']}
    (OUT / 'SYNTHETIC_PROOF.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
