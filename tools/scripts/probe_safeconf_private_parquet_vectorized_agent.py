#!/usr/bin/env python3
"""Version2: byte selection precedes INT64/DOUBLE materialization.

The pinned version1 reader is unchanged. This module reuses its opaque page,
codec and structural-level primitives, then vectorizes the role mask and gathers
only authorized eight-byte payload/dictionary entries before creating a typed
numeric view. It has no CLI accepting real dataset paths or expression permit.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.scripts import probe_safeconf_private_parquet_agent as v1

np, fastparquet, core, pt, enc = v1.np, v1.fastparquet, v1.core, v1.pt, v1.enc
hybrid, decompress_data = v1.hybrid, v1.decompress_data
VERSION = 'PRIVATE_NUMERIC_BYTE_SELECTION_V2'


def materialize_selected_bytes(raw, byte_indices, physical_rows, authorized, kind, trace,
                               forbidden_bytes=()):
    """The sole typed conversion boundary: gather byte blocks FIRST.

    raw can encode private dictionary/payload entries, so its only view here is
    uint8. Numeric views are created only on the gathered authorized byte copy.
    """
    if len(byte_indices) != len(physical_rows) or not np.all(authorized[physical_rows]):
        raise RuntimeError('Private physical row reached vectorized conversion boundary')
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8)
    if np.any(byte_indices < 0) or np.any(byte_indices >= len(blocks)):
        raise RuntimeError('Numeric byte routing index out of bounds')
    selected_bytes = blocks[byte_indices]  # advanced indexing makes a byte copy
    for sentinel in forbidden_bytes:
        if len(sentinel) != 8:
            raise RuntimeError('Synthetic sentinel must have exact eight-byte width')
        sentinel_bytes = np.frombuffer(sentinel, dtype=np.uint8)
        if np.any(np.all(selected_bytes == sentinel_bytes, axis=1)):
            raise RuntimeError('Forbidden synthetic numeric sentinel reached conversion')
    if trace.get('numeric_materialization_rows') is not None:
        trace['numeric_materialization_rows'].extend(physical_rows.tolist())
    trace['numeric_materialization_count'] = trace.get('numeric_materialization_count', 0) + len(byte_indices)
    # No numeric type is applied to raw or blocks; only selected_bytes is typed.
    return selected_bytes.reshape(-1).view('<i8' if kind == pt.Type.INT64 else '<f8')


def iter_selected_numeric_lists(path, name, mask, trace, forbidden_bytes=()):
    """Same generator API as version1; caller must bind all permissions/hashes."""
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
    authorized = np.asarray(mask, dtype=np.bool_)
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
        header = ph.data_page_header if ph.type == pt.PageType.DATA_PAGE else ph.data_page_header_v2
        if ph.type in {pt.PageType.DATA_PAGE, pt.PageType.DATA_PAGE_V2} and (
                header.num_values < 0 or header.num_values > entry_limit
                or header.num_values + entries > cmd.num_values):
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
            trace['opaque_numeric_bytes_decompressed'] += rlen + dlen + len(payload)
            coding, count = dh.encoding, dh.num_values
        else:
            raise RuntimeError('Unsupported page type')
        trace['data_pages'] += 1
        if reps is None or len(reps) != count or len(defs) != count:
            raise RuntimeError('Malformed LIST levels')
        if np.any(reps > 1) or np.any(defs > maxdef):
            raise RuntimeError('Invalid LIST level stream')
        # Some PyArrow V2 writers emit an empty data page between real rows.
        # It advances no physical row and has no numeric conversion boundary.
        if count == 0:
            if coding == pt.Encoding.PLAIN and len(payload) == 0:
                continue
            if (coding in {pt.Encoding.PLAIN_DICTIONARY, pt.Encoding.RLE_DICTIONARY}
                    and dictionary is not None and len(payload) == 1):
                continue
            raise RuntimeError('Malformed empty numeric page')
        leaf_levels = defs == maxdef
        nvalues = int(leaf_levels.sum())
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
        # This integer array is structural physical-row routing, never expression.
        level_rows = np.cumsum(reps == 0, dtype=np.int64) + row
        if level_rows[0] < 0 or level_rows[-1] >= len(mask):
            raise RuntimeError('Invalid repetition level or too many physical rows')
        allowed_levels = authorized[level_rows]
        selected_levels = leaf_levels & allowed_levels
        selected_value_indices = np.flatnonzero(allowed_levels[leaf_levels])
        selected_rows = level_rows[selected_levels]
        if codes is None:
            byte_indices, source = selected_value_indices, payload
        else:
            byte_indices, source = codes[selected_value_indices], dictionary
            trace['authorized_dictionary_lookups'] = trace.get('authorized_dictionary_lookups', 0) + len(byte_indices)
        selected_values = materialize_selected_bytes(source, byte_indices, selected_rows,
                                                     authorized, cmd.type, trace, forbidden_bytes)
        trace['private_numeric_values_skipped'] += nvalues-len(selected_values)
        selected_prefix = np.empty(count+1, dtype=np.int64)
        selected_prefix[0] = 0
        np.cumsum(selected_levels, dtype=np.int64, out=selected_prefix[1:])
        starts = np.flatnonzero(reps == 0)
        starts = np.concatenate(([0], starts[starts > 0]))
        ends = np.concatenate((starts[1:], [count]))
        # Only one Python iteration per physical row segment, not per value.
        for first, last in zip(starts, ends):
            segment_row = int(level_rows[first])
            if reps[first] == 0:
                if row >= 0 and mask[row]:
                    emitted += 1
                    yield row, row_values
                row = segment_row
                row_values = (None if list_nullable and defs[first] < listdef-1 else []) if mask[row] else None
            elif segment_row != row:
                raise RuntimeError('Cross-page row continuation differs')
            if not mask[row]:
                continue
            segment_defs = defs[first:last]
            values = selected_values[selected_prefix[first]:selected_prefix[last]]
            item_levels = segment_defs >= listdef
            if not item_levels.any():
                continue
            if row_values is None:
                raise RuntimeError('Null list contains numeric or item levels')
            if item_nullable and np.any(segment_defs[item_levels] < maxdef):
                item_defs = segment_defs[item_levels]
                items = np.empty(len(item_defs), dtype=object)
                items[item_defs == maxdef] = values
                items[item_defs < maxdef] = None
                row_values.extend(items.tolist())
            else:
                row_values.extend(values.tolist())
        entries += count
    if row + 1 != rg.num_rows or entries != cmd.num_values:
        raise RuntimeError('Physical row/value coverage mismatch')
    if row >= 0 and mask[row]:
        emitted += 1
        yield row, row_values
    if emitted != sum(mask):
        raise RuntimeError('Authorized row coverage mismatch')


def selected_numeric_lists(path, name, mask, trace, forbidden_bytes=()):
    return dict(iter_selected_numeric_lists(path, name, mask, trace, forbidden_bytes))
