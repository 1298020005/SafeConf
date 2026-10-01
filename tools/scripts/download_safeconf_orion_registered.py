#!/usr/bin/env python3
"""Acquire the preselected Orion release without materializing expression.

Only the registered forty shards and gene-ID metadata are allowed. Existing
verified files are reused; incomplete files resume; failures never trigger
selection of substitute shards. Completion requires publisher LFS SHA256.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/independent_asset_followup/ORION_CANDIDATE_FILES.json'
DESTINATION = Path('/home/yyf/data/safeconf_orion_frozen40_20261002')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def acquire(item, destination):
    path = destination / item['path']; path.parent.mkdir(parents=True, exist_ok=True)
    expected = item['lfs_sha256']; size = item['bytes']
    if path.exists():
        if path.stat().st_size != size or sha(path) != expected:
            raise RuntimeError(f'existing asset does not match registration: {path}')
        return {'path': item['path'], 'bytes': size, 'sha256': expected, 'status': 'VERIFIED_REUSED'}
    partial = path.with_suffix(path.suffix + '.part')
    transferred = 0; errors = []
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise RuntimeError(f'partial asset larger than registered size: {partial}')
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with requests.get(item['download_url'], headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError('server refused safe partial resume; no existing bytes overwritten')
                with partial.open('ab' if offset else 'wb') as stream:
                    for block in response.iter_content(2 * 1024 * 1024):
                        if block:
                            if stream.tell() + len(block) > size:
                                raise RuntimeError('server sent more bytes than registered file')
                            stream.write(block); transferred += len(block)
            if partial.stat().st_size == size:
                break
        except Exception as error:
            errors.append(f'{type(error).__name__}: {error}')
            if attempt < 2:
                time.sleep(2)
    if not partial.exists() or partial.stat().st_size != size:
        raise RuntimeError(f'download incomplete: {item["path"]}; {errors}')
    observed = sha(partial)
    if observed != expected:
        raise RuntimeError(f'publisher SHA256 mismatch; preserve partial for diagnosis: {partial}')
    os.replace(partial, path)
    return {'path': item['path'], 'bytes': size, 'sha256': observed, 'status': 'VERIFIED_DOWNLOADED',
            'transferred_bytes_this_run': transferred, 'retry_errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    args = parser.parse_args()
    payload = args.manifest.read_bytes()
    manifest = json.loads(payload)
    items = [*manifest['files'], manifest['gene_metadata']]
    expected = sum(x['bytes'] for x in items)
    if len(manifest['files']) != 40 or expected > 100_000_000_000:
        raise RuntimeError('registered file count or user download ceiling failed')
    args.destination.mkdir(parents=True, exist_ok=True)
    registration = args.destination / 'DOWNLOAD_REGISTRATION.json'
    record = {'manifest_path': str(args.manifest), 'manifest_sha256': sha(args.manifest),
        'release_sha': manifest['release_sha'], 'expected_bytes': expected,
        'n_registered_files': len(items), 'selection_may_change': False,
        'expression_materialization_permitted': False,
        'new_upstream_training_started': False, 'workers': 3}
    if registration.exists():
        old = json.loads(registration.read_text())
        # A byte-identical snapshot may live at another path; the original
        # registration remains immutable, including its recorded source path.
        compare = dict(record, manifest_path=old['manifest_path'])
        if old != compare:
            raise RuntimeError('download registration changed')
    else:
        registration.write_text(json.dumps(record, indent=2) + '\n')
    snapshot = args.destination / f'INPUT_MANIFEST_{record["manifest_sha256"]}.json'
    if snapshot.exists() and snapshot.read_bytes() != payload:
        raise RuntimeError('immutable input snapshot changed')
    if not snapshot.exists():
        snapshot.write_bytes(payload)
    completed = []; failures = []; started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(acquire, item, args.destination): item for item in items}
        for future in as_completed(futures):
            try:
                result = future.result(); completed.append(result)
                print(f'Verified {len(completed)}/{len(items)}: {result["path"]} ({result["bytes"]} bytes)', flush=True)
            except Exception as error:
                failures.append({'path': futures[future]['path'], 'error': str(error)})
                print(f'Registered file incomplete: {futures[future]["path"]}', flush=True)
            terminal = len(completed) + len(failures) == len(items)
            status = {'status': ('INCOMPLETE' if failures else 'COMPLETE') if terminal else 'RUNNING',
                'updated_utc': datetime.now(timezone.utc).isoformat(),
                'verified_files': len(completed), 'registered_files': len(items),
                'verified_bytes': sum(x['bytes'] for x in completed),
                'elapsed_seconds': time.perf_counter() - started,
                'expression_materialized': False, 'failures': failures,
                'files': sorted(completed, key=lambda x: x['path'])}
            temporary = args.destination / '.DOWNLOAD_STATUS.json.tmp'
            temporary.write_text(json.dumps(status, indent=2) + '\n')
            os.replace(temporary, args.destination / 'DOWNLOAD_STATUS.json')
    if failures:
        raise RuntimeError('registered acquisition incomplete; retry the same manifest without substituting files')


if __name__ == '__main__':
    main()
