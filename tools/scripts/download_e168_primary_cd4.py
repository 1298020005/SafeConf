#!/usr/bin/env python3
"""Resumable, range-sharded byte acquisition for the frozen E168 source.

This downloader treats the H5AD as opaque bytes.  It never opens HDF5 and
never interprets expression values.  Each completed range is independently
length/ETag checked, making recovery robust to the proxy's frequent TLS EOFs.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import crcmod
except ImportError:  # The auditable table fallback is slower but exact.
    crcmod = None


URL = (
    "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/"
    "marson2025_data/GWCD4i.pseudobulk_merged.h5ad"
)
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
SIZE = 44_566_657_140
ETAG = "010c14e0af0dccbc2524529d28ca517e-5313"
VERSION_ID = "BWCjgMRhH80BOFIid2.0kbCr2o8wNVmn"
CRC64NVME_BASE64 = "E2slkXBEb2c="
DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
CHUNK_ROOT = DATA_ROOT / "range_chunks_64m"
FINAL = DATA_ROOT / "source/GWCD4i.pseudobulk_merged.h5ad"
STATUS = DATA_ROOT / "E168_BYTE_ACQUISITION_STATUS.json"
PRINT_LOCK = threading.Lock()

# CRC-64/NVME is the full-object checksum published by S3.  The reflected
# polynomial below corresponds to the normal-form polynomial
# 0xAD93D23594C93659.  Parameters: init/xorout all ones, refin/refout true.
CRC64NVME_REFLECTED_POLY = 0x9A6C9329AC4BC9B5
CRC64_MASK = (1 << 64) - 1


def _crc64nvme_table() -> tuple[int, ...]:
    rows: list[int] = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ CRC64NVME_REFLECTED_POLY if crc & 1 else crc >> 1
        rows.append(crc & CRC64_MASK)
    return tuple(rows)


CRC64NVME_TABLE = _crc64nvme_table()


def crc64nvme_update(crc: int, payload: bytes) -> int:
    value = crc
    for byte in payload:
        value = CRC64NVME_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value & CRC64_MASK


def crc64nvme_base64(crc_state: int) -> str:
    finalized = crc_state ^ CRC64_MASK
    return base64.b64encode(finalized.to_bytes(8, "big")).decode("ascii")


class _TableCRC64NVME:
    def __init__(self) -> None:
        self.state = CRC64_MASK

    def update(self, payload: bytes) -> None:
        self.state = crc64nvme_update(self.state, payload)

    @property
    def crc_value(self) -> int:
        return self.state ^ CRC64_MASK


def crc64nvme_hasher() -> tuple[object, str]:
    if crcmod is not None:
        # crcmod's initCrc is the externally visible empty-message CRC, hence
        # zero here with an all-ones xorOut represents the NVME all-ones
        # internal initial register.
        value = crcmod.Crc(
            0x1AD93D23594C93659,
            initCrc=0,
            rev=True,
            xorOut=CRC64_MASK,
        )
        return value, f"crcmod-{importlib.metadata.version('crcmod')}-compiled"
    return _TableCRC64NVME(), "audited_python_table_fallback"


def code_freeze_attestation() -> dict[str, object]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if branch == "HEAD":
        raise RuntimeError("E168 assembly requires a named Git branch")
    relative = SCRIPT.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{head}:{relative}"], cwd=ROOT
    )
    local = SCRIPT.read_bytes()
    if local != committed:
        raise RuntimeError("E168 downloader differs from committed HEAD")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        subprocess.run(
            [
                "git", "fetch", "--quiet", remote,
                f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
            ],
            cwd=ROOT,
            check=True,
        )
        remote_head = subprocess.check_output(
            ["git", "rev-parse", f"refs/remotes/{remote}/{branch}"],
            cwd=ROOT,
            text=True,
        ).strip()
        contained = subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        )
        if contained.returncode:
            raise RuntimeError(f"Code freeze {head} is absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return {
        "git_head": head,
        "git_branch": branch,
        "remote_heads": remote_heads,
        "downloader_path": relative,
        "downloader_sha256": hashlib.sha256(local).hexdigest(),
    }


def session(method: str) -> requests.Session:
    attempts = 2 if method == "GET" else 8
    retry = Retry(
        total=attempts,
        connect=attempts,
        read=attempts,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({method}),
    )
    value = requests.Session()
    value.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1))
    return value


def source_head() -> dict[str, object]:
    value = session("HEAD")
    response = value.head(URL, headers={"x-amz-checksum-mode": "ENABLED"}, timeout=(30, 90))
    response.raise_for_status()
    observed = {
        "size": int(response.headers["Content-Length"]),
        "etag": response.headers["ETag"].strip('"'),
        "version_id": response.headers.get("x-amz-version-id", ""),
        "checksum_crc64nvme_base64": response.headers.get("x-amz-checksum-crc64nvme", ""),
        "last_modified": response.headers.get("Last-Modified", ""),
    }
    expected = {
        "size": SIZE,
        "etag": ETAG,
        "version_id": VERSION_ID,
        "checksum_crc64nvme_base64": CRC64NVME_BASE64,
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise RuntimeError(f"source identity mismatch: observed={observed}; expected={expected}")
    value.close()
    return observed


def expected_chunk_size(index: int, chunk_size: int) -> int:
    start = index * chunk_size
    return max(0, min(chunk_size, SIZE - start))


def completed_bytes(chunk_root: Path, chunk_size: int) -> int:
    total = 0
    n_chunks = (SIZE + chunk_size - 1) // chunk_size
    for index in range(n_chunks):
        path = chunk_root / f"chunk_{index:04d}.bin"
        expected = expected_chunk_size(index, chunk_size)
        if path.exists() and path.stat().st_size == expected:
            total += expected
    return total


def write_status(stage: str, chunk_root: Path, chunk_size: int, extra: dict | None = None) -> None:
    complete = completed_bytes(chunk_root, chunk_size)
    payload = {
        "stage": stage,
        "source_url": URL,
        "source_size": SIZE,
        "source_etag": ETAG,
        "chunk_size": chunk_size,
        "complete_range_bytes": complete,
        "complete_fraction": complete / SIZE,
        "hdf5_opened": False,
        "expression_values_decoded": False,
        "updated_at_epoch": time.time(),
        **(extra or {}),
    }
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def download_chunk(index: int, chunk_root: Path, chunk_size: int) -> tuple[int, int]:
    expected = expected_chunk_size(index, chunk_size)
    start = index * chunk_size
    end = start + expected - 1
    final = chunk_root / f"chunk_{index:04d}.bin"
    partial = chunk_root / f"chunk_{index:04d}.part"
    if final.exists() and final.stat().st_size == expected:
        return index, expected
    if final.exists():
        final.unlink()
    current = partial.stat().st_size if partial.exists() else 0
    if current > expected:
        partial.unlink()
        current = 0
    client = session("GET")
    failures = 0
    while current < expected:
        absolute_start = start + current
        try:
            response = client.get(
                URL,
                headers={"Range": f"bytes={absolute_start}-{end}"},
                stream=True,
                timeout=(20, 45),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"chunk {index}: expected 206, got {response.status_code}")
            etag = response.headers.get("ETag", "").strip('"')
            if etag != ETAG:
                raise RuntimeError(f"chunk {index}: ETag changed to {etag}")
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or tuple(map(int, match.groups())) != (absolute_start, end, SIZE):
                raise RuntimeError(f"chunk {index}: bad Content-Range {content_range}")
            with partial.open("ab") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
            new_size = partial.stat().st_size
            if new_size <= current:
                raise RuntimeError(f"chunk {index}: request made no progress")
            current = new_size
            failures = 0
        except Exception as exc:
            failures += 1
            with PRINT_LOCK:
                print(f"[E168 download] chunk={index} offset={current}/{expected} retry={failures}: {exc}", flush=True)
            time.sleep(min(2**min(failures, 5), 30))
            current = partial.stat().st_size if partial.exists() else 0
    if partial.stat().st_size != expected:
        raise RuntimeError(f"chunk {index}: final size mismatch")
    os.replace(partial, final)
    client.close()
    with PRINT_LOCK:
        print(f"[E168 download] complete chunk {index}: {expected} bytes", flush=True)
    return index, expected


def download_all(chunk_root: Path, chunk_size: int, workers: int) -> None:
    chunk_root.mkdir(parents=True, exist_ok=True)
    n_chunks = (SIZE + chunk_size - 1) // chunk_size
    write_status("DOWNLOADING_RANGES", chunk_root, chunk_size, {"workers": workers, "n_chunks": n_chunks})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_chunk, index, chunk_root, chunk_size) for index in range(n_chunks)]
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            if number % workers == 0 or number == n_chunks:
                write_status(
                    "DOWNLOADING_RANGES",
                    chunk_root,
                    chunk_size,
                    {"workers": workers, "n_chunks": n_chunks, "completed_chunks": number},
                )
    if completed_bytes(chunk_root, chunk_size) != SIZE:
        raise RuntimeError("range download finished with incomplete bytes")


def assemble(chunk_root: Path, chunk_size: int, final: Path) -> dict[str, object]:
    if completed_bytes(chunk_root, chunk_size) != SIZE:
        raise RuntimeError("cannot assemble incomplete chunk set")
    code_freeze = code_freeze_attestation()
    before = source_head()
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_suffix(final.suffix + ".assembled.tmp")
    sha = hashlib.sha256()
    crc64_hasher, crc64_implementation = crc64nvme_hasher()
    n_chunks = (SIZE + chunk_size - 1) // chunk_size
    written = 0
    with temporary.open("wb") as target:
        for index in range(n_chunks):
            source = chunk_root / f"chunk_{index:04d}.bin"
            expected = expected_chunk_size(index, chunk_size)
            if source.stat().st_size != expected:
                raise RuntimeError(f"chunk {index} changed before assembly")
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    target.write(block)
                    sha.update(block)
                    crc64_hasher.update(block)
                    written += len(block)
        target.flush()
        os.fsync(target.fileno())
    if written != SIZE or temporary.stat().st_size != SIZE:
        raise RuntimeError(f"assembled size mismatch: {written}, {temporary.stat().st_size}")
    after = source_head()
    if before != after:
        raise RuntimeError("source object changed across assembly")
    crc64_value = getattr(crc64_hasher, "crcValue", None)
    if crc64_value is None:
        crc64_value = crc64_hasher.crc_value
    observed_crc64 = base64.b64encode(
        int(crc64_value).to_bytes(8, "big")
    ).decode("ascii")
    if observed_crc64 != CRC64NVME_BASE64:
        raise RuntimeError(
            "assembled CRC64NVME mismatch: "
            f"observed={observed_crc64}, expected={CRC64NVME_BASE64}"
        )
    os.replace(temporary, final)
    aria2 = Path(str(final) + ".aria2")
    if aria2.exists():
        aria2.unlink()
    result = {
        "source_head_before": before,
        "source_head_after": after,
        "code_freeze": code_freeze,
        "assembled_path": str(final),
        "assembled_size": SIZE,
        "sha256": sha.hexdigest(),
        "computed_crc64nvme_base64": observed_crc64,
        "official_crc64nvme_base64": CRC64NVME_BASE64,
        "crc64nvme_matches_official_full_object_checksum": True,
        "crc64nvme_implementation": crc64_implementation,
        "hdf5_opened": False,
        "expression_values_decoded": False,
    }
    attestation = DATA_ROOT / "E168_SOURCE_BYTE_ATTESTATION.json"
    attestation.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_status("ASSEMBLED_SHA256_COMPLETE", chunk_root, chunk_size, result)
    return result


def self_test() -> None:
    state = CRC64_MASK
    state = crc64nvme_update(state, b"1234")
    state = crc64nvme_update(state, b"56789")
    observed = crc64nvme_base64(state)
    expected = base64.b64encode(
        0xAE8B14860A799888.to_bytes(8, "big")
    ).decode("ascii")
    if observed != expected:
        raise AssertionError(f"CRC64NVME check-vector mismatch: {observed} != {expected}")
    hasher, _implementation = crc64nvme_hasher()
    hasher.update(b"1234")
    hasher.update(b"56789")
    accelerated_value = getattr(hasher, "crcValue", None)
    if accelerated_value is None:
        accelerated_value = hasher.crc_value
    if int(accelerated_value) != 0xAE8B14860A799888:
        raise AssertionError("Selected CRC64NVME implementation failed the check vector")
    print("E168 downloader self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size-mib", type=int, default=64)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    chunk_size = args.chunk_size_mib * 1024 * 1024
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if args.download_only and args.assemble_only:
        raise ValueError("choose at most one of --download-only/--assemble-only")
    if args.self_test:
        self_test()
        return
    print(json.dumps(source_head(), ensure_ascii=False, indent=2), flush=True)
    if not args.assemble_only:
        download_all(CHUNK_ROOT, chunk_size, args.workers)
    if not args.download_only:
        print(json.dumps(assemble(CHUNK_ROOT, chunk_size, FINAL), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
