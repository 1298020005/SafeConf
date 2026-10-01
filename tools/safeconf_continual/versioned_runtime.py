"""Immutable runtime snapshots around the existing SafeConf memory classes.

This module adds storage, integrity and serving operations. It does not fit
models, define release gates, read evaluation tasks or change memory items.
Public ``load`` returns the original four-tuple (table, effects, controls,
manifest); Error ``load`` returns the exact-model-version table. Model
``current`` resolves a verified immutable copy of the registered artifact.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from . import memory as core
from .contracts import ErrorMemoryItem, PublicMemoryItem


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pointer.",
                                     delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


@contextmanager
def _locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runtime.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _reason(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a nonempty operation reason is required")
    return value.strip()


def _events(root: Path) -> list[dict]:
    path = root / "runtime_events.jsonl"
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                raise RuntimeError("empty or truncated runtime event")
            event = json.loads(line)
            digest = event.pop("event_sha256")
            previous = events[-1]["event_sha256"] if events else None
            if (event.get("previous_event_sha256") != previous or
                    event.get("sequence") != len(events) + 1 or _digest(event) != digest):
                raise RuntimeError("runtime event integrity failed")
            _reason(event["reason"])
            events.append(event | {"event_sha256": digest})
    return events


def _event(root: Path, action: str, payload: dict, reason: str) -> dict:
    events = _events(root)
    event = {"sequence": len(events) + 1, "created_utc": core._now(),
             "action": action, "payload": payload, "reason": _reason(reason),
             "previous_event_sha256": events[-1]["event_sha256"] if events else None}
    event["event_sha256"] = _digest(event)
    with (root / "runtime_events.jsonl").open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def _bound_event(root: Path, pointer: dict, action: str, payload: dict) -> None:
    matches = [event for event in _events(root)
               if event["event_sha256"] == pointer.get("event_sha256")]
    if len(matches) != 1 or matches[0]["action"] != action or matches[0]["payload"] != payload:
        raise RuntimeError("CURRENT is not bound to a verified runtime event")


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise RuntimeError("runtime path escapes its immutable root")
    return path


def _files(root: Path, names: Iterable[str]) -> list[dict]:
    return [{"path": name, "sha256": core._sha(_inside(root, name)),
             "bytes": _inside(root, name).stat().st_size} for name in names]


def _check_files(root: Path, records: list[dict]) -> None:
    if len({record["path"] for record in records}) != len(records):
        raise RuntimeError("duplicate declared snapshot file")
    for record in records:
        path = _inside(root, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or core._sha(path) != record["sha256"]:
            raise RuntimeError(f"snapshot integrity failed: {path}")


def _seal_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def _copy_files(source: Path, destination: Path, names: Iterable[str]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)


def _core_call(method: str, manifest: dict) -> dict:
    return {"method": method, "implementation_path": str(Path(core.__file__).resolve()),
            "implementation_sha256": core._sha(Path(core.__file__)),
            "returned_manifest": manifest}


def _public_names(manifest: dict) -> list[str]:
    names = ["public_memory.parquet", "effect_vectors.npy", "control_vectors.npy",
             "gene_ids.json", "manifest.json"]
    if manifest.get("eligibility") is not None:
        if manifest["eligibility"]["path"] != "eligibility.parquet":
            raise RuntimeError("unexpected Public eligibility artifact path")
        names.append("eligibility.parquet")
    return names


def _finite_arrays(effects, controls, n_rows: int, n_genes: int) -> None:
    if effects.ndim != 2 or controls.ndim != 2 or effects.shape != (n_rows, n_genes) or controls.shape != effects.shape:
        raise ValueError("Public effects/controls must align with items and the gene axis")
    for start in range(0, n_rows, 512):
        if not np.isfinite(effects[start:start + 512]).all() or not np.isfinite(controls[start:start + 512]).all():
            raise ValueError("Public effects and controls must both be finite")


def _eligibility(frame: pd.DataFrame | None, ids: set[str]) -> None:
    if frame is None:
        return
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.duplicated().any():
        raise ValueError("eligibility must be a nonempty unique relation")
    fields = [field for field in ("public_experiment_id", "experiment_id") if field in frame]
    if not fields:
        raise ValueError("eligibility requires a supported Public experiment ID column")
    if len(fields) == 2 and not frame[fields[0]].eq(frame[fields[1]]).all():
        raise ValueError("eligibility experiment ID columns conflict")
    for field in ("public_experiment_id", "experiment_id"):
        if field in frame and not set(frame[field]).issubset(ids):
            raise ValueError("eligibility references an absent Public experiment")


class VersionedPublicRuntime:
    def __init__(self, root: Path | str, gene_ids: list[str]):
        self.root = Path(root).resolve()
        self.gene_ids = list(gene_ids)
        if not self.gene_ids or len(set(self.gene_ids)) != len(self.gene_ids) or any(not isinstance(g, str) or not g for g in self.gene_ids):
            raise ValueError("the Public gene axis must contain unique nonempty IDs")
        self.current_path = self.root / "CURRENT.json"
        self.store = core.PublicMemoryStore(self.root)

    def _version(self, version: int) -> Path:
        if not isinstance(version, int) or version < 1:
            raise ValueError("Public version must be a positive integer")
        return self.root / "versions" / f"v{version:04d}"

    def _snapshot(self, source: Path, version: int, parent: int | None, call: dict) -> Path:
        final = self._version(version)
        if final.exists():
            raise FileExistsError(f"immutable Public version exists: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        manifest = _json(source / "manifest.json")
        names = _public_names(manifest)
        with tempfile.TemporaryDirectory(prefix=".public_snapshot.", dir=final.parent) as temp:
            stage = Path(temp) / "snapshot"
            _copy_files(source, stage, names)
            _atomic(stage / "CORE_CALL.json", call)
            names.append("CORE_CALL.json")
            _atomic(stage / "runtime_snapshot.json", {
                "schema": "SafeConf-PublicRuntime-Snapshot-v1", "version": version,
                "parent_version": parent, "files": _files(stage, names)})
            pointer = {"version": version, "path": str(final.relative_to(self.root)),
                       "manifest_sha256": core._sha(stage / "manifest.json"),
                       "snapshot_sha256": core._sha(stage / "runtime_snapshot.json")}
            self._read_version(stage, pointer)
            stage.rename(final)
            _seal_tree(final)
        return final

    def _read_version(self, directory: Path, pointer: dict):
        if core._sha(directory / "manifest.json") != pointer["manifest_sha256"] or core._sha(directory / "runtime_snapshot.json") != pointer["snapshot_sha256"]:
            raise RuntimeError("Public CURRENT manifest/snapshot hash failed")
        snapshot = _json(directory / "runtime_snapshot.json")
        if snapshot.get("schema") != "SafeConf-PublicRuntime-Snapshot-v1" or snapshot["version"] != pointer["version"]:
            raise RuntimeError("Public snapshot version mismatch")
        _check_files(directory, snapshot["files"])
        manifest = _json(directory / "manifest.json")
        expected = set(_public_names(manifest)) | {"CORE_CALL.json", "runtime_snapshot.json"}
        if {p.name for p in directory.iterdir()} != expected or {x["path"] for x in snapshot["files"]} != expected - {"runtime_snapshot.json"}:
            raise RuntimeError("Public snapshot file set changed")
        if manifest.get("version") != pointer["version"]:
            raise RuntimeError("Public manifest version mismatch")
        genes = _json(directory / "gene_ids.json")["gene_ids"]
        if genes != self.gene_ids or manifest["gene_ids_sha256"] != core._sha(directory / "gene_ids.json"):
            raise RuntimeError("Public gene-axis identity/integrity failed")
        frame, effects, controls, manifest = core.PublicMemoryStore(directory).load()
        if len(frame) != manifest["n_items"] or manifest["n_genes"] != len(genes) or not frame.experiment_id.is_unique or not np.array_equal(frame.effect_vector_row.to_numpy(int), np.arange(len(frame))):
            raise RuntimeError("Public metadata/vector rows changed")
        _finite_arrays(effects, controls, len(frame), len(genes))
        if manifest.get("eligibility") is not None:
            eligibility = pd.read_parquet(directory / "eligibility.parquet")
            binding = manifest["eligibility"]
            if len(eligibility) != binding["n_rows"] or core._sha(directory / "eligibility.parquet") != binding["sha256"]:
                raise RuntimeError("Public eligibility binding failed")
            _eligibility(eligibility, set(frame.experiment_id))
        return frame, effects, controls, manifest

    def _load(self):
        pointer = _json(self.current_path)
        if pointer.get("schema") != "SafeConf-PublicRuntime-CURRENT-v1" or pointer["path"] != str(self._version(pointer["version"]).relative_to(self.root)):
            raise RuntimeError("Public CURRENT path/version mismatch")
        payload = {k: pointer[k] for k in ("version", "path", "manifest_sha256", "snapshot_sha256")}
        _bound_event(self.root, pointer, "public_activate", payload)
        return self._read_version(_inside(self.root, pointer["path"]), pointer)

    def _activate(self, version: int, reason: str) -> dict:
        directory = self._version(version)
        pointer = {"version": version, "path": str(directory.relative_to(self.root)),
                   "manifest_sha256": core._sha(directory / "manifest.json"),
                   "snapshot_sha256": core._sha(directory / "runtime_snapshot.json")}
        self._read_version(directory, pointer)
        event = _event(self.root, "public_activate", pointer, reason)
        pointer = pointer | {"schema": "SafeConf-PublicRuntime-CURRENT-v1", "event_sha256": event["event_sha256"]}
        _atomic(self.current_path, pointer)
        return pointer

    def create(self, items: Iterable[PublicMemoryItem], effects, controls,
               source_manifest: dict, eligibility: pd.DataFrame | None = None) -> dict:
        items = list(items)
        effects, controls = np.asarray(effects, dtype=np.float32), np.asarray(controls, dtype=np.float32)
        _finite_arrays(effects, controls, len(items), len(self.gene_ids))
        _eligibility(eligibility, {item.experiment_id for item in items})
        with _locked(self.root):
            if self.current_path.exists() or (self.root / "manifest.json").exists() or (self.root / "versions").exists():
                raise FileExistsError("refusing to overwrite a Public runtime")
            returned = self.store.create(items, effects, controls, self.gene_ids, source_manifest, eligibility)
            call = _core_call("PublicMemoryStore.create", returned)
            manifest = returned | {"version": 1}
            _atomic(self.root / "manifest.json", manifest)
            self._snapshot(self.root, 1, None, call)
            self._activate(1, "initial snapshot from actual PublicMemoryStore.create")
            return manifest

    def append(self, items: Iterable[PublicMemoryItem], effects, controls,
               source_manifest: dict, full_eligibility: pd.DataFrame | None = None) -> dict:
        items = list(items)
        effects, controls = np.asarray(effects, dtype=np.float32), np.asarray(controls, dtype=np.float32)
        _finite_arrays(effects, controls, len(items), len(self.gene_ids))
        with _locked(self.root):
            old, _, _, old_manifest = self._load()
            _eligibility(full_eligibility, set(old.experiment_id) | {item.experiment_id for item in items})
            old_root = self._version(old_manifest["version"])
            if old_manifest.get("eligibility") is not None:
                previous = pd.read_parquet(old_root / "eligibility.parquet")
                if full_eligibility is None or list(full_eligibility.columns) != list(previous.columns):
                    raise ValueError("append must retain the full existing eligibility relation")
                retained = previous.merge(full_eligibility, how="left", on=list(previous.columns), indicator=True)
                if not retained["_merge"].eq("both").all():
                    raise ValueError("append cannot drop previous eligibility rows")
            version = max(int(p.name[1:]) for p in (self.root / "versions").glob("v[0-9][0-9][0-9][0-9]")) + 1
            with tempfile.TemporaryDirectory(prefix=".public_append.", dir=self.root) as temp:
                stage = Path(temp) / "core"
                _copy_files(old_root, stage, _public_names(old_manifest))
                returned = core.PublicMemoryStore(stage).append(items, effects, controls, source_manifest, full_eligibility)
                call = _core_call("PublicMemoryStore.append", returned)
                generated = stage / "versions" / f"v{returned['version']:04d}"
                manifest = returned | {"version": version}
                _atomic(generated / "manifest.json", manifest)
                self._snapshot(generated, version, old_manifest["version"], call)
            self._activate(version, "append snapshot from actual PublicMemoryStore.append")
            return manifest

    def load(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
        with _locked(self.root):
            return self._load()

    def activate(self, version: int, reason: str) -> dict:
        with _locked(self.root):
            if self.current_path.exists():
                self._load()
            return self._activate(version, _reason(reason))

    def rollback_to(self, version: int, reason: str) -> dict:
        with _locked(self.root):
            self._load()
            current = _json(self.current_path)["version"]
            if version >= current:
                raise ValueError("Public rollback must target an earlier immutable version")
            return self._activate(version, _reason(reason))


class VersionedErrorRuntime:
    def __init__(self, root: Path | str, output_contract_id: str):
        self.root = Path(root).resolve()
        self.output_contract_id = _reason(output_contract_id)
        self.registry = core.ErrorMemoryRegistry(self.root)
        with _locked(self.root):
            path = self.root / "OUTPUT_CONTRACT.json"
            binding = {"schema": "SafeConf-ErrorRuntime-OutputContract-v1", "output_contract_id": self.output_contract_id}
            if path.exists():
                if _json(path) != binding:
                    raise RuntimeError("Error runtime output contract differs")
            else:
                if any(p.is_dir() for p in self.root.iterdir()):
                    raise RuntimeError("cannot adopt unbound existing error history")
                _atomic(path, binding)
                path.chmod(0o444)

    def _contract(self):
        if _json(self.root / "OUTPUT_CONTRACT.json") != {"schema": "SafeConf-ErrorRuntime-OutputContract-v1", "output_contract_id": self.output_contract_id}:
            raise RuntimeError("Error runtime output contract changed")

    def _read(self, upstream: str, version: str):
        self._contract()
        directory = self.registry._key_dir(upstream, version)
        pointer = _json(directory / "CURRENT.json")
        if pointer.get("schema") != "SafeConf-ErrorRuntime-CURRENT-v1" or pointer["upstream_model_id"] != upstream or pointer["model_version"] != version or pointer["output_contract_id"] != self.output_contract_id:
            raise RuntimeError("Error CURRENT identity/contract mismatch")
        if pointer["output_contract_sha256"] != core._sha(self.root / "OUTPUT_CONTRACT.json"):
            raise RuntimeError("Error output-contract sidecar integrity failed")
        revision = _inside(directory, pointer["path"])
        if pointer["path"] != f"revisions/r{pointer['revision']:04d}" or core._sha(revision / "runtime_revision.json") != pointer["revision_sha256"]:
            raise RuntimeError("Error CURRENT revision binding failed")
        record = _json(revision / "runtime_revision.json")
        payload = {k: pointer[k] for k in ("upstream_model_id", "model_version", "output_contract_id", "output_contract_sha256", "revision", "path", "revision_sha256", "batch_id")}
        _bound_event(self.root, pointer, "error_append", payload)
        if (record.get("schema") != "SafeConf-ErrorRuntime-Revision-v1" or record["revision"] != pointer["revision"] or record["batch_id"] != pointer["batch_id"] or
                record["output_contract_id"] != self.output_contract_id or record["upstream_model_id"] != upstream or record["model_version"] != version):
            raise RuntimeError("Error immutable revision identity differs")
        if record["output_contract_sha256"] != pointer["output_contract_sha256"]:
            raise RuntimeError("Error revision output-contract binding differs")
        _check_files(revision, record["files"])
        snapshot_registry = core.ErrorMemoryRegistry(revision)
        bank = snapshot_registry._key_dir(upstream, version)
        if core._sha(bank / "manifest.json") != pointer["manifest_sha256"]:
            raise RuntimeError("Error CURRENT manifest hash failed")
        manifest = _json(bank / "manifest.json")
        frame = snapshot_registry.load(upstream, version)
        self._validate_items(frame, upstream, version)
        if len(frame) != manifest["n_items"]:
            raise RuntimeError("Error revision row count changed")
        # The legacy live files remain verifiable mirrors of the current revision.
        for name in ("manifest.json", "error_memory.parquet"):
            if core._sha(directory / name) != core._sha(bank / name):
                raise RuntimeError("Error live mirror differs from its pinned revision")
        return frame, manifest, pointer, revision

    @staticmethod
    def _validate_items(frame: pd.DataFrame, upstream: str, version: str) -> None:
        if frame.empty or not frame.prediction_id.is_unique or set(frame.upstream_model_id) != {upstream} or set(frame.model_version) != {version} or not frame.is_oof_or_heldout.eq(True).all():
            raise ValueError("exact-version unique legal error records required")
        values = frame[["shared_risk", "realised_error", "error_rank", "shared_risk_residual"]].to_numpy(float)
        if not np.isfinite(values).all() or (frame.realised_error < 0).any():
            raise ValueError("error records must contain finite values and nonnegative realised errors")

    def append(self, items: Iterable[ErrorMemoryItem], batch_id: str) -> dict:
        items = list(items)
        batch_id = _reason(batch_id)
        frame = pd.DataFrame([item.as_record() for item in items])
        if frame.empty or len(frame[["upstream_model_id", "model_version"]].drop_duplicates()) != 1:
            raise ValueError("append requires one nonempty exact model version")
        upstream, version = map(str, frame[["upstream_model_id", "model_version"]].iloc[0])
        self._validate_items(frame, upstream, version)
        with _locked(self.root):
            self._contract()
            directory = self.registry._key_dir(upstream, version)
            if (directory / "CURRENT.json").exists():
                _, _, pointer, previous = self._read(upstream, version)
                revision = pointer["revision"] + 1
                for old in (directory / "revisions").glob("r[0-9][0-9][0-9][0-9]"):
                    if _json(old / "runtime_revision.json")["batch_id"] == batch_id:
                        raise ValueError("Error batch ID is immutable for this exact model version")
            else:
                if directory.exists():
                    raise RuntimeError("cannot adopt an unverified existing Error key")
                previous = None
                revision = 1
            final = directory / "revisions" / f"r{revision:04d}"
            if final.exists():
                raise FileExistsError("immutable Error revision already exists")
            final.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".error_append.", dir=self.root) as temp:
                working = Path(temp) / "core"
                working.mkdir()
                actual = core.ErrorMemoryRegistry(working)
                bank = actual._key_dir(upstream, version)
                if previous is not None:
                    old_bank = core.ErrorMemoryRegistry(previous)._key_dir(upstream, version)
                    _copy_files(old_bank, bank, ["manifest.json", "error_memory.parquet"])
                returned = actual.append(items)
                actual.load(upstream, version)
                _atomic(working / "CORE_CALL.json", _core_call("ErrorMemoryRegistry.append", returned))
                names = [str((bank / name).relative_to(working)) for name in ("manifest.json", "error_memory.parquet")] + ["CORE_CALL.json"]
                _atomic(working / "runtime_revision.json", {
                    "schema": "SafeConf-ErrorRuntime-Revision-v1", "revision": revision,
                    "parent_revision": revision - 1 if previous else None, "batch_id": batch_id,
                    "upstream_model_id": upstream, "model_version": version,
                    "output_contract_id": self.output_contract_id,
                    "output_contract_sha256": core._sha(self.root / "OUTPUT_CONTRACT.json"),
                    "files": _files(working, names)})
                working.rename(final)
                _seal_tree(final)
            bank = core.ErrorMemoryRegistry(final)._key_dir(upstream, version)
            for name in ("manifest.json", "error_memory.parquet"):
                with tempfile.NamedTemporaryFile(dir=directory, delete=False, prefix=".live.") as handle:
                    temporary = Path(handle.name)
                shutil.copyfile(bank / name, temporary)
                os.replace(temporary, directory / name)
            payload = {"upstream_model_id": upstream, "model_version": version,
                       "output_contract_id": self.output_contract_id, "revision": revision,
                       "output_contract_sha256": core._sha(self.root / "OUTPUT_CONTRACT.json"),
                       "path": str(final.relative_to(directory)), "revision_sha256": core._sha(final / "runtime_revision.json"),
                       "batch_id": batch_id}
            event = _event(self.root, "error_append", payload, f"append immutable error batch {batch_id} through actual ErrorMemoryRegistry.append")
            pointer = payload | {"schema": "SafeConf-ErrorRuntime-CURRENT-v1",
                                 "manifest_sha256": core._sha(bank / "manifest.json"), "event_sha256": event["event_sha256"]}
            _atomic(directory / "CURRENT.json", pointer)
            self._read(upstream, version)
            return returned | {"runtime_revision": revision, "runtime_revision_path": str(final),
                               "runtime_revision_sha256": pointer["revision_sha256"],
                               "batch_id": batch_id, "output_contract_id": self.output_contract_id}

    def load(self, upstream_model_id: str, model_version: str) -> pd.DataFrame:
        with _locked(self.root):
            return self._read(upstream_model_id, model_version)[0]

    def manifest(self, upstream_model_id: str, model_version: str) -> dict:
        with _locked(self.root):
            _, manifest, pointer, revision = self._read(upstream_model_id, model_version)
            return manifest | {"runtime_revision": pointer["revision"], "runtime_revision_path": str(revision),
                               "runtime_revision_sha256": pointer["revision_sha256"],
                               "batch_id": pointer["batch_id"], "output_contract_id": self.output_contract_id}


class ServingModelRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.registry = core.ModelRegistry(self.root)
        self.current_path = self.root / "SERVING_CURRENT.json"

    def _binding(self, component: str, version: str) -> Path:
        return self.root / "models" / _digest(component)[:20] / _digest(version)[:20]

    def _entries(self) -> list[dict]:
        entries = self.registry.entries()
        mutations = [event for event in _events(self.root)
                     if event["action"] in ("model_register", "model_rollback")]
        witnessed = [event["payload"]["registry_entry"] for event in mutations]
        if entries != witnessed or len({(e["component"], e["version"]) for e in entries}) != len(entries):
            raise RuntimeError("actual ModelRegistry ledger differs from runtime event bindings")
        if mutations and core._sha(self.registry.path) != mutations[-1]["payload"]["registry_file_sha256"]:
            raise RuntimeError("actual ModelRegistry ledger bytes changed")
        return entries

    def _released(self, component: str, version: str) -> tuple[dict, Path]:
        entries = [e for e in self._entries() if e["component"] == component and e["version"] == version]
        if len(entries) != 1 or entries[0]["status"] != "RELEASED":
            raise ValueError("serving requires an existing RELEASED exact model version")
        directory = self._binding(component, version)
        binding = _json(directory / "binding.json")
        if binding["registry_entry"] != entries[0]:
            raise RuntimeError("model artifact binding differs from actual registry entry")
        witnessed = [event for event in _events(self.root) if event["action"] == "model_register"
                     and event["payload"]["registry_entry"] == entries[0]]
        if len(witnessed) != 1 or witnessed[0]["payload"]["binding_sha256"] != core._sha(directory / "binding.json"):
            raise RuntimeError("model binding differs from immutable registration event")
        artifact = _inside(directory, binding["artifact_file"])
        if core._sha(artifact) != entries[0]["artifact_sha256"] or artifact.stat().st_size != binding["artifact_bytes"]:
            raise RuntimeError("registered serving artifact integrity failed")
        return binding, artifact

    def _current(self, component: str) -> Path:
        pointers = _json(self.current_path)
        if pointers.get("schema") != "SafeConf-ServingRuntime-CURRENT-v1":
            raise RuntimeError("serving CURRENT schema differs")
        pointer = pointers["components"][component]
        binding, artifact = self._released(component, pointer["version"])
        if pointer["component"] != component or pointer["artifact_path"] != str(artifact) or pointer["artifact_sha256"] != binding["registry_entry"]["artifact_sha256"] or pointer["binding_sha256"] != core._sha(self._binding(component, pointer["version"]) / "binding.json"):
            raise RuntimeError("serving CURRENT artifact binding failed")
        payload = {k: pointer[k] for k in ("component", "version", "artifact_path", "artifact_sha256", "binding_sha256")}
        _bound_event(self.root, pointer, "model_publish", payload)
        return artifact

    def _publish(self, component: str, version: str, reason: str) -> dict:
        binding, artifact = self._released(component, version)
        pointers = (_json(self.current_path) if self.current_path.exists() else
                    {"schema": "SafeConf-ServingRuntime-CURRENT-v1", "components": {}})
        if not isinstance(pointers, dict) or pointers.get("schema") != "SafeConf-ServingRuntime-CURRENT-v1" or not isinstance(pointers.get("components"), dict):
            raise RuntimeError("existing serving CURRENT schema/components are invalid")
        for name in pointers["components"]:
            self._current(name)
        payload = {"component": component, "version": version, "artifact_path": str(artifact),
                   "artifact_sha256": binding["registry_entry"]["artifact_sha256"],
                   "binding_sha256": core._sha(self._binding(component, version) / "binding.json")}
        event = _event(self.root, "model_publish", payload, reason)
        pointer = payload | {"event_sha256": event["event_sha256"]}
        pointers["components"][component] = pointer
        _atomic(self.current_path, pointers)
        return pointer

    def register(self, component: str, version: str, artifact_path: Path | str,
                 split_hash: str, feature_schema_hash: str, parent_version: str | None = None,
                 status: str = "RELEASED") -> dict:
        component, version = _reason(component), _reason(version)
        split_hash, feature_schema_hash = _reason(split_hash), _reason(feature_schema_hash)
        if status not in {"RELEASED", "REJECTED"}:
            raise ValueError("new model status must be RELEASED or REJECTED")
        source = Path(artifact_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with _locked(self.root):
            entries = self._entries()
            if any(e["component"] == component and e["version"] == version for e in entries):
                raise ValueError("registered model versions are immutable")
            if parent_version is not None:
                self._released(component, parent_version)
            final = self._binding(component, version)
            if final.exists():
                raise FileExistsError("immutable model binding already exists")
            final.parent.mkdir(parents=True, exist_ok=True)
            entry = core.RegistryEntry(component=component, version=version, status=status,
                artifact_sha256=core._sha(source), split_hash=split_hash,
                feature_schema_hash=feature_schema_hash, parent_version=parent_version, created_utc=core._now())
            with tempfile.TemporaryDirectory(prefix=".model_binding.", dir=final.parent) as temp:
                stage = Path(temp) / "model"
                stage.mkdir()
                artifact = stage / ("artifact" + source.suffix)
                shutil.copyfile(source, artifact)
                if core._sha(artifact) != entry.artifact_sha256:
                    raise RuntimeError("model artifact changed during registration")
                binding = {"schema": "SafeConf-ServingRuntime-Binding-v1", "registry_entry": asdict(entry),
                           "source_artifact_path": str(source), "source_artifact_sha256": entry.artifact_sha256,
                           "artifact_file": artifact.name, "artifact_bytes": artifact.stat().st_size,
                           "core_method": "ModelRegistry.register", "core_implementation_sha256": core._sha(Path(core.__file__))}
                _atomic(stage / "binding.json", binding)
                self.registry.register(entry)
                stage.rename(final)
                _seal_tree(final)
            _event(self.root, "model_register", {"registry_entry": asdict(entry),
                   "registry_file_sha256": core._sha(self.registry.path),
                   "binding_sha256": core._sha(final / "binding.json")}, f"register {status} artifact {component}/{version}")
            self._entries()
            return asdict(entry) | {"artifact_path": str(final / artifact.name), "source_artifact_path": str(source),
                                   "binding_path": str(final / "binding.json")}

    def publish(self, component: str, version: str, reason: str) -> dict:
        with _locked(self.root):
            return self._publish(component, version, _reason(reason))

    def current(self, component: str) -> Path:
        with _locked(self.root):
            return self._current(component)

    def rollback(self, component: str, to_version: str, reason: str) -> dict:
        with _locked(self.root):
            _reason(reason)
            self._current(component)
            self._released(component, to_version)
            self.registry.rollback(component, to_version)
            entry = self.registry.entries()[-1]
            _event(self.root, "model_rollback", {"registry_entry": entry, "target_version": to_version,
                   "registry_file_sha256": core._sha(self.registry.path)}, reason)
            return self._publish(component, to_version, reason)
