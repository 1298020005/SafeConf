"""Append-only, versioned storage for the two SafeConf memory banks."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import ErrorMemoryItem, PublicMemoryItem


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


class PublicMemoryStore:
    """Model-independent biological experiments plus aligned effect vectors."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.table_path = self.root / "public_memory.parquet"
        self.vector_path = self.root / "effect_vectors.npy"
        self.control_path = self.root / "control_vectors.npy"
        self.manifest_path = self.root / "manifest.json"

    def create(
        self,
        items: Iterable[PublicMemoryItem],
        effect_vectors: np.ndarray,
        control_vectors: np.ndarray,
        gene_ids: list[str],
        source_manifest: dict,
        eligibility_frame: pd.DataFrame | None = None,
    ) -> dict:
        if any(path.exists() for path in (self.table_path, self.vector_path, self.manifest_path)):
            raise FileExistsError(f"refusing to overwrite public memory: {self.root}")
        rows = [item.as_record() for item in items]
        frame = pd.DataFrame(rows)
        if frame.empty or frame.experiment_id.duplicated().any():
            raise ValueError("public memory IDs must be non-empty and unique")
        effects = np.asarray(effect_vectors, dtype=np.float32)
        controls = np.asarray(control_vectors, dtype=np.float32)
        if effects.shape != controls.shape or effects.shape[0] != len(frame):
            raise ValueError("memory metadata/effect/control alignment failed")
        if effects.shape[1] != len(gene_ids) or not np.isfinite(effects).all():
            raise ValueError("public-memory gene axis or effect values failed")
        expected_rows = np.arange(len(frame))
        if not np.array_equal(frame.effect_vector_row.to_numpy(int), expected_rows):
            raise ValueError("effect_vector_row must be contiguous and aligned")
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(self.table_path, frame)
        _atomic_npy(self.vector_path, effects)
        _atomic_npy(self.control_path, controls)
        gene_path = self.root / "gene_ids.json"
        _atomic_json(gene_path, {"gene_ids": gene_ids})
        eligibility_record = None
        if eligibility_frame is not None:
            eligibility_path = self.root / "eligibility.parquet"
            if eligibility_frame.empty or eligibility_frame.duplicated().any():
                raise ValueError("eligibility relation must be non-empty and unique")
            _atomic_parquet(eligibility_path, eligibility_frame)
            eligibility_record = {
                "path": eligibility_path.name,
                "n_rows": len(eligibility_frame),
                "sha256": _sha(eligibility_path),
            }
        manifest = {
            "schema": "SafeConf-PublicMemory-v1",
            "created_utc": _now(),
            "n_items": len(frame),
            "n_genes": len(gene_ids),
            "table_sha256": _sha(self.table_path),
            "effect_vectors_sha256": _sha(self.vector_path),
            "control_vectors_sha256": _sha(self.control_path),
            "gene_ids_sha256": _sha(gene_path),
            "eligibility": eligibility_record,
            "source_manifest": source_manifest,
        }
        _atomic_json(self.manifest_path, manifest)
        return manifest

    def load(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
        current_path = self.root / "CURRENT.json"
        if current_path.exists():
            pointer = json.loads(current_path.read_text())
            version_root = self.root / pointer["path"]
            return PublicMemoryStore(version_root).load()
        manifest = json.loads(self.manifest_path.read_text())
        for path, field in (
            (self.table_path, "table_sha256"),
            (self.vector_path, "effect_vectors_sha256"),
            (self.control_path, "control_vectors_sha256"),
        ):
            if _sha(path) != manifest[field]:
                raise RuntimeError(f"public memory integrity failed: {path}")
        frame = pd.read_parquet(self.table_path)
        effects = np.load(self.vector_path, mmap_mode="r")
        controls = np.load(self.control_path, mmap_mode="r")
        if len(frame) != len(effects) or effects.shape != controls.shape:
            raise RuntimeError("public memory alignment changed")
        return frame, effects, controls, manifest

    def append(
        self,
        items: Iterable[PublicMemoryItem],
        effect_vectors: np.ndarray,
        control_vectors: np.ndarray,
        source_manifest: dict,
        eligibility_frame: pd.DataFrame | None = None,
    ) -> dict:
        """Publish an immutable version containing the old bank plus new items.

        The existing version is never rewritten. A CURRENT pointer is updated
        atomically only after the complete new version has passed integrity
        checks. Callers can retain the previous version directory for rollback.
        """
        old, old_effects, old_controls, old_manifest = self.load()
        additions = pd.DataFrame([item.as_record() for item in items])
        if additions.empty or additions.experiment_id.duplicated().any():
            raise ValueError("public-memory update must contain unique items")
        if set(additions.experiment_id) & set(old.experiment_id):
            raise ValueError("public-memory update contains an existing experiment")
        new_effects = np.asarray(effect_vectors, dtype=np.float32)
        new_controls = np.asarray(control_vectors, dtype=np.float32)
        if new_effects.shape != new_controls.shape or new_effects.shape[0] != len(additions):
            raise ValueError("public-memory update vector alignment failed")
        if new_effects.shape[1] != old_effects.shape[1] or not np.isfinite(new_effects).all():
            raise ValueError("public-memory update gene axis failed")
        additions = additions.copy()
        additions["effect_vector_row"] = np.arange(len(old), len(old) + len(additions))
        combined = pd.concat([old, additions], ignore_index=True)
        version = int(old_manifest.get("version", 1)) + 1
        version_root = self.root / "versions" / f"v{version:04d}"
        version_root.parent.mkdir(parents=True, exist_ok=True)
        manifest = PublicMemoryStore(version_root).create(
            [PublicMemoryItem(**row) for row in combined.to_dict(orient="records")],
            np.concatenate([old_effects, new_effects]),
            np.concatenate([old_controls, new_controls]),
            json.loads((self.root / "gene_ids.json").read_text())["gene_ids"],
            {**source_manifest, "parent_manifest": old_manifest},
            eligibility_frame=eligibility_frame,
        )
        manifest["version"] = version
        _atomic_json(version_root / "manifest.json", manifest)
        _atomic_json(self.root / "CURRENT.json", {
            "version": version,
            "path": str(Path("versions") / f"v{version:04d}"),
            "manifest_sha256": _sha(version_root / "manifest.json"),
        })
        return manifest


class ErrorMemoryRegistry:
    """Separate append-only error histories for exact upstream versions."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _key_dir(self, upstream_model_id: str, model_version: str) -> Path:
        key = hashlib.sha256(f"{upstream_model_id}\0{model_version}".encode()).hexdigest()[:20]
        return self.root / key

    def append(self, items: Iterable[ErrorMemoryItem]) -> dict:
        rows = [item.as_record() for item in items]
        if not rows:
            raise ValueError("cannot append an empty error-memory update")
        frame = pd.DataFrame(rows)
        keys = frame[["upstream_model_id", "model_version"]].drop_duplicates()
        if len(keys) != 1 or not frame.is_oof_or_heldout.all():
            raise ValueError("one update must contain one exact model version and legal errors")
        upstream, version = keys.iloc[0]
        directory = self._key_dir(str(upstream), str(version))
        table = directory / "error_memory.parquet"
        existing = pd.read_parquet(table) if table.exists() else pd.DataFrame()
        combined = pd.concat([existing, frame], ignore_index=True)
        if combined.prediction_id.duplicated().any():
            raise ValueError("duplicate prediction outcome in error memory")
        _atomic_parquet(table, combined)
        manifest = {
            "schema": "SafeConf-ErrorMemory-v1",
            "updated_utc": _now(),
            "upstream_model_id": str(upstream),
            "model_version": str(version),
            "n_items": len(combined),
            "table_sha256": _sha(table),
        }
        _atomic_json(directory / "manifest.json", manifest)
        return manifest

    def load(self, upstream_model_id: str, model_version: str) -> pd.DataFrame:
        directory = self._key_dir(upstream_model_id, model_version)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["upstream_model_id"] != upstream_model_id or manifest["model_version"] != model_version:
            raise RuntimeError("error-memory model identity mismatch")
        table = directory / "error_memory.parquet"
        if _sha(table) != manifest["table_sha256"]:
            raise RuntimeError("error-memory integrity failed")
        return pd.read_parquet(table)


@dataclass(frozen=True)
class RegistryEntry:
    component: str
    version: str
    status: str
    artifact_sha256: str
    split_hash: str
    feature_schema_hash: str
    parent_version: str | None
    created_utc: str


class ModelRegistry:
    """Append-only publication/rollback ledger for continually updated models."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.path = self.root / "model_registry.jsonl"

    def register(self, entry: RegistryEntry) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        existing = self.entries()
        if any(x["component"] == entry.component and x["version"] == entry.version for x in existing):
            raise ValueError("model registry versions are immutable")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def latest(self, component: str) -> dict | None:
        entries = [entry for entry in self.entries() if entry["component"] == component]
        return entries[-1] if entries else None

    def rollback(self, component: str, to_version: str) -> None:
        target = [
            entry for entry in self.entries()
            if entry["component"] == component and entry["version"] == to_version
        ]
        if not target:
            raise ValueError(f"unknown immutable version for rollback: {component}/{to_version}")
        latest = self.latest(component)
        if latest is not None and latest["version"] == to_version and latest["status"] == "RELEASED":
            raise ValueError("rollback target is already the current released version")
        entry = target[-1]
        self.register(RegistryEntry(
            component=component,
            version=f"rollback-to-{to_version}-{len(self.entries())}",
            status="ROLLED_BACK",
            artifact_sha256=entry["artifact_sha256"],
            split_hash=entry["split_hash"],
            feature_schema_hash=entry["feature_schema_hash"],
            parent_version=latest["version"] if latest else None,
            created_utc=_now(),
        ))
