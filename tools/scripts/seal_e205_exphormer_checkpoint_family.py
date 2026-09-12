#!/usr/bin/env python3
"""Verify and seal the 16 E205 Exphormer checkpoints without target truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path


TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEEDS = (1, 2, 3, 4)
TXPERT_COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
ADAPTER_SHA256 = "fe4e1a2e6861403fe5c650cc11ac0c58177fddb0ea9cf6f0ed0e48578613a285"
EXPECTED = {
    "K562": {
        "rows": 294_951,
        "train_batches": 4_608,
        "val_rows": 80_340,
        "val_batches": 1_256,
        "view_sha256": "8a19d3d4048800c06827e2f28e983bfa6f67b1945d2081692ce3d69a45d471db",
    },
    "RPE1": {
        "rows": 273_003,
        "train_batches": 4_265,
        "val_rows": 76_950,
        "val_batches": 1_203,
        "view_sha256": "9aeec2bdc56461713e9039348428d63aa1b98f6000835ed3253c35d6d85a387d",
    },
    "hepg2": {
        "rows": 314_391,
        "train_batches": 4_912,
        "val_rows": 89_117,
        "val_batches": 1_393,
        "view_sha256": "1f0dc20806bd40cd151ebfebb59a9fdac5ad14c4e223a655ae0ed6de890ed891",
    },
    "jurkat": {
        "rows": 282_132,
        "train_batches": 4_408,
        "val_rows": 78_682,
        "val_batches": 1_230,
        "view_sha256": "5a944ec0f114e2398f2058072121d130deee5af1f97def926619f5ee30c231fb",
    },
}


class SealFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def data_path(path: Path, data_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise SealFailure(f"artifact is outside data root: {path}") from exc
    return "DATA/" + relative.as_posix()


def checkpoint_schema(checkpoint: dict) -> tuple[str, bool, int, int]:
    import torch

    schema = []
    finite = True
    elements = 0
    for key, tensor in sorted(checkpoint["state_dict"].items()):
        schema.append(
            {"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        finite = finite and bool(torch.isfinite(tensor).all())
        elements += int(tensor.numel())
    return canonical_hash(schema), finite, len(schema), elements


def inspect_run(
    runs_root: Path, data_root: Path, target: str, seed: int
) -> dict:
    import torch

    run_dir = runs_root / target / f"seed_{seed}"
    status_path = run_dir / "E205_RUN_STATUS.json"
    if not status_path.is_file():
        raise SealFailure(f"missing status: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected = EXPECTED[target]
    tracking = status.get("safeconf_tracking_commits", {})
    model = status.get("resolved_model_config", {}).get("pert_model", {})
    graph = status.get("resolved_graph_config", {}).get("graph_cfg", {}).get(
        "graph1", {}
    )
    gates = {
        "complete": status.get("status") == "COMPLETE",
        "formal": status.get("kind") == "formal",
        "experiment": status.get("experiment") == "E205_cross_family_exphormer",
        "target": status.get("target") == target,
        "seed": int(status.get("seed", -1)) == seed,
        "family": status.get("model_family") == "TxPert-Exphormer",
        "comparison_family": status.get("comparison_family")
        == "TxPert-STRING-GAT",
        "architecture_only": status.get("architecture_change_only") is True,
        "base_config": status.get("base_config") == "config-x-cell-gat",
        "architecture_config": status.get("architecture_config")
        == "config-exphormer",
        "model_type": model.get("model_type") == "exphormer",
        "layer_type": model.get("layer_type") == "exphormer_w_mpnn",
        "graph": graph.get("graph_type") == "string",
        "txpert_commit": status.get("txpert_commit") == TXPERT_COMMIT,
        "txpert_clean": status.get("txpert_clean") is True,
        "adapter_sha256": status.get("training_adapter_sha256") == ADAPTER_SHA256,
        "adapter_committed": status.get("training_adapter_committed") is True,
        "tracking_refs": set(tracking) == {"origin", "github"}
        and all(value == status.get("safeconf_commit") for value in tracking.values()),
        "zero_target_training": int(
            status.get("blind_view_manifest", {}).get("n_target_treatments", -1)
        )
        == 0,
        "zero_target_access": int(status.get("target_perturbed_cells_accessed", -1))
        == 0,
        "test_not_constructed": status.get("target_test_dataset_constructed")
        is False,
        "view_hash": status.get("blind_view_manifest", {})
        .get("files", [{}])[0]
        .get("sha256")
        == expected["view_sha256"],
        "train_rows": int(status.get("n_train_dataset_rows", -1))
        == expected["rows"],
        "train_batches": int(status.get("n_train_batches", -1))
        == expected["train_batches"],
        "validation_rows": int(status.get("n_validation_dataset_rows", -1))
        == expected["val_rows"],
        "validation_batches": int(status.get("n_validation_batches", -1))
        == expected["val_batches"],
        "max_epochs": int(status.get("max_epochs", -1)) == 80,
        "current_epoch": int(status.get("current_epoch", -1)) == 80,
        "global_step": int(status.get("global_step", -1))
        == expected["train_batches"] * 80,
        "finite_fit_time": math.isfinite(
            float(status.get("fit_wall_seconds", math.nan))
        )
        and float(status.get("fit_wall_seconds", 0)) > 0,
        "finite_best_score": math.isfinite(
            float(status.get("best_model_score", math.nan))
        ),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise SealFailure(f"{target}/seed_{seed} gates failed: {failed}")

    entries = {item.get("role"): item for item in status.get("checkpoint_files", [])}
    if set(entries) != {"last", "best_source_validation"}:
        raise SealFailure(f"unexpected checkpoint roles: {target}/seed_{seed}")
    inspected = {}
    for role in ("last", "best_source_validation"):
        entry = entries[role]
        path = Path(str(entry.get("path", ""))).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(entry.get("bytes", -1))
            or sha256_file(path) != entry.get("sha256")
        ):
            raise SealFailure(f"checkpoint changed: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if role == "last" and (
            int(checkpoint.get("epoch", -1)) != 79
            or int(checkpoint.get("global_step", -1))
            != expected["train_batches"] * 80
        ):
            raise SealFailure(f"invalid last checkpoint counters: {path}")
        if role == "best_source_validation" and not (
            0 <= int(checkpoint.get("epoch", -1)) <= 79
        ):
            raise SealFailure(f"invalid best checkpoint epoch: {path}")
        schema_hash, finite, tensor_count, element_count = checkpoint_schema(checkpoint)
        if not finite:
            raise SealFailure(f"non-finite state: {path}")
        inspected[role] = {
            "path": data_path(path, data_root),
            "bytes": path.stat().st_size,
            "sha256": entry["sha256"],
            "epoch_zero_based": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "state_schema_sha256": schema_hash,
            "state_tensor_count": tensor_count,
            "state_element_count": element_count,
        }
    if (
        inspected["last"]["state_schema_sha256"]
        != inspected["best_source_validation"]["state_schema_sha256"]
    ):
        raise SealFailure(f"checkpoint schemas differ: {target}/seed_{seed}")
    return {
        "target": target,
        "seed": seed,
        "run_status_path": data_path(status_path, data_root),
        "run_status_sha256": sha256_file(status_path),
        "safeconf_commit": status["safeconf_commit"],
        "training_adapter_sha256": status["training_adapter_sha256"],
        "fit_wall_seconds": float(status["fit_wall_seconds"]),
        "best_source_validation_score": float(status["best_model_score"]),
        "last": inspected["last"],
        "best_source_validation": inspected["best_source_validation"],
    }


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    data_root = args.data_root.resolve()
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    for path in (output_json, output_csv):
        if path.exists():
            raise SealFailure(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        inspect_run(runs_root, data_root, target, seed)
        for target in TARGETS
        for seed in SEEDS
    ]
    schemas = {record["last"]["state_schema_sha256"] for record in records}
    if len(schemas) != 1:
        raise SealFailure("state schema differs across E205 jobs")
    payload = {
        "experiment": "E205_cross_family_exphormer",
        "status": "SEALED_16_EXPHORMER_CHECKPOINTS",
        "sealed_at": now(),
        "target_truth_opened": False,
        "targets": list(TARGETS),
        "seeds": list(SEEDS),
        "n_jobs": len(records),
        "txpert_commit": TXPERT_COMMIT,
        "training_adapter_sha256": ADAPTER_SHA256,
        "model_family": "TxPert-Exphormer",
        "comparison_family": "TxPert-STRING-GAT",
        "state_schema_sha256": next(iter(schemas)),
        "records_sha256": canonical_hash(records),
        "records": records,
    }
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, output_json)
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        columns = (
            "target",
            "seed",
            "safeconf_commit",
            "fit_wall_seconds",
            "best_source_validation_score",
            "last_path",
            "last_bytes",
            "last_sha256",
            "best_path",
            "best_bytes",
            "best_sha256",
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "target": record["target"],
                    "seed": record["seed"],
                    "safeconf_commit": record["safeconf_commit"],
                    "fit_wall_seconds": record["fit_wall_seconds"],
                    "best_source_validation_score": record[
                        "best_source_validation_score"
                    ],
                    "last_path": record["last"]["path"],
                    "last_bytes": record["last"]["bytes"],
                    "last_sha256": record["last"]["sha256"],
                    "best_path": record["best_source_validation"]["path"],
                    "best_bytes": record["best_source_validation"]["bytes"],
                    "best_sha256": record["best_source_validation"]["sha256"],
                }
            )
    os.replace(temporary_csv, output_csv)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
