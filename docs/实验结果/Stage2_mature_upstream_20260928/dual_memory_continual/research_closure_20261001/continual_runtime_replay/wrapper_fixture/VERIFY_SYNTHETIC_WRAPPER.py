"""Bounded synthetic verification; no production data, fits or evaluation reads."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.contracts import PublicMemoryItem, ErrorMemoryItem
from tools.safeconf_continual import memory as core
from tools.safeconf_continual.versioned_runtime import VersionedPublicRuntime, VersionedErrorRuntime, ServingModelRegistry

calls = {}
checks = []


def traced(cls, name):
    original = getattr(cls, name)
    def invoke(self, *args, **kwargs):
        key = cls.__name__ + "." + name
        calls[key] = calls.get(key, 0) + 1
        return original(self, *args, **kwargs)
    return invoke


def public_item(identifier, row):
    return PublicMemoryItem(identifier, "SYNTHETIC", "CTX", "CRISPR", identifier, "mock",
        row, "synthetic_effect_v1", "mockNTC", "synthetic3genes", 40, 2, None, 1,
        None, None, None, None, "synthetic_fixture", True, "2026-10-02")


def error_item(identifier, model="model-v1"):
    return ErrorMemoryItem("UPSTREAM", model, identifier, "task-" + identifier,
        "gene-" + identifier, .2, .3, .4, .2, "2026-10-02", "synthetic_fixture", True)


def reject(label, fn, errors=(RuntimeError, ValueError, FileNotFoundError, KeyError)):
    try:
        fn()
    except errors:
        checks.append(label)
    else:
        raise AssertionError("did not reject: " + label)


def tamper(path, fn, label, mutation=None):
    old = path.read_bytes()
    mode = path.stat().st_mode & 0o777
    path.chmod(0o644)
    if mutation:
        value = json.loads(old)
        mutation(value)
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(old + b" ")
    try:
        reject(label, fn)
    finally:
        path.write_bytes(old)
        path.chmod(mode)


def main():
    with patch.object(core.PublicMemoryStore, "create", traced(core.PublicMemoryStore, "create")), \
         patch.object(core.PublicMemoryStore, "append", traced(core.PublicMemoryStore, "append")), \
         patch.object(core.ErrorMemoryRegistry, "append", traced(core.ErrorMemoryRegistry, "append")), \
         patch.object(core.ModelRegistry, "register", traced(core.ModelRegistry, "register")), \
         patch.object(core.ModelRegistry, "rollback", traced(core.ModelRegistry, "rollback")):
        with tempfile.TemporaryDirectory(prefix="safeconf_versioned_wrapper_fixture_") as name:
            tmp = Path(name)
            genes = ["g1", "g2", "g3"]
            public = VersionedPublicRuntime(tmp / "public", genes)
            initial = [public_item("p1", 0), public_item("p2", 1)]
            effects = np.array([[1., 2., 3.], [3., 4., 5.]], np.float32)
            controls = effects / 10
            eligibility = pd.DataFrame({"experiment_id": ["p1", "p2"], "target_context": ["CTX", "CTX"]})
            first = public.create(initial, effects, controls, {"synthetic": True}, eligibility)
            assert first["version"] == 1 and json.loads(public.current_path.read_text())["version"] == 1
            v1 = public.root / "versions/v0001"
            old = {p.name: core._sha(p) for p in v1.iterdir()}
            reject("public_create_no_overwrite", lambda: public.create(initial, effects, controls, {}, eligibility), (FileExistsError,))
            full = pd.concat([eligibility, pd.DataFrame({"experiment_id": ["p3"], "target_context": ["CTX"]})], ignore_index=True)
            second = public.append([public_item("p3", 0)], [[4., 5., 6.]], [[.4, .5, .6]], {"synthetic": True}, full)
            assert second["version"] == 2 and json.loads(public.current_path.read_text())["version"] == 2
            frame, e, c, manifest = public.load()
            assert len(frame) == 3 and e.shape == c.shape == (3, 3)
            assert all(core._sha(v1 / name) == digest for name, digest in old.items())
            checks.extend(["public_actual_append_current_v2", "public_previous_snapshot_bytes_preserved", "public_load_complete_axes"])
            reject("public_drop_old_eligibility_rejected", lambda: public.append([public_item("p4", 0)], [[7., 8., 9.]], [[.7, .8, .9]], {}, full.iloc[2:]))
            reject("public_nonfinite_control_rejected", lambda: public.append([public_item("p4", 0)], [[7., 8., 9.]], [[float("nan"), .8, .9]], {}, full))
            reject("public_eligibility_unsupported_ID_schema_rejected", lambda: public.append([public_item("p4", 0)], [[7., 8., 9.]], [[.7, .8, .9]], {}, pd.DataFrame({"foo": [1]})))
            reject("public_eligibility_conflicting_ID_columns_rejected", lambda: public.append([public_item("p4", 0)], [[7., 8., 9.]], [[.7, .8, .9]], {}, full.assign(public_experiment_id=["p2", "p3", "p1"])))
            reject("public_wrong_gene_axis_rejected", lambda: VersionedPublicRuntime(public.root, list(reversed(genes))).load())
            for artifact in ["manifest.json", "effect_vectors.npy", "control_vectors.npy", "gene_ids.json", "eligibility.parquet", "public_memory.parquet"]:
                tamper(public.root / "versions/v0002" / artifact, public.load, "public_tamper_" + artifact)
            tamper(public.current_path, public.load, "public_CURRENT_manifest_SHA_tamper", lambda value: value.update(manifest_sha256="0" * 64))
            public.rollback_to(1, "synthetic rollback verification")
            assert len(public.load()[0]) == 2
            public.activate(2, "synthetic reactivation verification")
            assert len(public.load()[0]) == 3
            checks.extend(["public_real_rollback_pointer_v1", "public_real_reactivation_pointer_v2"])

            error = VersionedErrorRuntime(tmp / "error", "synthetic_rmse_axis3_v1")
            revision1 = error.append([error_item("e1")], "initial")
            r1 = Path(revision1["runtime_revision_path"])
            old_error = {str(p.relative_to(r1)): core._sha(p) for p in r1.rglob("*") if p.is_file()}
            revision2 = error.append([error_item("e2")], "update")
            assert revision2["runtime_revision"] == 2 and len(error.load("UPSTREAM", "model-v1")) == 2
            assert all(core._sha(r1 / name) == digest for name, digest in old_error.items())
            assert error.manifest("UPSTREAM", "model-v1")["runtime_revision"] == 2
            checks.extend(["error_actual_append_revision2", "error_previous_revision_bytes_preserved", "error_manifest_revision2"])
            reject("error_wrong_model_version_lookup_rejected", lambda: error.load("UPSTREAM", "model-v2"))
            reject("error_duplicate_batch_rejected", lambda: error.append([error_item("e3")], "update"))
            reject("error_duplicate_prediction_rejected", lambda: error.append([error_item("e1")], "newbatch"))
            reject("error_wrong_output_contract_rejected", lambda: VersionedErrorRuntime(error.root, "wrong_contract"))
            tamper(error.root / "OUTPUT_CONTRACT.json", lambda: error.load("UPSTREAM", "model-v1"), "error_contract_sidecar_tamper")
            live = error.registry._key_dir("UPSTREAM", "model-v1")
            for artifact in ["manifest.json", "error_memory.parquet"]:
                tamper(live / artifact, lambda: error.load("UPSTREAM", "model-v1"), "error_live_" + artifact + "_tamper")
            bank = core.ErrorMemoryRegistry(Path(revision2["runtime_revision_path"]))._key_dir("UPSTREAM", "model-v1")
            tamper(bank / "error_memory.parquet", lambda: error.load("UPSTREAM", "model-v1"), "error_revision_table_tamper")

            models = ServingModelRegistry(tmp / "models")
            prior = tmp / "old_model.bin"
            failed = tmp / "failed_model.bin"
            prior.write_bytes(b"actual synthetic fitted payload v1")
            failed.write_bytes(b"failed synthetic candidate payload")
            models.register("risk", "v1", prior, "split1", "features1")
            models.publish("risk", "v1", "synthetic initial publication")
            assert models.current("risk").read_bytes() == prior.read_bytes()
            current_bytes = models.current_path.read_bytes()
            event_bytes = (models.root / "runtime_events.jsonl").read_bytes()
            for label, value in [("wrong_schema", {"schema": "wrong", "components": {}}),
                                 ("missing_schema", {"components": {}}),
                                 ("wrong_components_type", {"schema": "SafeConf-ServingRuntime-CURRENT-v1", "components": []})]:
                models.current_path.write_text(json.dumps(value))
                malformed = models.current_path.read_bytes()
                reject("model_CURRENT_" + label + "_rejects_before_publish", lambda: models.publish("risk", "v1", "malformed CURRENT fixture"))
                assert models.current_path.read_bytes() == malformed
                assert (models.root / "runtime_events.jsonl").read_bytes() == event_bytes
                models.current_path.write_bytes(current_bytes)
            before = models.current_path.read_bytes()
            models.register("risk", "failed-v2", failed, "split1", "features1", "v1", status="REJECTED")
            reject("model_failed_candidate_not_publishable", lambda: models.publish("risk", "failed-v2", "attempt rejected fixture"))
            assert models.current_path.read_bytes() == before
            models.register("risk", "marker-v2", prior, "split1", "features1", "v1", status="RELEASED")
            models.publish("risk", "marker-v2", "administrative same-artifact marker drill")
            models.rollback("risk", "v1", "restore actual prior artifact pointer")
            assert models.current("risk").read_bytes() == prior.read_bytes()
            assert models.registry.entries()[-1]["status"] == "ROLLED_BACK"
            checks.extend(["model_actual_register_and_publish", "model_rejected_candidate_preserves_pointer", "model_actual_core_rollback_ledger", "model_rollback_restores_exact_payload"])
            reject("model_version_no_overwrite", lambda: models.register("risk", "v1", prior, "split1", "features1"))
            tamper(models.current("risk"), lambda: models.current("risk"), "model_artifact_tamper")
            tamper(models._binding("risk", "v1") / "binding.json", lambda: models.current("risk"), "model_binding_tamper")
            tamper(models.registry.path, lambda: models.current("risk"), "model_actual_registry_ledger_tamper")
            tamper(models.current_path, lambda: models.current("risk"), "model_CURRENT_artifact_SHA_tamper",
                   lambda value: value["components"]["risk"].update(artifact_sha256="0" * 64))
            reject("model_reason_required", lambda: models.publish("risk", "v1", " "))
            assert [json.loads((public.root / "versions" / f"v{i:04d}" / "CORE_CALL.json").read_text())["method"] for i in (1, 2)] == ["PublicMemoryStore.create", "PublicMemoryStore.append"]

    result = {"status": "PASS", "synthetic_only": True, "no_source_or_Orion_data_read": True,
        "wrapper_sha256": core._sha(ROOT / "tools/safeconf_continual/versioned_runtime.py"),
        "original_memory_sha256": core._sha(Path(core.__file__)),
        "actual_existing_core_method_calls": calls, "n_successful_checks": len(checks), "checks": checks,
        "old_public_snapshot_bytes_preserved": True, "old_error_revision_bytes_preserved": True,
        "rejected_scientific_candidate_never_published": True,
        "administrative_marker_disclosure": "second RELEASED marker references identical prior artifact; no new fit or quality improvement",
        "rollback_restored_artifact_payload": True}
    (Path(__file__).parent / "SYNTHETIC_WRAPPER_PROOF.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
