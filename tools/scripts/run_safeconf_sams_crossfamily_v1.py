#!/usr/bin/env python3
"""One registered SAMS-VAE upstream; truth-free generation and isolated risk followup.

This entrypoint preserves PerturBench c84038bc's architecture, ELBO, optimizer,
and McFaline training recipe. It never calls trainer.test or the official
CounterfactualWithReference loader. Prediction needs only task metadata, the
frozen transform and a zero-valued shape placeholder, never treated truth.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

REPO = Path(__file__).resolve().parents[2]
PB = Path("/home/yyf/archive/external/PerturBench")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PB / "src"))
ROOT = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/sams_v1")
DOCS = REPO / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/sams"
H5AD = Path("/home/yyf/data/perturbench_mcfaline23_official/mcfaline23_gxe_processed.h5ad")
SPLIT = Path("/home/yyf/data/perturbench_mcfaline23_official/splits/mcfaline23_gxe_splits/full_covariate_split.csv")
EXPERIMENT = PB / "src/perturbench/configs/experiment/neurips2025/mcfaline23/sams_best_params_mcfaline23_full.yaml"
COMMON = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis")
SEED = 20261003


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def receipt(name: str, value):
    if name in {"EXPERIMENT_CONTRACT.json", "POSTTRAIN_PROTOCOL.json"} and (ROOT/name).exists():
        old=json.loads((ROOT/name).read_text()); old.pop("time_utc",None)
        if old != json.loads(json.dumps(value,default=str)):
            raise ValueError(f"immutable frozen receipt differs: {name}; use a separately versioned amendment")
        print(json.dumps({"receipt":name,"status":"REUSED_IDENTICAL_FROZEN_CONTRACT"}),flush=True)
        return
    value = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **value}
    dump(ROOT / name, value)
    dump(DOCS / name, value)
    if name in {"TRAINING_START.json","RNG_RESUME_RECEIPT.json","TRAINING_COMPLETION.json","TRAINING_STATUS.json"}:
        dump(ROOT/"attempts"/str(os.getpid())/name,value)
        dump(DOCS/"attempts"/str(os.getpid())/name,value)
    print(json.dumps({"receipt": name, **value}, ensure_ascii=False, default=str), flush=True)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def array_hash(x):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def read_metadata():
    """Read obs/var and split only. No expression object is obtained."""
    import h5py
    import numpy as np
    import pandas as pd
    from anndata.io import read_elem
    with h5py.File(H5AD) as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
    split = pd.read_csv(SPLIT, header=None, names=["cell_id", "role"], dtype=str).set_index("cell_id")
    if not obs.index.is_unique or not split.index.is_unique:
        raise ValueError("nonunique cell identifiers")
    roles = split.role.reindex(obs.index)
    if roles.isna().any() or not set(roles) <= {"train", "val", "test"}:
        raise ValueError("official split does not align")
    obs = obs.copy()
    obs["official_role"] = roles.to_numpy()
    return obs, var


def recipe():
    from omegaconf import OmegaConf
    base = OmegaConf.load(PB / "src/perturbench/configs/model/sams_vae.yaml")
    exp = OmegaConf.load(EXPERIMENT)
    model = OmegaConf.merge(base, exp.model)
    out = OmegaConf.to_container(model, resolve=True)
    out.pop("_target_")
    out.pop("n_genes")
    out.pop("n_perts")
    if not out["generative_counterfactual"]:
        raise ValueError("generative path required")
    return out


def seed_everything(seed=SEED):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_verified_engineering_optimization():
    result_path = ROOT / "INDEXING_SPEEDUP_RESULT.json"
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text())
    if result["status"] != "ADOPT_EQUIVALENT_OPTIMIZATION":
        return
    helper = Path(__file__).with_name("benchmark_safeconf_sams_indexing_v1.py")
    if sha(helper) != result["script_sha256"]:
        raise ValueError("approved engineering helper changed after equivalence benchmark")
    official = PB / "src/perturbench/modelcore/models/sams_vae.py"
    contract = json.loads((ROOT / "EXPERIMENT_CONTRACT.json").read_text())
    if sha(official) != contract["input_hashes"][str(official)]:
        raise ValueError("official forward source changed after scientific freeze")
    from tools.scripts.benchmark_safeconf_sams_indexing_v1 import install_vectorized_forward
    install_vectorized_forward()
    receipt("INDEXING_APPLIED.json", {"status":"VERIFIED_EQUIVALENT_OPTIMIZATION_INSTALLED",
        "helper_sha256":sha(helper),"official_source_unchanged":True,
        "loss_architecture_and_hyperparameters_unchanged":True,
        "approval_receipt_sha256":sha(result_path)})


class ReplayableRandomSampler:
    """Uniform per-epoch permutation with explicit completed-row state."""
    def __init__(self,n,seed=SEED):
        self.n=n;self.seed=seed;self.epoch=None;self.permutation=None;self.completed=0
    def __len__(self):
        return self.n
    def set_epoch(self,epoch):
        import torch
        if self.epoch != epoch:
            generator=torch.Generator().manual_seed(self.seed+int(epoch))
            self.permutation=torch.randperm(self.n,generator=generator)
            self.completed=0;self.epoch=int(epoch)
    def __iter__(self):
        if self.permutation is None:self.set_epoch(0)
        yield from self.permutation[self.completed:].tolist()
    def state_dict(self):
        return {"n":self.n,"seed":self.seed,"epoch":self.epoch,"permutation":self.permutation,"completed":self.completed}
    def load_state_dict(self,state):
        if state["n"]!=self.n or state["seed"]!=self.seed:raise ValueError("sampler identity changed")
        self.epoch=state["epoch"];self.permutation=state["permutation"];self.completed=state["completed"]


def generation_batch(model, perturbation: str, cell_type: str, treatment: str, n: int):
    """Only metadata is accepted; no truth/data accessor argument exists."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from perturbench.data.types import Batch
    context = model.training_record["train_context"]
    labels = [] if perturbation == "control" else perturbation.split("+")
    known = set(context["perturbation_uniques"])
    if not set(labels) <= known:
        raise ValueError(f"untrained perturbation: {labels}")
    batch = Batch(
        gene_expression=csr_matrix((n, model.n_genes), dtype=np.float32),
        perturbations=[labels] * n,
        covariates={"cell_type": [cell_type] * n, "treatment": [treatment] * n},
    )
    batch = model.training_record["transform"](batch)
    return batch._replace(
        gene_expression=batch.gene_expression.to(model.device),
        perturbations=batch.perturbations.to(model.device),
        covariates={k: v.to(model.device) for k, v in batch.covariates.items()},
    )


def predict_metadata(model, task, n=1000, seed=SEED, chunk=250):
    import numpy as np
    import torch
    seed_everything(seed)
    model.eval()
    arrays = []
    with torch.inference_mode():
        for start in range(0, n, chunk):
            batch = generation_batch(model, str(task["perturbation"]), str(task["context"]),
                                     str(task["treatment"]), min(chunk, n - start))
            output = model.predict(batch)
            if not isinstance(output, torch.Tensor):
                output = output.mean
            arrays.append(output.float().cpu().numpy())
    x = np.concatenate(arrays, axis=0)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite generated predictions")
    return x.mean(axis=0, dtype=np.float64).astype("float32"), x.var(axis=0).astype("float32")


def preflight():
    import inspect
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import OmegaConf
    from perturbench.data.transforms.pipelines import LinearModelPipeline
    from perturbench.modelcore.models.sams_vae import SparseAdditiveVAE
    torch.set_num_threads(4)
    seed_everything()
    context = {
        "perturbation_uniques": ["GENE_A", "GENE_B", "GENE_C"],
        "covariate_uniques": {"cell_type": ["CELL_A", "CELL_B"], "treatment": ["VEHICLE", "DRUG"]},
        "perturbation_counts": pd.Series([100, 110, 120], index=["GENE_A", "GENE_B", "GENE_C"]),
    }
    transform = LinearModelPipeline(context["perturbation_uniques"], context["covariate_uniques"])
    model = SparseAdditiveVAE(n_genes=12, n_perts=3, transform=transform, context=context,
                             evaluation=OmegaConf.create({}), hidden_dim_x=16, latent_dim=4,
                             n_layers_encoder_x=2, n_layers_encoder_e=2, n_layers_decoder=2,
                             dropout=.1, generative_counterfactual=True,
                             inject_covariates_encoder=True, inject_covariates_decoder=True)
    model.eval()
    b = generation_batch(model, "GENE_A", "CELL_A", "VEHICLE", 8)
    modified = b._replace(gene_expression=torch.linspace(-50, 50, 96).reshape(8, 12))
    with torch.inference_mode():
        seed_everything(); p0 = model.predict(b).numpy()
        seed_everything(); p1 = model.predict(modified).numpy()
        model.generative_counterfactual = False
        seed_everything(); neg0 = model.predict(b).numpy()
        seed_everything(); neg1 = model.predict(modified).numpy()
        model.generative_counterfactual = True
    np.testing.assert_allclose(p0, p1, rtol=1e-5, atol=1e-7)
    if np.allclose(neg0, neg1, rtol=1e-5, atol=1e-7):
        raise AssertionError("negative reconstruction-path sensitivity control failed")
    task = {"perturbation": "GENE_A", "context": "CELL_A", "treatment": "VEHICLE"}
    pred, _ = predict_metadata(model, task, n=8, chunk=8)
    np.testing.assert_allclose(pred, p0.mean(axis=0), rtol=1e-5, atol=1e-7)
    obs, var = read_metadata()
    counts = obs.groupby(["official_role"], observed=True).size().to_dict()
    train = obs.official_role.eq("train")
    val = obs.official_role.eq("val")
    train_perts = set(obs.loc[train, "condition"].astype(str))
    unknown_val = sorted(set(obs.loc[val, "condition"].astype(str)) - train_perts)
    hashes = {str(p): sha(p) for p in [SPLIT, EXPERIMENT,
        PB / "src/perturbench/modelcore/models/sams_vae.py",
        PB / "src/perturbench/data/datasets/inmemory/sc_perturbation.py"]}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PB, text=True).strip()
    receipt("PREDICTION_INPUT_TESTS.json", {
        "status": "PASS", "architecture": "official SAMS-VAE synthetic miniature; not a second upstream",
        "generative_changed_expression_max_abs": float(np.abs(p0-p1).max()),
        "reconstruction_negative_control_max_abs": float(np.abs(neg0-neg1).max()),
        "truth_removed_generation_finite": bool(np.isfinite(pred).all()),
        "metadata_only_predictor_parameters": list(inspect.signature(predict_metadata).parameters),
        "forward_notes": "official forward computes discarded reconstruction/encoder terms; output samples prior and is expression-invariant",
        "official_repo_commit": commit, "input_hashes": hashes,
        "source_expression_accessed": False, "cell_counts_by_role": counts,
        "n_genes": len(var), "unknown_validation_conditions": unknown_val,
        "rtol": 1e-5, "atol": 1e-7,
    })
    gene_names = var.index.astype(str).tolist()
    dump(ROOT / "NATIVE_GENE_AXIS.json", gene_names)
    receipt("EXPERIMENT_CONTRACT.json", {
        "status": "FROZEN_BEFORE_TRAINING", "official_repo_commit": commit,
        "seed": SEED, "gpu_physical_id": 0, "model": recipe(),
        "batch_size": 256, "precision": "32-true", "max_epochs": 400, "min_epochs": 5,
        "early_stopping_patience": 50, "checkpoint_metric": "val_loss reconstruction ELBO",
        "competence_metric": "truth-free generative validation task error vs strong simple baseline",
        "generated_samples_per_task": 1000, "initial_gpu_hours": 24, "candidate_max_gpu_hours": 48,
        "permanent_test_seen_status": "SEEN retrospective; not new confirmation",
        "input_hashes": hashes, "axis_sha256": sha(ROOT / "NATIVE_GENE_AXIS.json"),
        "test_expressions_prohibited_during_training": True,
        "scientific_change": "new upstream family only; official architecture and objective fixed",
    })
    return model


def training_data():
    import h5py
    import numpy as np
    from types import SimpleNamespace
    from anndata.io import sparse_dataset
    from omegaconf import OmegaConf
    from perturbench.data.datasets import SingleCellPerturbation
    from perturbench.data.modules import AnnDataLitModule
    from perturbench.data.transforms.pipelines import LinearModelPipeline
    from perturbench.data.utils import get_covariates, parse_perturbation_combinations
    from scipy.sparse import csr_matrix
    obs, var = read_metadata()
    data_iter = functools.partial(SingleCellPerturbation.from_anndata,
        perturbation_key="condition", perturbation_combination_delimiter="+",
        perturbation_control_value="control", covariate_keys=["cell_type", "treatment"],
        embedding_key=None, feature_filter_path=None)
    dm = AnnDataLitModule(data=OmegaConf.create({"filename": str(H5AD)}), data_iter_factory=data_iter,
        mode="batch", loader=OmegaConf.create({"batch_size": 256, "num_workers": 0}),
        transform=None, splitter=OmegaConf.create({"save": False}),
        evaluation=OmegaConf.create({"save_dir": str(ROOT / "evaluation"), "split_value_to_evaluate": "val"}))
    dm.splits = {k: np.flatnonzero(obs.official_role.eq(k)).tolist() for k in ("train", "val")}
    allowed = set(dm.splits["train"]) | set(dm.splits["val"])
    test = set(np.flatnonzero(obs.official_role.eq("test")))
    if allowed & test:
        raise AssertionError("train/val expression read overlaps test")
    access = []
    handle = h5py.File(H5AD, "r")
    class GuardedX:
        """Do not use read_h5ad(backed): it eagerly loads the counts layer."""
        def __getitem__(self, selected):
            idx = np.arange(len(obs))[selected]
            if not set(np.asarray(idx).reshape(-1)) <= allowed:
                raise PermissionError("expression row outside train/validation requested")
            return sparse_dataset(handle["X"])[selected]
    dm.data_handle = SimpleNamespace(
        X=GuardedX(), obs=obs, obs_names=obs.index, var_names=var.index,
        shape=(len(obs), len(var)), obsm={},
    )
    dm.obs_df = obs
    dm.transform = None
    def guarded_factory(adata, split):
        idx = np.asarray(split, dtype=int)
        if not set(idx) <= allowed or set(idx) & test:
            raise PermissionError("test expression read forbidden")
        roles = sorted(set(obs.official_role.iloc[idx]))
        if len(roles) != 1:
            raise AssertionError("role-mixed expression request")
        access.append({"role": roles[0], "n_rows": len(idx), "row_index_sha256": array_hash(idx)})
        print(f"[SAMS] reading allowed {roles[0]} expression rows={len(idx)}", flush=True)
        role = roles[0]
        cache = ROOT / "role_expression_cache" / role
        cache.mkdir(parents=True, exist_ok=True)
        manifest_path = cache / "MANIFEST.json"
        expected = {"row_index_sha256": array_hash(idx), "n_rows": len(idx), "n_genes": len(var),
                    "split_sha256": sha(SPLIT), "input_expression": "X log-normalized; no counts layer"}
        if manifest_path.exists():
            cached = json.loads(manifest_path.read_text())
            if any(cached.get(k) != v for k, v in expected.items()):
                raise ValueError("role cache provenance changed")
            nnz = cached["nnz"]
        else:
            # Sparse fancy indexing of all 7e5 rows caused a preparation OOM.
            # Materialize only eligible rows in 2048-row blocks to disk-backed
            # float32 CSR, preserving the official input values and row order.
            ptr = np.asarray(handle["X"]["indptr"], dtype=np.int64)
            local_ptr = np.r_[0, np.cumsum(ptr[idx+1]-ptr[idx], dtype=np.int64)]
            nnz = int(local_ptr[-1])
            vals = np.memmap(cache / "data.f32", mode="w+", dtype="float32", shape=(nnz,))
            cols = np.memmap(cache / "indices.i32", mode="w+", dtype="int32", shape=(nnz,))
            out_ptr = np.memmap(cache / "indptr.i64", mode="w+", dtype="int64", shape=(len(idx)+1,))
            out_ptr[:] = local_ptr
            for start in range(0, len(idx), 2048):
                end = min(start+2048, len(idx))
                block = adata.X[idx[start:end]]
                lo, hi = int(local_ptr[start]), int(local_ptr[end])
                if hi-lo != block.nnz:
                    raise AssertionError("guarded CSR nnz mismatch")
                vals[lo:hi] = block.data
                cols[lo:hi] = block.indices
                if start % (2048*20) == 0:
                    vals.flush(); cols.flush()
                    print(f"[SAMS streaming cache] {role} {start}:{end}/{len(idx)}", flush=True)
            vals.flush(); cols.flush(); out_ptr.flush()
            dump(manifest_path, {**expected, "nnz": nnz})
            del vals, cols, out_ptr
        vals = np.memmap(cache / "data.f32", mode="r", dtype="float32", shape=(nnz,))
        cols = np.memmap(cache / "indices.i32", mode="r", dtype="int32", shape=(nnz,))
        ptr = np.memmap(cache / "indptr.i64", mode="r", dtype="int64", shape=(len(idx)+1,))
        matrix = csr_matrix((vals, cols, ptr), shape=(len(idx),len(var)), copy=False)
        part = obs.iloc[idx]
        perts, unique_perts, counts = parse_perturbation_combinations(part.condition, "+", "control")
        covs, unique_covs = get_covariates(part, ["cell_type", "treatment"])
        context = {"perturbation_uniques": unique_perts, "covariate_uniques": unique_covs,
            "perturbation_key": "condition", "covariate_keys": ["cell_type", "treatment"],
            "perturbation_combination_delimiter": "+", "perturbation_control_value": "control",
            "perturbation_counts": counts, "features_keep": np.arange(len(var))}
        ds = SingleCellPerturbation(matrix, perts, covs, cell_ids=part.index.tolist(),
                                   gene_names=var.index.astype(str).tolist(), perturbation_control_value="control")
        return ds, context
    dm.data_iter_factory = guarded_factory
    dm.setup("fit")
    dm.transform = LinearModelPipeline(dm.context["perturbation_uniques"], dm.context["covariate_uniques"])
    dm.train_iterator.transform = dm.transform
    dm.val_iterator.transform = dm.transform
    receipt("TRAINING_ACCESS_AUDIT.json", {
        "status": "PASS", "expression_reads": access, "test_expression_rows_read": 0,
        "all_metadata_available_for_split_alignment": True, "train_rows": len(dm.train_iterator),
        "validation_rows": len(dm.val_iterator), "native_genes": dm.num_genes,
        "n_train_perts": dm.num_perturbations, "num_workers": 0,
        "data_memory_bytes": int(dm.train_iterator.gene_expression.data.nbytes + dm.train_iterator.gene_expression.indices.nbytes),
        "reader": "HDF5 X-only guarded rows; layers/counts and TEST expression never opened",
        "rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    })
    handle.close()
    return dm


def train(args):
    from datetime import timedelta
    import lightning as L
    import numpy as np
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from perturbench.modelcore.models.sams_vae import SparseAdditiveVAE
    ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    tests = json.loads((ROOT / "PREDICTION_INPUT_TESTS.json").read_text())
    if tests["status"] != "PASS":
        raise AssertionError("preflight missing")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") == "0":
        raise ValueError("physical GPU0 only; require CUDA_VISIBLE_DEVICES=0")
    torch.set_num_threads(4)
    L.seed_everything(SEED, workers=True)
    apply_verified_engineering_optimization()
    started = time.time()
    dm = training_data()
    replay_sampler=ReplayableRandomSampler(len(dm.train_iterator))
    original_train_loader=dm.train_dataloader
    def replayable_train_loader():
        loader=original_train_loader()
        loader.batch_sampler.sampler=replay_sampler
        # Iterator base-seed generation is separated from the model RNG.
        loader.generator=torch.Generator().manual_seed(SEED+7919)
        return loader
    dm.train_dataloader=replayable_train_loader
    model = SparseAdditiveVAE(n_genes=dm.num_genes, n_perts=dm.num_perturbations,
        transform=dm.transform, context=dm.context, evaluation=dm.evaluation, **recipe())
    best = ModelCheckpoint(dirpath=ROOT / "checkpoints", monitor="val_loss", mode="min",
                           save_top_k=1, save_last=True, filename="epoch{epoch:03d}-step{step}")
    early = EarlyStopping(monitor="val_loss", mode="min", patience=50, check_finite=True)
    class Monitor(L.Callback):
        def __init__(self):
            self.start = None
            self.last = 0
            self.first_epoch_seconds = None
            self.last_recovery = 0
            self.start_step = 0
            self.pending_rng = None
        def on_fit_start(self, trainer, pl_module):
            self.start = time.time()
            self.start_step = trainer.global_step
            self.emit(trainer, "TRAINING_RUNNING")
        def on_load_checkpoint(self, trainer, pl_module, checkpoint):
            self.pending_rng = checkpoint.get("safeconf_rng_state")
            if "safeconf_data_order" in checkpoint:
                replay_sampler.load_state_dict(checkpoint["safeconf_data_order"])
        def on_train_start(self, trainer, pl_module):
            import random
            self.start_step = trainer.global_step
            if self.pending_rng is not None:
                random.setstate(self.pending_rng["python"])
                np.random.set_state(self.pending_rng["numpy"])
                torch.set_rng_state(self.pending_rng["torch_cpu"])
                torch.cuda.set_rng_state_all(self.pending_rng["torch_cuda"])
                restoration = "EXPLICIT_SAVED_RNG_RESTORED"
            elif args.resume:
                seed_everything(SEED)
                restoration = "OFFICIAL_PARENT_HAS_NO_GLOBAL_RNG; FIXED_REGISTERED_CONTINUATION_SEED"
            else:
                restoration = "FRESH_REGISTERED_SEED"
            receipt("RNG_RESUME_RECEIPT.json", {"status":restoration,
                "global_step":trainer.global_step,"epoch":trainer.current_epoch,
                "continuation_seed":SEED,"parent_checkpoint":str(args.resume) if args.resume else None,
                "bitwise_original_rng_trajectory_claimed":self.pending_rng is not None})
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            import random
            checkpoint["safeconf_rng_state"]={"python":random.getstate(),"numpy":np.random.get_state(),
                "torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all()}
            checkpoint["safeconf_data_order"]=replay_sampler.state_dict()
        def on_train_epoch_start(self, trainer, pl_module):
            replay_sampler.set_epoch(trainer.current_epoch)
        def emit(self, trainer, status):
            elapsed = time.time() - self.start if self.start else 0
            metrics = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                       if getattr(v, "numel", lambda: 0)() == 1}
            value = {"status": status, "pid": os.getpid(), "epoch": trainer.current_epoch,
                "global_step": trainer.global_step, "elapsed_fit_seconds": elapsed,
                "gpu_hours": elapsed / 3600, "initial_budget_gpu_hours": args.gpu_hours,
                "rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "metrics": metrics, "best_checkpoint": best.best_model_path,
                "last_checkpoint": best.last_model_path,
                "validation_loss_is_reconstruction": True,
                "upstream_competence_status": "PENDING_TRUE_GENERATION",
                "segment_start_step": self.start_step,
                "segment_train_steps": trainer.global_step-self.start_step,
                "seconds_per_train_step_so_far": elapsed / max(1, trainer.global_step-self.start_step),
                "first_epoch_seconds": self.first_epoch_seconds}
            receipt("TRAINING_STATUS.json", value)
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            replay_sampler.completed=min(replay_sampler.n,(int(batch_idx)+1)*256)
            if trainer.global_step-self.start_step in (1, 10, 200) or time.time() - self.last > 180:
                self.last = time.time()
                self.emit(trainer, "TRAINING_RUNNING")
            if trainer.global_step-self.start_step == 200 or (self.last_recovery and time.time() - self.last_recovery > 3600):
                trainer.save_checkpoint(ROOT / "checkpoints/recovery.ckpt")
                self.last_recovery = time.time()
            if time.time() - self.start > args.gpu_hours * 3600:
                trainer.should_stop = True
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 > 70:
                trainer.should_stop = True
        def on_validation_end(self, trainer, pl_module):
            if not trainer.sanity_checking:
                if self.first_epoch_seconds is None:
                    self.first_epoch_seconds = time.time() - self.start
                self.emit(trainer, "TRAINING_VALIDATED_EPOCH")
        def on_exception(self, trainer, pl_module, exception):
            self.emit(trainer, "TRAINING_EXCEPTION")
            if trainer.global_step > 0:
                try:
                    trainer.save_checkpoint(ROOT / "checkpoints/exception_resumable.ckpt")
                except Exception:
                    pass
    monitor = Monitor()
    class SafeCSVLogger(CSVLogger):
        def log_hyperparams(self, params):
            # Official transform is a dict subclass with a required-argument
            # constructor; Lightning's YAML traversal cannot recreate it.
            # Exact recipe/context are preserved in contract + checkpoint.
            return
    trainer = L.Trainer(default_root_dir=ROOT, accelerator="gpu", devices=1,
        precision="32-true", max_epochs=400, min_epochs=5, deterministic=False,
        callbacks=[best, early, monitor], logger=SafeCSVLogger(ROOT, name="lightning"),
        enable_progress_bar=False, enable_model_summary=False,
        log_every_n_steps=10, num_sanity_val_steps=2, check_val_every_n_epoch=1,
        max_time=timedelta(hours=args.gpu_hours),
    )
    resume = str(args.resume) if args.resume else None
    receipt("TRAINING_START.json", {
        "status": "STARTED", "pid": os.getpid(), "physical_gpu": 0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "native_genes": dm.num_genes, "train_steps_per_epoch": len(dm.train_dataloader()),
        "validation_steps": len(dm.val_dataloader()), "resume": resume,
        "preparation_seconds": time.time() - started, "read_only_inputs_unchanged": True,
    })
    trainer.fit(model, datamodule=dm, ckpt_path=resume, weights_only=False)
    final_path = ROOT / "checkpoints/final_resumable.ckpt"
    trainer.save_checkpoint(final_path)
    receipt("TRAINING_COMPLETION.json", {
        "status": "TRAINING_STOPPED", "epoch": trainer.current_epoch, "global_step": trainer.global_step,
        "best_checkpoint": best.best_model_path, "final_checkpoint": str(final_path),
        "early_stop": bool(early.stopped_epoch), "budget_stop": not bool(early.stopped_epoch) and trainer.current_epoch < 400,
        "elapsed_total_seconds": time.time() - started,
        "best_val_reconstruction_loss": float(best.best_model_score),
        "competence_status": "PENDING_TRUE_GENERATION",
    })
    args.checkpoint = Path(best.best_model_path)
    followup(args)


def evaluate(args):
    """Generate first; score only after prediction arrays and hash are persisted."""
    import numpy as np
    import pandas as pd
    import torch
    from perturbench.modelcore.models.sams_vae import SparseAdditiveVAE
    torch.set_num_threads(4)
    apply_verified_engineering_optimization()
    if not args.checkpoint:
        payload = json.loads((ROOT / "TRAINING_COMPLETION.json").read_text())
        args.checkpoint = Path(payload["best_checkpoint"])
    model = SparseAdditiveVAE.load_from_checkpoint(args.checkpoint, weights_only=False, map_location="cpu")
    model = model.to("cuda").eval()
    axis = json.loads((ROOT / "NATIVE_GENE_AXIS.json").read_text())
    tasks = pd.read_csv(COMMON / "VALIDATION_COMPETENCE_TASKS.csv",
                        usecols=["task_id", "perturbation", "context", "treatment", "stratum"])
    if "perturbation" not in tasks:
        tasks["perturbation"] = tasks.get("condition", tasks.get("target_gene_id"))
    predictions, variances = [], []
    started = time.time()
    for i, row in tasks.iterrows():
        pred, var = predict_metadata(model, row, n=1000, seed=SEED + i)
        predictions.append(pred); variances.append(var)
        if i % 50 == 0:
            print(f"[SAMS true-generation] validation {i}/{len(tasks)} elapsed={time.time()-started:.1f}", flush=True)
    matrix = np.stack(predictions)
    np.save(ROOT / "VALIDATION_NATIVE_PREDICTED_STATES.npy", matrix)
    np.save(ROOT / "VALIDATION_NATIVE_SAMPLING_VARIANCE.npy", np.stack(variances))
    tasks.to_csv(ROOT / "VALIDATION_GENERATED_TASKS.csv", index=False)
    receipt("VALIDATION_PREDICTION_SEAL.json", {
        "status": "PREDICTIONS_SAVED_BEFORE_SCORING", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint), "rows": len(tasks), "native_genes": len(axis),
        "generated_samples_per_task": 1000, "prediction_sha256": sha(ROOT / "VALIDATION_NATIVE_PREDICTED_STATES.npy"),
        "expression_input": "zeros; metadata only", "test_expression_accessed": False,
        "elapsed_generation_seconds": time.time()-started,
    })
    score_validation(tasks, matrix, axis)


def score_validation(tasks, native_predictions, native_axis):
    import numpy as np
    import pandas as pd
    genes = json.loads((COMMON / "GENE_IDS.json").read_text())
    if isinstance(genes, dict):
        genes = genes.get("gene_ids", genes.get("genes"))
    idx = pd.Series(np.arange(len(native_axis)), index=native_axis).loc[genes].to_numpy(int)
    controls = np.load(COMMON / "VALIDATION_CONTROLS.npy")
    truth = np.load(COMMON / "VALIDATION_TRUE_EFFECTS.npy")
    baseline = np.load(COMMON / "VALIDATION_TRAIN_STATE_MEAN_EFFECTS.npy")
    if len(tasks) != len(truth):
        raise ValueError("competence task order requires exact common-axis provenance alignment")
    effect = native_predictions[:, idx] - controls
    model_rmse = np.sqrt(np.mean((effect-truth)**2, axis=1))
    mean_rmse = np.sqrt(np.mean((baseline-truth)**2, axis=1))
    nochange_rmse = np.sqrt(np.mean(truth**2, axis=1))
    result = tasks.copy()
    result["model_rmse"] = model_rmse
    result["state_mean_rmse"] = mean_rmse
    result["nochange_rmse"] = nochange_rmse
    if "stratum" not in result:
        result["stratum"] = result.context.astype(str) + "::" + result.treatment.astype(str)
    strata = result.groupby("stratum")[["model_rmse", "state_mean_rmse", "nochange_rmse"]].mean()
    strongest = "state_mean_rmse" if mean_rmse.mean() <= nochange_rmse.mean() else "nochange_rmse"
    strata["strong_baseline_rmse"] = strata[strongest]
    strata["relative_gap"] = strata.model_rmse / strata.strong_baseline_rmse - 1
    result.to_csv(ROOT / "VALIDATION_TASK_ERRORS.csv", index=False)
    strata.to_csv(DOCS / "VALIDATION_COMPETENCE_STRATA.csv")
    np.save(ROOT / "VALIDATION_PREDICTED_EFFECTS.npy", effect.astype("float32"))
    # Bootstrap whole biological genes jointly across strata, never cells.
    gene = result.perturbation.astype(str).to_numpy()
    unique, inverse = np.unique(gene, return_inverse=True)
    rng = np.random.default_rng(SEED)
    gaps = []
    for _ in range(5000):
        weight = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[inverse]
        b = mean_rmse if strongest == "state_mean_rmse" else nochange_rmse
        m, b = [np.average(x, weights=weight) for x in (model_rmse, b)]
        gaps.append(m/b-1)
    baseline_error = mean_rmse if strongest == "state_mean_rmse" else nochange_rmse
    gap = model_rmse.mean()/baseline_error.mean()-1
    ci = np.quantile(gaps, [.025,.975]).tolist()
    point_pass = gap <= .02 and float((strata.relative_gap <= .02).mean()) >= .6 and ci[0] <= .02
    receipt("VALIDATION_COMPETENCE.json", {
        "status": "OPERATIONAL_GATE_PASS" if point_pass else "OPERATIONAL_GATE_FAIL",
        "primary_gene_contract": "same existing McFaline common gene axis", "n_tasks": len(result),
        "n_clusters": len(unique), "n_strata": len(strata), "model_mean_rmse": float(model_rmse.mean()),
        "nochange_mean_rmse": float(nochange_rmse.mean()), "state_mean_rmse": float(mean_rmse.mean()),
        "strongest_simple_baseline_fixed_before_bootstrap": strongest,
        "relative_gap_to_strong_simple": float(gap), "relative_gap_cluster_bootstrap_95ci": ci,
        "fraction_strata_within_2pct": float((strata.relative_gap <= .02).mean()),
        "all_finite": bool(np.isfinite(effect).all()),
        "prediction_effect_rms": float(np.sqrt(np.mean(effect**2))),
        "across_task_prediction_std": float(np.std(effect,axis=0).mean()),
        "bootstrap_draws": 5000, "full_original_ci_gate_review_required": False,
        "passes_original_operational_criteria": bool(point_pass),
        "original_gate_formula": "gap <= .02 and strata_fraction >= .60 and bootstrap_ci95_lower <= .02",
        "validation_reconstruction_not_used_as_competence": True,
    })


def original_axis_competence(tasks, native_predictions, native_axis):
    """Preserve the original 512-gene competence contract as a second panel.

    Validation controls are recomputed from the permitted role-only CSR. The
    frozen validation task truths/baseline errors are read only after sealing
    the generated prediction matrix. No original H5AD test row is requested.
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    old = Path("/home/yyf/runtime_artifacts/safeconf_mcfaline23/validation_competence_512/decoder")
    genes = json.loads((old / "GENE_IDS.json").read_text())
    if isinstance(genes, dict):
        genes = genes["gene_ids"]
    axis = pd.Series(np.arange(len(native_axis)), index=native_axis)
    gi = axis.loc[genes].to_numpy(int)
    obs, _ = read_metadata()
    part = obs.loc[obs.official_role.eq("val")].reset_index(drop=True)
    cache = ROOT / "role_expression_cache/val"
    manifest = json.loads((cache / "MANIFEST.json").read_text())
    nnz = manifest["nnz"]
    values = np.memmap(cache / "data.f32", mode="r", dtype="float32", shape=(nnz,))
    indices = np.memmap(cache / "indices.i32", mode="r", dtype="int32", shape=(nnz,))
    ptr = np.memmap(cache / "indptr.i64", mode="r", dtype="int64", shape=(len(part)+1,))
    x = csr_matrix((values, indices, ptr), shape=(len(part), len(native_axis)), copy=False)
    controls = {}
    for key, group in part.loc[part.control.astype(str).eq("1")].groupby(["cell_type", "treatment"], observed=True):
        controls[tuple(map(str,key))] = np.asarray(x[group.index.to_numpy()][:,gi].mean(axis=0)).ravel()
    truth_table = pd.read_csv(old / "TASK_ERRORS.csv.gz")
    truth = np.load(old / "VALIDATION_TRUE_EFFECTS.npy")
    task_positions = pd.Series(np.arange(len(tasks)), index=tasks.task_id)
    if set(truth_table.task_id) != set(tasks.task_id):
        raise ValueError("original competence task universe changed")
    predictions = native_predictions[task_positions.loc[truth_table.task_id].to_numpy()][:,gi]
    control_matrix = np.stack([controls[(str(r.context),str(r.treatment))] for r in truth_table.itertuples()])
    errors = np.sqrt(np.mean((predictions-control_matrix-truth)**2, axis=1))
    table = truth_table.copy()
    table["SAMS_error"] = errors
    baseline_name = min(["no_change", "train_state_mean_effect"], key=lambda c: table[c].mean())
    baseline = table[baseline_name].to_numpy(float)
    strata = table.groupby("stratum")[["SAMS_error",baseline_name]].mean()
    fraction = float((strata.SAMS_error <= 1.02*strata[baseline_name]).mean())
    groups = [g.index.to_numpy() for _,g in table.groupby("perturbation",sort=True)]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(5000):
        rows = np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        draws.append(float(errors[rows].mean()/baseline[rows].mean()-1))
    ci = np.quantile(draws,[.025,.975]).tolist()
    gap = float(errors.mean()/baseline.mean()-1)
    passed = gap <= .02 and fraction >= .6 and ci[0] <= .02
    table.to_csv(ROOT / "VALIDATION_512_TASK_ERRORS.csv", index=False)
    strata.to_csv(DOCS / "VALIDATION_512_STRATA.csv")
    receipt("VALIDATION_ORIGINAL_512_COMPETENCE.json", {
        "status": "OPERATIONAL_GATE_PASS" if passed else "OPERATIONAL_GATE_FAIL",
        "gene_axis": 512, "n_tasks": len(table), "n_clusters":len(groups),
        "strongest_simple_baseline":baseline_name, "model_mean_rmse":float(errors.mean()),
        "baseline_mean_rmse":float(baseline.mean()), "relative_gap":gap,
        "bootstrap_ci95":ci, "noninferior_strata_fraction":fraction,
        "passes_original_operational_criteria":passed, "bootstrap_draws":5000,
        "role": "validation original contract; common2840 remains risk comparison axis",
    })


def public_features(meta, predictions, realized, upstream, model_version):
    import math
    import numpy as np
    import pandas as pd
    from tools.safeconf_continual.research import P, PUBLIC
    frame = meta.copy().reset_index(drop=True)
    absolute = np.abs(predictions)
    frame[P] = np.column_stack([np.sqrt(np.mean(predictions**2,axis=1)), absolute.mean(1),
        predictions.mean(1), predictions.std(1), np.quantile(absolute,.95,axis=1),
        np.mean(absolute<=1e-8,axis=1)])
    memory = pd.read_parquet(COMMON / "public_mcfaline_trainval/public_memory.parquet").sort_values("effect_vector_row").reset_index(drop=True)
    history = np.load(COMMON / "reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy", mmap_mode="r")
    by_gene = memory.groupby("perturbation_target",sort=False).indices
    for row, task in enumerate(frame.itertuples()):
        ix = np.asarray(by_gene.get(str(task.gene),[]),int)
        m = memory.iloc[ix]
        legal = ~((m.context.astype(str)==str(task.context)) & (m.condition.astype(str)==str(task.treatment))).to_numpy()
        ix=ix[legal]; m=memory.iloc[ix]
        if not len(ix):
            raise ValueError("registered MC risk task lacks legal history")
        cells = m.n_cells.to_numpy(float)
        same = (m.context.astype(str)==str(task.context)).to_numpy()
        w=cells*(same if same.any() else 1); w=w/w.sum()
        h=np.asarray(history[ix],float); mu=w@h; p=predictions[row]
        variance=float(w@np.mean((h-mu)**2,axis=1))
        distance=float(np.mean((p-mu)**2)); direct=float(w@np.mean((h-p)**2,axis=1))
        if not np.isclose(variance+distance,direct,rtol=1e-10,atol=1e-12):
            raise AssertionError("weighted historical distance identity failed")
        norm=float(np.linalg.norm(p)*np.linalg.norm(mu))
        values=[float(np.sqrt(np.mean(mu**2))), math.sqrt(distance), float(p@mu/norm) if norm>1e-12 else 0.,
            math.sqrt(max(variance,0)), math.log1p(cells.sum()), float(1/(w@w)),
            float(np.sqrt(np.mean((h-h.mean(0))**2,axis=1)).mean())]
        frame.loc[row,PUBLIC]=values
        frame.loc[row,"simple_history_risk"]=math.sqrt(max(direct,0))
        frame.loc[row,"support_risk"]=-math.log1p(cells.sum())
    frame["true_error_rmse"]=realized
    frame["upstream"]=upstream
    frame["model_version"]=model_version
    return frame


def crossfamily(args, model, native_axis):
    import joblib
    import numpy as np
    import pandas as pd
    from tools.safeconf_continual.research import P, PUBLIC, SEEDS, rank_labels, fit_risk, shuffled_labels, summarize, bootstrap_u20, ids_hash
    closure = REPO / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001"
    old = closure / "common_gene_axis/results"
    tasks = pd.read_csv(COMMON / "TEST_TASKS.csv")
    tasks["perturbation"]=tasks.gene
    output = ROOT / "TEST_NATIVE_PREDICTED_STATES.npy"
    if not output.exists():
        array=[]; sampling_variances=[]
        for i,row in tasks.iterrows():
            pred,sampling_variance=predict_metadata(model,row,n=1000,seed=SEED+10000+i)
            array.append(pred); sampling_variances.append(sampling_variance)
            if i%50==0: print(f"[SAMS true-generation] SEEN heldout {i}/{len(tasks)}",flush=True)
        np.save(output,np.stack(array))
        np.save(ROOT/"TEST_NATIVE_SAMPLING_VARIANCE.npy",np.stack(sampling_variances))
        receipt("HELDOUT_PREDICTION_SEAL.json", {"status":"SAVED_BEFORE_ERROR_COMPUTATION",
            "prediction_sha256":sha(output),"checkpoint_sha256":sha(args.checkpoint),
            "n_tasks":len(tasks),"n_generated_samples_per_task":1000,
            "raw_h5ad_test_treated_expression_accessed":False,"role":"SEEN_RETROSPECTIVE"})
    else:
        seal=json.loads((ROOT / "HELDOUT_PREDICTION_SEAL.json").read_text())
        if seal["checkpoint_sha256"]!=sha(args.checkpoint) or seal["prediction_sha256"]!=sha(output):
            raise ValueError("heldout prediction checkpoint binding changed")
    # The next reads are explicitly after saving all generated task states.
    truth=np.asarray(np.load(COMMON/"TEST_TRUE_EFFECTS.npy"),float)
    control=np.asarray(np.load(COMMON/"TEST_CONTROLS.npy"),float)
    gene_axis=json.loads((COMMON/"GENE_IDS.json").read_text())
    gi=pd.Series(np.arange(len(native_axis)),index=native_axis).loc[gene_axis].to_numpy(int)
    prediction=np.asarray(np.load(output),float)[:,gi]-control
    np.save(ROOT/"TEST_PREDICTED_EFFECTS.npy",prediction.astype("float32"))
    meta=pd.read_parquet(COMMON/"risk_cache/external_Learned.parquet")
    if not np.array_equal(meta.task_id,tasks.task_id):raise AssertionError("canonical task order differs")
    strict=pd.read_csv(old/"STRICT_FEEDBACK_TASK_PREDICTIONS.csv.gz")
    anchor=strict[(strict.method=="Shared")&(strict.seed==SEEDS[0])&(strict.budget==0)]
    query_ids=set(anchor.task_id)
    is_query=meta.task_id.isin(query_ids).to_numpy()
    if (int(is_query.sum()),int((~is_query).sum()),meta.loc[is_query,"gene"].nunique(),meta.loc[~is_query,"gene"].nunique())!=(212,331,152,228):
        raise AssertionError("risk training/evaluation identity changed")
    if set(meta.loc[is_query,"gene"]) & set(meta.loc[~is_query,"gene"]):raise AssertionError("risk gene leakage")
    decoder_prediction=np.asarray(np.load(COMMON/"TEST_CALIBRATED_EFFECTS.npy"),float)
    decoder_error=np.sqrt(np.mean((decoder_prediction-truth)**2,axis=1))
    if not np.allclose(decoder_error,meta.true_error_rmse,rtol=1e-6,atol=1e-9):
        raise AssertionError("canonical decoder error failed to reproduce")
    models={
        "DecoderOnly":public_features(meta,decoder_prediction,decoder_error,"DecoderOnly",str(meta.model_version.iloc[0])),
        "SAMS_VAE":public_features(meta,prediction,np.sqrt(np.mean((prediction-truth)**2,axis=1)),"SAMS_VAE",sha(args.checkpoint)),
    }
    (ROOT/"risk_models").mkdir(exist_ok=True)
    all_predictions=[];cdf_audit=[];null_audit=[];fits=[]
    for name,frame in models.items():frame.to_parquet(ROOT/f"{name}_RISK_FEATURES.parquet",index=False)
    for source,target in [("DecoderOnly","SAMS_VAE"),("SAMS_VAE","DecoderOnly")]:
        line=f"{source}_to_{target}"
        trainframe=models[source].loc[~is_query].reset_index(drop=True)
        query=models[target].loc[is_query].reset_index(drop=True)
        label,audit=rank_labels(trainframe,line)
        cdf_audit.extend(audit)
        scores={"Magnitude":query.predicted_magnitude.to_numpy(),
                "PublicDirectDistance":query.prediction_prior_rmse.to_numpy(),
                "WeightedHistoryDistance":query.simple_history_risk.to_numpy(),
                "NegativeHistorySupport":query.support_risk.to_numpy()}
        if target == "SAMS_VAE":
            native_var=np.load(ROOT/"TEST_NATIVE_SAMPLING_VARIANCE.npy")
            scores["SAMS_GeneratedSampleDisagreement"]=np.sqrt(native_var[is_query][:,gi].mean(axis=1))
        for kind in ["ridge","hgb"]:
            begin=time.time(); learner=fit_risk(trainframe,label,P+PUBLIC,kind,SEEDS[0])
            name=f"Source_{kind}"
            scores[name]=learner.predict(query)
            joblib.dump(learner,ROOT/f"risk_models/{line}_{name}.joblib")
            fits.append({"line":line,"method":name,"train_rows":len(trainframe),"train_genes":trainframe.gene.nunique(),
                         "elapsed_seconds":time.time()-begin,"source_label_hash":array_hash(label),
                         "source_task_hash":ids_hash(trainframe.task_id),"target_task_hash":ids_hash(query.task_id)})
        for offset in range(5):
            null,null_info=shuffled_labels(trainframe,label,SEED+offset)
            null_audit.append({"line":line,**null_info})
            begin=time.time(); learner=fit_risk(trainframe,null,P+PUBLIC,"hgb",SEEDS[0])
            name=f"ShuffledSource_hgb_{offset}"
            scores[name]=learner.predict(query)
            joblib.dump(learner,ROOT/f"risk_models/{line}_{name}.joblib")
            fits.append({"line":line,"method":name,"train_rows":len(trainframe),"train_genes":trainframe.gene.nunique(),
                         "elapsed_seconds":time.time()-begin,"source_label_hash":array_hash(null),
                         "source_task_hash":ids_hash(trainframe.task_id),"target_task_hash":ids_hash(query.task_id)})
        for method,score in scores.items():
            part=query[["task_id","gene","target","true_error_rmse"]].copy()
            part["line"]=line;part["method"]=method;part["seed"]=SEEDS[0];part["risk"]=score
            all_predictions.append(part)
    all_predictions=pd.concat(all_predictions,ignore_index=True)
    all_predictions.to_csv(ROOT/"CROSSFAMILY_TASK_PREDICTIONS.csv.gz",index=False)
    strata,macro=summarize(all_predictions)
    strata.to_csv(DOCS/"CROSSFAMILY_STRATA.csv",index=False)
    macro.to_csv(DOCS/"CROSSFAMILY_MACRO.csv",index=False)
    pd.DataFrame(cdf_audit).to_csv(DOCS/"CROSSFAMILY_CDF_AUDIT.csv",index=False)
    pd.DataFrame(null_audit).to_csv(DOCS/"CROSSFAMILY_SHUFFLE_AUDIT.csv",index=False)
    pd.DataFrame(fits).to_csv(DOCS/"CROSSFAMILY_FIT_AUDIT.csv",index=False)
    differences=[]
    for line,group in all_predictions.groupby("line"):
        frame=group[group.method=="Magnitude"].sort_values("task_id").reset_index(drop=True)
        wide=group.pivot(index="task_id",columns="method",values="risk").loc[frame.task_id]
        contrasts=[("WeightedHistoryDistance","Magnitude"),("Source_hgb","WeightedHistoryDistance"),
            ("Source_ridge","WeightedHistoryDistance")]+[("Source_hgb",f"ShuffledSource_hgb_{i}") for i in range(5)]
        for a,b in contrasts:
            result=bootstrap_u20(frame,wide[a].to_numpy(),wide[b].to_numpy(),replicates=5000,seed=SEED)
            differences.append({"line":line,"method_a":a,"method_b":b,**result})
    pd.DataFrame(differences).to_csv(DOCS/"CROSSFAMILY_PAIRED_DIFFERENCES.csv",index=False)
    receipt("CROSSFAMILY_STATUS.json", {"status":"COMPLETE","risk_fits":len(fits),
        "lines":["DecoderOnly_to_SAMS_VAE","SAMS_VAE_to_DecoderOnly"],
        "source_errors_from":"frozen upstream heldout tasks only", "source_train_rows":331,
        "source_train_genes":228,"query_rows":212,"query_genes":152,"gene_axis":2840,
        "bootstrap_draws":5000,"new_confirmation_claim":False,"role":"SEEN_RETROSPECTIVE",
        "upstream_validation_information":"Decoder uses legacy validation shrinkage; SAMS selected by reconstruction val_loss; no target risk errors in source fitting"})


def followup(args):
    import numpy as np
    import pandas as pd
    import torch
    from perturbench.modelcore.models.sams_vae import SparseAdditiveVAE
    sys.path.insert(0,str(REPO))
    torch.set_num_threads(4)
    apply_verified_engineering_optimization()
    if not args.checkpoint:
        args.checkpoint=Path(json.loads((ROOT/"TRAINING_COMPLETION.json").read_text())["best_checkpoint"])
    if not (ROOT/"VALIDATION_PREDICTION_SEAL.json").exists():evaluate(args)
    seal=json.loads((ROOT/"VALIDATION_PREDICTION_SEAL.json").read_text())
    if seal["checkpoint_sha256"]!=sha(args.checkpoint):raise ValueError("validation checkpoint mismatch")
    native=np.load(ROOT/"VALIDATION_NATIVE_PREDICTED_STATES.npy")
    tasks=pd.read_csv(ROOT/"VALIDATION_GENERATED_TASKS.csv")
    axis=json.loads((ROOT/"NATIVE_GENE_AXIS.json").read_text())
    score_validation(tasks,native,axis)
    original_axis_competence(tasks,native,axis)
    common=json.loads((ROOT/"VALIDATION_COMPETENCE.json").read_text())
    original=json.loads((ROOT/"VALIDATION_ORIGINAL_512_COMPETENCE.json").read_text())
    if not common["passes_original_operational_criteria"]:
        receipt("CROSSFAMILY_STATUS.json", {"status":"UPSTREAM_NOT_YET_QUALIFIED",
            "common2840_competence":common["status"],"original512_competence":original["status"],
            "source_transfer_conclusion":None,"next_action":"review convergence and one allowed upstream diagnostic before deciding pressure-only followup"})
        return
    model=SparseAdditiveVAE.load_from_checkpoint(args.checkpoint,weights_only=False,map_location="cpu").to("cuda").eval()
    crossfamily(args,model,axis)


def watch(args):
    """Durable followup: use fresh on-disk code once upstream fit/eval exits."""
    train_unit="safeconf-sams-v1-stream-20261003.service"
    while True:
        status=subprocess.run(["systemctl","--user","is-active",train_unit],capture_output=True,text=True).stdout.strip()
        if status not in {"active","activating"}:
            if (ROOT/"TRAINING_COMPLETION.json").exists():
                followup(args)
            else:
                receipt("FOLLOWUP_STATUS.json", {"status":"TRAINING_EXIT_WITHOUT_COMPLETION",
                    "unit_status":status,"next_action":"inspect LAST_FAILURE/systemd log and resume same candidate"})
            return
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "train", "evaluate", "followup", "watch"])
    parser.add_argument("--gpu-hours", type=float, default=24)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not 0 < args.gpu_hours <= 48:
        raise ValueError("SAMS candidate budget must be within (0,48] GPU-hours")
    try:
        {"preflight": preflight, "train": lambda: train(args), "evaluate": lambda: evaluate(args),
         "followup":lambda:followup(args),"watch":lambda:watch(args)}[args.mode]()
    except Exception as exc:
        import traceback
        receipt("LAST_FAILURE.json", {"mode": args.mode, "exception_type": type(exc).__name__,
                                      "message": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
