#!/usr/bin/env python3
"""Bounded equivalence/timing test for one SAMS GPU-indexing optimization.

Only torch.cat([m[perturbation[i].bool()] ...]) becomes m[z_p_index_pert].
Model structure, loss, randomness, masks and training rows are unchanged.
"""
from __future__ import annotations
import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
sys.path.insert(0,"/home/yyf/archive/external/PerturBench/src")


def make_vectorized_forward():
    from perturbench.modelcore.models import sams_vae
    original=sams_vae.SparseAdditiveVAE.forward
    if getattr(original,"_safeconf_vectorized_indexing",False):
        return original
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source)
    class Replace(ast.NodeTransformer):
        changed=0
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="m_t":
                if not (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                        and node.value.func.attr=="cat"):
                    raise AssertionError("official m_t expression differs from audited implementation")
                self.changed+=1
                return ast.copy_location(ast.parse("m_t = m[z_p_index_pert]").body[0],node)
            return self.generic_visit(node)
    replace=Replace();tree=replace.visit(tree);ast.fix_missing_locations(tree)
    if replace.changed!=1:raise AssertionError("expected exactly one official indexed expression")
    namespace=dict(vars(sams_vae))
    exec(compile(tree,str(Path(__file__)) + ":equivalent_forward","exec"),namespace)
    function=namespace["forward"]
    function._safeconf_vectorized_indexing=True
    return function


def install_vectorized_forward():
    from perturbench.modelcore.models.sams_vae import SparseAdditiveVAE
    SparseAdditiveVAE.forward=make_vectorized_forward()


def main():
    from tools.scripts import run_safeconf_sams_crossfamily_v1 as driver
    import numpy as np
    import torch
    from scipy.sparse import csr_matrix
    from perturbench.data.types import Batch
    from perturbench.modelcore.models.sams_vae import SparseAdditiveVAE
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,default=driver.ROOT/"checkpoints/recovery.ckpt")
    parser.add_argument("--steps",type=int,default=20)
    args=parser.parse_args()
    started=time.time()
    torch.set_num_threads(4)
    model=SparseAdditiveVAE.load_from_checkpoint(args.checkpoint,map_location="cpu",weights_only=False).to("cuda")
    original=SparseAdditiveVAE.forward
    optimized=make_vectorized_forward()
    obs,_=driver.read_metadata()
    part=obs[obs.official_role.eq("train")].reset_index(drop=True)
    rng=np.random.default_rng(driver.SEED)
    rows=rng.choice(len(part),size=256,replace=False)
    cache=driver.ROOT/"role_expression_cache/train"
    config=json.loads((cache/"MANIFEST.json").read_text()); nnz=config["nnz"]
    data=np.memmap(cache/"data.f32",mode="r",dtype="float32",shape=(nnz,))
    indices=np.memmap(cache/"indices.i32",mode="r",dtype="int32",shape=(nnz,))
    ptr=np.memmap(cache/"indptr.i64",mode="r",dtype="int64",shape=(len(part)+1,))
    x=csr_matrix((data,indices,ptr),shape=(len(part),model.n_genes),copy=False)
    selected=part.iloc[rows]
    batch=Batch(gene_expression=x[rows],perturbations=[[] if str(v)=="control" else str(v).split("+") for v in selected.condition],
        covariates={k:selected[k].astype(str).tolist() for k in ["cell_type","treatment"]})
    batch=model.training_record["transform"](batch)
    batch=batch._replace(gene_expression=batch.gene_expression.to("cuda"),perturbations=batch.perturbations.to("cuda"),
                         covariates={k:v.to("cuda") for k,v in batch.covariates.items()})
    def fit_step(function, seed=None):
        SparseAdditiveVAE.forward=function
        model.train();model.zero_grad(set_to_none=True)
        if seed is not None:driver.seed_everything(seed)
        _,_,_,elbo=model(batch.gene_expression,batch.perturbations,batch.covariates)
        loss=-elbo;loss.backward()
        return loss
    loss_original=fit_step(original,driver.SEED).detach().cpu()
    grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    loss_optimized=fit_step(optimized,driver.SEED).detach().cpu()
    loss_equal=torch.allclose(loss_original,loss_optimized,rtol=1e-5,atol=1e-6)
    differences=[]
    for name,p in model.named_parameters():
        if name in grads:
            current=p.grad.detach().cpu()
            difference=float((current-grads[name]).abs().max())
            okay=torch.allclose(current,grads[name],rtol=1e-4,atol=1e-5)
            differences.append({"parameter":name,"max_abs_gradient_difference":difference,"pass":okay})
        elif p.grad is not None:
            raise AssertionError("gradient presence changed")
    gradient_equal=all(d["pass"] for d in differences)
    del grads
    model.eval()
    with torch.inference_mode():
        SparseAdditiveVAE.forward=original;driver.seed_everything(driver.SEED)
        pred0=model.predict(batch).detach().cpu()
        SparseAdditiveVAE.forward=optimized;driver.seed_everything(driver.SEED)
        pred1=model.predict(batch).detach().cpu()
    prediction_equal=torch.allclose(pred0,pred1,rtol=1e-5,atol=1e-7)
    import gc
    del pred0,pred1
    gc.collect();torch.cuda.empty_cache()
    measurements={"original":[],"vectorized":[]}
    # Alternate order to avoid conflating GPU thermal/ramp effects with code.
    for repeat,order in enumerate([("original","vectorized"),("vectorized","original"),("original","vectorized")]):
        for name in order:
            fn=original if name=="original" else optimized
            for _ in range(3):fit_step(fn)
            torch.cuda.synchronize();t=time.time()
            for _ in range(args.steps):fit_step(fn)
            torch.cuda.synchronize()
            measurements[name].append((time.time()-t)/args.steps)
    original_seconds=float(np.median(measurements["original"]))
    vectorized_seconds=float(np.median(measurements["vectorized"]))
    speedup=(original_seconds-vectorized_seconds)/original_seconds
    passed=bool(loss_equal and gradient_equal and prediction_equal and speedup>=.15)
    driver.receipt("INDEXING_SPEEDUP_RESULT.json",{
        "status":"ADOPT_EQUIVALENT_OPTIMIZATION" if passed else "KEEP_OFFICIAL_INDEXING",
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":driver.sha(args.checkpoint),
        "change":"m_t = m[z_p_index_pert] replaces row-wise mask indexing + torch.cat",
        "official_source_file_modified":False,"new_upstream_attempt":False,
        "loss_original":float(loss_original),"loss_vectorized":float(loss_optimized),"loss_equal":loss_equal,
        "gradient_equal":gradient_equal,"gradient_tolerance":{"rtol":1e-4,"atol":1e-5},
        "prediction_equal":prediction_equal,"prediction_tolerance":{"rtol":1e-5,"atol":1e-7},
        "gradient_details":differences,"timing_seconds_per_forward_backward":measurements,
        "median_original_seconds":original_seconds,"median_vectorized_seconds":vectorized_seconds,
        "fraction_time_reduction":speedup,"minimum_time_reduction":.15,
        "benchmark_gpu_hours":(time.time()-started)/3600,"physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_sha256":driver.sha(Path(__file__)),"train_batch_row_hash":driver.array_hash(rows),
    })


if __name__=="__main__":main()
