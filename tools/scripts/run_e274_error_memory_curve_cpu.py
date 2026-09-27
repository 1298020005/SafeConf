#!/usr/bin/env python3
"""E274: perturbation-cold error-memory availability curve.

Uses only fold 0 train/validation rows from the frozen development matrix.
For each held perturbation bucket, the error-memory pool is deterministically
thinned to 0/5/10/25/50/100 percent. The held bucket never contributes labels.
This is a feasibility/sensitivity analysis, not a blind test.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_e273_dual_history_cpu import (SOURCE_DEFAULT, PREDICTORS, PREDICTION, QUALITY,
    HISTORY, bucket, percentile, utility)

E = ("error_hist_pert_mean", "error_hist_pert_std", "error_hist_pert_n",
     "error_hist_context_mean", "error_hist_context_std", "error_hist_context_n")
BASE = PREDICTION + QUALITY + HISTORY

def available(key: str, fraction: float) -> bool:
    if fraction <= 0: return False
    if fraction >= 1: return True
    h = hashlib.blake2b(("E274-memory:" + str(key)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % 10000 < int(round(fraction * 10000))

def make_memory(source: pd.DataFrame, query: pd.DataFrame, leave_one_out: bool) -> pd.DataFrame:
    # Group sufficient statistics let us remove a fit row's own label exactly.
    y = pd.to_numeric(source.true_error_rmse, errors="coerce").to_numpy(float)
    finite = np.isfinite(y)
    global_sum = float(np.nansum(y)); global_sq = float(np.nansum(y*y)); global_n = int(finite.sum())
    global_mean = global_sum / global_n if global_n else 0.0
    global_var = max(0.0, global_sq / global_n - global_mean**2) if global_n else 0.0
    def stats(col):
        out = {}
        for key, ix in source.groupby(col, sort=False).groups.items():
            vals = pd.to_numeric(source.loc[ix, "true_error_rmse"], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            out[str(key)] = [float(vals.sum()), float(np.dot(vals, vals)), int(len(vals))]
        return out
    pert, context = stats("perturbation"), stats("context")
    source_by_task = {str(k): i for i, k in enumerate(source.task_key.astype(str))}
    rows = []
    for idx, row in query.iterrows():
        def one(group, key, remove):
            s, q, n = group.get(str(key), [global_sum, global_sq, global_n])
            if remove and str(row.task_key) in source_by_task:
                yy = float(row.true_error_rmse)
                if np.isfinite(yy): s, q, n = s-yy, q-yy*yy, n-1
            if n <= 0: return (np.nan, np.nan, 0)
            m = s/n; sd = np.sqrt(max(0.0, q/n-m*m))
            return (m, sd, n)
        p = one(pert, row.perturbation, leave_one_out)
        c = one(context, row.context, leave_one_out)
        rows.append({"error_hist_pert_mean":p[0],"error_hist_pert_std":p[1],"error_hist_pert_n":p[2],
                     "error_hist_context_mean":c[0],"error_hist_context_std":c[1],"error_hist_context_n":c[2]})
    return pd.DataFrame(rows, index=query.index)

def rank_frame(fit, test, cols):
    a, b = fit.copy(), test.copy()
    for col in cols:
        a["r_"+col] = percentile(fit[col], fit[col])
        b["r_"+col] = percentile(fit[col], test[col])
    return a, b

def main(args):
    out=args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True)
    use=["dataset_name","fold_id","split","task_key","context","perturbation","predictor_name","true_error_rmse",*PREDICTION,*QUALITY,*HISTORY]
    raw=pd.read_csv(args.source,usecols=list(dict.fromkeys(use)))
    raw=raw.loc[raw.fold_id.eq(0)&raw.split.isin(("train","val"))&raw.predictor_name.isin(PREDICTORS)].copy()
    raw["bucket"]=raw.perturbation.map(lambda x: bucket(x,args.n_buckets))
    rows=[]; audit=[]
    fractions=(0.0,0.05,0.10,0.25,0.50,1.0)
    for (dataset,predictor), task in raw.groupby(["dataset_name","predictor_name"], sort=True):
        for held in range(args.n_buckets):
            fit=task.loc[task.bucket.ne(held)].copy(); test=task.loc[task.bucket.eq(held)].copy()
            if len(fit)<50 or len(test)<20: continue
            if set(fit.perturbation)&set(test.perturbation): raise AssertionError("perturbation overlap")
            for frac in fractions:
                source=fit.loc[fit.task_key.map(lambda x: available(x,frac))].copy()
                fit_e=make_memory(source,fit,True) if len(source) else pd.DataFrame(np.nan,index=fit.index,columns=E)
                test_e=make_memory(source,test,False) if len(source) else pd.DataFrame(np.nan,index=test.index,columns=E)
                fit2=fit.copy(); test2=test.copy(); fit2[list(E)]=fit_e.to_numpy(); test2[list(E)]=test_e.to_numpy()
                fr,tr=rank_frame(fit2,test2,BASE+E)
                cols=["r_"+c for c in BASE+E]
                model=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),Ridge(alpha=10.0))
                model.fit(fr[cols], percentile(fr.true_error_rmse,fr.true_error_rmse))
                score=model.predict(tr[cols]); truth=test2.true_error_rmse.to_numpy(float)
                rows.append({"dataset":dataset,"predictor":predictor,"held_bucket":held,"memory_fraction":frac,
                    "n_memory_rows":len(source),"n_fit":len(fit),"n_test":len(test),"utility20":utility(score,truth),
                    "spearman":float(spearmanr(score,truth).statistic)})
            print(f"{dataset}/{predictor}/bucket={held}: complete",flush=True)
    result=pd.DataFrame(rows); result.to_csv(out/"FOLD_RESULTS.csv",index=False)
    summary=(result.groupby(["memory_fraction"],as_index=False).agg(n_cells=("utility20","size"),
        utility20_mean=("utility20","mean"),spearman_mean=("spearman","mean"),n_memory_rows_mean=("n_memory_rows","mean")))
    summary.to_csv(out/"SUMMARY.csv",index=False)
    paired=result.pivot_table(index=["dataset","predictor","held_bucket"],columns="memory_fraction",values="utility20")
    deltas=[]
    for frac in fractions:
        if frac==0: continue
        vals=paired[frac]-paired[0.0]
        deltas.append({"memory_fraction":frac,"delta_utility20_vs_zero_mean":float(vals.mean()),"positive_cells":int((vals>0).sum()),"n_cells":int(vals.notna().sum())})
    pd.DataFrame(deltas).to_csv(out/"DELTA_VS_ZERO.csv",index=False)
    status={"status":"DEVELOPMENT_ONLY_TRAIN_VAL_PERTURBATION_COLD","source":str(args.source),"no_final_test_rows_read":True,
      "memory_fractions":list(fractions),"n_rows":len(raw),"note":"Memory pool is deterministically thinned; held perturbation bucket never contributes labels.",
      "limits":["Development feature matrix; no blind external panel.","Availability simulation does not reconstruct raw biological provenance."]}
    (out/"STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    print(summary.to_string(index=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,default=SOURCE_DEFAULT); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--n-buckets",type=int,default=5); main(p.parse_args())
