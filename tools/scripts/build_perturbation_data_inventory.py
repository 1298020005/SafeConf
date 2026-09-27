#!/usr/bin/env python3
"""Build a metadata-only inventory of locally available perturbation datasets.

No expression matrix is loaded: h5ad files are opened backed and only obs
metadata is inspected. This is an inventory/audit, not a preprocessing run.
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path
import pandas as pd

try:
    import anndata as ad
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"anndata is required: {exc}")

PERT_COLS = ["perturbation", "drug_dose", "drug", "drug_name", "condition", "gene", "target", "guide_identity", "guide_id", "sgRNA"]
CONTEXT_COLS = ["cell_line", "cell_line_id", "celltype", "cell_type", "cell_types", "donor", "donor_id", "patient", "patient_id", "condition"]
SPLIT_COLS = ["split", "split_name", "partition", "set", "phase"]

def actual_path(p: str) -> Path:
    q = Path(p)
    if q.exists(): return q
    s = str(q).replace("/home/yyf/datasets/", "/home/yyf/data/")
    return Path(s)

def first_existing(cols, candidates):
    for c in candidates:
        if c in cols: return c
    return ""

def unique_count(obs: pd.DataFrame, col: str) -> int:
    if not col: return 0
    return int(obs[col].astype(str).replace({"nan": pd.NA}).dropna().nunique())

def values_preview(obs: pd.DataFrame, col: str, n=8) -> str:
    if not col: return ""
    vals = sorted(map(str, obs[col].dropna().unique()))
    return " | ".join(vals[:n]) + (" | ..." if len(vals) > n else "")

def classify_subtype(study: str, path: str, obs: pd.DataFrame, ptype: str) -> str:
    text=(study+" "+path).lower()
    cols=" ".join(map(str,obs.columns)).lower()
    if ptype.startswith("chemical"): return "药物/小分子"
    labels=[]
    if "crispra" in text or "activation" in text or "crispra" in cols: labels.append("激活/CRISPRa")
    if "crispri" in text or "inhibition" in text or "crispri" in cols: labels.append("抑制/CRISPRi")
    if "knockout" in text or "ko" in text or "essential" in text or "gwps" in text: labels.append("敲除/损失功能候选")
    if "overexpression" in text or "overexpression" in cols: labels.append("过表达候选")
    if "+" in " ".join(values_preview(obs,c,20) for c in PERT_COLS if c in obs.columns): labels.append("组合扰动")
    return ";".join(dict.fromkeys(labels)) or "基因扰动（具体机制需按原研究协议核对）"

def scan_record(rec):
    path=actual_path(rec.get("local_path",""))
    out={"study_family":rec.get("study_family",""),"dataset_family":rec.get("dataset_family",""),"perturbation_type_manifest":rec.get("perturbation_type",""),"modality_manifest":rec.get("modality",""),"path":str(path),"exists":path.exists(),"read_status":"NOT_READ"}
    if not path.exists():
        out["read_status"]="MISSING"; return out
    try:
        obj=ad.read_h5ad(path, backed="r")
        obs=obj.obs
        cols=list(map(str,obs.columns))
        pcol=first_existing(cols,PERT_COLS)
        ccol=first_existing(cols,CONTEXT_COLS)
        scol=first_existing(cols,SPLIT_COLS)
        nobs=int(obj.n_obs); nvars=int(obj.n_vars)
        if ccol and pcol:
            n_tasks=int(obs[[ccol,pcol]].astype(str).drop_duplicates().shape[0])
        elif pcol: n_tasks=unique_count(obs,pcol)
        else: n_tasks=0
        split_values=values_preview(obs,scol,20)
        dose_cols=[c for c in cols if re.search(r"dose|conc|concentration",c,re.I)]
        time_cols=[c for c in cols if re.search(r"time|hour|duration|day",c,re.I)]
        drug_cols=[c for c in cols if re.search(r"drug|compound|smile|moa|chemical",c,re.I)]
        out.update({"read_status":"READ_BACKED","n_cells":nobs,"n_genes_or_vars":nvars,"obs_columns":";".join(cols),"perturbation_column":pcol,"context_column":ccol,"split_column":scol,"n_context":unique_count(obs,ccol),"n_perturbation":unique_count(obs,pcol),"n_tasks_context_x_perturbation":n_tasks,"perturbation_preview":values_preview(obs,pcol),"context_preview":values_preview(obs,ccol),"split_values":split_values,"dose_fields":";".join(dose_cols),"time_fields":";".join(time_cols),"drug_related_fields":";".join(drug_cols),"subtype_inference":classify_subtype(out["study_family"],str(path),obs,out["perturbation_type_manifest"])})
        obj.file.close()
    except Exception as exc:
        out.update({"read_status":"READ_ERROR","error":f"{type(exc).__name__}: {exc}"})
    return out

def main(a):
    manifest=json.loads(a.manifest.read_text())
    rows=[scan_record(r) for r in manifest]
    df=pd.DataFrame(rows).sort_values(["dataset_family","study_family","path"])
    a.out_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.out_dir/"PERTURBATION_DATA_INVENTORY.csv",index=False)
    status={"status":"METADATA_ONLY_INVENTORY_COMPLETE","manifest":str(a.manifest),"records":len(df),"existing":int(df.exists.sum()),"read_backed":int((df.read_status=="READ_BACKED").sum()),"read_errors":int((df.read_status=="READ_ERROR").sum()),"missing":int((df.read_status=="MISSING").sum()),"no_expression_loaded":True,"no_new_download":True}
    (a.out_dir/"STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(status,ensure_ascii=False,indent=2))
    print(df.groupby(["perturbation_type_manifest"],dropna=False).size().to_string())
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=Path("/home/yyf/data/singlecell_perturbation_atlas/manifests/dataset_manifest.json")); p.add_argument("--out-dir",type=Path,required=True); main(p.parse_args())
