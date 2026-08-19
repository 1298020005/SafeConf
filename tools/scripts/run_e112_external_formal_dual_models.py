#!/usr/bin/env python3
"""E112: formal scGPT/GEARS replication on Lara ex vivo and Santinha."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, random, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
CODE_ROOT=ROOT/"code/20260426_154505_perturb_transport_final_push";sys.path.insert(0,str(CODE_ROOT))
E106_SCRIPT=ROOT/"tools/scripts/run_e106_frangieh_context_scgpt.py";E107_SCRIPT=ROOT/"tools/scripts/run_e107_frangieh_context_gears.py"
CONTRACT=ROOT/"docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv"
SOURCE_AUDIT=ROOT/"docs/实验结果/E99_multicontext_external_contract_20260713/tables/E99_SOURCE_AUDIT.csv"
CACHE=Path("/home/yyf/data/safeconf_e100_gene_external")
OUT=ROOT/"docs/实验结果/E112_external_formal_dual_models_20260713"
SCGPT_VOCAB=Path("/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human/vocab.json")
ALIASES={"Gltscr1":"BICRA","Dgcr14":"ESS2"};EXPRESSION_ALIASES={"Gltscr1":"Bicra","Dgcr14":"Ess2"};SEED=202607112;N_GENES=512

import anndata as ad,numpy as np,pandas as pd,scipy.sparse as sp,torch
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from safetrans_confidence.data.records import validate_prediction_record_artifacts


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
E106=module("e106_for_e112",E106_SCRIPT);E107=module("e107_for_e112",E107_SCRIPT);E65=E106.import_e65()

SPECS={
 "Lara_exvivo":{"source":Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/LaraAstiasoHuntly2023_exvivo.h5ad"),"context":"celltype"},
 "Santinha":{"source":Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/SantinhaPlatt2023.h5ad"),"context":"cell_types"},
}

def seed(value):
    os.environ["PYTHONHASHSEED"]=str(value);random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def rmse(a,b):return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
def cosine(a,b):
    d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(np.dot(a,b)/d) if d>1e-12 else 0.
def cosine_error(a,b):return 1.-cosine(a,b)
def rho(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3 or np.unique(a[m]).size<2 or np.unique(b[m]).size<2:return float("nan")
    return float(np.corrcoef(rankdata(a[m]),rankdata(b[m]))[0,1])
def robust_z(v,r):
    r=np.asarray(r,float);c=float(np.median(r));mad=float(np.median(np.abs(r-c)));s=max(1.4826*mad,float(np.std(r)),1e-8);return np.clip((np.asarray(v,float)-c)/s,-5,5)
def human_token(g):return ALIASES.get(str(g),str(g).upper())


class Assets:
    def __init__(self,dataset):
        cache_dir=Path("/home/yyf/data/safeconf_e112_external");cache_dir.mkdir(parents=True,exist_ok=True);cache=cache_dir/f"{dataset}_CONTROL_ONLY_512.npz"
        if not cache.exists():
            manifest=pd.read_csv(CONTRACT,keep_default_na=False);manifest=manifest[manifest.dataset.eq(dataset)];contexts=sorted(manifest.context.astype(str).unique());perts=sorted(manifest.perturbation.astype(str).unique());spec=SPECS[dataset];raw=ad.read_h5ad(spec["source"]);obs_c=raw.obs[spec["context"]].astype(str).to_numpy();obs_p=raw.obs["perturbation"].astype(str).to_numpy();keep=np.isin(obs_c,contexts)&(np.isin(obs_p,perts)|(obs_p=="control"));x=raw.X[keep];x=sp.csr_matrix(x) if not sp.issparse(x) else x.tocsr();x=normalize_log1p(x);genes=raw.var_names.astype(str).tolist();gix={g:i for i,g in enumerate(genes)};control=np.asarray(x[obs_p[keep]=="control"].mean(axis=0)).ravel();vocab=json.loads(SCGPT_VOCAB.read_text());required=[gix[EXPRESSION_ALIASES.get(p,p)] for p in perts];ranked=np.argsort(-control,kind="stable");selected=[];seen=set()
            for i in required+ranked.astype(int).tolist():
                token=human_token(genes[i])
                if token in vocab and token not in seen:selected.append(i);seen.add(token)
                if len(selected)==N_GENES:break
            selected=np.asarray(selected,int);raw_genes=np.asarray([genes[i] for i in selected],dtype=str);tokens=np.asarray([human_token(g) for g in raw_genes],dtype=str);xs=x[:,selected];labels=np.asarray([f"{c}\x1f{p}" for c,p in zip(obs_c[keep],obs_p[keep])]);groups,codes=np.unique(labels,return_inverse=True);membership=sp.csr_matrix((np.ones(len(codes),np.float32),(codes,np.arange(len(codes)))),shape=(len(groups),len(codes)));sums=membership@xs;counts=np.bincount(codes,minlength=len(groups)).astype(np.float32);means=np.asarray(sums.multiply((1/counts)[:,None]).toarray(),np.float32);mean_map={label:means[i] for i,label in enumerate(groups)};controls=np.stack([mean_map[f"{c}\x1fcontrol"] for c in contexts]);effects=np.stack([mean_map[f"{c}\x1f{p}"]-mean_map[f"{c}\x1fcontrol"] for c in contexts for p in perts]);np.savez_compressed(cache,contexts=np.asarray(contexts,dtype=str),perturbations=np.asarray(perts,dtype=str),raw_genes=raw_genes,tokens=tokens,controls=controls,effects=effects)
        with np.load(cache) as z:
            self.contexts=z["contexts"].astype(str).tolist();self.perts=z["perturbations"].astype(str).tolist();self.raw_genes=z["raw_genes"].astype(str).tolist();self.genes=z["tokens"].astype(str).tolist();self.controls=np.asarray(z["controls"],np.float32);self.effects=np.asarray(z["effects"],np.float32).reshape(len(self.contexts),len(self.perts),-1)
        self.cix={v:i for i,v in enumerate(self.contexts)};self.pix={v:i for i,v in enumerate(self.perts)};self.gix={v:i for i,v in enumerate(self.genes)}
    def control(self,c):return self.controls[self.cix[c]]
    def effect(self,c,p):return self.effects[self.cix[c],self.pix[p]]


def graphs_for(tasks,assets):
    graphs=[]
    for r in tasks.itertuples(index=False):
        basal=assets.control(str(r.context));target=basal+assets.effect(str(r.context),str(r.perturbation));flag=np.zeros(N_GENES,np.float32);flag[assets.gix[human_token(r.perturbation)]]=1
        graphs.append(Data(x=torch.from_numpy(np.stack([basal,flag],axis=1)),y=torch.from_numpy(target).unsqueeze(0),pert=f"{r.context}::{r.perturbation}",context=str(r.context),perturbation=str(r.perturbation),split=str(r.split),setting=str(r.setting)))
    return {s:[g for g in graphs if g.split==s] for s in ("train","val","test")}


def normalize_log1p(x):
    x=x.tocsr().astype(np.float32);tot=np.asarray(x.sum(axis=1)).ravel();scale=np.divide(1e4,tot,out=np.zeros_like(tot,dtype=np.float32),where=tot>0);x=(sp.diags(scale)@x).tocsr();x.data=np.log1p(x.data);return x


def control_matrix(dataset,assets,source_contexts):
    spec=SPECS[dataset];raw=ad.read_h5ad(spec["source"]);obs_c=raw.obs[spec["context"]].astype(str);obs_p=raw.obs["perturbation"].astype(str);mask=obs_c.isin(source_contexts).to_numpy()&obs_p.eq("control").to_numpy();idx={g:i for i,g in enumerate(raw.var_names.astype(str))};columns=[idx[g] for g in assets.raw_genes];x=raw.X[mask][:,columns];x=sp.csr_matrix(x) if not sp.issparse(x) else x.tocsr();return normalize_log1p(x),int(mask.sum())


def coexpression_from_matrix(x,genes,k=10,threshold=.4):
    corr=np.nan_to_num(np.abs(np.corrcoef(x.toarray(),rowvar=False)),nan=0,posinf=0,neginf=0);rows=[]
    for ti,t in enumerate(genes):
        for si in np.argsort(corr[:,ti])[-(k+1):][::-1]:
            v=float(corr[si,ti])
            if v>=threshold or si==ti:rows.append({"source":genes[int(si)],"target":t,"importance":v})
    frame=pd.DataFrame(rows);ei,ew=E107.edge_tensors(frame,genes);return frame,ei,ew


def fit_models(graphs,assets,dataset,source_contexts,device):
    model,_,meta=E65.load_model(device);gene_ids=E65.make_gene_ids(assets.genes,meta["vocab"]);train=DataLoader(graphs["train"],16,shuffle=True,generator=torch.Generator().manual_seed(SEED));val=DataLoader(graphs["val"],16,shuffle=False);opt=torch.optim.Adam(model.parameters(),lr=1e-4);scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda");best=1e99;state=None;stale=0;hist=[]
    for epoch in range(1,11):
        tr=E65.train_one_epoch(model,train,gene_ids,opt,scaler,device,device.type=="cuda");va=E65.evaluate_mse(model,val,gene_ids,device,device.type=="cuda");hist.append({"model":"scGPT","epoch":epoch,"train_mse":tr,"validation_mse":va})
        if va<best-1e-7:best=va;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=3:break
    model.load_state_dict(state);model.to(device).eval()
    go,goi,gow=E107.build_go(assets.genes,20);ctrl,nctrl=control_matrix(dataset,assets,source_contexts);co,coi,cow=coexpression_from_matrix(ctrl,assets.genes)
    from gears.model import GEARS_Model
    cfg={"hidden_size":64,"num_go_gnn_layers":1,"num_gene_gnn_layers":1,"decoder_hidden_size":16,"uncertainty":False,"G_go":goi,"G_go_weight":gow,"G_coexpress":coi,"G_coexpress_weight":cow,"device":str(device),"num_genes":N_GENES};ge=GEARS_Model(cfg).to(device);opt=torch.optim.Adam(ge.parameters(),lr=1e-3);best=1e99;state=None;stale=0
    for epoch in range(1,41):
        tr=E107.mse_epoch(ge,train,device,opt);va=E107.mse_epoch(ge,val,device,None);hist.append({"model":"GEARS","epoch":epoch,"train_mse":tr,"validation_mse":va})
        if va<best-1e-7:best=va;stale=0;state={k:v.detach().cpu().clone() for k,v in ge.state_dict().items()}
        else:
            stale+=1
            if stale>=6:break
    ge.load_state_dict(state);ge.to(device).eval();return model,gene_ids,ge,pd.DataFrame(hist),{"n_go_edges":len(go),"n_coexpression_edges":len(co),"n_source_control_cells":nctrl,"scgpt_pretrained_tensors":meta["matched_pretrained_parameter_tensors"]}


def predict(sc_model,gene_ids,ge_model,query,device):
    result={};loader=DataLoader(query,16,shuffle=False)
    with torch.no_grad():
        for batch in loader:
            sp,truth=E65.model_forward(sc_model,batch,gene_ids,device,device.type=="cuda",False);batch=batch.to(device);gp=ge_model(batch);basal=batch.x[:,0].reshape(len(batch.pert),-1)
            for task,split,setting,context,pert,s,g,t,b in zip(batch.pert,batch.split,batch.setting,batch.context,batch.perturbation,sp.detach().cpu().numpy(),gp.detach().cpu().numpy(),truth.detach().cpu().numpy(),basal.detach().cpu().numpy()):result[str(task)]={"split":str(split),"setting":str(setting),"context":str(context),"perturbation":str(pert),"scgpt":s-b,"gears":g-b,"truth":t-b}
    return result


def run_dataset(dataset,device):
    start=time.time();root=OUT/dataset;root.mkdir(parents=True,exist_ok=True);(root/"arrays").mkdir(exist_ok=True);manifest=pd.read_csv(CONTRACT,keep_default_na=False);manifest=manifest[manifest.dataset.eq(dataset)].copy();manifest["context"]=manifest.context.astype(str);manifest["perturbation"]=manifest.perturbation.astype(str);assets=Assets(dataset);all_tasks=[];records=[];pred_arrays={};true_arrays={};histories=[];fold_meta=[]
    for fi,(fold,tasks) in enumerate(manifest.groupby("fold_id",sort=True)):
        seed(SEED+fi);tasks=tasks[~tasks.split.eq("train")|tasks.in_train_fraction_100.astype(bool)].copy();graphs=graphs_for(tasks,assets);source=sorted(tasks.loc[tasks.split.eq("train"),"context"].unique());sc_model,gene_ids,ge_model,history,meta=fit_models(graphs,assets,dataset,source,device);history.insert(0,"fold_id",fold);histories.append(history);result=predict(sc_model,gene_ids,ge_model,graphs["val"]+graphs["test"],device)
        rows=[]
        for key,v in result.items():
            ensemble=(v["scgpt"]+v["gears"])/2;rows.append({"fold_id":fold,"task_id":key,"split":v["split"],"setting":v["setting"],"context":v["context"],"perturbation":v["perturbation"],"risk_model_disagreement":rmse(v["scgpt"],v["gears"]),"baseline_predicted_magnitude":float(np.sqrt(np.mean(ensemble**2)))})
        scored=pd.DataFrame(rows);train_perts=tasks.loc[tasks.split.eq("train"),"perturbation"].value_counts();train_contexts=source;dist=[1-cosine(assets.control(a),assets.control(b)) for i,a in enumerate(assets.contexts) for b in assets.contexts[i+1:]];scale=max(float(np.median([d for d in dist if d>1e-10])),1e-8)
        scored["context_novelty_scaled"]=[0 if c in train_contexts else min((1-max(cosine(assets.control(c),assets.control(x)) for x in train_contexts))/scale,5) for c in scored.context];scored["training_support_count"]=[int(train_perts.get(p,0)) for p in scored.perturbation];scored["perturbation_novelty"]=1/(1+scored.training_support_count)
        val=scored.split.eq("val");scored["risk_disagreement_z"]=robust_z(scored.risk_model_disagreement,scored.loc[val,"risk_model_disagreement"]);scored["predicted_magnitude_z"]=robust_z(scored.baseline_predicted_magnitude,scored.loc[val,"baseline_predicted_magnitude"]);scored["safeconf_frozen_pair_risk"]=scored.risk_disagreement_z+scored.context_novelty_scaled+scored.perturbation_novelty
        yval=[]
        for key in scored.loc[val,"task_id"]:v=result[key];yval.append(np.mean([rmse(v["scgpt"],v["truth"]),rmse(v["gears"],v["truth"])]))
        cal=Ridge(alpha=1,positive=True).fit(scored.loc[val,["risk_disagreement_z","predicted_magnitude_z"]],yval);base=cal.predict(scored[["risk_disagreement_z","predicted_magnitude_z"]]);struct_scale=max(float(np.std(yval)),.05*float(np.mean(yval)),1e-6);scored["safeconf_calibrated_pair_risk"]=base+struct_scale*(scored.context_novelty_scaled+np.maximum(scored.perturbation_novelty-float(scored.loc[val,"perturbation_novelty"].median()),0));scored["test_truth_used_for_score_or_threshold"]=False
        for r in scored[scored.split.eq("test")].itertuples(index=False):
            v=result[r.task_id];errors=[rmse(v["scgpt"],v["truth"]),rmse(v["gears"],v["truth"])];row=r._asdict();row.update({"dataset":dataset,"error_scgpt_rmse":errors[0],"error_gears_rmse":errors[1],"error_two_predictor_mean_rmse":float(np.mean(errors)),"error_two_predictor_max_rmse":float(np.max(errors))});all_tasks.append(row);truth_key=f"E112::{dataset}::{fold}::{r.task_id}::truth";true_arrays[truth_key]=v["truth"].astype(np.float32);n_cells=int(tasks[(tasks.context==r.context)&(tasks.perturbation==r.perturbation)].n_cells.iloc[0])
            for name,vector,error in [("scGPT_context_mean_finetuned",v["scgpt"],errors[0]),("GEARS_context_mean_trainonly_graphs",v["gears"],errors[1])]:
                pred_key=f"E112::{dataset}::{fold}::{r.task_id}::{name}::prediction";pred_arrays[pred_key]=vector.astype(np.float32);records.append({"schema_version":"safeconf_prediction_record_v1","record_id":pred_key.removesuffix("::prediction"),"task_id":r.task_id,"task_key":f"E112::{dataset}::{fold}::{r.task_id}","dataset_name":f"{dataset}_E99_formal512","dataset_group":"external_gene_context_cartesian_formal","fold_id":fold,"split":"test","context":r.context,"perturbation":r.perturbation,"predictor_name":name,"run_type":"formal","gene_panel_id":f"{dataset}_control_only_scgpt512","gene_order_hash":"sha256:"+hashlib.sha256("\n".join(assets.genes).encode()).hexdigest(),"effect_definition":"mean_diff","normalization_id":f"{dataset}_normalize_total_1e4_log1p_same_context_ctrl512_v1","error_normalization":"raw_rmse","predicted_effect_key":pred_key,"true_effect_key":truth_key,"true_error_rmse":error,"true_error_cosine":cosine_error(vector,v["truth"]),"n_cells":n_cells})
        fold_meta.append({"fold_id":fold,"source_contexts":"+".join(source),**meta,"test_truth_used_for_training_calibration_or_score":False});del sc_model,ge_model;torch.cuda.empty_cache();print(f"[E112] {dataset} {fold} complete",flush=True)
    task=pd.DataFrame(all_tasks);rec=pd.DataFrame(records);task.to_csv(root/"TASK_RISK_TABLE.csv",index=False);rec.to_csv(root/"PREDICTION_RECORDS.csv",index=False);pd.concat(histories).to_csv(root/"TRAINING_HISTORY.csv",index=False);pd.DataFrame(fold_meta).to_csv(root/"FOLD_AUDIT.csv",index=False);pd.DataFrame({"raw_gene":assets.raw_genes,"scgpt_token":assets.genes}).to_csv(root/"GENE_PANEL.csv",index=False);np.savez_compressed(root/"arrays/predicted_effects.npz",**pred_arrays);np.savez_compressed(root/"arrays/true_effects.npz",**true_arrays)
    issues=validate_prediction_record_artifacts(root,records=rec,strict=True);pd.DataFrame({"strict_issue":issues}).to_csv(root/"STRICT_ISSUES.csv",index=False);summary=[]
    for (fold,setting),g in task.groupby(["fold_id","setting"]):
        for score in ["safeconf_calibrated_pair_risk","safeconf_frozen_pair_risk","risk_model_disagreement","baseline_predicted_magnitude"]:summary.append({"fold_id":fold,"setting":setting,"score":score,"n_tasks":len(g),"spearman":rho(g[score],g.error_two_predictor_mean_rmse)})
    pd.DataFrame(summary).to_csv(root/"SETTING_SUMMARY.csv",index=False);status={"dataset":dataset,"status":"complete","generated_at":datetime.now().isoformat(timespec="seconds"),"n_folds":int(task.fold_id.nunique()),"n_test_tasks":len(task),"n_records":len(rec),"strict_issue_count":len(issues),"test_truth_used_for_training_calibration_score_or_threshold":False,"wall_seconds":time.time()-start};(root/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n");return status

def validate_existing(dataset):
    root=OUT/dataset;records=pd.read_csv(root/"PREDICTION_RECORDS.csv");issues=validate_prediction_record_artifacts(root,records=records,strict=True);pd.DataFrame({"strict_issue":issues}).to_csv(root/"STRICT_ISSUES.csv",index=False);status=json.loads((root/"RUN_STATUS.json").read_text());status["strict_issue_count"]=len(issues);status["revalidated_at"]=datetime.now().isoformat(timespec="seconds");(root/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n");return status


def aggregate():
    statuses=[];frames=[]
    for d in SPECS:
        p=OUT/d/"RUN_STATUS.json"
        if p.exists():statuses.append(json.loads(p.read_text()));f=pd.read_csv(OUT/d/"TASK_RISK_TABLE.csv");frames.append(f)
    status={"experiment":"E112_external_formal_dual_models","generated_at":datetime.now().isoformat(timespec="seconds"),"status":"complete" if len(statuses)==2 else "partial","datasets":[s["dataset"] for s in statuses],"test_truth_used_for_training_calibration_score_or_threshold":False};(OUT/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    if frames:
        allf=pd.concat(frames,ignore_index=True);allf.to_csv(OUT/"E112_ALL_TASKS.csv",index=False);rows=[]
        for (dataset,fold),g in allf.groupby(["dataset","fold_id"]):
            for score in ["safeconf_calibrated_pair_risk","safeconf_frozen_pair_risk","risk_model_disagreement","baseline_predicted_magnitude"]:rows.append({"dataset":dataset,"fold_id":fold,"score":score,"spearman_mean_error":rho(g[score],g.error_two_predictor_mean_rmse),"spearman_gears_error":rho(g[score],g.error_gears_rmse),"spearman_scgpt_error":rho(g[score],g.error_scgpt_rmse)})
        pd.DataFrame(rows).to_csv(OUT/"E112_FOLD_SUMMARY.csv",index=False)
    (OUT/"E112_REPORT.md").write_text("# E112｜外部五背景正式双模型复制\n\nLara ex vivo 与 Santinha 均使用 E99 冻结合同、control-only 512 基因面板、同背景基础状态、正式微调 scGPT 和按外层训练背景构建共表达图的 GEARS。测试扰动标签只在风险冻结后计算误差。\n");return status


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="all",choices=["all",*SPECS]);p.add_argument("--device",default="cuda:0");p.add_argument("--aggregate-only",action="store_true");p.add_argument("--validate-only",action="store_true");a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.aggregate_only:print(json.dumps(aggregate(),ensure_ascii=False,indent=2));return
    if a.validate_only:
        if a.dataset=="all":raise ValueError("--validate-only requires one dataset")
        print(json.dumps(validate_existing(a.dataset),ensure_ascii=False,indent=2));return
    device=torch.device(a.device if torch.cuda.is_available() else "cpu");selected=list(SPECS) if a.dataset=="all" else [a.dataset]
    for d in selected:print(json.dumps(run_dataset(d,device),ensure_ascii=False,indent=2))
    print(json.dumps(aggregate(),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
