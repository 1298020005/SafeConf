#!/usr/bin/env python3
"""Archive corrected evidence and produce a meeting receipt, never pick on holdout."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import bootstrap_u20,metrics
from tools.scripts.run_safeconf_system_freeze_v1 import macro
RUN=Path('/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1')
DOC=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/review_repair_20261007_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DOC)
    out=ap.parse_args().output;out.mkdir(parents=True,exist_ok=True)
    roots={'source_gate':RUN/'source_gate_v4','public_reliability':RUN/'public_reliability_v3',
           'native_public':RUN/'native_public_current_20261007_v1'}
    files={
        'source_gate':['SOURCE_GATE_RUN_STATUS.json','SOURCE_GATE_METRICS.csv','SOURCE_GATE_PAIRED_BOOTSTRAP.csv','SOURCE_GATE_FIT_AUDIT.csv'],
        'public_reliability':['E258_VALIDATION_DECISION.json','E258_REFERENCE_RELIABILITY.csv',
            'E258_REFERENCE_RELIABILITY_SUMMARY.csv','MCFALINE_DEV_RISK_COMPARISON.csv',
            'MCFALINE_FEATURE_STATUS.json','MCFALINE_DEV_JACKKNIFE_FEATURES.csv','MCFALINE_HOLDOUT_JACKKNIFE_FEATURES.csv'],
        'native_public':['EXECUTION_CONFIG.json','RUN_STATUS.json','MACRO.csv','STRATA.csv','SEED_SUMMARY.csv',
            'PAIRED_BOOTSTRAP.csv','FIT_AND_INFORMATION_LEDGER.csv','ERROR_LABEL_CDF_AUDIT.csv','OOF_CALIBRATION_AUDIT.csv']}
    manifest=[]
    for group,names in files.items():
        (out/group).mkdir(exist_ok=True)
        for name in names:
            src=roots[group]/name;dst=out/group/name
            shutil.copy2(src,dst)
            manifest.append({'group':group,'source':str(src),'file':str(dst.relative_to(out)),
                'bytes':src.stat().st_size,'sha256':sha(src)})
    base=pd.read_parquet(RUN/'system_freeze_v4/SYSTEM_RANKING.parquet')
    native=pd.read_parquet(roots['native_public']/'TASK_PREDICTIONS.parquet')
    # Current native baseline/public pair; seed and budget fixed in advance.
    for b in [.5,1.]:
        for f in ['Native61','Native61_Public']:
            q=native[(native.budget==b)&(native.seed==20260930)&(native.output_type=='raw')&(native.feature_set==f)]
            if q.task_id.duplicated().any() or set(q.task_id)!=set(base.task_id):
                raise ValueError('current comparison lost exact task identity')
            if not np.allclose(q.set_index('task_id').loc[base.task_id].true_error_rmse,base.true_error_rmse,rtol=1e-12,atol=1e-14):
                raise ValueError('current truth mismatch')
            base[f'PertEMA_{f}_{int(b*100)}_corrected']=base.task_id.map(q.set_index('task_id').risk)
    j=pd.read_csv(roots['public_reliability']/'MCFALINE_HOLDOUT_JACKKNIFE_FEATURES.csv').set_index('task_id')
    aligned=base.task_id.map(j.public_mean_jackknife)
    base['PublicMeanJackknife_corrected_diagnostic']=aligned.where(aligned.notna(),base.PublicRule)
    base['J_available']=base.task_id.map(j.jackknife_mean_stability).notna()
    base['FinalSafeConf']=base.PublicRule.to_numpy(float)
    methods=['Amplitude','PublicRule','FinalSafeConf','TargetH1F1_50','TargetX0F2_50','TargetTabPFN50',
        'PertEMA_Native61_50_corrected','PertEMA_Native61_Public_50_corrected',
        'PertEMA_Native61_100_corrected','PertEMA_Native61_Public_100_corrected',
        'PublicMeanJackknife_corrected_diagnostic']
    comparison=[];discovery=[];paired=[]
    for method in methods:
        s=base[method].to_numpy(float)
        comparison.append({'method':method,'n_tasks':len(base),'n_genes':base.gene.nunique(),
            'selection_role':'frozen_default' if method=='FinalSafeConf' else 'fixed_candidate_diagnostic',
            **macro(base,s)})
        ids=base.task_id.astype(str).to_numpy();y=base.true_error_rmse.to_numpy(float)
        k=int(np.ceil(.2*len(base)));hi=np.lexsort((ids,-s))[:k];oracle=np.lexsort((ids,-y))[:k]
        discovery.append({'method':method,'n_tasks':len(base),'review_budget':k,
            'high_error_found':len(set(hi)&set(oracle)),
            'high_error_recall':len(set(hi)&set(oracle))/k,
            'remaining_mean_error':float(np.delete(y,hi).mean())})
        if method in ['PertEMA_Native61_Public_50_corrected','PertEMA_Native61_Public_100_corrected',
                      'PublicMeanJackknife_corrected_diagnostic']:
            paired.append({'method':method,'comparator':'PublicRule',
                **bootstrap_u20(base,s,base.PublicRule.to_numpy(float),5000,20260930)})
    c=pd.DataFrame(comparison);a=pd.DataFrame(discovery)
    c.to_csv(out/'SYSTEM_COMPARISON_CORRECTED.csv',index=False)
    a.to_csv(out/'GLOBAL_20_PERCENT_REVIEW.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'SYSTEM_PAIRED_BOOTSTRAP_CORRECTED.csv',index=False)
    base.to_parquet(out/'SYSTEM_RANKING_CORRECTED.parquet',index=False)
    m=pd.read_csv(roots['source_gate']/'SOURCE_GATE_METRICS.csv')
    source=m[~m.shuffle].copy()
    null=m[m.shuffle&m.method.eq('always_source')].groupby(['source','target'],as_index=False).u20.agg(['mean','min','max']).reset_index()
    source.to_csv(out/'SOURCE_REAL_COMPARISON.csv',index=False)
    null.to_csv(out/'SOURCE_SHUFFLED_SUMMARY.csv',index=False)
    nm=pd.read_csv(roots['native_public']/'MACRO.csv');nm=nm[nm.output_type.eq('raw')]
    budget_table=nm.groupby(['budget','feature_set'],as_index=False).utility20.mean().pivot(index='budget',columns='feature_set',values='utility20').reset_index()
    budget_table['public_rule_no_feedback']=float(c.loc[c.method.eq('PublicRule'),'u20'].iloc[0])
    budget_table.to_csv(out/'NATIVE_PUBLIC_BUDGET_TABLE.csv',index=False)
    decision={'status':'REVIEW_REPAIRS_COMPLETE_RESEARCH_CONTINUES',
        'default_current':'PublicRule','default_selection_changed_using_holdout':False,
        'source_gate_decision':'retain as conditional diagnostic; always_source scores higher on this same-family source DEV',
        'jackknife_decision':'formula and response/reference contracts repaired; J predicts reference deviation but largely duplicates V',
        'pertema_decision':'Native61_Public clearly exceeds same-learner Native61; does not displace no-feedback PublicRule',
        'evidence_identity':'current McFaline and Source are DEV/SEEN; E182 is an independent-study retrospective boundary',
        'full_official_pertema_reproduced':False,'new_download_bytes':0,'new_gpu_hours':0,
        'permanent_test_truth_opened':False,'paper_pdf_work_started':False,
        'next_priority':'test reusable error supervision across a competent independent predictor/study after corrected source configuration freeze; preserve all negative transfer evidence'}
    json_write(out/'COMPONENT_DECISION_CORRECTED.json',decision)
    json_write(out/'REPAIR_STATUS.json',decision|{'source_gate':str(roots['source_gate']),
        'public_reliability':str(roots['public_reliability']),'native_public':str(roots['native_public']),
        'reporting_date':'2026-10-08','incomplete_prior_supervision_admitted':True})
    json_write(DOC.parent/'implementation_v1/REVIEW_REPAIR_STATUS.json',{'status':decision['status'],
        'authoritative_corrections':str(out),'historical_bundle_retained':True,
        'old_source_v3_superseded_by':'source_gate_v4','old_jackknife_superseded_by':'public_reliability_v3',
        'native_public_comparison':'native_public_current_20261007_v1'})
    report='# SafeConf：10月8日汇报摘要\n\n'
    report+='## 当前结论\n\n公共真实实验是目前最稳的风险参照；旧模型错误在同体系预测器之间提供明确增量；加入公共信息能增强较完整的PertEMA特征适配。跨研究仍存在迁移边界，目标反馈目前保留为候选。当前默认继续采用PublicRule。\n\n'
    report+='## 一、上次三项修复的真实状态\n\n'
    report+='- Source：已完成通道训练侧经验秩变换与整簇标签置乱，两个方向、五个置乱种子均有结果；整簇移动率记录在FIT_AUDIT。\n'
    report+='- Jackknife：删除多余平均项，另修正McFaline历史聚合合同；现在J、V及缓存预测距离对应同一参照，且验证了方差一致性。\n'
    report+='- PertEMA：本次已补Native61与Native61＋Public，五个预算、三个种子、3-fold gene OOF，30个配置、120个XGBoost拟合，主种子配对5000次gene bootstrap。\n'
    report+='- 交接事实：10月5日两项重算已落盘，之后没有持续SafeConf训练，第三项未及时完成；本次补齐并更新Git，不把上次阶段标记当成投稿目标完成。\n\n'
    report+='## 二、修正后的Source比较\n\n| 方向 | 任务／gene clusters | 公共秩通道U20 | 全Source U20 | 条件切换U20 |\n|---|---:|---:|---:|---:|\n'
    for (src,tgt),part in source.groupby(['source','target']):
        vals=part.set_index('method').u20
        report+=f'| {src} → {tgt} | 1808／575 | {vals.always_public:.3f} | {vals.always_source:.3f} | {vals.selected_gate:.3f} |\n'
    report+='\n这是完整外层gene留出的同family跨预测器实验。真实Source模型超过五个整簇置乱对照。条件切换低于始终Source，因此gate没有额外算法价值。公共分数为各外层训练侧转换后的通道，和旧原始距离值分开记录。\n\n'
    report+='## 三、PertEMA适配加入Public的实际增量\n\n以下为同一212任务、152个gene clusters、同一反馈记录、同一官方树配方。表格为三个种子的U20均值；原始每次结果与主种子CI分别保存。\n\n| 反馈gene预算 | Native61 | Native61＋Public | 无反馈PublicRule |\n|---|---:|---:|---:|\n'
    for row in budget_table.itertuples(index=False):
        report+=f'| {int(row.budget*100)}% | {row.Native61:.3f} | {row.Native61_Public:.3f} | {row.public_rule_no_feedback:.3f} |\n'
    report+='\n五个预算中，主种子Native61＋Public相对Native61的配对CI均为正。PublicRule仍是更强起点，不因反馈学习器使用了标签就自动替换它。\n\n'
    report+='实现另落实两项公平条件：native_prediction_abs_mean重算自当前冻结预测；原来全缺失的training similarity在每个训练分区内重建。两个输入版本都使用相同gene等权。CD4特征转为McFaline控制特征、donor variance替换plate proxy、Pearson目标替换RMSE/CDF等仍是适配；本次没有复现完整官方conformal流程。\n\n'
    report+='## 四、固定复核20%的实际用途\n\n'
    p=a[a.method.eq('PublicRule')].iloc[0];amp=a[a.method.eq('Amplitude')].iloc[0]
    report+=f'当前212任务，全局复核{int(p.review_budget)}项：PublicRule发现真实最高误差任务{int(p.high_error_found)}项，幅度基线{int(amp.high_error_found)}项，多发现{int(p.high_error_found-amp.high_error_found)}项。排除后剩余平均误差相对幅度降低{100*(amp.remaining_mean_error-p.remaining_mean_error)/amp.remaining_mean_error:.2f}%。\n\n'
    report+='U20主表按context等权汇总；上述收益为全部任务统一排序。两种评价单列，避免把context内筛选与全局复核混为同一个数字。2,993任务混合历史队列的旧冻结回顾另保留：公共方法相对幅度多发现33项严重错误。\n\n'
    report+='## 五、可靠性和独立研究边界\n\n'
    report+='E258的1,530个合法验证记录用于独立donor参照偏差诊断。修正J仍与偏差正相关，但与历史分散度V的Spearman约0.9998；因此没有把它包装成新可靠性核心。McFaline候选按正确效应聚合重新计算。\n\n'
    report+='E182有36/40任务、18/20 genes可评分，公共规则没有稳定超过幅度；SAMS已完成不同预测机制检验，但Source尚未稳定超过强公共规则。这些结果用于定位迁移条件；当前没有取得新的未见研究正向确认。\n\n'
    report+='## 六、建议明天口头汇报\n\n> 我们研究的是：一个新扰动预测器尚无自身错误记录时，能否借助公共真实实验和旧模型错误，在有限复核预算下发现严重错误。目前公共参照在McFaline上有明确实际收益；加入公共历史能显著增强现有错误学习器；真实旧模型错误在GAT与Exphormer双向迁移中超过公共和置乱对照。当前采用公共规则作为默认，Source和Target按场景验证后启用。下一步集中补不同family／研究的迁移条件和冻结后确认，不再堆新网络。\n\n'
    report+='## 七、下一项具体工作\n\n1. 将已完成SAMS同任务结果与修正Source证据绑定，明确同family和跨family两种证据层级。\n2. 从已登记资产中锁定一个预测能力合格且信息合同闭合的外研究迁移验证；无新大型训练。\n3. 先登记任务、Public来源、Source错误与采用门，再评分；保留E182负结果。\n4. 正文和PDF继续暂停。\n\n'
    report+='## 复现\n\n```bash\nOMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python tools/scripts/run_safeconf_native_public_current_v1.py --output /home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/native_public_current_reproduce_v1\nOMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python tools/scripts/build_safeconf_meeting_receipt_20261008.py\n/home/miniconda/bin/python -m unittest tools.tests.test_safeconf_review_repair -v\n```\n'
    (out/'REPORT_FOR_20261008.md').write_text(report)
    (out/'FAILURE_AND_ACTIONS.md').write_text('''# 修复记录

- Source v3混合不同尺度与逐行置乱：v4改为训练侧通道CDF与兼容整gene block交换，重跑。
- Jackknife额外除以m：改为sum，等权样本方差／n回归测试通过。
- 后续核查发现McFaline J沿用旧effect aggregation，而D来自cell-weighted reference：v3分别复用DEV和holdout的精确合同，并断言V匹配。
- Native61旧预测幅度与当前预测轴不同，similarity列全缺失：本轮统一到当前预测，similarity仅训练分区重建，两版本同样修正。
- XGBoost/sklearn wrapper模型保存异常：改存native Booster；不改变模型拟合、参数或预测。
- 先前未落实连续监管和补全第三项：本回执如实记录空档，不声称持续运行。
''')
    pd.DataFrame(manifest).to_csv(out/'ARTIFACT_MANIFEST.csv',index=False)
    print(c.to_string(index=False));print(a.to_string(index=False));print(str(out),flush=True)


if __name__=='__main__':
    main()
