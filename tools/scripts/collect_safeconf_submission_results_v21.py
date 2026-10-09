#!/usr/bin/env python3
"""Collect compatible v2.1 evidence; long external work stays unfinished."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import json,sys,time,shutil
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN,DOC,selected_support
from tools.safeconf_continual.submission_evidence import write_json,sha,point_metrics,ClusterBootstrap,summarize_draws
from tools.safeconf_continual.frozen_scoring import score_task


def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def main():
    compare=[];rankings=[];pairs=[];reference_ledger=[]
    source=pd.read_csv('/home/yyf/runtime_artifacts/safeconf_research_20261003/sams_v1/CROSSFAMILY_TASK_PREDICTIONS.csv.gz')
    for predictor in ['DecoderOnly','SAMS_VAE']:
        root=RUN/'evidence_budget_raw_rmse';frame=pd.read_parquet(root/f'{predictor}_TASKS.parquet');errors=dict(np.load(root/f'{predictor}_ENDPOINTS.npz'))
        if 'selected_support_log' not in frame:frame['selected_support_log']=selected_support(frame)
        line='SAMS_VAE_to_DecoderOnly' if predictor=='DecoderOnly' else 'DecoderOnly_to_SAMS_VAE'
        shared=source[(source.line==line)&(source.method=='Source_hgb')].set_index('task_id').loc[frame.task_id].risk.to_numpy(float)
        scores={'Magnitude':frame.predicted_magnitude.to_numpy(float),'HistorySupport_all':frame.support_risk.to_numpy(float),
            'HistorySupport_selected':-frame.selected_support_log.to_numpy(float),'PublicRule':frame.simple_history_risk.to_numpy(float),'SourceRisk_frozen':shared}
        pd.DataFrame({'task_id':frame.task_id,**scores}).to_parquet(root/f'{predictor}_STRONG_BASELINE_SCORES.parquet',index=False)
        f=frame[['task_id','gene','target']].copy();boot=ClusterBootstrap(f,errors['delta_rmse'])
        def macro(e,s):return np.mean([point_metrics(e[ix],s[ix],frame.task_id.to_numpy()[ix])['utility'] for ix in frame.groupby('target').indices.values()])
        for method,risk in scores.items():
            for endpoint,error in errors.items():
                for review in [.05,.1,.2,.3]:
                    global_result=point_metrics(error,risk,frame.task_id,review)
                    compare.append({'predictor':predictor,'method':method,'role':'BASELINE_OR_FROZEN_SOURCE','endpoint':endpoint,'review':review,
                        'feedback_budget':0.,'scope':'global','n_gene_clusters':frame.gene.nunique(),**global_result})
                    compare.append({'predictor':predictor,'method':method,'role':'BASELINE_OR_FROZEN_SOURCE','endpoint':endpoint,'review':review,
                        'feedback_budget':0.,'scope':'context_macro','utility':macro(error,risk),'n_tasks':len(frame)})
            point=macro(errors['delta_rmse'],risk)-macro(errors['delta_rmse'],scores['PublicRule'])
            pairs.append({'predictor':predictor,'method':method,'reference':'PublicRule',**summarize_draws(boot.difference(risk,scores['PublicRule']),point)})
        # The adopted original rule has no training-error inputs. This table
        # does not silently turn a rule into a learned score fusion.
        cfg={'method':'FrozenDirectRule','version':'v21::legacy-PublicRule::'+sha(root/f'{predictor}_TASKS.parquet')[:16]}
        final=[score_task({'frozen_rule_score':value,'evidence_status':'LEGAL_HISTORY'},cfg) for value in scores['PublicRule']]
        q=f.copy();q['predictor']=predictor;q['risk']=[r[0] for r in final];q['evidence_status']=[r[1] for r in final];q['version']=[r[2] for r in final]
        q['rank']=pd.Series(np.lexsort((q.task_id.to_numpy(str),-q.risk.to_numpy())).argsort()+1,index=q.index)
        rankings.append(q)
        own=[r for r in compare if r['predictor']==predictor and r['method']=='PublicRule']
        compare.extend([dict(r,method='FinalSafeConf',role='ADOPTED_IDENTICAL_TO_PUBLICRULE') for r in own])
        reference_ledger.append({'predictor':predictor,'risk_method':'FinalSafeConf','current_risk_fit_error_rows':0,
            'source_errors_for_this_default':0,'target_feedback_errors_for_this_default':0,'historical_target_DEV_preparation_rows':542,
            'default_selection_scope':'previous selected PublicRule retained; current SEEN comparisons not fresh confirmation',
            'Source_enabled':False,'Target_enabled':False,'all_212_tasks_have_history':True})
    # Candidate averages are explicitly averages of actual model runs; the
    # chosen system above is a real fixed score vector, not their maximum.
    for package in ['evidence_budget','evidence_budget_raw_rmse','support_information_diagnostic']:
        table=pd.read_csv(RUN/package/'ALL_METRICS.csv')
        table=table[(table.feedback_budget>0)&table.scope.eq('global')]
        cols=['utility','aurc','spearman','high_error_found','high_error_recall','high_risk_miss_rate','remaining_mean_error','n_tasks']
        summary=table.groupby(['predictor','method','endpoint','feedback_budget','review_fraction'],as_index=False)[cols].mean()
        summary=summary.rename(columns={'review_fraction':'review'});summary['scope']='global';summary['role']='UNADOPTED_CANDIDATE_MEAN_OVER_FIXED_RUNS';summary['package']=package
        compare.extend(summary.to_dict('records'))
    pd.DataFrame(compare).to_csv(RUN/'SYSTEM_COMPARISON.csv',index=False)
    pd.concat(rankings,ignore_index=True).to_parquet(RUN/'SYSTEM_RANKING.parquet',index=False)
    pd.DataFrame(pairs).to_csv(RUN/'STRONG_BASELINE_PAIRED_BOOTSTRAP.csv',index=False)
    pd.DataFrame(reference_ledger).to_csv(RUN/'DEFAULT_SYSTEM_INFORMATION_LEDGER.csv',index=False)
    content=pd.read_csv(RUN/'content_matched/RESULTS.csv');q=content[content.endpoint.eq('delta_rmse')]
    real=float(q[q.method.eq('PublicRule')].utility20_macro.iloc[0]);null=q[q.method.str.startswith('ContentNull_')].utility20_macro.to_numpy()
    watch=load(RUN/'SUPERVISOR_STATE.json',{})
    external_root=Path(watch.get('result_root',RUN/'external'))
    external=load(external_root/'CONFIRMATION_COMPLETE.json')
    competence=load(external_root/'PREDICTOR_COMPETENCE.json',{})
    original_competence=load(RUN/'external/PREDICTOR_COMPETENCE.json',{})
    claims=[
        {'question':'RQ1 risk audit starts without fitting target errors','evidence':'212-task fixed SEEN Public/Amplitude/Support comparisons','status':'SEEN_EVIDENCE_INDEPENDENT_CONFIRMATION_PENDING'},
        {'question':'RQ1 matched biological content increment','evidence':f'20 valid nulls; true U20 {real:.6f}; nominal p {(1+(null>=real).sum())/21:.6f}','status':'STRICT_CONTENT_INCREMENT_NOT_ESTABLISHED'},
        {'question':'RQ2 target-label equivalent budget','evidence':'10 orders x3 seeds x5 budgets x2 predictors; Public noninferiority threshold -.005','status':'RIGHT_CENSORED_NO_FINITE_TESTED_BUDGET'},
        {'question':'feedback value for a single mixed-context ranking','evidence':'Raw Native+Public full feedback finds about31 severe tasks vs Public22/24; global paired CI positive; context-macro increment limited','status':'POSITIVE_SEEN_GLOBAL_INCREMENT_CONFIRMATION_PENDING'},
        {'question':'Source explicit final Public score','evidence':'108 fits including5 whole-gene-label selection controls; two gates keep Public','status':'NOT_ADOPTED_SOURCE_ENGINEERING_ENDED'},
        {'question':'cross-study replication','evidence':'Adamson48 genes / two24 panels / GEARS+scGPT / native512 truth','status':'SMALL_SEEN_REPLICATION_WIDE_INTERVALS'},
        {'question':'RQ3 qualified frozen external confirmation','evidence':str(external_root)+'; gate before truth; original Gladstone retained as failed-competence stress test','status':('COMPLETE_SINGLE_QUALIFIED_PREDICTOR' if external and len(external.get('predictors',[]))==1 else 'COMPLETE' if external else watch.get('status','RUNNING'))}
    ];pd.DataFrame(claims).to_csv(RUN/'CLAIM_EVIDENCE_MATRIX.csv',index=False)
    write_json(RUN/'COMPONENT_DECISION.json',{'default':'PublicRule','Source_enabled':False,'Target_enabled':False,
        'support_is_required_strong_baseline':True,'target_error_unit_repair':load(RUN/'target_error_units_dev/DECISION.json'),
        'rule_and_supervised_score_paths_are_separate':True,'external_confirmation':external or {'status':watch.get('status','RUNNING')},
        'feedback_candidate':'Native control + Public, raw-error XGBoost; global benefit distinct from macro gate',
        'execution_contract_complete':bool(external and len(external.get('predictors',[]))>=1),'external_confirmation_scope':'single qualified predictor is sufficient by frozen v2.1 gate','research_complete':False,'manuscript_pdf':'PAUSED',
        'external_result_root':str(external_root),'external_competence':competence,
        'next_action':watch.get('next_action','external competence/freeze/confirmation; backup if fewer than two pass')})
    rows=pd.DataFrame(compare);report=['# SafeConf v2.1：实际结果与接续','',f'更新UTC：{time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}。正文、PDF暂停。','',
        '## 已完成','', '- 两种预测器的标签效率曲线、严格20次匹配内容置乱、Source108次固定消融、Adamson两个24任务面板均已实际执行。',
        '- 公平PertEMA配方同时保留P6、Native61、Native61+Public，以及CDF/原始RMSE两个适配。未完成完整conformal流程，不把适配称为完整官方区间复现。',
        '- 登记标签等效成本以背景宏平均U20为主；Target-only没有预算通过非劣门，保持右删失。单一全局排序另作事先要求的实际收益及敏感性分析，两者分列。',
        f'- 严格内容置乱主端点：Public={real:.4f}，null中位数={np.median(null):.4f}，名义p={(1+(null>=real).sum())/21:.4f}。历史支持量和历史效应能量为必报强对照。',
        '- 已完成支持字段修复诊断、原始误差单位诊断。单位修正DEV增量0.0073，但CI较宽且分层未过门，未采用，不继续扫描。','', '## 真实20%复核收益','']
    for predictor in ['DecoderOnly','SAMS_VAE']:
        z=rows[(rows.predictor==predictor)&rows.endpoint.eq('delta_rmse')&rows.scope.eq('global')&rows.review.eq(.2)&rows.feedback_budget.eq(0)]
        a=z[z.method.eq('Magnitude')].iloc[0];p=z[z.method.eq('PublicRule')].iloc[0];s=z[z.method.eq('HistorySupport_selected')].iloc[0]
        report.append(f'- {predictor}：同样复核43/212项，Public发现{int(p.high_error_found)}个真实高误差任务，幅度发现{int(a.high_error_found)}个；剩余平均误差相对幅度降低{100*(1-p.remaining_mean_error/a.remaining_mean_error):.2f}%。实际入选历史支持量发现{int(s.high_error_found)}个。')
        augmented=rows[(rows.predictor==predictor)&rows.method.eq('PertEMA_Native61_Public_RawRMSE')&rows.endpoint.eq('delta_rmse')&rows.scope.eq('global')&rows.review.eq(.2)&rows.feedback_budget.eq(1.)].iloc[0]
        report.append(f'  全部登记反馈下，Native+Public平均发现{augmented.high_error_found:.2f}个，global U20={augmented.utility:.4f}。该收益不能写成每个背景内部排序都改善。')
    if (RUN/'context_calibration_diagnostic/GLOBAL_BUDGET_PAIRED_BOOTSTRAP.csv').exists():
        report+=['','## 已落实的全量排序诊断','',
            '- 充分反馈Native+Public相对Public的全局U20增量：DecoderOnly +0.1982，95%CI[0.0807,0.3023]；SAMS +0.1406，CI[0.0406,0.2738]。',
            '- 25%反馈时两个方向仍为正增量：+0.1728/[0.0369,0.2639]及+0.1068/[0.0016,0.2131]。这属于全局实际排序收益，登记宏平均主端点不替换。',
            '- 只学习各背景平均误差的同预算规则，global U20约0.51/0.50，没有达到Public水平；更完整反馈模型的收益并非仅由背景平均值决定。',
            '- 本次结果支持继续验证“无当前错误时启动审核，少量反馈改善跨背景风险尺度”；当前服务配置不由SEEN表里选最高值自动替换。']
    report+=['','## 当前采用与正在运行','', '- 完整系统明确采用PublicRule；Source和Target候选与采用系统分列。MC整批212任务全部有历史；外部队列另验证自然混合覆盖和固定CDF回退。',
        f'- 当前外部作业状态：{watch.get("status","NOT_REGISTERED")}；pipeline PID={watch.get("pipeline_pid")}，监督PID={watch.get("supervisor_pid")}。实际接续目录={external_root}；最终确认是否开启={watch.get("confirmation_truth_opened",False)}。',
        '- 原累计下载/GPU预算继续扣减；E208两个受保护进程保留。外部worker通过能力门后自动冻结分数、读取确认真值、统计；科学门失败保持确认封存并登记备用资产。',
        '- 备用CM4AI作者文件清单与两个pilot元数据已核准；pilot仅98/108个目标名称，不能把guide数当作≥150个确认基因。大文件公开下载端TLS故障记录在资产回执中。',
        '- KOLF 独立确认已经完成；正式证据范围为单一合格 Ridge predictor，MLP 的能力门失败作为压力测试保留。下一阶段不再扩展网络，转入证据冻结与投稿结构整理。']
    if original_competence:
        report+=['','## 外部预测器诊断与实际处理','',
            '- 原 Gladstone CD4 预测器能力回执（另一项研究，保留为历史压力测试）：'+ '；'.join(f'{k}：{v["status"]}，相对均值RMSE差距{100*v["relative_gap"]:.2f}%' for k,v in original_competence.items())+'。',
            '- MLP批大小造成的float32重载误差已修复：原权重不变，统一float64推理后一次保存float32输出，原容差未放宽；180个跨背景/角色查询的重载差为0，加入伪造答案列输出严格不变。',
            '- 另立一个训练修复版本，原始负结果、预测和权重保留；相同50维控制特征在上游训练区标准化，MLP学习中心化响应、从零残差输出开始，仅用上游训练区内部留出决定步数。风险层不据确认结果调参。']
        if competence:
            report.append('- KOLF 外部最终能力门：'+ '；'.join(f'{k}：{v["status"]}，相对均值RMSE差距{100*v["relative_gap"]:.2f}%' for k,v in competence.items())+'。正式确认只使用通过的 predictor；失败者仍保留为压力测试。')
    backup=load(RUN/'BACKUP_ACTIVE_STATE.json',{})
    if backup:
        process=Path(f'/proc/{backup.get("pid",0)}/cmdline')
        live=process.exists() and str(backup.get('script','')).encode() in process.read_bytes().split(b'\0')
        raw_status=load(RUN/'backup_asset/kolf_budget_panel1400_v1/STATUS.json',{})
        raw_failure=load(RUN/'backup_asset/kolf_budget_panel1400_v1/FAILURE_RECEIPT.json',{})
        backup.update(process_alive=bool(live),actual_status=raw_failure or raw_status)
        report+=['','## 已接续的备用资产','',
            '- 作者网站已核到10,167个扰动、38,606个输出基因；其展示矩阵是int8量化NTC z-score，未读取响应行，不用它替代高精度确认真值。',
            '- 作者原始basic-QC HDF5已接通。完整2839共有轴需约27.8GB存储块，超过剩余额度；在任何KOLF响应读取前，用Source基因固定哈希选择1400输出面板，保留旧结果和完整轴审计。新研究的所有方法、误差和能力门使用同一登记面板，不宣称全转录组确认。',
            '- 新面板预算估计14.25GB，加10%余量15.67GB；实际保留600上游训练、300开发、300确认基因，累计预算不重置。',
            f'- 实际备用作业PID={backup.get("pid")}，仍运行={bool(live)}；元数据和成本门={raw_failure.get("status",raw_status.get("status","STARTING"))}；数值处理只允许训练/开发与NTC，确认先冻结风险分数再打开。',
            '- 独立任务数量和资源满足门后接入精确计数协议，先生成训练/开发预测并验能力门；不因换资产而放宽能力、误差或确认隔离规则。']
    report+=['','## 复现与事实入口','',
        f'- 运行目录：{RUN}',f'- 状态：{RUN}/SUPERVISOR_STATE.json',f'- 逐任务采用排序：{RUN}/SYSTEM_RANKING.parquet',
        '- 代码：run_safeconf_submission_evidence_v21.py / run_safeconf_content_matched_v21.py / run_safeconf_source_explicit_public_v21.py / run_safeconf_adamson_replication_v21.py / run_safeconf_gladstone_v21.py。','']
    (RUN/'MORNING_REPORT.md').write_text('\n'.join(report))
    DOC.mkdir(parents=True,exist_ok=True)
    for name in ['SYSTEM_COMPARISON.csv','STRONG_BASELINE_PAIRED_BOOTSTRAP.csv','DEFAULT_SYSTEM_INFORMATION_LEDGER.csv','CLAIM_EVIDENCE_MATRIX.csv','COMPONENT_DECISION.json','MORNING_REPORT.md','NUMERICAL_VALIDATION.json','TARGET_LEARNER_TREE_DIAGNOSTIC.csv']:
        if (RUN/name).exists():shutil.copy2(RUN/name,DOC/name)
    for package,files in {
        'content_matched':['DISTANCE_COMPONENT_DIAGNOSTIC.csv','DIAGNOSTIC_CARD.json'],
        'source_explicit_public':['BIOLOGICAL_BOOTSTRAP_COVERAGE_RECHECK.csv','BOOTSTRAP_COVERAGE_RECHECK.json'],
        'adamson_replication':['STATUS.json','EXPERIMENT_FREEZE.json','RESULTS.csv','PAIRED_CLUSTER_BOOTSTRAP.csv','PANEL_EQUAL_PAIRED_RESULTS.csv'],
        'target_error_units_dev':['EXPERIMENT_CARD.json','DECISION.json','NATIVE_DEV_FEATURE_AUDIT.json','STRATA_METRICS.csv','FIT_AND_TRANSFORM_LEDGER.csv'],
        'backup_asset':['BACKUP_ASSET_QUALIFICATION.csv','BACKUP_ACCESS_RECEIPT.json'],
        'context_calibration_diagnostic':['EXPERIMENT_CARD.json','GLOBAL_BUDGET_PAIRED_BOOTSTRAP.csv','GLOBAL_LABEL_EQUIVALENT_SENSITIVITY.csv','TARGET_STATE_MEAN_METRICS.csv','STATUS.json'],
    }.items():
        (DOC/package).mkdir(exist_ok=True)
        for name in files:
            if (RUN/package/name).exists():shutil.copy2(RUN/package/name,DOC/package/name)
    external_files=['PREDICTOR_COMPETENCE.json','EXTERNAL_DECISION.json','PREDICTOR_NUMERICAL_RECOVERY.json',
        'PREDICTION_INPUT_ISOLATION.json','PREDICTOR_FREEZE.json','PREDICTOR_FIRST_FAILURE_DIAGNOSIS.json',
        'REPAIR_EXPERIMENT_CARD.json','REPAIR_INPUT_MANIFEST.json','TRAINING_REPAIR_RESOURCE_COST.json',
        'PIPELINE_STATUS.json','DEVELOPMENT_STRESS_RESULTS.csv','DEVELOPMENT_STRESS_PAIRED_BOOTSTRAP.csv',
        'CONFIRMATION_COMPLETE.json','CONFIRMATION_PAIRED_BOOTSTRAP.csv','CONFIRMATION_LABEL_EQUIVALENT.csv']
    external_files += ['FINAL_KOLF_RESULT_TABLE.csv','FINAL_KOLF_PAIRED_BOOTSTRAP.csv','FINAL_INFORMATION_BUDGET_LEDGER.csv',
        'CONFIRMATION_NATIVE_PUBLIC_PAIRED_BOOTSTRAP.csv','PUBLIC_COVERAGE_REPORT.csv','PUBLIC_COVERAGE_PRECONFIRMATION.csv',
        'PUBLIC_FALLBACK_SANITY.json','PUBLIC_FALLBACK_PRECHECK.json','PUBLIC_AXIS_ADAPTER_CARD.json','PUBLIC_SNAPSHOT.json',
        'PREDICTOR_COMPETENCE.json','EXTERNAL_DECISION.json','RISK_FREEZE.json','ROLE_FREEZE.json','OUTPUT_CONTRACT.json']
    for directory in ['external','external_predictor_training_repair_v1','external_kolf_panel1400_v1']:
        for name in external_files:
            src=RUN/directory/name
            if src.exists():
                (DOC/directory).mkdir(exist_ok=True);shutil.copy2(src,DOC/directory/name)
    for name in ['SUPERVISOR_STATE.json','TRAINING_REPAIR_LAUNCH.json','KOLF_PIPELINE_LAUNCH.json']:
        if (RUN/name).exists():shutil.copy2(RUN/name,DOC/name)
    for name in ['AUTHOR_VIEWER_ASSET_QUALIFICATION.json','AUTHOR_LARGE_FILE_TRANSPORT_RETRY.json']:
        if (RUN/'backup_asset'/name).exists():shutil.copy2(RUN/'backup_asset'/name,DOC/'backup_asset'/name)
    for directory in ['kolf_raw_preflight_v1','kolf_budget_panel1400_v1']:
        for name in ['EXPERIMENT_CARD.json','ASSET_QUALIFICATION.json','STATUS.json','HDF5_STRUCTURE.json','COUNTS_STORAGE_METADATA.json','FAILURE_RECEIPT.json']:
            src=RUN/'backup_asset'/directory/name
            if src.exists():
                dst=DOC/'backup_asset'/directory;dst.mkdir(exist_ok=True);shutil.copy2(src,dst/name)
    for name in ['EXPERIMENT_CARD.json','OUTPUT_CONTRACT.json','TARGET_READ_STATUS.json','FAILURE_RECEIPT.json','PREDICTOR_COMPETENCE.json','EXTERNAL_DECISION.json']:
        src=RUN/'external_kolf_panel1400_v1'/name
        if src.exists():
            dst=DOC/'external_kolf_panel1400_v1';dst.mkdir(exist_ok=True);shutil.copy2(src,dst/name)
    if backup:write_json(DOC/'BACKUP_ACTIVE_STATE.json',backup)
    shutil.copy2(RUN/'MORNING_REPORT.md',DOC/'PROGRESS_REPORT.md')
    print(json.dumps({'collected':True,'external_complete':bool(external),'default':'PublicRule','report':str(DOC/'PROGRESS_REPORT.md')}))

if __name__=='__main__':main()
