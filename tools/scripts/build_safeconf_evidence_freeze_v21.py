#!/usr/bin/env python3
"""Freeze actual SafeConf evidence after KOLF scoring, without new fits.

Only already-saved task scores, endpoints and receipts are read.  This script
does not fetch datasets, refit predictors, alter risk scores or open raw truth.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse,json,shutil,sys,time
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.submission_evidence import sha,write_json,point_metrics,ClusterBootstrap,summarize_draws
RUN=Path('/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21')
EXT=RUN/'external_kolf_panel1400_v1'
DEFAULT=ROOT/'docs/研究推进/20261009_投稿证据_v21/evidence_freeze_v1'
FR_DOC=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/frangieh_cross_family_v1'


def load(path):return json.loads(Path(path).read_text())


def verify_confirmation():
    freeze=load(EXT/'RISK_FREEZE.json');complete=load(EXT/'CONFIRMATION_COMPLETE.json')
    if not complete.get('predictors') or complete.get('risk_configuration_changed_after_confirmation'):
        raise RuntimeError('qualified unchanged confirmation receipt required')
    for filename,key in [('FROZEN_CONFIRMATION_SCORES.parquet','confirmation_score_sha256'),
        ('PREDICTOR_FREEZE.json','predictor_freeze_sha256'),
        ('PUBLIC_COVERAGE_PRECONFIRMATION.csv','public_coverage_preconfirmation_sha256'),
        ('PUBLIC_FALLBACK_PRECHECK.json','public_fallback_precheck_sha256')]:
        if sha(EXT/filename)!=freeze[key]:raise RuntimeError(f'changed frozen artifact: {filename}')
    event=load(EXT/'CONFIRMATION_EVALUATION_OPEN_EVENT.json')
    if event['risk_freeze_sha256']!=sha(EXT/'RISK_FREEZE.json'):raise RuntimeError('open event refers to another freeze')
    if load(EXT/'PUBLIC_FALLBACK_SANITY.json')['status']!='PASS':raise RuntimeError('fallback verification failed')
    scores=pd.read_parquet(EXT/'FROZEN_CONFIRMATION_SCORES.parquet')
    tasks=pd.read_parquet(EXT/'CONFIRMATION_TASKS.parquet')
    if tasks.task_id.duplicated().any() or len(tasks)!=300 or tasks.gene.nunique()!=300:
        raise RuntimeError('confirmation identity registry is not the frozen 300 genes')
    return freeze,complete,scores,tasks


def paired_mc(out):
    """Same Native inputs with and without Public, all existing model runs."""
    root=RUN/'evidence_budget_raw_rmse';pred=pd.read_parquet(root/'PREDICTIONS.parquet');rows=[]
    for predictor in ['DecoderOnly','SAMS_VAE']:
        f=pd.read_parquet(root/f'{predictor}_TASKS.parquet');error=np.load(root/f'{predictor}_ENDPOINTS.npz')['delta_rmse']
        ids=f.task_id.to_numpy(str);engine=ClusterBootstrap(f,error);cache={}
        def evaluate(s,scope):
            key=(scope,tuple(np.lexsort((ids,-s))))
            if key not in cache:cache[key]=engine.utility(s,macro=scope=='context_macro')
            if scope=='global':point=point_metrics(error,s,ids)['utility']
            else:point=np.mean([point_metrics(error[ix],s[ix],ids[ix])['utility'] for ix in f.groupby('target').indices.values()])
            return point,cache[key]
        q=pred[pred.predictor.eq(predictor)]
        for budget in [.1,.25,.5,.75,1.]:
            source={}
            for method in ['PertEMA_Native61_RawRMSE','PertEMA_Native61_Public_RawRMSE']:
                source[method]={}
                for (order,seed),part in q[q.method.eq(method)&q.feedback_budget.eq(budget)].groupby(['order_seed','learner_seed'],sort=True):
                    if part.task_id.duplicated().any():raise RuntimeError('duplicate paired task scores')
                    source[method][(order,seed)]=part.set_index('task_id').loc[ids].risk_score.to_numpy(float)
            a=source['PertEMA_Native61_Public_RawRMSE'];b=source['PertEMA_Native61_RawRMSE']
            if not a or set(a)!=set(b):raise RuntimeError('paired Native/Public algorithm runs differ')
            for scope in ['context_macro','global']:
                point=[];draw=[]
                for key in sorted(a):
                    ap,ad=evaluate(a[key],scope);bp,bd=evaluate(b[key],scope)
                    point.append(ap-bp);draw.append(ad-bd)
                rows.append({'study':'McFaline','predictor':predictor,'feedback_budget':budget,
                    'scope':scope,'endpoint':'delta_rmse','review_fraction':.2,
                    'comparison':'Native61_Public-minus-Native61','n_tasks':len(f),
                    'n_gene_clusters':f.gene.nunique(),'algorithm_runs':len(a),
                    **summarize_draws(np.mean(draw,axis=0),float(np.mean(point)))})
            print(json.dumps({'paired_saved_scores':predictor,'budget':budget}),flush=True)
    pd.DataFrame(rows).to_csv(out/'MC_NATIVE_PUBLIC_PAIRED_BOOTSTRAP.csv',index=False)


def summary(out,ext):
    mc=pd.read_csv(RUN/'evidence_budget_raw_rmse/ALL_METRICS.csv')
    mc['study']='McFaline';ext=ext.copy();ext['study']='KOLF'
    all_rows=pd.concat([mc[mc.scope.eq('global')],ext[ext.scope.eq('global')]],ignore_index=True)
    columns=['utility','aurc','spearman','high_error_found','high_error_recall','high_risk_miss_rate','remaining_mean_error','n_tasks','review_k','severe_k']
    keys=['study','predictor','endpoint','method','feedback_budget','review_fraction']
    table=all_rows.groupby(keys,as_index=False)[columns].mean()
    table['algorithm_runs']=all_rows.groupby(keys).size().to_numpy()
    table['task_role']=np.where(table.study.eq('KOLF'),'FROZEN_INDEPENDENT_CONFIRMATION','SEEN')
    table['method_role']=np.where(table.method.eq('PublicRule'),'ADOPTED_FROZEN_SYSTEM',
        np.where(table.feedback_budget.gt(0),'UNADOPTED_REGISTERED_CANDIDATE','FIXED_BASELINE'))
    table.to_csv(out/'UNIFIED_BENCHMARK_TABLE.csv',index=False)
    extkeys=['predictor','endpoint','scope','method','feedback_budget','review_fraction']
    stability=ext.groupby(extkeys,as_index=False)[columns].agg(['mean','median','min','max'])
    stability.columns=['_'.join(str(c) for c in x if c) if isinstance(x,tuple) else x for x in stability.columns]
    stability.to_csv(out/'KOLF_ALGORITHM_STABILITY.csv',index=False)
    fr_macro=pd.read_csv(FR_DOC/'MACRO_RESULTS.csv')
    fr_boot=pd.read_csv(FR_DOC/'PAIRED_BOOTSTRAP.csv')
    fr_comp=pd.read_csv(FR_DOC/'UPSTREAM_COMPETENCE.csv')
    keep_macro=fr_macro[(fr_macro.method.isin(['PublicHGB','WeightedHistoryDistance','Magnitude'])) &
                        (fr_macro.scope.isin(['all_test','heldout_context'])) &
                        (fr_macro.aggregation.eq('risk_fold'))]
    keep_boot=fr_boot[(fr_boot.method.eq('PublicHGB')) &
                      (fr_boot.baseline.isin(['Magnitude','WeightedHistoryDistance'])) &
                      (fr_boot.scope.isin(['all_test','heldout_context']))]
    keep_macro.to_csv(out/'FRANGIEH_CROSS_FAMILY_MACRO.csv',index=False)
    keep_boot.to_csv(out/'FRANGIEH_CROSS_FAMILY_PAIRED.csv',index=False)
    fr_comp.to_csv(out/'FRANGIEH_CROSS_FAMILY_COMPETENCE.csv',index=False)
    fr_summary=[]
    for _,r in keep_boot.iterrows():
        fr_summary.append({'study':'Frangieh','direction':r.direction,'scope':r.scope,
            'method':'PublicHGB','baseline':r.baseline,'delta_utility20':r.delta_utility20,
            'ci95_lower':r.ci95_lower,'ci95_upper':r.ci95_upper,
            'n_gene_clusters':r.n_paired_gene_clusters,'upstream_competence':'FAILED_ALL_SIX'})
    pd.DataFrame(fr_summary).to_csv(out/'FRANGIEH_CROSS_FAMILY_STRESS_SUMMARY.csv',index=False)
    return table


def frozen_ranking(out,scores,tasks):
    q=scores[scores.method.eq('PublicRule')].merge(tasks[['task_id','gene','context']],on='task_id',validate='one_to_one')
    q=q.sort_values(['risk','task_id'],ascending=[False,True]);q['rank']=np.arange(1,len(q)+1)
    q['adopted_system']='PublicRule_with_feature_CDF_amplitude_fallback'
    q.to_parquet(out/'KOLF_ADOPTED_SYSTEM_RANKING.parquet',index=False)


def figures(out,table):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'svg.fonttype':'none','savefig.bbox':'tight'})
    green='#277C71';orange='#B66A43';blue='#4C7899';grey='#84939A'
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=220);fig.savefig(folder/(name+'.svg'));plt.close(fig)
    primary=table[table.endpoint.eq('delta_rmse')&table.review_fraction.eq(.2)]
    # Figure1 is an actual triage result, not an imagined monotonically rising diagram.
    cold=primary[primary.feedback_budget.eq(0)&primary.method.isin(['Magnitude','HistorySupport','HistorySupport_selected','HistoryEnergy','PublicRule'])].copy()
    cold.to_csv(folder/'FIG1_ZERO_ERROR_SOURCE_DATA.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.8))
    for ax,(study,predictor) in zip(axes,[('McFaline','DecoderOnly'),('McFaline','SAMS_VAE'),('KOLF','ridge')]):
        q=cold[cold.study.eq(study)&cold.predictor.eq(predictor)]
        order=['Magnitude','HistorySupport','HistorySupport_selected','HistoryEnergy','PublicRule']
        q=q.set_index('method').reindex([m for m in order if m in set(q.method)]).reset_index()
        labels={'HistorySupport_selected':'Selected support','HistorySupport':'Support','HistoryEnergy':'History energy','PublicRule':'Public','Magnitude':'Magnitude'}
        ax.bar(np.arange(len(q)),q.high_error_found/q.severe_k,color=[green if m=='PublicRule' else grey for m in q.method],width=.65)
        ax.set_xticks(np.arange(len(q)),[labels.get(m,m) for m in q.method],rotation=35,ha='right');ax.set_ylim(0,.7)
        for i,r in enumerate(q.itertuples()):ax.text(i,r.high_error_found/r.severe_k+.015,f'{r.high_error_found:.0f}/{r.severe_k:.0f}',ha='center',fontsize=8)
        ax.set_title(study+' / '+predictor,loc='left',fontweight='bold');ax.set_ylabel('Recall of true top20% errors')
    fig.suptitle('Fixed20% review: observed severe-error retrieval',x=.06,ha='left',fontweight='bold',fontsize=11)
    fig.subplots_adjust(top=.8,bottom=.26,wspace=.4);save(fig,'FIG1_ZERO_ERROR')
    # Figure2 keeps the nonsignificant content result and strong support baseline visible.
    content=pd.read_csv(RUN/'content_matched/RESULTS.csv');content.to_csv(folder/'FIG2_CONTENT_SOURCE_DATA.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10.5,3))
    for ax,endpoint in zip(axes,['delta_rmse','pearson_error','effect_top200_rmse']):
        q=content[content.endpoint.eq(endpoint)];null=q[q.method.str.startswith('ContentNull_')].utility20_macro.to_numpy()
        real=float(q[q.method.eq('PublicRule')].utility20_macro.iloc[0]);support=float(q[q.method.eq('SupportSelected')].utility20_macro.iloc[0])
        ax.scatter(null,np.linspace(-.1,.1,len(null)),s=18,color=grey);ax.scatter([real],[.28],s=85,marker='*',color=green)
        ax.axvline(support,color=blue,lw=1,ls='--');ax.set_yticks([]);ax.spines['left'].set_visible(False)
        ax.set_title(endpoint,loc='left');ax.set_xlabel('Context-macro U20');ax.set_ylim(-.2,.45)
    fig.suptitle('20 support-matched nulls; strict content increment not established',x=.06,ha='left',fontsize=10)
    fig.subplots_adjust(top=.73,bottom=.25,wspace=.3);save(fig,'FIG2_CONTENT')
    # Figure3 explicitly separates the SEEN global endpoint from the registered macro endpoint.
    budget=primary[primary.study.eq('McFaline')].copy();budget.to_csv(folder/'FIG3_FEEDBACK_SOURCE_DATA.csv',index=False)
    macro=pd.read_csv(RUN/'evidence_budget_raw_rmse/MACRO_PER_RUN.csv')
    macro=macro[macro.endpoint.eq('delta_rmse')&macro.review_fraction.eq(.2)]
    macro.to_csv(folder/'FIG3_REGISTERED_MACRO_SOURCE_DATA.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for col,predictor in enumerate(['DecoderOnly','SAMS_VAE']):
        for row,scope in enumerate(['global','registered context macro']):
            ax=axes[row,col];q=budget[budget.predictor.eq(predictor)] if row==0 else macro[macro.predictor.eq(predictor)]
            public=q[q.method.eq('PublicRule')].utility.mean();ax.axhline(public,color=green,label='Public: zero risk-fit target errors')
            for method,color,label in [('Target_P6_RawRMSE',grey,'Target P6'),('PertEMA_Native61_RawRMSE',blue,'Native + Target'),('PertEMA_Native61_Public_RawRMSE',orange,'Native + Public + Target')]:
                z=q[q.method.eq(method)].groupby('feedback_budget').utility.mean()
                ax.plot(z.index*100,z.values,'o-',color=color,lw=1.2,ms=3,label=label)
            ax.set_title(predictor+' / '+scope,loc='left',fontsize=9);ax.set_xlabel('Feedback genes available (%)');ax.set_ylabel('U20')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('McFaline SEEN evidence: global and macro outcomes stay separate',x=.06,ha='left',fontsize=11,fontweight='bold')
    fig.subplots_adjust(top=.86,bottom=.19,hspace=.5,wspace=.3);save(fig,'FIG3_FEEDBACK')
    # Figure4 uses the actual independent confirmation, including uncertainty.
    external=primary[primary.study.eq('KOLF')];external.to_csv(folder/'FIG4_KOLF_SOURCE_DATA.csv',index=False)
    pairs=pd.read_csv(EXT/'CONFIRMATION_PAIRED_BOOTSTRAP.csv');pairs.to_csv(folder/'FIG4_KOLF_CI_SOURCE_DATA.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7));ax=axes[0]
    pub=float(external[external.method.eq('PublicRule')].utility.iloc[0]);ax.axhline(pub,color=green,label='Public')
    for method,color,label in [('PertEMA_native_control',blue,'Native + Target'),('PertEMA_native_control_Public',orange,'Native + Public + Target')]:
        q=external[external.method.eq(method)].sort_values('feedback_budget');ax.plot(q.feedback_budget*100,q.utility,'o-',color=color,ms=3,label=label)
    ax.set_xlabel('Feedback genes available (%)');ax.set_ylabel('U20');ax.set_title('300 independent confirmation genes',loc='left',fontsize=9);ax.legend(frameon=False,fontsize=7)
    q=pairs[pairs.feedback_budget.eq(0)&pairs.method.isin(['Magnitude','HistorySupport','HistoryEnergy'])].copy()
    q['label']=q.method;ax=axes[1]
    for i,r in enumerate(q.itertuples()):
        ax.plot([r.ci95_lower,r.ci95_upper],[i,i],color=grey,lw=1.3);ax.plot(r.delta_point,i,'o',color=blue)
    ax.set_yticks(np.arange(len(q)),q.label);ax.axvline(0,color=grey,lw=.7,ls='--');ax.set_xlabel('Baseline minus Public U20');ax.set_title('95% gene-cluster CI',loc='left',fontsize=9)
    fig.suptitle('KOLF: point gains, broad intervals; no extra severe-error hits',x=.06,ha='left',fontsize=11,fontweight='bold')
    fig.subplots_adjust(top=.8,bottom=.23,wspace=.58);save(fig,'FIG4_KOLF')
    fr=pd.read_csv(out/'FRANGIEH_CROSS_FAMILY_STRESS_SUMMARY.csv')
    fig,ax=plt.subplots(figsize=(7.3,3.8));
    y=np.arange(len(fr));colors=[blue if b=='Magnitude' else grey for b in fr.baseline]
    for i,r in fr.iterrows():
        ax.plot([r.ci95_lower,r.ci95_upper],[i,i],color=colors[i],lw=2)
        ax.plot(r.delta_utility20,i,'o',color=colors[i],ms=5)
    labels=[f'{r.direction.replace("_","→")} / {r.scope}\nvs {r.baseline}' for _,r in fr.iterrows()]
    ax.set_yticks(y,labels);ax.axvline(0,color='#777',ls='--',lw=.7);ax.set_xlabel('PublicHGB paired U20 increment');ax.set_title('Frangieh cross-family stress evidence (SEEN; upstream competence failed)',loc='left',fontweight='bold')
    fig.subplots_adjust(left=.36,right=.97,top=.84,bottom=.13);save(fig,'FIG5_FRANGIEH_STRESS')


def documentation(out,table,freeze,complete):
    primary=table[table.endpoint.eq('delta_rmse')&table.review_fraction.eq(.2)&table.study.eq('KOLF')]
    def row(method,budget=0.):return primary[primary.method.eq(method)&primary.feedback_budget.eq(budget)].iloc[0]
    pub,mag,energy=row('PublicRule'),row('Magnitude'),row('HistoryEnergy')
    paired=pd.read_csv(EXT/'CONFIRMATION_PAIRED_BOOTSTRAP.csv');a=paired[paired.method.eq('Magnitude')].iloc[0]
    decisions={'default':'PublicRule','Source_enabled':False,'Target_default_enabled':False,
        'external_predictors':complete['predictors'],'MLP':'competence_failed_variance_gate',
        'KNN_fallback_executed':False,'external_evaluation_complete':True,'publication_objective_complete':False,
        'adopted_config_unchanged_after_confirmation':True,'KOLF_public_supported':228,'KOLF_no_history':72,
        'external_primary_delta_public_minus_magnitude':float(-a.delta_point),
        'external_primary_ci95':[-float(a.ci95_upper),-float(a.ci95_lower)],
        'external_severe_hits_gain':float(pub.high_error_found-mag.high_error_found),
        'Target_increase_on_independent_study':'not established; same-budget native/Public intervals cross zero',
        'stop_unregistered_method_search':True,'next_scientific_gap':'predictor-specific benefit beyond support/history-energy on a qualified independent study'}
    write_json(out/'COMPONENT_DECISION.json',decisions)
    claims=[
        ['Public enables auditing without risk-fit target errors','McFaline:22/24 severe hits vs4/5 amplitude;542 prior DEV labels separately recorded','positive SEEN'],
        ['Public audits naturally mixed history coverage','KOLF:228 supported +72 fallback, unique frozen ranking over300 genes','implementation and observed behavior verified'],
        ['Public exceeds amplitude on independent primary endpoint',f'KOLF delta U20={-a.delta_point:.6f},CI[{ -a.ci95_upper:.6f},{-a.ci95_lower:.6f}];hits17 vs17','point-positive; superiority not established'],
        ['Public adds gene-specific content beyond support/energy','20 matched nulls nominalp0.285714;history energy stronger onKOLF','not established'],
        ['Target adds value on top of Public','McFaline global positive;registered macro andKOLF separate','scope limited to SEEN global outcome'],
        ['Finite target-label-equivalent saving','Target-only never passes tested noninferiority;unstable curves andwideCI','right-censored;no exact saved budget'],
        ['Transferable Source errors add beyond Public','real-label models exceed null butdefaultselectsalwaysPublic','not adopted'],
        ['Public signal transfers across predictor families','Frangieh GEARS↔scGPT PublicHGB vs Magnitude: +0.460 [0.317,0.704] and +0.309 [0.079,0.440];189 gene clusters','SEEN stress evidence; all six upstream competence checks failed'],
    ]
    pd.DataFrame(claims,columns=['candidate_claim','evidence','decision']).to_csv(out/'CLAIM_EVIDENCE_MATRIX.csv',index=False)
    methods='''# Methods 实验合同（证据规格，不是论文正文）

- 主问题：风险层没有当前预测器错误记录时，公共真实扰动实验能提供什么审核能力；反馈回来后还有什么增量。
- McFaline：212 SEEN任务/152基因；331反馈任务/228基因；542开发任务单列准备成本。DecoderOnly和SAMS预测已冻结。
- KOLF：600上游训练、300开发反馈、300确认基因。基于固定scGPT基因表示、仅训练基因PCA50的Ridge通过原能力门；这不是完整scGPT扰动预测模型。MLP未通过方差门。
- KOLF主预测/真值/能力门：1400注册输出基因的pooled-count log1pCP10k delta。GWPS实际缺34输出基因；Public比较仅在实际共同1366轴，预测delta+NTC经inverse-log和相同轴重归一化。缺失不补零，主端点不截轴。负log表达只在Public比较视图投影到0并另作诊断。
- Public：Replogle K562 GWPS；KOLF研究排除。一个历史screen/context每扰动，单历史分散度0不是实测独立重复质量。
- Public可用时用公共距离的训练侧featureCDF；无历史时用幅度的featureCDF。CDF不拟合错误标签，但不是相同百分位即相同真实风险。全300任务唯一排序。
- 上游资格门读取300开发误差，单列准备成本；Public规则风险拟合/选型/校准的KOLF错误使用数0。不要把这写成整个流程未读任何KOLF答案。
- 反馈候选是固定PertEMA代码配方的XGBoost输入适配，包含prediction、control、NTC-onlySVD50、相似性及Public增量。所有臂相同任务/真值/预算/簇权重。尚未完成官方conformal和区间流程。
- 反馈预算10/25/50/75/100%，10个预固定顺序、3个固定种子；它们是算法稳定性不是30个独立生物样本。
- 主复核20%，辅5/10/30%；主误差deltaRMSE，辅1-Pearson和评分侧真实效应Top200RMSE。Top200不进入模型。
- Utility以oracle选出同数量高误差任务的收益归一化。真实top20%严重错误集合与review预算分别定义。宏平均与全量global分别报告。
- 5000次基因簇配对bootstrap，同一抽样同步应用所有背景/算法运行。区间条件于已拟合模型和本研究，不代替新研究间不确定性。
- 风险分数、模型及哈希冻结后一次打开confirmation；当前证据包只读取保存评分，新增拟合0、GPU0、下载0。
'''
    (out/'METHODS_CONTRACT.md').write_text(methods)
    index='''# Supplement / 实验索引

1. `MC_NATIVE_PUBLIC_PAIRED_BOOTSTRAP.csv`：较完整Native输入增加Public的同预算比较，macro与global分开。
2. `KOLF_CONFIRMATION_METRICS.csv`：三端点、四复核预算、全量/有历史/无历史分层和所有固定运行。
3. `KOLF_NATIVE_PUBLIC_PAIRED_BOOTSTRAP.csv`：KOLF公平Native vsNative+Public控制比较。
4. `KOLF_ALGORITHM_STABILITY.csv`：算法变异；不作为生物CI。
5. `PUBLIC_COVERAGE_PRECONFIRMATION.csv`、`PUBLIC_FALLBACK_SANITY.json`：确认前覆盖及72任务回退逐分数不变。
6. `PUBLIC_AXIS_ADAPTER_CARD.json`：34缺失基因、1366共有轴、1400主端点保持。
7. 上级目录`content_matched/`：20次严格支持匹配置乱，移动率门和p0.285714。
8. 上级目录`source_explicit_public/`：真实及5次整基因标签置乱的完整选择，不采用Source。
9. 上级目录`adamson_replication/`：两个24任务panel，小型SEEN复现不是新的主确认。
10. 上级目录`external/`：原Gladstone能力失败压力测试。与KOLF资格回执分开，不混成修复前后同一个研究。
11. `FAILURE_AND_ACTION_LOG.md`：数据轴、传输、进程认领的修复，原负结果与失败回执保留。
'''
    (out/'SUPPLEMENT_INDEX.md').write_text(index)
    (out/'REPRODUCTION.md').write_text(f'''# 复现入口

工作树：`{ROOT}`。解释器：`/home/miniconda/bin/python`。

仅重算保存分数、重新生成表图及哈希（不训练、不下载、不再次读取原始确认数据）：

```bash
cd {ROOT}
/home/miniconda/bin/python tools/scripts/build_safeconf_evidence_freeze_v21.py --output {out}
```

原单路线入口（用于独立环境复现整个登记协议，不用于本机再开新确认）：

```bash
/home/miniconda/bin/python tools/scripts/run_safeconf_kolf_panel_v21.py --phase complete
```

原输入与保存分数路径见`ARTIFACT_MANIFEST.json`；所有部署排序见`KOLF_ADOPTED_SYSTEM_RANKING.parquet`。
图上均值为实际运行平均，不是一个可部署的平均排序。实际采用系统固定为PublicRule。
环境：Python{sys.version.split()[0]} / NumPy{np.__version__} / pandas{pd.__version__} / matplotlib{matplotlib.__version__}。
''')
    text=f'''# 负责人结果交付：2026-10-09

## 实际完成

KOLF 600/300/300流程完成；Ridge通过原能力门，MLP方差门失败。300确认基因在风险分数冻结后评分；Public覆盖228、无历史72。未触发KNN备用。

## 实际效果

- McFaline：20%复核Public命中22/24项，幅度4/5项；充分反馈Native+Public平均约31/30.67项。这是SEEN证据，global与登记macro分开。
- KOLF：Public U20={pub.utility:.6f}，幅度{mag.utility:.6f}，增量{-a.delta_point:.6f}，95%CI[{-a.ci95_upper:.6f},{-a.ci95_lower:.6f}]。同样复核60项，二者都命中17项；历史能量命中18项。
- KOLF移除高风险20%后，Public剩余平均误差比幅度低{100*(1-pub.remaining_mean_error/mag.remaining_mean_error):.3f}%。这与严重错误命中是不同量。
- KOLFNative+Public相对Native在充分反馈的点增量+0.156005，95%CI[-0.039604,0.331345]；没有建立独立的稳定反馈增量。
- Target-only标签等效预算仍右删失，不写成已省掉某个精确比例的标签。

## 负责人采用决定

当前默认保持PublicRule；Source不采用；Target保留已验证的McFaline全量排序范围，不因看到KOLF而换默认。KOLF执行完成与投稿贡献完成分开：本轮最明确缺口是**超越历史支持量/效应能量的预测器特异收益**。这一缺口已有数据指向，不以继续调KOLF确认集填补。

## 已处理问题和接续

已修复公共轴34基因缺失、相同归一化投影、监管器相对路径误认和陈旧SSL日志触发错误重试；原失败和冻结结果保留。现在提交完整图表、输入/监督账本、Methods合同、Supplement索引和复现入口。正文、PDF暂缓；研究Goal保持活动。后续资源投向这个唯一科学缺口的开发诊断及冻结后验证，不再开展散乱网络搜索。
'''
    (out/'MORNING_REPORT.md').write_text(text + '\n## Cross-family stress evidence\n\nFrangieh native512 contains two directions (GEARS→scGPT and scGPT→GEARS) and 189 gene clusters. PublicHGB exceeds Magnitude with paired U20 increments +0.460 [0.317, 0.704] and +0.309 [0.079, 0.440]. All six upstream competence checks fail, so these results support transfer of public evidence as stress evidence and define a competence boundary; they are not a qualified independent confirmation.\n')
    (out/'FAILURE_AND_ACTION_LOG.md').write_text('''# 修复与科学取舍

- 下载发生SSL失败：缓存断点继续，未改变角色。监管器曾误把旧SSL日志用于新KeyError；已改为读取当前失败类型，完整保留日志。
- 1400注册输出坐标中34个GWPS表达基因不存在：在KOLF上游拟合/确认评分前固定1366共有轴适配。主误差和预测器仍1400；两边Public比较按同1366分母重归一化，未补零。
- 监管器按绝对字符串识别相对命令路径失败：修为解析路径认领；用真实子进程测试。没有触碰E208。
- Ridge通过误差非劣及方差门；MLP方差门失败：采用一个合格predictor确认，不为凑双模型放松门。KNN候选未触发，不能写成实测结果。
- Public与幅度均17严重错误命中：不宣称多命中；U20点正而CI跨零单列。Target和内容负结果不覆盖。
- 汇总曾错误写成KOLFPublic18命中：18是HistoryEnergy；交付表、图和短报一律17。历史commentary明确更正。
''')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    freeze,complete,scores,tasks=verify_confirmation()
    metrics=pd.read_csv(EXT/'CONFIRMATION_METRICS.csv')
    paired_mc(out);table=summary(out,metrics);frozen_ranking(out,scores,tasks)
    copies={'CONFIRMATION_METRICS.csv':'KOLF_CONFIRMATION_METRICS.csv',
        'CONFIRMATION_NATIVE_PUBLIC_PAIRED_BOOTSTRAP.csv':'KOLF_NATIVE_PUBLIC_PAIRED_BOOTSTRAP.csv',
        'CONFIRMATION_PAIRED_BOOTSTRAP.csv':'KOLF_PUBLIC_PAIRED_BOOTSTRAP.csv',
        'PUBLIC_COVERAGE_PRECONFIRMATION.csv':'PUBLIC_COVERAGE_PRECONFIRMATION.csv',
        'PUBLIC_FALLBACK_SANITY.json':'PUBLIC_FALLBACK_SANITY.json',
        'PUBLIC_AXIS_ADAPTER_CARD.json':'PUBLIC_AXIS_ADAPTER_CARD.json',
        'INFORMATION_BUDGET_LEDGER.csv':'KOLF_INFORMATION_BUDGET_LEDGER.csv',
        'CHANNEL_SCORE_CDF_AUDIT.csv':'KOLF_CHANNEL_SCORE_CDF_AUDIT.csv',
        'PREDICTOR_COMPETENCE.json':'KOLF_PREDICTOR_COMPETENCE.json'}
    for src,dst in copies.items():shutil.copy2(EXT/src,out/dst)
    figures(out,table);documentation(out,table,freeze,complete)
    inputs=[EXT/'RISK_FREEZE.json',EXT/'CONFIRMATION_COMPLETE.json',EXT/'FROZEN_CONFIRMATION_SCORES.parquet',EXT/'CONFIRMATION_METRICS.csv',
        RUN/'evidence_budget_raw_rmse/PREDICTIONS.parquet',RUN/'evidence_budget_raw_rmse/ALL_METRICS.csv',RUN/'content_matched/RESULTS.csv']
    manifest={'status':'EVIDENCE_ARTIFACTS_GENERATED_PUBLICATION_OBJECTIVE_ACTIVE','time_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'inputs':{str(p):sha(p) for p in inputs},'outputs':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='ARTIFACT_MANIFEST.json'},
        'generator_path':str(Path(__file__).resolve()),'generator_sha256':sha(__file__),
        'new_model_fits':0,'new_gpu_hours':0,'new_download_bytes':0,'wall_seconds':time.monotonic()-start,
        'paper_body_pdf_created':False,'confirmation_score_sha256':freeze['confirmation_score_sha256']}
    write_json(out/'ARTIFACT_MANIFEST.json',manifest)
    print(json.dumps({'evidence_freeze':str(out),'outputs':len(manifest['outputs']),'new_fits':0}),flush=True)


if __name__=='__main__':main()
