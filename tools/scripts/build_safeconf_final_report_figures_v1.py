#!/usr/bin/env python3
"""Export separate, checked figures for the two final Markdown documents."""
from pathlib import Path
import importlib.util
import json
import hashlib
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/组会汇报/20261008_最终报告'
FIG=OUT/'figures'
BASE=ROOT/'docs/组会汇报/20261008_科学示意_v2'
RECEIPT=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/review_repair_20261007_v1'
LAYOUT=[];LABELS=[]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def checked_save(fig,out,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[];seen=set()
    for obj in fig.findobj(match=Text):
        if not obj.get_visible() or not obj.get_text().strip() or id(obj) in seen:continue
        seen.add(id(obj));bounds=obj.get_window_extent(renderer)
        if bounds.width<=0 or bounds.height<=0:continue
        texts.append((obj.get_text(),bounds))
    overlaps=[];clipped=[]
    fb=fig.bbox
    for i,(s,a) in enumerate(texts):
        LABELS.append({'figure':name,'label':s})
        if a.x0<fb.x0-1 or a.y0<fb.y0-1 or a.x1>fb.x1+1 or a.y1>fb.y1+1:clipped.append(s)
        for t,b in texts[i+1:]:
            wi=min(a.x1,b.x1)-max(a.x0,b.x0);he=min(a.y1,b.y1)-max(a.y0,b.y0)
            if wi>1 and he>1:overlaps.append({'a':s,'b':t,'width_px':float(wi),'height_px':float(he)})
    result={'figure':name,'texts':len(texts),'text_overlaps':overlaps,'clipped_labels':clipped}
    LAYOUT.append(result)
    (OUT/'evidence/LAYOUT_CHECK.json').write_text(json.dumps(LAYOUT,ensure_ascii=False,indent=2)+'\n')
    if overlaps or clipped:raise RuntimeError(json.dumps(result,ensure_ascii=False))
    fig.savefig(out/(name+'.png'),dpi=260,facecolor='white')
    # Outlined SVG is independent of fonts installed on the receiving machine.
    with plt.rc_context({'svg.fonttype':'path'}):fig.savefig(out/(name+'.svg'),facecolor='white')
    with plt.rc_context({'svg.fonttype':'none'}):fig.savefig(out/(name+'_text.svg'),facecolor='white')
    for p in [out/(name+'.svg'),out/(name+'_text.svg')]:
        p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
    plt.close(fig)


def result_figures():
    src=pd.read_csv(RECEIPT/'SOURCE_REAL_COMPARISON.csv')
    mag=pd.read_csv(BASE/'SOURCE_MAGNITUDE_POINT_COMPARISON.csv')
    bud=pd.read_csv(RECEIPT/'NATIVE_PUBLIC_BUDGET_TABLE.csv')
    review=pd.read_csv(RECEIPT/'GLOBAL_20_PERCENT_REVIEW.csv')
    dirs=[('TxPert_Exphormer','TxPert_GAT'),('TxPert_GAT','TxPert_Exphormer')]
    fig,ax=plt.subplots(figsize=(10.7,6.5))
    for i,(m,color,label) in enumerate([('always_public','#81909a','Public rank'),('Magnitude','#d7924b','Magnitude'),('always_source','#3e78b0','Source HGB'),('selected_gate','#a59bc3','Selected gate')]):
        vals=[]
        for s,t in dirs:
            df=mag if m=='Magnitude' else src
            vals.append(df[(df.source==s)&(df.target==t)&(df.method==m)].u20.iloc[0])
        bars=ax.bar(np.arange(2)+(i-1.5)*.19,vals,width=.18,color=color,label=label)
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=11)
    ax.set_xticks([0,1],['Exphormer → GAT','GAT → Exphormer']);ax.set_ylim(0,1.01)
    ax.set_ylabel('Context-macro U20')
    ax.set_title('来源错误迁移与强预测侧基线\n1,808 tasks / 575 gene clusters · DEV/SEEN',fontsize=15,pad=61)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.025),ncol=4,frameon=False,fontsize=10)
    finish_axis(ax);fig.subplots_adjust(left=.11,right=.97,bottom=.15,top=.69)
    checked_save(fig,FIG,'03_SOURCE')
    fig,ax=plt.subplots(figsize=(10.7,6.5))
    ax.plot(bud.budget*100,bud.Native61,'o-',color='#81909a',lw=2,label='Native61 adaptation')
    ax.plot(bud.budget*100,bud.Native61_Public,'o-',color='#3e78b0',lw=2,label='Native61 + Public')
    ax.axhline(bud.public_rule_no_feedback.iloc[0],color='#268d86',ls='--',label='PublicRule · no new feedback')
    ax.set_xlim(6,104);ax.set_ylim(-.04,1.01);ax.set_xticks([10,25,50,75,100])
    ax.set_xlabel('Feedback gene budget (%)');ax.set_ylabel('Context-macro U20 · three-seed mean')
    ax.set_title('公共输入增强目标错误学习\n212 tasks / 152 gene clusters · current adaptation',fontsize=15,pad=67)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.025),ncol=2,frameon=False,fontsize=10)
    finish_axis(ax);fig.subplots_adjust(left=.12,right=.97,bottom=.15,top=.66)
    checked_save(fig,FIG,'04_TARGET')
    fig,ax=plt.subplots(figsize=(10.7,6.5))
    methods=['Amplitude','PublicRule','PertEMA_Native61_Public_50_corrected']
    vals=[review[review.method==m].high_error_found.iloc[0] for m in methods]
    bars=ax.bar([0,1,2],vals,color=['#81909a','#268d86','#3e78b0'],width=.60)
    ax.bar_label(bars,fmt='%d',padding=6,fontsize=18)
    ax.set_xticks([0,1,2],['Amplitude','PublicRule','Native61 + Public\n50% feedback'])
    ax.set_ylim(0,30);ax.set_ylabel('True high-error tasks discovered')
    ax.axhline(43*43/212,color='#9e9e9e',ls=':',lw=1.0)
    ax.text(-.28,43*43/212+1.0,'Random expectation: 8.72',fontsize=9,ha='left',color='#67747a')
    ax.set_title('同样复核43项预测，公共规则多发现18项大误差\n43 / 212 reviewed tasks · fixed first seed',fontsize=15,pad=18)
    finish_axis(ax);fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.83)
    checked_save(fig,FIG,'05_REVIEW')


def finish_axis(ax):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)


def main():
    FIG.mkdir(parents=True,exist_ok=True);(FIG/'originals').mkdir(exist_ok=True);(OUT/'evidence').mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('scientific_v2',ROOT/'tools/scripts/build_safeconf_scientific_figures_v2.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.setup()
    original_label=m.label
    def improved_label(ax,x,y,s,*args,**kw):
        swaps={'Public perturbation experiments':'Public historical reference · default',
            '合法公共扰动实验':'公共历史参照 · 当前默认',
            'Shared risk learner':'Source\nconditional','共享风险学习器':'Source\n条件增量',
            'Optional adaptation for later predictions':'Target · feedback candidate',
            '可选个性化更新，服务后续任务':'Target · 有反馈的候选'}
        if s in ['Optional transfer','可选错误经验迁移']:
            s='Transfer' if s=='Optional transfer' else '经验迁移';x,y=9.45,2.12;kw.pop('rotation',None)
        if s==r'PublicRule: $\sqrt{D^2+V}$':
            s=r'$R=\sqrt{D^2+V}$';x,y=10.63,2.44
        return original_label(ax,x,y,swaps.get(s,s),*args,**kw)
    m.label=improved_label
    def saving(fig,out,name):
        if name=='APPENDIX_STRONG_BASELINES':
            ax=fig.axes[0]
            handles,labels=ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.025),ncol=3,frameon=False,fontsize=10)
            ax.set_title(ax.get_title(loc='left'),loc='left',fontsize=13,pad=48)
            right=fig.axes[1]
            right.set_title(right.get_title(loc='left'),loc='left',fontsize=13,pad=48)
            fig.subplots_adjust(top=.69)
        name={'METHOD_OVERVIEW':'02_METHOD_EN','METHOD_OVERVIEW_ZH':'02_METHOD_ZH','EVIDENCE_DRIVEN_EVOLUTION':'01_EVOLUTION','APPENDIX_STRONG_BASELINES':'06_STRONG_BASELINES'}.get(name,name)
        checked_save(fig,FIG,name)
    m.save=saving
    m.method_overview(FIG);m.method_overview(FIG,True);m.evolution(FIG)
    result_figures()
    m.strong_baselines(BASE)
    current=Path('/home/yyf/runtime_artifacts/safeconf_goal_report_20261008_v1/current_public_content/RESULTS.csv')
    if current.exists():
        data=pd.read_csv(current);null=data[data.method.str.startswith('CurrentContentNull')]
        fig,ax=plt.subplots(figsize=(10.7,6.5))
        ax.axhline(0,color='#aeb7bd',lw=1)
        x=np.arange(5);val=null.delta_utility20.to_numpy()
        ax.errorbar(x,val,yerr=np.vstack([val-null.ci95_lower,null.ci95_upper-val]),fmt='o',color='#268d86',capsize=5,lw=1.7)
        ax.set_xticks(x,[f'Order {i+1}' for i in range(5)])
        ax.set_ylabel('ΔU20 · real minus shuffled content')
        ax.set_title('当前默认：真实历史与支持匹配内容置乱\n212 tasks / 152 gene clusters · all five fixed orders',fontsize=15,pad=19)
        finish_axis(ax);fig.subplots_adjust(left=.12,right=.97,bottom=.15,top=.80)
        checked_save(fig,FIG,'07_CONTENT_MECHANISM')
    # Original plots remain immutable; the final exported variants only fix
    # layout and display the already-audited stronger baseline.
    hashes={}
    for n in ['03_SOURCE_RESULT.png','04_TARGET_RESULT.png','05_REVIEW_RESULT.png','MEETING_RESULTS.png']:
        hashes[n]=sha(BASE/n);shutil.copyfile(BASE/n,FIG/'originals'/n)
        assert sha(FIG/'originals'/n)==hashes[n]
    (OUT/'evidence/FIGURE_LABELS.json').write_text(json.dumps(LABELS,ensure_ascii=False,indent=2)+'\n')
    (OUT/'evidence/FIGURE_MANIFEST.json').write_text(json.dumps({'figures':{p.name:sha(p) for p in FIG.iterdir() if p.is_file()},
        'original_hashes_unchanged':hashes,'outlined_svg_exported':True,'script_sha256':sha(Path(__file__)),
        'figure_role':'diagram elements illustrative; result plots use released current-contract scores',
        'text_bbox_overlap_checks':len(LAYOUT)},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':len(LAYOUT),'overlaps':sum(len(r['text_overlaps']) for r in LAYOUT),'originals_unchanged':True},ensure_ascii=False))


if __name__=='__main__':main()
