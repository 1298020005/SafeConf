#!/usr/bin/env python3
"""Original white-background explanatory diagrams; no experimental data invented."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/方法设计/20260922_条件风险模型/figures'
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
INK='#243342'; LINE='#9ba7af'; BLUE='#446988'; TEAL='#25867e'; LIGHT='#eef5f4'
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,
                     'svg.fonttype':'path','font.size':12})

def setup(width=16,height=8):
    fig,ax=plt.subplots(figsize=(width,height))
    ax.set_xlim(0,width); ax.set_ylim(0,height); ax.axis('off')
    fig.subplots_adjust(0,0,1,1)
    return fig,ax

def text(ax,x,y,label,size=14,color=INK,weight='normal',ha='center'):
    ax.text(x,y,label,fontproperties=FONT,fontsize=size,color=color,
            weight=weight,ha=ha,va='center',linespacing=1.6)

def box(ax,x,y,w,h,title,body='',color=LINE,fill='white',dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.06',
            ec=color,fc=fill,lw=1.2,linestyle='--' if dashed else '-'))
    text(ax,x+w/2,y+h*.69 if body else y+h/2,title,14,weight='bold')
    if body: text(ax,x+w/2,y+h*.31,body,11)

def arrow(ax,start,end,color=LINE,style='-',curve=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,lw=1.4,
        color=color,linestyle=style,connectionstyle=f'arc3,rad={curve}'))

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,facecolor='white')
    plt.close(fig)

def overview():
    fig,ax=setup()
    box(ax,.35,5.6,2.8,1.35,'新任务','细胞背景 · 扰动\n匹配对照表达',BLUE)
    box(ax,4.0,5.6,3.0,1.35,'固定上游预测器','一个或多个预测向量',BLUE)
    box(ax,8.0,5.05,3.2,2.2,'特征提取','预测幅度 M · 分歧 D\n预测—历史差异 G',TEAL)
    box(ax,.35,1.35,3.1,1.5,'允许使用的历史实验','对照 · 效应 · 来源记录',BLUE)
    box(ax,4.,1.35,3.,1.5,'历史证据','均值 · 离散度\n样本量 · 背景覆盖',TEAL)
    box(ax,8.,1.35,3.2,1.5,'历史证据与条件','离散 · 数量 · 覆盖\n训练覆盖 · 缺失标记',TEAL)
    box(ax,12.25,3.6,3.4,1.6,'SafeConf 评分器','已有规则 / 候选小模型',TEAL,LIGHT)
    box(ax,12.25,.55,3.4,1.5,'任务级输出','风险分数 · 复核顺序\n证据缺口',BLUE)
    arrow(ax,(3.15,6.28),(4.,6.28))
    arrow(ax,(7.,6.28),(8.,6.28))
    arrow(ax,(3.45,2.1),(4.,2.1))
    arrow(ax,(7.,2.1),(8.,2.1))
    arrow(ax,(6.7,2.85),(8.3,5.05),TEAL)
    text(ax,7.0,4.05,'来源效应',10)
    arrow(ax,(11.2,6.0),(12.25,4.75),TEAL)
    arrow(ax,(11.2,2.1),(12.25,4.0),TEAL)
    arrow(ax,(13.95,3.6),(13.95,2.05))
    text(ax,.35,7.65,'输入与风险评估流程',16,ha='left',weight='bold')
    text(ax,.35,.5,'新任务的真实扰动结果仅在独立评价阶段使用',11,ha='left')
    save(fig,'01_overview')

def candidate():
    fig,ax=setup(16,7)
    text(ax,.35,6.6,'候选条件风险模型',16,ha='left',weight='bold')
    text(ax,15.65,6.6,'待完整验证',11,ha='right')
    box(ax,.4,4.5,3.,1.3,'预测幅度 M',color=BLUE)
    box(ax,.4,2.45,3.,1.4,'补充证据','分歧 · 历史偏离\n离散 · 数量 · 覆盖',BLUE)
    box(ax,.4,.5,3.,1.3,'覆盖与缺失条件 z',color=BLUE)
    box(ax,4.45,4.5,3.3,1.3,'基础评分 b(M)',color=TEAL)
    box(ax,4.45,2.45,3.3,1.4,'补充校正 h(证据)',color=TEAL,dashed=True)
    box(ax,4.45,.5,3.3,1.3,'条件权重 a(z)',color=TEAL,dashed=True)
    box(ax,9.,1.5,2.1,1.8,'条件校正','a × h',TEAL,LIGHT,dashed=True)
    box(ax,12.15,3.1,3.45,1.5,'风险分数 R','b + a × h',TEAL,LIGHT,dashed=True)
    arrow(ax,(3.4,5.15),(4.45,5.15)); arrow(ax,(3.4,3.15),(4.45,3.15))
    arrow(ax,(3.4,1.15),(4.45,1.15))
    arrow(ax,(7.75,3.15),(9.,2.8)); arrow(ax,(7.75,1.15),(9.,2.0))
    arrow(ax,(7.75,5.15),(12.15,4.2),TEAL)
    arrow(ax,(11.1,2.4),(12.15,3.55),TEAL)
    text(ax,11.9,.55,'权重由开发数据学习；不改变上游预测结果',11)
    save(fig,'02_candidate_model')

def validation():
    fig,ax=setup(16,7)
    text(ax,.35,6.6,'开发与外部评价分开',16,ha='left',weight='bold')
    text(ax,.5,5.9,'开发阶段',12,ha='left',color=TEAL,weight='bold')
    box(ax,.4,4.1,3.1,1.4,'历史折外预测','可用特征 + 已知误差',BLUE)
    box(ax,4.25,4.1,3.1,1.4,'按研究分组训练','拟合与消融',TEAL)
    box(ax,8.1,4.1,3.1,1.4,'内部验证选择','固定候选与预算',TEAL)
    box(ax,11.95,4.1,3.6,1.4,'冻结版本','公式 · 权重 · 来源 · 阈值',TEAL,LIGHT)
    arrow(ax,(3.5,4.8),(4.25,4.8)); arrow(ax,(7.35,4.8),(8.1,4.8)); arrow(ax,(11.2,4.8),(11.95,4.8))
    ax.plot([.4,15.6],[3.65,3.65],color=LINE,lw=.8)
    text(ax,.5,3.25,'外部评价阶段',12,ha='left',color=BLUE,weight='bold')
    box(ax,.4,1.2,3.1,1.4,'新研究预测与证据','不含测试误差',BLUE)
    box(ax,4.25,1.2,3.1,1.4,'固定评分器','不重新挑权重',TEAL)
    box(ax,8.1,1.2,3.1,1.4,'风险排序封存','文件与版本可核对',TEAL)
    box(ax,11.95,1.2,3.6,1.4,'一次性评价','对照真实结果 · 报告边界',BLUE)
    arrow(ax,(3.5,1.9),(4.25,1.9)); arrow(ax,(7.35,1.9),(8.1,1.9)); arrow(ax,(11.2,1.9),(11.95,1.9))
    text(ax,13.75,.4,'真实结果只进入评价',11)
    arrow(ax,(13.75,.68),(13.75,1.2),BLUE)
    save(fig,'03_development_validation')

def results():
    e226=ROOT/'docs/实验结果/E226_local_history_calibration_20260922'
    e227=ROOT/'docs/实验结果/E227_local_nonlinear_ranker_20260922'
    a=pd.read_csv(e226/'INTERVALS.csv')
    a=a[(a.method=='MDH_condition')&(a.baseline=='magnitude')&(a.metric=='utility')&(a.budget==.2)].sort_values('history_fraction')
    b=pd.read_csv(e227/'INTERVALS.csv')
    methods=['tree_direct_MDH_condition','tree_residual_MDH_condition','tree_residual_MD','ridge_MDH_condition']
    b=b[(b.baseline=='magnitude')&(b.metric=='utility')&(b.budget==.2)].set_index('method').loc[methods]
    fig,axes=plt.subplots(1,2,figsize=(16,6.6),gridspec_kw={'width_ratios':[1,1.25]})
    fig.subplots_adjust(left=.07,right=.97,bottom=.2,top=.82,wspace=.56)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_color(LINE)
        ax.tick_params(colors=INK,labelsize=11)
        ax.grid(False)
    x=np.arange(4); y=a.delta.to_numpy()
    axes[0].axhline(0,lw=.8,color=LINE)
    axes[0].errorbar(x,y,yerr=np.vstack([y-a.ci95_lower.to_numpy(),a.ci95_upper.to_numpy()-y]),
        fmt='o-',color=TEAL,lw=1.5,capsize=4,ms=6)
    axes[0].set_xticks(x,['10%','25%','50%','100%'])
    axes[0].set_xlabel('可用本地历史组比例',fontproperties=FONT,labelpad=10)
    axes[0].set_ylabel('相对幅度的复核效用差值',fontproperties=FONT,labelpad=10)
    axes[0].set_title('历史预算：条件岭回归',fontproperties=FONT,pad=18,fontsize=14)
    axes[0].set_ylim(-.015,.035)
    axes[0].annotate('预定主预算',xy=(2,y[2]),xytext=(2,.03),ha='center',fontproperties=FONT,
                     fontsize=10,color=INK,arrowprops={'arrowstyle':'-','color':LINE,'lw':.8})
    y=b.delta.to_numpy(); pos=np.arange(len(b))
    axes[1].axvline(0,lw=.8,color=LINE)
    axes[1].errorbar(y,pos,xerr=np.vstack([y-b.ci95_lower.to_numpy(),b.ci95_upper.to_numpy()-y]),
        fmt='o',color=BLUE,lw=1.4,capsize=4,ms=6)
    axes[1].set_yticks(pos,['直接树：全条件','残差树：全条件','残差树：幅度＋分歧','岭回归：全条件'],fontproperties=FONT)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('相对幅度的复核效用差值',fontproperties=FONT,labelpad=10)
    axes[1].set_title('同一历史预算：非线性对照',fontproperties=FONT,pad=18,fontsize=14)
    fig.text(.07,.94,'已完成的历史开发检验',fontproperties=FONT,fontsize=17,color=INK)
    fig.text(.07,.07,'复核预算20% · 全部4个研究 · 区间为描述性研究簇重采样 · 不代表独立外部确认',
             fontproperties=FONT,fontsize=11,color=INK)
    save(fig,'04_history_and_models')

if __name__=='__main__':
    overview(); candidate(); validation(); results()
    print(OUT)
