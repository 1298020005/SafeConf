#!/usr/bin/env python3
"""White-background plots from all four E230 targets; no chosen-best configuration."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_safeconf_design_figures_20260922 import FONT, BLUE, TEAL, INK, LINE, OUT
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/实验结果/E230_history_capacity_20260922'
COLORS={'K562':BLUE,'RPE1':TEAL,'hepg2':'#9e674d','jurkat':'#817697'}


def main():
    a=pd.read_csv(DATA/'CAPACITY_CURVES.csv')
    full=pd.read_csv(DATA/'FULL_CAPACITY_COMPARATORS.csv').set_index('target')
    fig,axs=plt.subplots(1,3,figsize=(17,6.3))
    fig.subplots_adjust(left=.065,right=.985,top=.77,bottom=.22,wspace=.4)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(LINE)
        ax.tick_params(colors=INK,labelsize=10)
        ax.axhline(0,color=LINE,lw=.9,zorder=0)
    for target,color in COLORS.items():
        b=a[(a.target==target)&(a.history_mode=='fraction')&(a.k_sources==3)&(a.method=='m_plus_history')].sort_values('fraction')
        # Paired changes against the identical target's magnitude, not pooled absolute utilities.
        baseline=float(full.loc[target,'magnitude'])
        axs[0].plot(np.arange(3),b.utility-baseline,'o-',color=color,lw=1.5,ms=5,label=target)
        b=a[(a.target==target)&(a.history_mode=='matched30')&(a.method=='m_plus_history')].sort_values('k_sources')
        axs[1].plot(b.k_sources,b.utility-float(b.iloc[0].utility),'o-',color=color,lw=1.5,ms=5)
    axs[0].set_xticks([0,1,2],['25%','50%','100%'])
    axs[0].set_xlabel('每个来源的历史细胞比例',fontproperties=FONT,labelpad=12)
    axs[0].set_ylabel('组合评分相对幅度的复核效用差值',fontproperties=FONT,labelpad=10)
    axs[0].set_title('固定三个来源背景',fontproperties=FONT,fontsize=14,pad=18)
    axs[1].set_xticks([1,2,3])
    axs[1].set_xlabel('历史来源背景数',fontproperties=FONT,labelpad=12)
    axs[1].set_ylabel('相对单来源的复核效用变化',fontproperties=FONT,labelpad=10)
    axs[1].set_title('固定总计30个历史细胞',fontproperties=FONT,fontsize=14,pad=18)
    d=full.m_plus_history-full.magnitude
    pos=np.arange(len(d))
    axs[2].bar(pos,d,width=.52,color=[COLORS[t] for t in d.index])
    axs[2].set_xticks(pos,d.index)
    axs[2].set_ylabel('组合评分相对幅度的复核效用差值',fontproperties=FONT,labelpad=10)
    axs[2].set_title('完整历史库：四个目标均保留',fontproperties=FONT,fontsize=14,pad=18)
    for x,y in zip(pos,d):
        axs[2].text(x,y+.001,f'{y:+.4f}',ha='center',fontsize=10,color=INK)
    axs[2].set_ylim(0,.031)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.985,.995),
               ncol=4,frameon=False,fontsize=11)
    fig.text(.065,.955,'历史数量与来源多样性：实际抽样结果',fontproperties=FONT,fontsize=18,color=INK)
    fig.text(.065,.09,'左、右：1808个主任务；中：每个目标243个有完整来源支持的主任务。上游预测保持不变。',
             fontproperties=FONT,fontsize=11,color=INK)
    fig.text(.065,.045,'复核预算20% · 三次预定抽样取均值 · 已知资料上的开发分析 · 曲线连接离散配置，不表示连续规律',
             fontproperties=FONT,fontsize=10,color=INK)
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'05_real_history_capacity.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    print(OUT/'05_real_history_capacity.png')


if __name__=='__main__':
    main()
