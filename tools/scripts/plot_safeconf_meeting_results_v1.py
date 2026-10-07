#!/usr/bin/env python3
"""Plot the three separate registered comparisons for the meeting receipt."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/review_repair_20261007_v1'


def main():
    src=pd.read_csv(OUT/'SOURCE_REAL_COMPARISON.csv')
    bud=pd.read_csv(OUT/'NATIVE_PUBLIC_BUDGET_TABLE.csv')
    review=pd.read_csv(OUT/'GLOBAL_20_PERCENT_REVIEW.csv')
    fig,ax=plt.subplots(1,3,figsize=(15.3,4.9))
    for i,(method,color,label) in enumerate(zip(
        ['always_public','always_source','selected_gate'],
        ['#98a7b6','#2b738d','#e5a64a'],['Public rank','Source HGB','Selected gate'])):
        values=[src[(src.source==s)&(src.target==t)&(src.method==method)].u20.iloc[0]
            for s,t in [('TxPert_Exphormer','TxPert_GAT'),('TxPert_GAT','TxPert_Exphormer')]]
        bars=ax[0].bar(np.arange(2)+(i-1)*.23,values,width=.22,color=color,label=label)
        ax[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax[0].set_xticks(np.arange(2),['Exphormer -> GAT','GAT -> Exphormer'],fontsize=9)
    ax[0].set_ylim(0,1);ax[0].set_ylabel('U20, higher is better')
    ax[0].set_title('A  Same-family transfer\n1,808 tasks / 575 gene clusters')
    ax[0].legend(loc='lower left',fontsize=8)
    ax[1].plot(bud.budget*100,bud.Native61,'o-',color='#98a7b6',label='Native61 adaptation')
    ax[1].plot(bud.budget*100,bud.Native61_Public,'o-',color='#2b738d',label='Native61 + Public')
    ax[1].axhline(bud.public_rule_no_feedback.iloc[0],color='#5b7956',linestyle='--',label='PublicRule, no feedback')
    ax[1].set_ylim(-.05,1);ax[1].set_xticks([10,25,50,75,100]);ax[1].set_xlabel('Feedback gene budget (%)')
    ax[1].set_ylabel('Context-macro U20 (three-seed mean)')
    ax[1].set_title('B  Current McFaline comparison\n212 tasks / 152 gene clusters');ax[1].legend(loc='lower right',fontsize=8)
    values=[review[review.method==m].high_error_found.iloc[0]
        for m in ['Amplitude','PublicRule','PertEMA_Native61_Public_50_corrected']]
    bars=ax[2].bar(np.arange(3),values,color=['#98a7b6','#5b7956','#2b738d'],width=.63)
    ax[2].bar_label(bars,fmt='%d',padding=3,fontsize=12)
    ax[2].set_xticks(np.arange(3),['Amplitude','PublicRule','Native + Public\n50% feedback'],fontsize=9)
    ax[2].set_ylim(0,30);ax[2].set_ylabel('True high-error tasks discovered')
    ax[2].set_title('C  Global 20% review\n43 reviewed tasks, fixed first seed')
    for a in ax:
        a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
        a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.text(.5,.015,'DEV/SEEN results. Panels use different registered task cohorts; they are not a staged improvement curve.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'MEETING_RESULTS.png',dpi=190);fig.savefig(OUT/'MEETING_RESULTS.svg')
    svg=OUT/'MEETING_RESULTS.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':
    main()
