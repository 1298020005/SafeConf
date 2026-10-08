#!/usr/bin/env python3
"""Whole-cohort ranking figure; kept separate from registered macro result."""
from pathlib import Path
import sys
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN


def main():
    raw=pd.read_csv(RUN/'evidence_budget_raw_rmse/ALL_METRICS.csv')
    ci=pd.read_csv(RUN/'context_calibration_diagnostic/GLOBAL_BUDGET_PAIRED_BOOTSTRAP.csv')
    mean=pd.read_csv(RUN/'context_calibration_diagnostic/TARGET_STATE_MEAN_METRICS.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,6.4))
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':9})
    for col,model in enumerate(['DecoderOnly','SAMS_VAE']):
        frame=raw[raw.predictor.eq(model)&raw.endpoint.eq('delta_rmse')&raw.scope.eq('global')&raw.review_fraction.eq(.2)]
        pub=frame[frame.method.eq('PublicRule')].utility.iloc[0]
        ax,lower=axes[0,col],axes[1,col];ax.axhline(pub,color='#26796D',label='PublicRule: no risk-training errors',lw=1.5)
        state=mean[mean.predictor.eq(model)].groupby('feedback_budget').utility.mean()
        ax.plot(state.index*100,state.values,color='#9A9390',ls=':',label='Budget-only state mean')
        for method,color,label in [('PertEMA_Native61_RawRMSE','#44789A','Native controls + target errors'),('PertEMA_Native61_Public_RawRMSE','#B66A43','Native controls + Public + target errors')]:
            stats=frame[frame.method.eq(method)].groupby('feedback_budget').utility.agg(['mean','min','max'])
            ax.plot(stats.index*100,stats['mean'],color=color,marker='o',ms=3,label=label)
            ax.fill_between(stats.index*100,stats['min'],stats['max'],color=color,alpha=.1)
            b=ci[ci.predictor.eq(model)&ci.method.eq(method)].sort_values('feedback_budget')
            lower.errorbar(b.feedback_budget*100,b.delta_point,
                yerr=np.stack([b.delta_point-b.ci95_lower,b.ci95_upper-b.delta_point]),fmt='o-',color=color,lw=1,ms=3,capsize=2)
        ax.set_title(model.replace('_',' '),loc='left',fontweight='bold');ax.set_ylabel('Global utility at 20% review')
        lower.axhline(0,color='#26796D',lw=1);lower.set_ylabel('Global difference from PublicRule')
        lower.set_title('95% paired gene-cluster CI',loc='left',fontsize=9)
        for a in [ax,lower]:
            a.set_xlabel('Feedback budget (%)');a.set_xticks([0,10,25,50,75,100]);a.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8)
    fig.suptitle('One ranking across contexts: feedback improves risk scale',x=.07,ha='left',fontsize=12,fontweight='bold')
    fig.text(.07,.92,'212 SEEN tasks / 152 genes. Secondary practical endpoint; primary macro label-cost result retained.',fontsize=8,color='#555')
    fig.subplots_adjust(top=.84,bottom=.18,hspace=.52,wspace=.35)
    for ext in ['png','svg']:
        p=RUN/'figures'/('GLOBAL_FEEDBACK_BUDGET.'+ext);fig.savefig(p,dpi=220,bbox_inches='tight')
        if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    plt.close(fig)

if __name__=='__main__':main()
