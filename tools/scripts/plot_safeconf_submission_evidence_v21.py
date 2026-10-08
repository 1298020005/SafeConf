#!/usr/bin/env python3
"""Scientific source-data figures for actual v2.1 experiment outputs."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN,selected_support
from tools.safeconf_continual.submission_evidence import point_metrics,write_json


def main():
    out=RUN/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','savefig.bbox':'tight'})
    data=RUN/'evidence_budget_raw_rmse';macro=pd.read_csv(data/'MACRO_PER_RUN.csv')
    bio=pd.read_csv(data/'BIOLOGICAL_CLUSTER_BOOTSTRAP.csv')
    rows=[]
    colors={'Target_P6_RawRMSE':'#89949C','PertEMA_Native61_RawRMSE':'#44789A',
            'PertEMA_Native61_Public_RawRMSE':'#B66A43'}
    names={'Target_P6_RawRMSE':'Prediction features + target errors',
        'PertEMA_Native61_RawRMSE':'Native controls + target errors',
        'PertEMA_Native61_Public_RawRMSE':'Native controls + Public + target errors'}
    fig,axes=plt.subplots(2,2,figsize=(10,7.2))
    for col,predictor in enumerate(['DecoderOnly','SAMS_VAE']):
        ax,lower=axes[0,col],axes[1,col]
        q=macro[(macro.predictor==predictor)&(macro.endpoint=='delta_rmse')&(macro.review_fraction==.2)]
        public=q[q.method=='PublicRule'].utility.iloc[0]
        f=pd.read_parquet(data/f'{predictor}_TASKS.parquet')
        support=-selected_support(f);support_macro=np.mean([point_metrics(f.true_error_rmse.to_numpy()[ix],support[ix],f.task_id.to_numpy()[ix])['utility'] for ix in f.groupby('target').indices.values()])
        ax.axhline(public,color='#26796D',lw=1.7,label='PublicRule: zero target-error risk training')
        ax.axhline(support_macro,color='#765A8B',lw=1.2,ls='--',label='Selected-history support: zero target errors')
        for method,color in colors.items():
            z=q[q.method==method].groupby('feedback_budget').utility.agg(['mean','min','max'])
            x=z.index.to_numpy(float)*100
            ax.plot(x,z['mean'],marker='o',ms=3,lw=1.4,color=color,label=names[method])
            ax.fill_between(x,z['min'].to_numpy(),z['max'].to_numpy(),color=color,alpha=.10,lw=0)
            b=bio[(bio.predictor==predictor)&(bio.method==method)].sort_values('feedback_budget')
            lower.errorbar(b.feedback_budget*100,b.delta_point,
                yerr=np.vstack([b.delta_point-b.ci95_lower,b.ci95_upper-b.delta_point]),
                fmt='o-',lw=1,color=color,ms=3,capsize=2,label=names[method])
            rows.extend([{'predictor':predictor,'method':method,'budget':float(t),'mean':float(r['mean']),
                          'algorithm_min':float(r['min']),'algorithm_max':float(r['max'])} for t,r in z.iterrows()])
        ax.set_title(predictor.replace('_',' '),loc='left',fontweight='bold')
        ax.set_ylabel('Utility at 20% review');ax.set_xticks([0,10,25,50,75,100]);ax.set_xlabel('Target-error feedback budget (%)')
        lower.axhline(0,color='#26796D',lw=1);lower.axhline(-.005,color='#AAA',lw=.6,ls=':')
        lower.set_ylabel('Utility difference from PublicRule');lower.set_xlabel('Target-error feedback budget (%)')
        lower.set_xticks([10,25,50,75,100]);lower.set_title('Biological uncertainty: 95% gene-cluster CI',loc='left',fontsize=9)
        ax.text(0,-.23,'Shading: variation across 10 feedback orders and 3 learner seeds',transform=ax.transAxes,fontsize=7)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.5,-.015))
    fig.suptitle('Target-error supervision and risk-audit performance',x=.06,ha='left',fontsize=13,fontweight='bold')
    fig.text(.06,.925,'212 SEEN evaluation tasks / 152 gene clusters. Historical preparation: 542 DEV tasks.',fontsize=8,color='#555')
    fig.subplots_adjust(top=.86,bottom=.18,hspace=.52,wspace=.3)
    fig.savefig(out/'LABEL_EFFICIENCY.png',dpi=220);fig.savefig(out/'LABEL_EFFICIENCY.svg');plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'LABEL_EFFICIENCY_SOURCE_DATA.csv',index=False)
    result=pd.read_csv(RUN/'content_matched/RESULTS.csv')
    fig,axes=plt.subplots(1,3,figsize=(10,3.2))
    rng=np.random.default_rng(0)
    for ax,endpoint,title in zip(axes,['delta_rmse','pearson_error','effect_top200_rmse'],['All-axis delta RMSE','1 - Pearson','EffectTop200 RMSE']):
        q=result[result.endpoint==endpoint];null=q[q.method.str.startswith('ContentNull_')].utility20_macro.to_numpy()
        real=q[q.method=='PublicRule'].utility20_macro.iloc[0]
        support=q[q.method=='SupportSelected'].utility20_macro.iloc[0]
        ax.scatter(null,rng.uniform(-.12,.12,len(null)),s=22,color='#A3B2BD',alpha=.8,label='20 matched-content nulls')
        ax.scatter([real],[.32],marker='*',s=110,color='#26796D',zorder=4,label='True PublicRule')
        ax.axvline(support,color='#765A8B',ls='--',lw=1,label='Selected-history support')
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Utility at 20% review');ax.set_yticks([]);ax.set_ylim(-.25,.48)
        ax.spines['left'].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig.suptitle('Strict content attribution: 542 DEV tasks / 377 gene clusters',x=.06,ha='left',fontweight='bold',fontsize=12)
    fig.subplots_adjust(top=.78,bottom=.25,wspace=.3)
    fig.savefig(out/'CONTENT_ATTRIBUTION.png',dpi=220);fig.savefig(out/'CONTENT_ATTRIBUTION.svg');plt.close(fig)
    result.to_csv(out/'CONTENT_ATTRIBUTION_SOURCE_DATA.csv',index=False)
    write_json(out/'FIGURE_MANIFEST.json',{'figures':['LABEL_EFFICIENCY','CONTENT_ATTRIBUTION'],
        'data_only_no_schematic_curves':True,'algorithm_variation_is_not_biological_CI':True,
        'manuscript_or_pdf_created':False})


if __name__=='__main__':main()
