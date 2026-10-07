#!/usr/bin/env python3
"""Original scientific vector schematics, informed by local paper Figure 1s.

Drawn profiles, cells and heatmaps are illustrative; released result plots are
copied byte-for-byte. No training, ground-truth opening, or service mutation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/组会汇报/20261008_科学示意_v2'
OLD=ROOT/'docs/组会汇报/20261008_演变与当前架构'
C=dict(ink='#25323b',grey='#81909a',pale='#dce3e7',teal='#268d86',teal_light='#e5f2ee',
       blue='#3e78b0',blue_light='#e7eff7',orange='#d7924b',orange_light='#fbefdf',
       red='#c86970',navy='#497ba0',white='#ffffff')
CMAP=LinearSegmentedColormap.from_list('effects',['#426e92','#f7f7f5','#ce7e7d'])


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def setup():
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':['DejaVu Sans','Noto Sans CJK JP'],
                         'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none'})


def canvas(w=16,h=8.5):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.subplots_adjust(left=.018,right=.982,bottom=.025,top=.975)
    ax.set_xlim(0,w);ax.set_ylim(0,h);ax.set_aspect('equal');ax.axis('off')
    return fig,ax


def label(ax,x,y,s,size=11,color='ink',weight='normal',ha='center',va='center',**kw):
    ax.text(x,y,s,fontsize=size,color=C.get(color,color),fontweight=weight,
            ha=ha,va=va,linespacing=1.35,**kw)


def panel(ax,x,y,letter,title):
    label(ax,x,y,letter,20,weight='bold',ha='left')
    label(ax,x+.38,y,title,14,weight='bold',ha='left')


def arrow(ax,a,b,color='grey',lw=1.5,dashed=False,rad=0,head=12):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=head,
                 linewidth=lw,color=C.get(color,color),linestyle=(0,(4,3)) if dashed else '-',
                 connectionstyle=f'arc3,rad={rad}',shrinkA=2,shrinkB=3,zorder=5))


def cell(ax,x,y,r=.3,color='teal',phase=0):
    angles=np.linspace(0,2*np.pi,80)
    rr=r*(1+.06*np.sin(5*angles+phase)+.035*np.cos(3*angles))
    p=np.column_stack([x+rr*np.cos(angles),y+rr*np.sin(angles)])
    path=MPath(np.vstack([p,p[:1]]),[MPath.MOVETO]+[MPath.LINETO]*79+[MPath.CLOSEPOLY])
    ax.add_patch(PathPatch(path,facecolor=C[color+'_light'] if color+'_light' in C else '#eef2f4',edgecolor=C[color],lw=1.15))
    ax.add_patch(Ellipse((x-.045*r,y+.02*r),r*.75,r*.58,angle=30,facecolor=C[color],alpha=.35,edgecolor='none'))
    for k in range(4):
        a=.6+k*1.2+phase
        xx=x+.62*r*np.cos(a);yy=y+.56*r*np.sin(a)
        ax.plot([xx-.055*r,xx+.055*r],[yy-.055*r,yy+.055*r],color=C[color],lw=.8)


def helix(ax,x,y,w=.65,h=.28,color='ink'):
    t=np.linspace(0,2*np.pi,100);xs=x+t/(2*np.pi)*w
    ax.plot(xs,y+.5*h*np.sin(t),color=C[color],lw=1)
    ax.plot(xs,y-.5*h*np.sin(t),color=C[color],lw=1)
    for ti in np.linspace(0,2*np.pi,9):
        ax.plot([x+ti/(2*np.pi)*w]*2,[y-.5*h*np.sin(ti),y+.5*h*np.sin(ti)],color=C[color],lw=.65)


def scissors(ax,x,y):
    ax.add_patch(Circle((x-.10,y-.14),.07,fill=False,edgecolor=C['ink'],lw=1.1))
    ax.add_patch(Circle((x+.08,y-.14),.07,fill=False,edgecolor=C['ink'],lw=1.1))
    ax.plot([x-.05,x+.15],[y-.09,y+.17],color=C['ink'],lw=1.1)
    ax.plot([x+.03,x-.17],[y-.09,y+.17],color=C['ink'],lw=1.1)


def model(ax,x,y,r=.36,name='C',color='ink'):
    ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=C[color],lw=1.5))
    label(ax,x,y,name,17,color,weight='bold')
    for ang in [-.7,0,.7]:
        xx=x+r*np.cos(ang); yy=y+r*np.sin(ang)
        ax.add_patch(Circle((xx,yy),.035,facecolor=C[color],edgecolor='none'))


def vector(ax,x,y,values,w=2.2,h=.28,alpha=1):
    values=np.asarray(values);n=len(values)
    for i,v in enumerate(values):
        ax.add_patch(Rectangle((x+i*w/n,y),w/n-.012,h,
                              facecolor=CMAP((v+1)/2),edgecolor='white',lw=.25,alpha=alpha))


def petri(ax,x,y,r=.27,color='teal'):
    ax.add_patch(Ellipse((x,y),2*r,r*.65,facecolor=C[color+'_light'],edgecolor=C[color],lw=1))
    ax.add_patch(Ellipse((x,y+.04),2*r,r*.65,fill=False,edgecolor=C[color],lw=.75))
    for dx,dy in [(-.12,.02),(.07,.045),(.14,-.005),(-.02,-.04)]:
        ax.add_patch(Circle((x+dx,y+dy),.033,facecolor=C[color],alpha=.6,edgecolor='none'))


def microscope(ax,x,y,s=.7,color='ink'):
    ax.plot([x-.33*s,x+.3*s],[y-.33*s]*2,color=C[color],lw=1.5)
    ax.plot([x-.06*s,x-.2*s,x-.22*s],[y-.32*s,y-.05*s,y+.16*s],color=C[color],lw=1.8)
    ax.add_patch(FancyArrowPatch((x-.2*s,y+.17*s),(x+.12*s,y+.33*s),
                               connectionstyle='arc3,rad=-.5',arrowstyle='-',color=C[color],lw=1.8))
    ax.plot([x+.09*s,x+.20*s],[y+.27*s,y+.12*s],color=C[color],lw=4)
    ax.plot([x-.11*s,x+.24*s],[y-.07*s]*2,color=C[color],lw=1.5)
    ax.add_patch(Circle((x-.12*s,y+.02*s),.055*s,edgecolor=C[color],facecolor='white',lw=1))


def profile(ax,x,y,w=2.7,h=1.3):
    g=np.arange(12)
    mu=np.array([.1,-.22,-.18,.35,.52,.10,-.43,-.14,.29,.38,.08,-.10])
    p=mu+np.array([.03,.13,.09,-.21,-.12,.16,.20,.25,.16,-.24,-.02,.08])
    spread=np.array([.08,.11,.06,.12,.08,.06,.08,.14,.10,.07,.10,.06])
    xx=x+g/(len(g)-1)*w;scale=h/.95
    ax.plot([x-.06,x+w+.06],[y,y],color=C['pale'],lw=.8)
    ax.fill_between(xx,y+(mu-spread)*scale,y+(mu+spread)*scale,color=C['teal'],alpha=.13,lw=0)
    ax.plot(xx,y+mu*scale,color=C['teal'],lw=1.8)
    ax.plot(xx,y+p*scale,color=C['orange'],lw=1.7)
    # A local discrepancy marker illustrates the prediction/reference comparison.
    j=7;yy1=y+mu[j]*scale;yy2=y+p[j]*scale
    ax.plot([xx[j],xx[j]],[yy1,yy2],color=C['ink'],lw=1.05)
    ax.plot([xx[j]-.06,xx[j]+.06],[yy1]*2,color=C['ink'],lw=.8)
    ax.plot([xx[j]-.06,xx[j]+.06],[yy2]*2,color=C['ink'],lw=.8)
    label(ax,xx[j]+.22,(yy1+yy2)/2,r'$\Delta$',11,weight='bold')
    label(ax,x+w/2,y-.72,'Genes',10,'grey')
    ax.plot([x,x+.25],[y+.93]*2,color=C['teal'],lw=2)
    label(ax,x+.33,y+.93,'Reference',10,'teal',ha='left')
    ax.plot([x+1.65,x+1.9],[y+.93]*2,color=C['orange'],lw=2)
    label(ax,x+1.98,y+.93,'Prediction',10,'orange',ha='left')


def residuals(ax,x,y,color='blue'):
    for i,v in enumerate([.18,.08,.26,.14,.05]):
        ax.plot([x+i*.14]*2,[y,y+v],color=C[color],lw=2)
    ax.plot([x-.08,x+.64],[y,y],color=C['pale'],lw=.7)


def learnt_reader(ax,x,y,color='blue'):
    for end in [(x-.20,y-.17),(x+.20,y-.17)]:
        ax.plot([x,end[0]],[y+.18,end[1]],color=C[color],lw=1.05)
    for xx,yy in [(x,y+.18),(x-.20,y-.17),(x+.20,y-.17)]:
        ax.add_patch(Circle((xx,yy),.063,facecolor=C[color],edgecolor='white',lw=.4))


def certificate_lane(ax,zh=False):
    ax.plot([.25,15.75],[.69,.69],color=C['pale'],lw=.7)
    label(ax,.35,.37,'Parallel family certificate' if not zh else '并行家族证书',10,'grey',weight='bold',ha='left')
    t=np.linspace(0,1,25)
    for i in range(3):
        ax.plot(4.2+t*.8,.31+.1*np.sin(6*t+i)+i*.035,color=C['grey'],lw=.65,alpha=.75)
    arrow(ax,(5.2,.37),(5.8,.37),lw=.8,head=9)
    label(ax,6.3,.37,r'$D_F \leq E_F$',12,'grey')
    label(ax,8.15,.37,'Registered family spread → error lower bound' if not zh else '注册家族的分歧 → 家族误差下界',10,'grey',ha='left')
    label(ax,15.7,.37,'Separate from ranking' if not zh else '与经验排序分开',9,'grey',ha='right')


def method_overview(out,zh=False):
    fig,ax=canvas(16,8.2)
    panel(ax,.25,7.79,'a','New prediction' if not zh else '新预测器与未知结果')
    panel(ax,4.7,7.79,'b','Reference-aware risk' if not zh else '以公共参照判断风险')
    panel(ax,12.85,7.79,'c','Prioritise review' if not zh else '有限复核与反馈')
    for x in [4.32,12.55]:ax.plot([x,x],[1.02,7.42],color=C['pale'],lw=.7)
    # a: a prediction is available before the perturbation outcome is observed.
    for dx,dy,r in [(-.28,.10,.28),(.28,.15,.27),(.0,-.22,.3)]:cell(ax,1.75+dx,6.47+dy,r,'teal',dx)
    helix(ax,2.45,6.67,.58,.26);scissors(ax,2.73,6.71)
    label(ax,1.91,5.91,'Perturbation + context' if not zh else '扰动与细胞背景',11)
    arrow(ax,(1.88,5.68),(1.88,5.2),'ink')
    model(ax,1.88,4.76,.39,'C')
    label(ax,2.52,4.79,'Frozen predictor' if not zh else '冻结预测器',11,ha='left')
    arrow(ax,(1.88,4.31),(1.88,3.80),'ink')
    vec=[.1,-.2,.7,.4,-.65,-.15,.29,.63,-.4,.1,.24,-.1]
    vector(ax,.65,3.34,vec,2.75,.33)
    label(ax,2.02,2.99,r'Predicted effect $\hat{y}_C$' if not zh else '预测生物效应',11,'orange')
    vector(ax,.65,1.95,np.zeros(12),2.75,.29,alpha=.30)
    label(ax,2.02,2.10,'?',24,'grey',weight='bold')
    label(ax,2.02,1.57,'Outcome not yet observed' if not zh else '当前任务真值未知',11,'grey')
    # b: historical biological responses are the main path.
    label(ax,7.19,7.1,'Public perturbation experiments' if not zh else '合法公共扰动实验',12,'teal',weight='bold')
    for i,x in enumerate([5.76,7.18,8.60]):
        petri(ax,x,6.60,.34)
        for j in range(2):cell(ax,x-.15+j*.30,6.23,.10,'teal',i+j)
        vector(ax,x-.52,5.77,np.roll(vec,i)/1.4,1.04,.23)
    label(ax,7.20,5.4,'Matched histories across experiments' if not zh else '匹配扰动的历史响应',10,'grey')
    arrow(ax,(7.17,5.18),(7.17,4.85),'teal',lw=2.3)
    profile(ax,5.15,3.25,3.43,1.19)
    arrow(ax,(3.51,3.50),(5.02,3.50),'orange',lw=1.75)
    label(ax,6.90,2.21,'Support · conflict · history variation' if not zh else '支持量 · 冲突 · 历史分散度',10,'teal')
    arrow(ax,(8.74,3.66),(9.80,3.66),'teal',lw=2.4)
    label(ax,9.29,3.97,r'$D, V$',10,'teal')
    ax.add_patch(Circle((10.63,3.69),.74,facecolor=C['teal_light'],edgecolor=C['teal'],lw=1.7))
    label(ax,10.63,3.84,'R',30,'teal',weight='bold')
    label(ax,10.63,3.35,'Risk',11,'teal')
    label(ax,10.63,4.87,'SafeConf',15,weight='bold')
    label(ax,10.63,2.69,r'PublicRule: $\sqrt{D^2+V}$',11,'teal')
    # Optional learnt readers act on risk, not on biological reference construction.
    model(ax,5.63,1.63,.24,'A','blue');model(ax,6.24,1.63,.24,'B','blue')
    residuals(ax,6.79,1.50)
    label(ax,6.30,1.04,'Predictions + errors' if not zh else '旧模型预测及真实错误',10,'blue')
    arrow(ax,(7.5,1.66),(7.9,1.66),'blue',lw=1.0,head=9)
    learnt_reader(ax,8.3,1.66,'blue')
    label(ax,8.3,1.04,'Shared risk learner' if not zh else '共享风险学习器',9,'blue')
    arrow(ax,(8.61,1.96),(10.09,3.19),'blue',dashed=True,rad=.12)
    label(ax,9.10,2.18,'Optional transfer' if not zh else '可选错误经验迁移',9,'blue',rotation=31)
    # c: one ordering, a limited experimental check and feedback for later tasks.
    arrow(ax,(11.36,4.00),(12.91,4.90),'teal',lw=2.3,rad=-.08)
    widths=[1.35,1.18,.98,.85,.73,.61,.52,.43,.34,.23]
    for i,ww in enumerate(widths):
        yy=6.27-i*.245
        color='red' if i<2 else 'grey'
        ax.add_patch(Circle((13.02,yy+.055),.045,facecolor=C[color],edgecolor='none'))
        ax.add_patch(Rectangle((13.17,yy),ww,.11,facecolor=C[color],edgecolor='none',alpha=.95 if i<2 else .60))
    label(ax,13.68,6.94,'Risk ranking' if not zh else '唯一风险排序',11,weight='bold')
    ax.plot([14.67,14.78,14.78,14.67],[6.36,6.36,5.97,5.97],color=C['red'],lw=1.1)
    arrow(ax,(14.87,6.15),(15.49,6.15),'red',lw=1.1,head=9)
    microscope(ax,15.3,5.59,.78)
    label(ax,14.97,5.03,'Review top 20%' if not zh else '复核前 20%',10,'red')
    label(ax,13.70,3.47,'High → low risk' if not zh else '高风险 → 低风险',10,'grey')
    arrow(ax,(15.19,4.75),(15.19,3.01),'orange',dashed=True,lw=1.1)
    petri(ax,15.15,2.68,.24,'orange')
    residuals(ax,13.90,2.49,'orange')
    label(ax,14.68,2.13,'Observed C errors' if not zh else 'C 的真实错误反馈',10,'orange')
    arrow(ax,(13.93,2.34),(13.31,1.88),'orange',lw=1.0,head=9)
    learnt_reader(ax,13.01,1.78,'orange')
    arrow(ax,(12.64,1.89),(11.07,3.10),'orange',dashed=True,rad=-.14)
    label(ax,12.42,1.29,'Optional adaptation for later predictions' if not zh else '可选个性化更新，服务后续任务',10,'orange')
    label(ax,.26,.88,'Solid: public reference pathway   Dashed: optional learnt risk readers' if not zh else '实线：公共参照主干　虚线：可选的风险学习器',9,'grey',ha='left')
    label(ax,15.74,.88,'Current McFaline default: PublicRule' if not zh else '当前 McFaline 默认：PublicRule',9,'teal',ha='right')
    certificate_lane(ax,zh)
    save(fig,out,'METHOD_OVERVIEW_ZH' if zh else 'METHOD_OVERVIEW')


def evolution(out):
    fig,ax=canvas(16,8.0)
    titles=['Does disagreement survive shift?','Does the score beat magnitude?',
            'Does better correlation improve review?','What can be reused for a new model?']
    for x,letter,title in zip([.25,4.28,8.30,12.30],list('abcd'),titles):
        label(ax,x,7.57,letter,20,weight='bold',ha='left')
        words=title.split();middle=len(words)//2
        label(ax,x+.35,7.56,' '.join(words[:middle])+'\n'+' '.join(words[middle:]),12,weight='bold',ha='left')
    for x in [4.08,8.1,12.13]:ax.plot([x,x],[1.4,7.15],color=C['pale'],lw=.7)
    # a: actual E189 ranges, not a confidence interval.
    names=['Random pair','Unseen perturbation','Unseen context','Double unseen']
    limits=[(.368,.412),(.210,.247),(-.095,-.013),(-.349,-.241)]
    x0=2.80;sc=1.50
    ax.plot([x0,x0],[3.7,6.50],color=C['pale'],lw=.85)
    for i,(n,(lo,hi)) in enumerate(zip(names,limits)):
        y=6.20-i*.63;color='teal' if i<2 else 'red'
        label(ax,.39,y+.10,n,9,ha='left')
        ax.plot([x0+lo*sc,x0+hi*sc],[y,y],color=C[color],lw=3,solid_capstyle='round')
        ax.add_patch(Circle((x0+(lo+hi)/2*sc,y),.047,facecolor=C[color],edgecolor='none'))
        label(ax,.39,y-.13,f'{lo:.2f} to {hi:.2f}',8.5,color,ha='left')
    label(ax,2.14,3.65,'E189 · ρ range across support budgets',9,'grey')
    # b: E201 marginal associations; conditional association is distinct.
    xx=[5.17,6.65];vals=[.40821841982841595,.618852826256313]
    for x,v,col,n in zip(xx,vals,['blue','orange'],['Old score','Magnitude']):
        ax.add_patch(Rectangle((x,4.22),.60,v*2.7,facecolor=C[col],edgecolor='none'))
        label(ax,x+.3,4.32+v*2.7,f'{v:.3f}',15,col,weight='bold')
        label(ax,x+.3,3.96,n,10)
    label(ax,6.1,6.5,'E201 · family RMS error',10,'grey')
    label(ax,6.1,3.32,'Partial ρ | magnitude = 0.250',11,'blue')
    # c: two separately labelled observed increments and their saved intervals.
    label(ax,10.06,6.41,'SafeConf-M: 0.8Q(M) + 0.2Q(S)',10,'grey')
    xzero=9.72;scale=26
    ax.plot([xzero,xzero],[3.62,5.91],color=C['pale'],lw=1)
    for y,point,lo,hi,n,col in [(5.52,.0183,.0091,.0271,'ΔSpearman','teal'),(4.36,.0134,-.0139,.0520,'ΔU20','orange')]:
        ax.plot([xzero+lo*scale,xzero+hi*scale],[y,y],color=C[col],lw=1.8)
        ax.plot([xzero+lo*scale]*2,[y-.06,y+.06],color=C[col],lw=1)
        ax.plot([xzero+hi*scale]*2,[y-.06,y+.06],color=C[col],lw=1)
        ax.add_patch(Circle((xzero+point*scale,y),.06,facecolor=C[col],edgecolor='white',lw=.5))
        label(ax,8.40,y+.35,n,10,ha='left')
        label(ax,10.04,y-.29,f'{point:+.4f}; CI [{lo:+.4f}, {hi:+.4f}]',8.4,col)
    label(ax,10.04,3.32,'Sept. DEV · observed paired increments',9,'grey')
    # d: the information condition made explicit by PertEMA.
    label(ax,14.05,6.72,'After reviewing PertEMA',10,'grey')
    model(ax,13.12,5.90,.30,'C');residuals(ax,13.79,5.76,'orange')
    label(ax,14.04,5.27,'Own errors → post-hoc learner',9,'grey')
    arrow(ax,(14.05,4.95),(14.05,4.51),'grey',head=10)
    for i,x in enumerate([13.18,13.91]):petri(ax,x,4.04,.24)
    model(ax,14.65,4.07,.22,'A','blue');model(ax,15.20,4.07,.22,'B','blue')
    label(ax,14.05,3.42,'Public experiments + other-model errors',9,'teal')
    # Evidence → method changes are the organising axis, not a project timeline.
    changes=[('Disagreement is setting-dependent','明确误差对象与适用设置'),
             ('Magnitude becomes the anchor','保留幅度，研究额外信息怎样使用'),
             ('Correlation and review differ','固定复核收益单独验收'),
             ('Test cross-predictor information','分别计量 Public / Source / Target')]
    for x,(en,cn) in zip([.30,4.31,8.33,12.30],changes):
        arrow(ax,(x+1.70,2.92),(x+1.70,2.50),'ink',head=11)
        label(ax,x+1.75,2.16,en,10,weight='bold')
        label(ax,x+1.75,1.70,cn,10,'grey')
    ax.plot([.25,15.75],[1.22,1.22],color=C['pale'],lw=.7)
    label(ax,.35,.85,'Parallel since July: registered-family bounds',10,'grey',weight='bold',ha='left')
    label(ax,15.6,.85,'E181: 2,393 tasks, zero lower-bound violations · E182 upper-bound gate FAIL',9,'grey',ha='right')
    label(ax,.35,.36,'Each panel uses its own registered cohort and error target. a: support-budget ranges; c: 95% paired CIs.',9,'grey',ha='left')
    save(fig,out,'EVIDENCE_DRIVEN_EVOLUTION')


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=260,facecolor='white')
    fig.savefig(out/(name+'.svg'),facecolor='white')
    p=out/(name+'.svg');p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
    plt.close(fig)


def strong_baselines(out):
    point=out/'SOURCE_MAGNITUDE_POINT_COMPARISON.csv'
    if not point.is_file():return
    source=pd.read_csv(point)
    ci=pd.read_csv(out/'SOURCE_MAGNITUDE_PAIRED_BOOTSTRAP.csv')
    support=pd.read_csv(out/'PUBLIC_SUPPORT_POINT_COMPARISON.csv')
    prior=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/review_repair_20261007_v1/SOURCE_REAL_COMPARISON.csv'
    public=pd.read_csv(prior)
    fig,axs=plt.subplots(1,2,figsize=(13.7,5.7),gridspec_kw={'width_ratios':[1.6,1]})
    directions=[('TxPert_Exphormer','TxPert_GAT'),('TxPert_GAT','TxPert_Exphormer')]
    for i,(name,col) in enumerate([('Public rank',C['grey']),('Magnitude',C['orange']),('SourceHGB',C['blue'])]):
        vals=[]
        for s,t in directions:
            part=public[(public.source==s)&(public.target==t)&(public.method=='always_public')] if name=='Public rank' else source[(source.source==s)&(source.target==t)&(source.method==name)]
            vals.append(part.u20.iloc[0])
        bars=axs[0].bar(np.arange(2)+(i-1)*.23,vals,width=.215,color=col,label=name)
        axs[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=11)
    axs[0].set_xticks([0,1],['Exphormer → GAT','GAT → Exphormer'])
    axs[0].set_title('a  Source versus the strong prediction baseline',loc='left',fontsize=13,weight='bold',pad=18)
    axs[0].legend(loc='lower left',frameon=False,fontsize=10)
    for i,(s,t) in enumerate(directions):
        r=ci[(ci.source==s)&(ci.target==t)].iloc[0]
        axs[0].text(i,-.15,f'Δ Source − magnitude: {r.delta_utility20:+.3f}\n95% CI [{r.ci95_lower:+.3f}, {r.ci95_upper:+.3f}]',ha='center',va='top',fontsize=9)
    vals=[support[support.method==m].u20.iloc[0] for m in ['NegativeHistorySupport','PublicRule']]
    bars=axs[1].bar([0,1],vals,color=[C['grey'],C['teal']],width=.58)
    axs[1].bar_label(bars,fmt='%.3f',padding=4,fontsize=12)
    axs[1].set_xticks([0,1],['History support','PublicRule'])
    axs[1].set_title('b  Public reference versus support',loc='left',fontsize=13,weight='bold',pad=18)
    axs[1].text(.5,-.15,'212 McFaline tasks / 152 gene clusters\nPoint comparison; support is not experimental quality',ha='center',va='top',fontsize=9)
    for ax in axs:
        ax.set_ylim(0,1.02);ax.set_ylabel('Context-macro U20')
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.06,right=.985,top=.84,bottom=.23,wspace=.30)
    save(fig,out,'APPENDIX_STRONG_BASELINES')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    setup();method_overview(out);method_overview(out,True);evolution(out);strong_baselines(out)
    originals={}
    for n in ['MEETING_RESULTS.png','MEETING_RESULTS.svg','03_SOURCE_RESULT.png','03_SOURCE_RESULT.svg',
              '04_TARGET_RESULT.png','04_TARGET_RESULT.svg','05_REVIEW_RESULT.png','05_REVIEW_RESULT.svg']:
        originals[n]=sha(OLD/n);shutil.copyfile(OLD/n,out/n);assert originals[n]==sha(out/n)
    papers={n:{'path':str(Path('/home/yyf/师姐论文')/n),'sha256':sha(Path('/home/yyf/师姐论文')/n),'figure':'Fig. 1',
               'pdf_page':p} for n,p in [('scGPT.pdf',3),('scfoundation.pdf',3),('C.Origami.pdf',2)]}
    manifest={'role':'original scientific concept diagrams for meeting','source_commit':'50ce0165b573802cb1e8c7351b674a8b3df76ed1',
              'style_references_inspected':papers,'reference_figures_not_redistributed':True,
              'illustrative_profiles_and_cells_not_empirical_results':True,'old_result_plots_unchanged':originals,
              'script_sha256':sha(Path(__file__)),
              'outputs':{p.name:sha(p) for p in out.iterdir() if p.suffix in ['.png','.svg']}}
    (out/'FIGURE_MANIFEST_V2.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out),'scientific_figures':3,'original_results_unchanged':True},ensure_ascii=False))


if __name__=='__main__':main()
