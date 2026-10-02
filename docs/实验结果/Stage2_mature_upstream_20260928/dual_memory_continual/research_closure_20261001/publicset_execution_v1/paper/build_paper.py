#!/usr/bin/env python3
"""Generate a standalone LaTeX draft and figures from the written manuscript/results."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import csv
import os
import time
import hashlib
import pandas as pd

P = Path(__file__).resolve().parent

def latex_text(text):
    replacements = {'–':'--','—':'---','→':r'$\rightarrow$', '×':r'$\times$',
                    '−':'-', '≥':r'$\geq$', '≤':r'$\leq$', '／':'/', '＋':'+',
                    '“':'``','”':"''", '’':"'", 'α':r'$\alpha$'}
    placeholders = {}
    def hold(s):
        key = f'ZZTOKEN{len(placeholders)}ZZ'
        placeholders[key] = s
        return key
    text = re.sub(r'\[@([A-Za-z0-9_]+)\]', lambda m:hold(r'\cite{'+m.group(1)+'}'), text)
    text = re.sub(r'`([^`]+)`', lambda m:hold(r'\texttt{\detokenize{'+m.group(1)+'}}'), text)
    text = re.sub(r'https?://[^\s,]+',lambda m:hold(r'\url{'+m.group(0).rstrip('.')+'}'+('.' if m.group(0).endswith('.') else '')),text)
    for old,new in replacements.items():
        text=text.replace(old,hold(new))
    escapes = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
               '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    text=''.join(escapes.get(c,c) for c in text)
    text=re.sub(r'\*\*(.+?)\*\*',lambda m:r'\textbf{'+m.group(1)+'}',text)
    for key,value in placeholders.items():text=text.replace(key,value)
    return text

def render_markdown(text):
    lines=text.splitlines()
    title=lines[0].removeprefix('# ')
    parts=[];i=1;abstract=False
    while i<len(lines):
        line=lines[i]
        if not line.strip() or line.startswith('Author names'):
            i+=1;continue
        if line=='## Abstract':
            parts.append(r'\begin{abstract}');abstract=True;i+=1;continue
        if line.startswith('Keywords:'):
            if abstract:parts.append(r'\end{abstract}');abstract=False
            parts.append(r'\noindent\textbf{Index terms---}'+latex_text(line.removeprefix('Keywords: ')));i+=1;continue
        if line.startswith('## '):
            heading=re.sub(r'^\d+\.\s*','',line[3:])
            parts.append(r'\section{'+latex_text(heading)+'}');i+=1;continue
        if line.startswith('### '):
            heading=re.sub(r'^\d+\.\d+\.?(?:\s*)','',line[4:])
            parts.append(r'\subsection{'+latex_text(heading)+'}');i+=1;continue
        if line.strip()=='$$':
            eq=[];i+=1
            while i<len(lines) and lines[i].strip()!='$$':eq.append(lines[i]);i+=1
            parts.append('\\[\n'+'\n'.join(eq)+'\n\\]');i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                i+=1
            n=len(rows[0]); width=round(0.93/n,3)
            parts.append(r'\begin{table*}[t]\centering\small\setlength{\tabcolsep}{2pt}')
            match=re.match(r'Table (\d+)\.\s*(.*)',rows[0][0])
            if match:
                parts.append(r'\caption{'+latex_text(match.group(2))+'}')
                parts.append(r'\label{tab:main'+match.group(1)+'}')
                first_headers={1:'Cohort',2:'Method',3:'Direction',4:'Method',
                               5:'Feedback budget',6:'Direction',7:'Reference',
                               8:'Reference / rule',9:'Scope and comparison',10:'Reader / information'}
                rows[0][0]=first_headers.get(int(match.group(1)),'Comparison')
            parts.append(r'\begin{tabular}{'+''.join(r'>{\raggedright\arraybackslash}p{'+str(width)+r'\textwidth}' for _ in range(n))+'}')
            parts.append(r'\toprule')
            for j,row in enumerate(rows):
                parts.append(' & '.join(latex_text(c) for c in row)+r' \\')
                if j==0:parts.append(r'\midrule')
            parts.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','$$')):
            paragraph.append(lines[i]);i+=1
        parts.append(latex_text(' '.join(paragraph))+'\n')

    return title, '\n\n'.join(parts)

title, body = render_markdown((P/'PAPER.md').read_text())

preamble=r'''% Generated from PAPER.md by build_paper.py. Do not edit this mirror by hand.
\IfFileExists{IEEEtran.cls}{\documentclass[journal]{IEEEtran}}{\documentclass[10pt,twocolumn]{article}\usepackage[margin=0.75in]{geometry}}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,hyperref}
\hypersetup{hidelinks}
\setlength{\emergencystretch}{2em}
\title{'''+latex_text(title)+r'''}
\author{Author names: [AUTHOR NAMES]\\Affiliation: [DEPARTMENT / INSTITUTION]\\Corresponding author: [NAME], [EMAIL]}
\date{}
\begin{document}
\maketitle
'''
tail=r'''
\clearpage
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/method_overview.pdf}
\caption{SafeConf separates biological supervision of the public reference, source-error supervision of the risk function, and target-feedback personalization. The four biological builders are assessed under common information and output contracts. Biological reconstruction and risk-reader increments are evaluated separately.}
\end{figure*}
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/core_information.pdf}
\caption{First registered seed on the common 2,840-gene axis. The useful information source changes across cohorts; the support-only comparator remains visible. These are point estimates, not independent replications or confidence intervals.}
\end{figure*}
\begin{figure*}[t]
\centering\includegraphics[width=.85\textwidth]{figures/feedback_information.pdf}
\caption{Strict McFaline feedback evaluation on 212 tasks and 152 genes. The percentage refers to the feedback pool. The evaluation pool is never included in the denominator. Curves display the first registered seed and do not imply monotonic improvement.}
\end{figure*}
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
'''
additional_figures = r'''
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/publicset_primary_utility.pdf}
\caption{Complete five-fold public-reference comparison on Source and McFaline. Metrics are calculated per seed, averaged within context-by-fold, then macro-averaged. Bars include 5,000 paired gene-bootstrap percentile intervals. Fixed references use the same eligible records. This is development/seen evidence.}
\end{figure*}
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/publicset_paired_readouts.pdf}
\caption{Paired representation increments under the primary weighted historical reader and auxiliary direct-distance reader. These fixed-reader comparisons do not determine the jointly supervised builder-reader result. Gene draws synchronize both Source upstreams and fitting seeds.}
\end{figure*}
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/publicset_biology_strata.pdf}
\caption{Biological reconstruction and its stratum variation in the completed four-arm experiment. Reconstruction is assessed independently from risk ranking; the constrained reconstruction diagnostic is not a risk oracle.}
\end{figure*}
'''
tail = tail.replace(r'\bibliographystyle', additional_figures + r'\bibliographystyle')
(P/'main.tex').write_text(preamble+body+tail)
supplement_title, supplement_body = render_markdown((P/'SUPPLEMENT.md').read_text())
supplement_preamble = r'''\documentclass[journal,onecolumn]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,hyperref}
\hypersetup{hidelinks}
\setlength{\emergencystretch}{2em}
\title{SafeConf: Supplementary Methods and Results}
\author{Author names: [AUTHOR NAMES]}
\date{}
\begin{document}
\maketitle
'''
(P/'supplement.tex').write_text(supplement_preamble+supplement_body+r'\end{document}'+'\n')
for original, target in [('FIG01_PRIMARY_UTILITY', 'publicset_primary_utility'),
                         ('FIG02_PAIRED_MAIN_AUXILIARY', 'publicset_paired_readouts'),
                         ('FIG03_BIOLOGY_STRATA', 'publicset_biology_strata')]:
    for ext in ['pdf', 'png']:
        shutil.copyfile(P.parent/'statistics_v1'/f'{original}.{ext}', P/'figures'/f'{target}.{ext}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig,ax=plt.subplots(figsize=(11,4.6))
ax.set_xlim(0,11);ax.set_ylim(0,4.6);ax.axis('off')
def box(x,y,w,h,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.08',
        facecolor=color,edgecolor='#3B4652',lw=.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.1,color='#3B4652'))
box(.2,3.35,2.35,.75,'Real public experiments\nBiological effects + provenance','#D8EBF4')
box(3.15,3.35,2.35,.75,'Public reference builder\nBiological reconstruction target','#D8EBF4')
box(6.1,3.35,2.35,.75,'Reference + dispersion\nSupport + original conflict','#D8EBF4')
arrow((2.55,3.72),(3.15,3.72));arrow((5.5,3.72),(6.1,3.72))
box(.2,1.9,2.35,.75,'Current prediction\nMagnitude + shape features','#EEEEEE')
box(3.15,1.9,2.35,.75,'Risk input\nPrediction + public discrepancy','#E9EEF3')
box(6.1,1.9,2.35,.75,'Shared risk learner\nSource-error supervision','#FAE4D0')
arrow((2.55,2.27),(3.15,2.27));arrow((5.5,2.27),(6.1,2.27));arrow((7.25,3.35),(4.35,2.65))
box(8.95,1.9,1.7,.75,'No target feedback\nRisk ranking','#F6EBCF')
arrow((8.45,2.27),(8.95,2.27))
box(.2,.35,2.35,.75,'Returned target outcomes\nHeld-out errors by model version','#DDEBDC')
box(3.15,.35,2.35,.75,'Target-specific learner\nPublic / shared inputs optional','#DDEBDC')
box(6.1,.35,2.35,.75,'Personalized risk\nSame feedback-label budget','#DDEBDC')
arrow((2.55,.72),(3.15,.72));arrow((5.5,.72),(6.1,.72));arrow((7.25,1.9),(4.35,1.1));arrow((4.05,1.9),(4.05,1.1))
ax.text(8.95,.72,'Versioned records\nTrain candidate\nValidate release',va='center',fontsize=9,color='#405344')
fig.tight_layout()
for suffix in ['pdf','png']:fig.savefig(P/f'figures/method_overview.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
core=pd.read_csv(P/'evidence/table2_core.csv')
support=pd.read_csv(P/'evidence/support_controls.csv')
fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
methods=['Magnitude','Prediction_hgb','Learned_WeightedHistoryDistance','Learned_hgb','NegativeHistorySupport']
labels=['Magnitude','Prediction HGB','Public distance','Shared HGB','Support only']
palette=['#666666','#A5A5A5','#337DB5','#D8772F','#609B63']
for ax,line in zip(axes,['Exphormer_to_GAT','GAT_to_Exphormer','TxPert_to_McFaline']):
    values=[]
    for method in methods:
        df=support if method=='NegativeHistorySupport' else core
        row=df[(df.line==line)&(df.method==method)&(df.seed==20260930)]
        if len(row)!=1:raise ValueError((line,method,len(row)))
        values.append(float(row.iloc[0].utility20))
    ax.bar(range(5),values,color=palette,width=.75)
    ax.set_xticks(range(5),labels,rotation=35,ha='right')
    ax.axhline(0,color='black',lw=.6)
    ax.set_title(line.replace('_to_',' → '))
    ax.set_ylim(-.3,1)
    for j,v in enumerate(values):ax.text(j,v+(.02 if v>=0 else -.025),f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
axes[0].set_ylabel('Utility@20 (higher is better)')
fig.tight_layout()
for suffix in ['pdf','png']:fig.savefig(P/f'figures/core_information.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)

feedback=pd.read_csv(P/'evidence/table4_feedback.csv')
fig,ax=plt.subplots(figsize=(7,4.1))
for method,label,color in [('Shared','Shared','#666666'),('TargetOnly_HGB','Target only','#9D658C'),
                           ('PublicTarget_HGB','Public + Target','#337DB5'),('SharedTarget_HGB','Shared + Target','#D8772F')]:
    df=feedback[(feedback.method==method)&(feedback.budget>0)].sort_values('budget')
    ax.plot(df.budget*100,df.utility20,'o-',label=label,color=color,lw=1.8,ms=4)
ax.set_xlabel('Opened feedback pool (%)');ax.set_ylabel('Utility@20 (higher is better)')
ax.set_xticks([10,25,50,75,100]);ax.set_ylim(0,.9);ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2)
fig.tight_layout()
for suffix in ['pdf','png']:fig.savefig(P/f'figures/feedback_information.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)

bib=(P/'references.bib').read_text()
keys=set(re.findall(r'@\w+\{([^,]+),',bib))
citations=set(re.findall(r'\[@([^\]]+)\]',(P/'PAPER.md').read_text()))
missing=sorted(citations-keys)
if missing:raise ValueError(f'Missing bibliography keys: {missing}')
status={'manuscript_words':len((P/'PAPER.md').read_text().split()),'citation_keys':sorted(citations),
        'missing_citations':missing,'figures_generated':6,'latex_generated':True,
        'latex_compiled':False,'latex_engine':shutil.which('pdflatex'),
        'new_model_fits':0,'new_test_truth_reads':0,
        'paper_sha256':hashlib.sha256((P/'PAPER.md').read_bytes()).hexdigest(),
        'supplement_sha256':hashlib.sha256((P/'SUPPLEMENT.md').read_bytes()).hexdigest()}
build=P/'build';build.mkdir(exist_ok=True)
tectonic=shutil.which('tectonic')
local_tectonic=build/'toolchain/tectonic'
if not tectonic and local_tectonic.is_file(): tectonic=str(local_tectonic)
commands=[]
bundle=None
env=os.environ.copy()
if status['latex_engine'] and shutil.which('bibtex'):
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex'],
              ['bibtex','build/main'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex']]
elif tectonic:
    status['latex_engine']=tectonic
    bundle_meta=build/'toolchain/TOOLCHAIN.json'
    bundle=json.loads(bundle_meta.read_text()).get('bundle') if bundle_meta.is_file() else None
    commands=[[tectonic,'--keep-logs','--keep-intermediates','--outdir',str(build)]+(['--bundle',bundle] if bundle else [])+['main.tex']]
    status['tectonic_bundle']=bundle
    env['XDG_CACHE_HOME']=str(build/'toolchain/cache')
start=time.time()
try:
    if commands:
        for j,cmd in enumerate(commands):
            with (build/f'compile_{j}.log').open('w') as log:
                run=subprocess.run(cmd,cwd=P,text=True,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180)
            if run.returncode: raise RuntimeError(f'LaTeX command failed (exit {run.returncode}): {cmd}')
        pdf=build/'main.pdf'
        if not pdf.is_file() or pdf.stat().st_size < 1000: raise RuntimeError('Compile returned success without a valid PDF artifact')
        status.update(latex_compiled=True,compile_status='COMPILED',pdf_bytes=pdf.stat().st_size,
                      pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
        if tectonic:
            supplementary_command = [tectonic, '--keep-logs', '--keep-intermediates', '--outdir', str(build)] + (['--bundle', bundle] if bundle else []) + ['supplement.tex']
        else:
            supplementary_command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory=build', 'supplement.tex']
        for repetition in range(2 if not tectonic else 1):
            with (build/f'compile_supplement_{repetition}.log').open('w') as log:
                run = subprocess.run(supplementary_command, cwd=P, text=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=180)
            if run.returncode:
                raise RuntimeError(f'Supplement compilation failed (exit {run.returncode})')
        supplementary_pdf = build/'supplement.pdf'
        if not supplementary_pdf.is_file() or supplementary_pdf.stat().st_size < 1000:
            raise RuntimeError('Supplement compilation returned no valid PDF')
        status.update(supplement_compiled=True, supplement_pdf_bytes=supplementary_pdf.stat().st_size,
                      supplement_pdf_sha256=hashlib.sha256(supplementary_pdf.read_bytes()).hexdigest(),
                      supplement_compile_command=supplementary_command)
    else:
        status['compile_status']='NOT_RUN_NO_LATEX_TOOLCHAIN'
except (RuntimeError,subprocess.TimeoutExpired) as exc:
    status.update(compile_status='FAILED',compile_error=str(exc))
finally:
    status['compile_seconds']=time.time()-start
    status['compile_commands']=commands
    cache=build/'toolchain/cache'
    status['toolchain_cache_bytes']=sum(f.stat().st_size for f in cache.rglob('*') if f.is_file()) if cache.is_dir() else 0
    (build/'BUILD_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status))
if status['compile_status']=='FAILED': raise SystemExit(1)
