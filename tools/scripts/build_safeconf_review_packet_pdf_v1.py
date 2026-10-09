#!/usr/bin/env python3
"""Build a review packet PDF from frozen SafeConf evidence.

This is a readable internal packet, not a journal-typeset manuscript. It never
refits models or modifies the frozen results.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import subprocess
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/研究推进/20261009_投稿证据_v21'
DRAFT=BASE/'submission_draft_v1'
EVID=BASE/'evidence_freeze_v1'
OUT=Path(os.environ.get('SAFECONF_REVIEW_OUT', str(EVID/'REVIEW_PACKET.pdf')))
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')


def page_text(pdf,title,paragraphs,fontsize=11):
    fig=plt.figure(figsize=(8.27,11.69));fig.patch.set_facecolor('white')
    fig.text(.08,.94,title,fontsize=18,fontweight='bold',va='top',color='#1e2d35',fontproperties=FONT)
    y=.88
    for para in paragraphs:
        lines=[]
        for line in para.split('\n'):
            lines.extend(textwrap.wrap(line,width=92) or [''])
        block='\n'.join(lines)
        fig.text(.08,y,block,fontsize=fontsize,va='top',linespacing=1.45,color='#263238',fontproperties=FONT)
        y-=.045*max(1,len(lines))+.025
        if y<.09:break
    pdf.savefig(fig,bbox_inches='tight');plt.close(fig)


def page_figure(pdf,path,title,caption):
    fig=plt.figure(figsize=(11.69,8.27));fig.patch.set_facecolor('white')
    ax=fig.add_axes([.04,.12,.92,.82]);ax.imshow(imread(path));ax.axis('off')
    fig.text(.05,.96,title,fontsize=16,fontweight='bold',va='top',color='#1e2d35',fontproperties=FONT)
    fig.text(.05,.045,caption,fontsize=9,va='bottom',color='#4d5c63',fontproperties=FONT)
    pdf.savefig(fig,bbox_inches='tight');plt.close(fig)


def main():
    manuscript=(DRAFT/'MANUSCRIPT_DRAFT.md').read_text()
    with PdfPages(OUT) as pdf:
        page_text(pdf,'SafeConf | review packet',[
            'Public perturbation evidence enables cold-start risk auditing for single-cell perturbation prediction',
            '投稿目标：IEEE/ACM Transactions on Computational Biology and Bioinformatics（TCBB，CCF-B方向）',
            f'证据冻结版本：2026-10-09；当前Git提交：{subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip()}；模型训练和数据下载均未由本PDF构建触发。',
            '本文件用于研究组审阅和投稿前校对，正文的最终期刊排版仍由LaTeX模板完成。'],12)
        page_text(pdf,'Abstract', [manuscript.split('## 1. Introduction')[0].replace('# Public perturbation evidence enables cold-start risk auditing for single-cell perturbation prediction','').strip()],10)
        page_text(pdf,'Study design and adopted method',[
            'SafeConf将公共真实扰动实验、Source错误和Target反馈登记为三类信息。默认系统只使用PublicRule；Source和Target作为有预算、有边界的候选扩展。',
            'McFaline：212任务/152基因簇；KOLF：600上游训练、300开发、300确认基因。公共历史不读取当前确认任务答案；风险分数在确认读取前冻结。',
            'KOLF共有公共响应轴为1366/1400。缺失坐标不补零；公共参照只在实际共同轴上计算。无历史任务走固定幅度回退。',
            '主端点是20%复核的U20、严重错误命中和复核后的剩余误差；区间按扰动基因簇进行5000次配对bootstrap。'],10)
        figs=[('FIG1_ZERO_ERROR.png','Figure 1 | Zero target-error risk auditing','McFaline与KOLF的固定20%复核结果；全量、任务、回退状态分开保存。'),
              ('FIG2_CONTENT.png','Figure 2 | Content attribution','20个支持匹配置乱；内容增量和历史支持/能量基线不混写。'),
              ('FIG3_FEEDBACK.png','Figure 3 | Feedback budget','McFaline全量global和context-macro端点分开展示。'),
              ('FIG4_KOLF.png','Figure 4 | KOLF frozen independent evaluation','Ridge通过能力门后完成的冻结独立评价；KOLF反馈增量区间跨零。'),
              ('FIG5_FRANGIEH_STRESS.png','Figure 5 | Cross-family stress evidence','GEARS↔scGPT压力证据；六个上游能力门失败，不能当作合格独立确认。')]
        for fn,title,caption in figs:page_figure(pdf,EVID/'figures'/fn,title,caption)
        page_text(pdf,'Key results and claim boundaries',[
            'McFaline：Public在DecoderOnly和SAMS上分别命中22/43与24/43个最高误差任务，幅度基线为4/43与5/43；剩余误差下降7.07%与6.84%。',
            'Frangieh：PublicHGB相对Magnitude的paired U20增量为+0.460 [0.317,0.704]与+0.309 [0.079,0.440]；上游能力全部失败，保留为stress evidence。',
            'KOLF：Ridge通过能力门；Public覆盖228/300；Public相对Magnitude点增量+0.136，95%区间跨零；严重错误命中17对17，因此解读为可运行性和信息账本验证，而非已确认普遍优越。',
            '当前论文主张是公共证据可启动风险审核和跨模型压力下仍有价值；不声称所有学习器、所有背景、所有反馈预算均单调改善。'],10)
    print(str(OUT))


if __name__=='__main__':main()
