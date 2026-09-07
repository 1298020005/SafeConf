#!/usr/bin/env python3
"""Universe of journals scored against locked E201/E204 evidence.

This is not every periodical on Earth. On 2026-09-07 the complete WoS SCIE
lists were opened:

* Mathematical & Computational Biology, 61 titles (impactfactor.cn sort-699, 3 pages)
* Biochemical Research Methods, 82 titles (impactfactor.cn sort-615, 3 pages)

plus Nature / Cell / Science / genomics / CCF titles a SafeConf manuscript
is routinely asked about. Each row is scored against the same eight evidence
lamps. Quartiles are not a publication guarantee.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
import numpy as np

PACK_REL = Path("docs/学习导航/20260906_论文审核与从零教学")
UNIVERSE_MD = "06_全球相关期刊逐本评价.md"
UNIVERSE_CSV = "journal_universe.csv"
OPENED_ON = "2026-09-07"
MCB_SOURCE = "impactfactor.cn sort-699（SCIE Mathematical & Computational Biology，完整 3 页，共 61 本）"
BRM_SOURCE = "impactfactor.cn sort-615（SCIE Biochemical Research Methods，完整 3 页，共 82 本）"

# Same lamps as paper_pack.EVIDENCE_HAVE. Tests require the two dicts equal.
EVIDENCE_ITEMS = (
    ("blind_e201", "四细胞系盲测封存"),
    ("magnitude_baseline", "报告预测幅度基线"),
    ("partial_info", "幅度之外仍有信息"),
    ("beat_magnitude", "单独排序超过幅度"),
    ("e204_formal", "风险训练正式效果"),
    ("e205_family", "跨架构模型家族"),
    ("new_external", "冻结后的新外部数据"),
    ("wet_mechanism", "湿实验或机制闭环"),
)
EVIDENCE_HAVE = {
    "blind_e201": 2,
    "magnitude_baseline": 2,
    "partial_info": 2,
    "beat_magnitude": 0,
    "e204_formal": 0,
    "e205_family": 0,
    "new_external": 0,
    "wet_mechanism": 0,
}
HAVE_FACT = {
    "blind_e201": "有：E201 主分析 1808 个任务，四细胞系先封存再解封",
    "magnitude_baseline": "有：预测幅度 Spearman 0.6189，20% 效用 0.5943",
    "partial_info": "有：控制幅度后偏 Spearman 0.2503，区间下限 0.2021",
    "beat_magnitude": "没有：SafeConf 0.4082 / 效用 0.3200，弱于幅度；效用差 -0.2743",
    "e204_formal": "没有：E204 只有工程验收，没有 80 轮正式效果",
    "e205_family": "没有：E205 未跑，四个种子仍是同一 GAT",
    "new_external": "没有：Frangieh 等不能再包装成冻结后的新外部",
    "wet_mechanism": "没有：当前主线没有湿实验或机制闭环",
}

VERDICT_CN = {
    "discuss_narrow": "可讨论（必须收窄主张）",
    "q2_not_ready": "二区候选但未就绪",
    "q1_off": "一区/顶刊不对轨或故事不够",
    "off_track": "学科不对轨",
    "review_only": "只收综述/方案，不收本类研究文",
}
VERDICT_ORDER = (
    "discuss_narrow",
    "q2_not_ready",
    "q1_off",
    "review_only",
    "off_track",
)

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "grey": "#4D4D4D",
}

# Official titles copied from the opened category pages (not memory).
MCB_OFFICIAL: list[tuple[str, str]] = [
    ("ACTA BIOTHEORETICA", "0001-5342"),
    ("Algorithms for Molecular Biology", ""),
    ("Annual Review of Biomedical Data Science", "2574-3414"),
    ("BULLETIN OF MATHEMATICAL BIOLOGY", "0092-8240"),
    ("BioData Mining", "1756-0381"),
    ("BIOINFORMATICS", "1367-4803"),
    ("BIOMETRICAL JOURNAL", "0323-3847"),
    ("BIOMETRICS", "0006-341X"),
    ("BIOMETRIKA", "0006-3444"),
    ("BIOSTATISTICS", "1465-4644"),
    ("BIOSYSTEMS", "0303-2647"),
    ("BMC BIOINFORMATICS", "1471-2105"),
    ("BRIEFINGS IN BIOINFORMATICS", "1467-5463"),
    ("COMPUTERS IN BIOLOGY AND MEDICINE", "0010-4825"),
    ("Current Bioinformatics", "1574-8936"),
    ("Database-The Journal of Biological Databases and Curation", "1758-0463"),
    ("Evolutionary Bioinformatics", "1176-9343"),
    ("Frontiers in Computational Neuroscience", ""),
    ("Frontiers in Neuroinformatics", ""),
    ("GENETIC EPIDEMIOLOGY", "0741-0395"),
    ("IEEE Journal of Biomedical and Health Informatics", "2168-2194"),
    ("IET Systems Biology", "1751-8849"),
    ("International Journal of Biomathematics", "1793-5245"),
    ("International Journal of Biostatistics", "2194-573X"),
    ("International Journal of Data Mining and Bioinformatics", "1748-5673"),
    ("International Journal for Numerical Methods in Biomedical Engineering", "2040-7939"),
    ("Interdisciplinary Sciences-Computational Life Sciences", "1913-2751"),
    ("JOURNAL OF AGRICULTURAL BIOLOGICAL AND ENVIRONMENTAL STATISTICS", "1085-7117"),
    ("Journal of Bioinformatics and Computational Biology", "0219-7200"),
    ("Journal of Biological Dynamics", "1751-3758"),
    ("JOURNAL OF BIOLOGICAL SYSTEMS", "0218-3390"),
    ("Journal of Biomedical Semantics", "2041-1480"),
    ("JOURNAL OF COMPUTATIONAL BIOLOGY", "1066-5277"),
    ("JOURNAL OF COMPUTATIONAL NEUROSCIENCE", "0929-5313"),
    ("JOURNAL OF MATHEMATICAL BIOLOGY", "0303-6812"),
    ("Journal of Mathematical Neuroscience", "2190-8567"),
    ("JOURNAL OF MOLECULAR GRAPHICS & MODELLING", "1093-3263"),
    ("JOURNAL OF THEORETICAL BIOLOGY", "0022-5193"),
    ("MATHEMATICAL BIOSCIENCES", "0025-5564"),
    ("Mathematical Biosciences and Engineering", "1547-1063"),
    ("MATHEMATICAL MEDICINE AND BIOLOGY-A JOURNAL OF THE IMA", "1477-8599"),
    ("Mathematical Modelling of Natural Phenomena", "0973-5348"),
    ("MEDICAL & BIOLOGICAL ENGINEERING & COMPUTING", "0140-0118"),
    ("Molecular Informatics", "1868-1743"),
    ("npj Systems Biology and Applications", ""),
    ("PLoS Computational Biology", "1553-734X"),
    ("Research Synthesis Methods", "1759-2879"),
    ("SAR AND QSAR IN ENVIRONMENTAL RESEARCH", "1062-936X"),
    ("Statistical Applications in Genetics and Molecular Biology", "2194-6302"),
    ("Statistics in Biopharmaceutical Research", "1946-6315"),
    ("Statistics and Its Interface", "1938-7989"),
    ("STATISTICS IN MEDICINE", "0277-6715"),
    ("STATISTICAL METHODS IN MEDICAL RESEARCH", "0962-2802"),
    ("THEORY IN BIOSCIENCES", "1431-7613"),
    ("THEORETICAL POPULATION BIOLOGY", "0040-5809"),
    ("Wiley Interdisciplinary Reviews-Computational Molecular Science", "1759-0876"),
    ("BMC Systems Biology", "1752-0509"),
    ("Computational Intelligence and Neuroscience", "1687-5265"),
    ("Computational and Mathematical Methods in Medicine", "1748-670X"),
    ("Journal of Medical Imaging and Health Informatics", "2156-7018"),
    ("Theoretical Biology and Medical Modelling", "1742-4682"),
]
BRM_OFFICIAL: list[tuple[str, str]] = [
    ("ACS Synthetic Biology", "2161-5063"),
    ("Acta Crystallographica Section D-Structural Biology", "2059-7983"),
    ("Acta Crystallographica Section F-Structural Biology Communications", ""),
    ("Algorithms for Molecular Biology", ""),
    ("ANALYTICAL AND BIOANALYTICAL CHEMISTRY", "1618-2642"),
    ("ANALYTICAL BIOCHEMISTRY", "0003-2697"),
    ("ASSAY AND DRUG DEVELOPMENT TECHNOLOGIES", "1540-658X"),
    ("Bioanalysis", "1757-6180"),
    ("BioChip Journal", "1976-0280"),
    ("BIOCONJUGATE CHEMISTRY", "1043-1802"),
    ("BIOINFORMATICS", "1367-4803"),
    ("BIOLOGICAL PROCEDURES ONLINE", ""),
    ("BIOLOGICALS", "1045-1056"),
    ("BIOMEDICAL CHROMATOGRAPHY", "0269-3879"),
    ("Biomedical Optics Express", "2156-7085"),
    ("Biomicrofluidics", "1932-1058"),
    ("BIOTECHNIQUES", "0736-6205"),
    ("Biotechnology Journal", "1860-6768"),
    ("BMC BIOINFORMATICS", "1471-2105"),
    ("BRIEFINGS IN BIOINFORMATICS", "1467-5463"),
    ("CHROMATOGRAPHIA", "0009-5893"),
    ("Clinical Proteomics", "1542-6416"),
    ("COMBINATORIAL CHEMISTRY & HIGH THROUGHPUT SCREENING", "1386-2073"),
    ("Current Bioinformatics", "1574-8936"),
    ("CURRENT ISSUES IN MOLECULAR BIOLOGY", "1467-3037"),
    ("CURRENT OPINION IN BIOTECHNOLOGY", "0958-1669"),
    ("Current Proteomics", "1570-1646"),
    ("CYTOMETRY PART A", "1552-4922"),
    ("Drug Testing and Analysis", "1942-7603"),
    ("ELECTROPHORESIS", "0173-0835"),
    ("Expert Review of Proteomics", "1478-9450"),
    ("IEEE-ACM Transactions on Computational Biology and Bioinformatics", "1545-5963"),
    ("IEEE TRANSACTIONS ON NANOBIOSCIENCE", "1536-1241"),
    ("IET Nanobiotechnology", "1751-8741"),
    ("JOURNAL OF THE AMERICAN SOCIETY FOR MASS SPECTROMETRY", "1044-0305"),
    ("Journal of Biological Engineering", "1754-1611"),
    ("JOURNAL OF BIOMEDICAL OPTICS", "1083-3668"),
    ("Journal of Biophotonics", "1864-063X"),
    ("Journal of Breath Research", "1752-7155"),
    ("JOURNAL OF CHROMATOGRAPHY A", "0021-9673"),
    ("JOURNAL OF CHROMATOGRAPHY B-ANALYTICAL TECHNOLOGIES IN THE BIOMEDICAL AND LIFE SCIENCES", "1570-0232"),
    ("JOURNAL OF CHROMATOGRAPHIC SCIENCE", "0021-9665"),
    ("JOURNAL OF COMPUTATIONAL BIOLOGY", "1066-5277"),
    ("JOURNAL OF FLUORESCENCE", "1053-0509"),
    ("JOURNAL OF IMMUNOLOGICAL METHODS", "0022-1759"),
    ("JOURNAL OF LABELLED COMPOUNDS & RADIOPHARMACEUTICALS", "0362-4803"),
    ("JOURNAL OF LIQUID CHROMATOGRAPHY & RELATED TECHNOLOGIES", "1082-6076"),
    ("JOURNAL OF MAGNETIC RESONANCE", "1090-7807"),
    ("JOURNAL OF MASS SPECTROMETRY", "1076-5174"),
    ("JOURNAL OF MICROBIOLOGICAL METHODS", "0167-7012"),
    ("JOURNAL OF MOLECULAR GRAPHICS & MODELLING", "1093-3263"),
    ("JOURNAL OF NEUROSCIENCE METHODS", "0165-0270"),
    ("JOURNAL OF PROTEOME RESEARCH", "1535-3893"),
    ("Journal of Proteomics", "1874-3919"),
    ("Journal of Spectroscopy", "2314-4920"),
    ("JOURNAL OF VIROLOGICAL METHODS", "0166-0934"),
    ("LAB ON A CHIP", "1473-0197"),
    ("METHODS", "1046-2023"),
    ("MOLECULAR AND CELLULAR PROBES", "0890-8508"),
    ("MOLECULAR & CELLULAR PROTEOMICS", "1535-9476"),
    ("Molecular Imaging", "1535-3508"),
    ("NATURE METHODS", "1548-7091"),
    ("Nature Protocols", "1754-2189"),
    ("New Biotechnology", "1871-6784"),
    ("PHYTOCHEMICAL ANALYSIS", "0958-0344"),
    ("Plant Methods", ""),
    ("PLANT MOLECULAR BIOLOGY REPORTER", "0735-9640"),
    ("PLoS Computational Biology", "1553-734X"),
    ("PREPARATIVE BIOCHEMISTRY & BIOTECHNOLOGY", "1082-6068"),
    ("PROTEIN EXPRESSION AND PURIFICATION", "1046-5928"),
    ("Proteomics Clinical Applications", "1862-8346"),
    ("Proteome Science", ""),
    ("PROTEOMICS", "1615-9853"),
    ("RAPID COMMUNICATIONS IN MASS SPECTROMETRY", "0951-4198"),
    ("SLAS Discovery", "2472-5552"),
    ("SLAS Technology", "2472-6303"),
    ("Synthetic Biology", "1939-7267"),
    ("TRANSGENIC RESEARCH", "0962-8819"),
    ("JOURNAL OF BIOMOLECULAR SCREENING", "1087-0571"),
    ("JALA", "2211-0682"),
    ("Methods in Enzymology", "0076-6879"),
    ("Methods in Microbiology", "0580-9517"),
]

TRACK_META = {
    "bioinfo_methods": {
        "cn": "生物信息方法",
        "they_want": "可复现的计算方法或评价协议、公开数据与代码、和最强简单基线比较。通常不强制新湿实验，但会问失败边界和别人能不能跑。",
    },
    "theory_mathbio": {
        "cn": "数理生物学理论",
        "they_want": "微分方程、随机过程或可证明的生物学理论模型，不是深度学习预测后的审计表。",
    },
    "biostats": {
        "cn": "生物统计",
        "they_want": "新的统计估计、检验或试验设计，通常要渐近性质、模拟校准或临床试验口径。",
    },
    "review": {
        "cn": "综述/约稿",
        "they_want": "编辑约稿的领域综述、方法引物或逐步实验方案，不收本类研究论文。",
    },
    "medical": {
        "cn": "医学计算/临床",
        "they_want": "临床决策、患者数据、医学影像或健康记录上的应用，不是细胞系扰动预测。",
    },
    "neuro": {
        "cn": "神经科学计算或方法",
        "they_want": "神经元、脑网络或神经实验方法。当前四个细胞系是 K562 / RPE1 / HepG2 / Jurkat，不是神经主场。",
    },
    "database": {
        "cn": "数据库 / Web 资源",
        "they_want": "可公开点击的数据库或 Web server，带资源更新或使用说明。没有对外服务就不对轨。",
    },
    "systems": {
        "cn": "系统生物学",
        "they_want": "网络、通路或细胞决策的机制模型，最好能改变生物学解释，而不只是误差排序相关。",
    },
    "engineering": {
        "cn": "生物医学工程",
        "they_want": "器件、数值求解、纳米生物或实验室自动化硬件/软件。",
    },
    "chemo": {
        "cn": "化学信息 / 分子模拟",
        "they_want": "对接、QSAR、分子图形或小分子性质预测。",
    },
    "ontology": {
        "cn": "本体与语义",
        "they_want": "生物医学本体、语义标注或知识图谱，不是扰动表达预测。",
    },
    "imaging": {
        "cn": "生物医学成像",
        "they_want": "光学、磁共振或医学影像重建与分析。",
    },
    "wet_lab": {
        "cn": "湿实验方法",
        "they_want": "实验室可执行的新实验方案、试剂、芯片或装置，通常要湿实验验证。",
    },
    "chromatography": {
        "cn": "色谱 / 电泳 / 分离分析",
        "they_want": "分离方法、定量验证、检测限、基质效应或保留时间，不是单细胞预测风险。",
    },
    "crystallography": {
        "cn": "晶体学",
        "they_want": "蛋白质或核酸晶体结构测定与方法。",
    },
    "proteomics": {
        "cn": "蛋白质组",
        "they_want": "质谱蛋白质组流程、鉴定定量或临床蛋白标志物。",
    },
    "mass_spec": {
        "cn": "质谱",
        "they_want": "质谱仪器方法、离子化或谱图解析。",
    },
    "multidisciplinary": {
        "cn": "综合 / 方法顶刊",
        "they_want": "完整故事：方法新、证据硬、最好成为领域默认用法，或能改变实验预算/生物学决策。",
    },
    "genomics": {
        "cn": "基因组学",
        "they_want": "新的基因组生物学发现，或能推动基因组实验决策的方法。",
    },
    "ml": {
        "cn": "机器学习",
        "they_want": "新的学习算法或理论，而不是把已有图网络接到事后审计协议上。",
    },
    "conference": {
        "cn": "计算机会议",
        "they_want": "算法新意、可复现实验、相对强基线的增量。应用审计单独通常不够。",
    },
}

# name, field, track, verdict, wos_name, reason
# wos_name must equal the opened official title for MCB/BRM members.
ROWS: list[tuple[str, str, str, str, str, str]] = [
    # ---- WoS Mathematical & Computational Biology (61) ----
    ("Acta Biotheoretica", "MCB", "theory_mathbio", "off_track", "ACTA BIOTHEORETICA",
     "理论生物学与生命哲学。SafeConf 是经验审计协议，没有理论定理可投。"),
    ("Algorithms for Molecular Biology", "MCB+BRM", "bioinfo_methods", "discuss_narrow", "Algorithms for Molecular Biology",
     "算法刊，可收窄成评价协议+四背景盲测。必须把幅度 0.6189 当主基线。不能写成二区一定能发。"),
    ("Annual Review of Biomedical Data Science", "MCB", "review", "review_only", "Annual Review of Biomedical Data Science",
     "约稿综述年刊，不收本类研究论文。"),
    ("Bulletin of Mathematical Biology", "MCB", "theory_mathbio", "off_track", "BULLETIN OF MATHEMATICAL BIOLOGY",
     "数理生物学理论模型。没有微分方程或证明，不对轨。"),
    ("BioData Mining", "MCB", "bioinfo_methods", "discuss_narrow", "BioData Mining",
     "数据挖掘方法。协议+负结果可讨论，缺新外部和训练收益，不是二区主投。"),
    ("Bioinformatics", "MCB+BRM", "bioinfo_methods", "q2_not_ready", "BIOINFORMATICS",
     "CCF-A / 中科院生物大类常见 2 区口径 / JCR Q1（年度表为准）。缺 E204 正式效果、新外部，且单独排序未超过幅度 0.6189。不能把二区写成一定能发。"),
    ("Biometrical Journal", "MCB", "biostats", "off_track", "BIOMETRICAL JOURNAL",
     "生物统计方法。不是单细胞扰动预测风险。"),
    ("Biometrics", "MCB", "biostats", "off_track", "BIOMETRICS",
     "统计理论与应用顶刊口径，不对轨。"),
    ("Biometrika", "MCB", "biostats", "off_track", "BIOMETRIKA",
     "数理统计顶刊，不对轨。"),
    ("Biostatistics", "MCB", "biostats", "off_track", "BIOSTATISTICS",
     "生物统计，不对轨。"),
    ("BioSystems", "MCB", "theory_mathbio", "off_track", "BIOSYSTEMS",
     "系统/理论模型，不对轨。"),
    ("BMC Bioinformatics", "MCB+BRM", "bioinfo_methods", "discuss_narrow", "BMC BIOINFORMATICS",
     "CCF-C，中科院常见 3 区口径。收窄成协议+四背景盲测+诚实负结果，有讨论空间。不是一定能发。"),
    ("Briefings in Bioinformatics", "MCB+BRM", "bioinfo_methods", "q2_not_ready", "BRIEFINGS IN BIOINFORMATICS",
     "2026 年常见口径：大类生物学 2 区、小类计算生物学/生化研究方法 1 区 Top；JCR Q1。缺训练收益、新外部、可运行软件叙事。比 Genome Biology 现实，但不等于现在能投就中。"),
    ("Computers in Biology and Medicine", "MCB", "medical", "off_track", "COMPUTERS IN BIOLOGY AND MEDICINE",
     "临床医学与医学数据计算，不是扰动预测风险。"),
    ("Current Bioinformatics", "MCB+BRM", "bioinfo_methods", "discuss_narrow", "Current Bioinformatics",
     "可收窄讨论。影响力与 Bioinformatics / Briefings 不是同一档，不能当二区主目标。"),
    ("Database: The Journal of Biological Databases and Curation", "MCB", "database", "off_track",
     "Database-The Journal of Biological Databases and Curation",
     "要可访问数据库或资源策展。当前没有对外库。"),
    ("Evolutionary Bioinformatics", "MCB", "bioinfo_methods", "off_track", "Evolutionary Bioinformatics",
     "进化基因组与系统发育，不是 CRISPRi 扰动预测风险。"),
    ("Frontiers in Computational Neuroscience", "MCB", "neuro", "off_track", "Frontiers in Computational Neuroscience",
     "计算神经，不对轨。"),
    ("Frontiers in Neuroinformatics", "MCB", "neuro", "off_track", "Frontiers in Neuroinformatics",
     "神经信息，不对轨。"),
    ("Genetic Epidemiology", "MCB", "biostats", "off_track", "GENETIC EPIDEMIOLOGY",
     "遗传流行病学，不对轨。"),
    ("IEEE Journal of Biomedical and Health Informatics", "MCB", "medical", "off_track",
     "IEEE Journal of Biomedical and Health Informatics",
     "医疗健康信息与临床信号，不是单细胞扰动。"),
    ("IET Systems Biology", "MCB", "systems", "off_track", "IET Systems Biology",
     "系统生物学网络模型，不是任务风险路由。"),
    ("International Journal of Biomathematics", "MCB", "theory_mathbio", "off_track", "International Journal of Biomathematics",
     "生物数学，不对轨。"),
    ("International Journal of Biostatistics", "MCB", "biostats", "off_track", "International Journal of Biostatistics",
     "生物统计，不对轨。"),
    ("International Journal of Data Mining and Bioinformatics", "MCB", "bioinfo_methods", "discuss_narrow",
     "International Journal of Data Mining and Bioinformatics",
     "数据挖掘交叉，可收窄。影响力偏低，不是二区主目标。"),
    ("International Journal for Numerical Methods in Biomedical Engineering", "MCB", "engineering", "off_track",
     "International Journal for Numerical Methods in Biomedical Engineering",
     "生物医学工程数值方法，不对轨。"),
    ("Interdisciplinary Sciences: Computational Life Sciences", "MCB", "bioinfo_methods", "discuss_narrow",
     "Interdisciplinary Sciences-Computational Life Sciences",
     "交叉计算生命。收窄协议稿可讨论，不是 CAS 保证二区。"),
    ("Journal of Agricultural Biological and Environmental Statistics", "MCB", "biostats", "off_track",
     "JOURNAL OF AGRICULTURAL BIOLOGICAL AND ENVIRONMENTAL STATISTICS",
     "农业环境统计，不对轨。"),
    ("Journal of Bioinformatics and Computational Biology", "MCB", "bioinfo_methods", "discuss_narrow",
     "Journal of Bioinformatics and Computational Biology",
     "可收窄，影响力偏低。"),
    ("Journal of Biological Dynamics", "MCB", "theory_mathbio", "off_track", "Journal of Biological Dynamics",
     "动力学理论，不对轨。"),
    ("Journal of Biological Systems", "MCB", "theory_mathbio", "off_track", "JOURNAL OF BIOLOGICAL SYSTEMS",
     "系统理论，不对轨。"),
    ("Journal of Biomedical Semantics", "MCB", "ontology", "off_track", "Journal of Biomedical Semantics",
     "本体与语义，不对轨。"),
    ("Journal of Computational Biology", "MCB+BRM", "bioinfo_methods", "discuss_narrow", "JOURNAL OF COMPUTATIONAL BIOLOGY",
     "算法刊。可讨论协议，不是中科院二区主投。"),
    ("Journal of Computational Neuroscience", "MCB", "neuro", "off_track", "JOURNAL OF COMPUTATIONAL NEUROSCIENCE",
     "计算神经，不对轨。"),
    ("Journal of Mathematical Biology", "MCB", "theory_mathbio", "off_track", "JOURNAL OF MATHEMATICAL BIOLOGY",
     "数理生物，不对轨。"),
    ("Journal of Mathematical Neuroscience", "MCB", "neuro", "off_track", "Journal of Mathematical Neuroscience",
     "数学神经，不对轨。"),
    ("Journal of Molecular Graphics & Modelling", "MCB+BRM", "chemo", "off_track", "JOURNAL OF MOLECULAR GRAPHICS & MODELLING",
     "分子图形与对接，不对轨。"),
    ("Journal of Theoretical Biology", "MCB", "theory_mathbio", "off_track", "JOURNAL OF THEORETICAL BIOLOGY",
     "理论生物学，不对轨。"),
    ("Mathematical Biosciences", "MCB", "theory_mathbio", "off_track", "MATHEMATICAL BIOSCIENCES",
     "数学生物医学，不对轨。"),
    ("Mathematical Biosciences and Engineering", "MCB", "theory_mathbio", "off_track", "Mathematical Biosciences and Engineering",
     "数学生物工程，不对轨。"),
    ("Mathematical Medicine and Biology", "MCB", "medical", "off_track", "MATHEMATICAL MEDICINE AND BIOLOGY-A JOURNAL OF THE IMA",
     "IMA 数学医学，不对轨。"),
    ("Mathematical Modelling of Natural Phenomena", "MCB", "theory_mathbio", "off_track", "Mathematical Modelling of Natural Phenomena",
     "自然现象建模，不对轨。"),
    ("Medical & Biological Engineering & Computing", "MCB", "engineering", "off_track", "MEDICAL & BIOLOGICAL ENGINEERING & COMPUTING",
     "医工计算，不对轨。"),
    ("Molecular Informatics", "MCB", "chemo", "off_track", "Molecular Informatics",
     "化学信息 / QSAR，不对轨。"),
    ("npj Systems Biology and Applications", "MCB", "systems", "off_track", "npj Systems Biology and Applications",
     "系统生物学应用。当前是审计现象，不是细胞决策机制。"),
    ("PLOS Computational Biology", "MCB+BRM", "bioinfo_methods", "q2_not_ready", "PLoS Computational Biology",
     "CCF-B。要生物学洞察，不只是排序相关。当前缺训练收益和机制，二区候选但未就绪。"),
    ("Research Synthesis Methods", "MCB", "biostats", "off_track", "Research Synthesis Methods",
     "荟萃分析和方法学综述，不对轨。"),
    ("SAR and QSAR in Environmental Research", "MCB", "chemo", "off_track", "SAR AND QSAR IN ENVIRONMENTAL RESEARCH",
     "环境毒理 QSAR，不对轨。"),
    ("Statistical Applications in Genetics and Molecular Biology", "MCB", "biostats", "off_track",
     "Statistical Applications in Genetics and Molecular Biology",
     "遗传统计，不对轨。"),
    ("Statistics in Biopharmaceutical Research", "MCB", "biostats", "off_track", "Statistics in Biopharmaceutical Research",
     "制药统计，不对轨。"),
    ("Statistics and Its Interface", "MCB", "biostats", "off_track", "Statistics and Its Interface",
     "统计交叉，不对轨。"),
    ("Statistics in Medicine", "MCB", "biostats", "off_track", "STATISTICS IN MEDICINE",
     "医学统计，不对轨。"),
    ("Statistical Methods in Medical Research", "MCB", "biostats", "off_track", "STATISTICAL METHODS IN MEDICAL RESEARCH",
     "医学统计方法，不对轨。"),
    ("Theory in Biosciences", "MCB", "theory_mathbio", "off_track", "THEORY IN BIOSCIENCES",
     "理论生物，不对轨。"),
    ("Theoretical Population Biology", "MCB", "theory_mathbio", "off_track", "THEORETICAL POPULATION BIOLOGY",
     "种群理论，不对轨。"),
    ("WIREs Computational Molecular Science", "MCB", "review", "review_only",
     "Wiley Interdisciplinary Reviews-Computational Molecular Science",
     "约稿综述，不收本类研究文。"),
    ("BMC Systems Biology", "MCB", "systems", "off_track", "BMC Systems Biology",
     "系统生物学；部分年份停刊或合并，轨道也不对。"),
    ("Computational Intelligence and Neuroscience", "MCB", "neuro", "off_track", "Computational Intelligence and Neuroscience",
     "神经计算。预警/口碑需自查，且学科不对轨。"),
    ("Computational and Mathematical Methods in Medicine", "MCB", "medical", "off_track",
     "Computational and Mathematical Methods in Medicine",
     "医学数学，不对轨。"),
    ("Journal of Medical Imaging and Health Informatics", "MCB", "imaging", "off_track",
     "Journal of Medical Imaging and Health Informatics",
     "医学影像，不对轨。"),
    ("Theoretical Biology and Medical Modelling", "MCB", "theory_mathbio", "off_track",
     "Theoretical Biology and Medical Modelling",
     "理论医学模型，不对轨。"),
    # ---- WoS Biochemical Research Methods extras ----
    ("ACS Synthetic Biology", "BRM", "wet_lab", "off_track", "ACS Synthetic Biology",
     "合成生物学湿实验，不对轨。"),
    ("Acta Crystallographica Section D", "BRM", "crystallography", "off_track",
     "Acta Crystallographica Section D-Structural Biology",
     "晶体学结构生物学，不对轨。"),
    ("Acta Crystallographica Section F", "BRM", "crystallography", "off_track",
     "Acta Crystallographica Section F-Structural Biology Communications",
     "晶体学快报，不对轨。"),
    ("Analytical and Bioanalytical Chemistry", "BRM", "chromatography", "off_track",
     "ANALYTICAL AND BIOANALYTICAL CHEMISTRY",
     "分析化学，不对轨。"),
    ("Analytical Biochemistry", "BRM", "wet_lab", "off_track", "ANALYTICAL BIOCHEMISTRY",
     "生化分析实验，不对轨。"),
    ("Assay and Drug Development Technologies", "BRM", "wet_lab", "off_track",
     "ASSAY AND DRUG DEVELOPMENT TECHNOLOGIES",
     "药物筛选实验，不对轨。"),
    ("Bioanalysis", "BRM", "chromatography", "off_track", "Bioanalysis",
     "生物分析检测，不对轨。"),
    ("BioChip Journal", "BRM", "wet_lab", "off_track", "BioChip Journal",
     "生物芯片硬件，不对轨。"),
    ("Bioconjugate Chemistry", "BRM", "wet_lab", "off_track", "BIOCONJUGATE CHEMISTRY",
     "生物偶联化学，不对轨。"),
    ("Biological Procedures Online", "BRM", "wet_lab", "off_track", "BIOLOGICAL PROCEDURES ONLINE",
     "实验规程，不对轨。"),
    ("Biologicals", "BRM", "wet_lab", "off_track", "BIOLOGICALS",
     "生物制品质控，不对轨。"),
    ("Biomedical Chromatography", "BRM", "chromatography", "off_track", "BIOMEDICAL CHROMATOGRAPHY",
     "色谱，不对轨。"),
    ("Biomedical Optics Express", "BRM", "imaging", "off_track", "Biomedical Optics Express",
     "生物医学光学，不对轨。"),
    ("Biomicrofluidics", "BRM", "wet_lab", "off_track", "Biomicrofluidics",
     "微流控，不对轨。"),
    ("BioTechniques", "BRM", "wet_lab", "off_track", "BIOTECHNIQUES",
     "实验室技术短文，不对轨。"),
    ("Biotechnology Journal", "BRM", "wet_lab", "off_track", "Biotechnology Journal",
     "生物技术，不对轨。"),
    ("Chromatographia", "BRM", "chromatography", "off_track", "CHROMATOGRAPHIA",
     "色谱方法。SafeConf 没有分离分析，完全不对轨。"),
    ("Clinical Proteomics", "BRM", "proteomics", "off_track", "Clinical Proteomics",
     "临床蛋白质组，不对轨。"),
    ("Combinatorial Chemistry & High Throughput Screening", "BRM", "wet_lab", "off_track",
     "COMBINATORIAL CHEMISTRY & HIGH THROUGHPUT SCREENING",
     "组合化学筛选，不对轨。"),
    ("Current Issues in Molecular Biology", "BRM", "wet_lab", "off_track", "CURRENT ISSUES IN MOLECULAR BIOLOGY",
     "分子生物学议题，不对轨。"),
    ("Current Opinion in Biotechnology", "BRM", "review", "review_only", "CURRENT OPINION IN BIOTECHNOLOGY",
     "约稿评论，不收本类研究文。"),
    ("Current Proteomics", "BRM", "proteomics", "off_track", "Current Proteomics",
     "蛋白质组，不对轨。"),
    ("Cytometry Part A", "BRM", "wet_lab", "off_track", "CYTOMETRY PART A",
     "流式细胞术，不对轨。"),
    ("Drug Testing and Analysis", "BRM", "chromatography", "off_track", "Drug Testing and Analysis",
     "药物检测，不对轨。"),
    ("Electrophoresis", "BRM", "chromatography", "off_track", "ELECTROPHORESIS",
     "电泳，不对轨。"),
    ("Expert Review of Proteomics", "BRM", "review", "review_only", "Expert Review of Proteomics",
     "蛋白质组综述，不对轨。"),
    ("IEEE/ACM Transactions on Computational Biology and Bioinformatics", "BRM", "bioinfo_methods", "discuss_narrow",
     "IEEE-ACM Transactions on Computational Biology and Bioinformatics",
     "CCF 交叉常见 B 档口径。可收窄成算法/系统+可复现协议。缺训练收益，不是二区主投保证。"),
    ("IEEE Transactions on Nanobioscience", "BRM", "engineering", "off_track", "IEEE TRANSACTIONS ON NANOBIOSCIENCE",
     "纳米生物器件，不对轨。"),
    ("IET Nanobiotechnology", "BRM", "engineering", "off_track", "IET Nanobiotechnology",
     "纳米生物技术，不对轨。"),
    ("Journal of the American Society for Mass Spectrometry", "BRM", "mass_spec", "off_track",
     "JOURNAL OF THE AMERICAN SOCIETY FOR MASS SPECTROMETRY",
     "质谱，不对轨。"),
    ("Journal of Biological Engineering", "BRM", "engineering", "off_track", "Journal of Biological Engineering",
     "生物工程，不对轨。"),
    ("Journal of Biomedical Optics", "BRM", "imaging", "off_track", "JOURNAL OF BIOMEDICAL OPTICS",
     "生物医学光学，不对轨。"),
    ("Journal of Biophotonics", "BRM", "imaging", "off_track", "Journal of Biophotonics",
     "生物光子，不对轨。"),
    ("Journal of Breath Research", "BRM", "medical", "off_track", "Journal of Breath Research",
     "呼气检测，不对轨。"),
    ("Journal of Chromatography A", "BRM", "chromatography", "off_track", "JOURNAL OF CHROMATOGRAPHY A",
     "色谱 A，不对轨。"),
    ("Journal of Chromatography B", "BRM", "chromatography", "off_track",
     "JOURNAL OF CHROMATOGRAPHY B-ANALYTICAL TECHNOLOGIES IN THE BIOMEDICAL AND LIFE SCIENCES",
     "色谱 B，不对轨。"),
    ("Journal of Chromatographic Science", "BRM", "chromatography", "off_track", "JOURNAL OF CHROMATOGRAPHIC SCIENCE",
     "色谱，不对轨。"),
    ("Journal of Fluorescence", "BRM", "wet_lab", "off_track", "JOURNAL OF FLUORESCENCE",
     "荧光，不对轨。"),
    ("Journal of Immunological Methods", "BRM", "wet_lab", "off_track", "JOURNAL OF IMMUNOLOGICAL METHODS",
     "免疫方法，不对轨。"),
    ("Journal of Labelled Compounds & Radiopharmaceuticals", "BRM", "wet_lab", "off_track",
     "JOURNAL OF LABELLED COMPOUNDS & RADIOPHARMACEUTICALS",
     "标记化合物，不对轨。"),
    ("Journal of Liquid Chromatography & Related Technologies", "BRM", "chromatography", "off_track",
     "JOURNAL OF LIQUID CHROMATOGRAPHY & RELATED TECHNOLOGIES",
     "液相色谱，不对轨。"),
    ("Journal of Magnetic Resonance", "BRM", "imaging", "off_track", "JOURNAL OF MAGNETIC RESONANCE",
     "磁共振，不对轨。"),
    ("Journal of Mass Spectrometry", "BRM", "mass_spec", "off_track", "JOURNAL OF MASS SPECTROMETRY",
     "质谱，不对轨。"),
    ("Journal of Microbiological Methods", "BRM", "wet_lab", "off_track", "JOURNAL OF MICROBIOLOGICAL METHODS",
     "微生物方法，不对轨。"),
    ("Journal of Neuroscience Methods", "BRM", "neuro", "off_track", "JOURNAL OF NEUROSCIENCE METHODS",
     "神经方法，不对轨。"),
    ("Journal of Proteome Research", "BRM", "proteomics", "off_track", "JOURNAL OF PROTEOME RESEARCH",
     "蛋白质组，不对轨。"),
    ("Journal of Proteomics", "BRM", "proteomics", "off_track", "Journal of Proteomics",
     "蛋白质组，不对轨。"),
    ("Journal of Spectroscopy", "BRM", "wet_lab", "off_track", "Journal of Spectroscopy",
     "光谱，不对轨。"),
    ("Journal of Virological Methods", "BRM", "wet_lab", "off_track", "JOURNAL OF VIROLOGICAL METHODS",
     "病毒学方法，不对轨。"),
    ("Lab on a Chip", "BRM", "wet_lab", "off_track", "LAB ON A CHIP",
     "微流控芯片，不对轨。"),
    ("Methods", "BRM", "wet_lab", "off_track", "METHODS",
     "实验方法汇编。不是预测风险研究论文轨道。"),
    ("Molecular and Cellular Probes", "BRM", "wet_lab", "off_track", "MOLECULAR AND CELLULAR PROBES",
     "分子探针，不对轨。"),
    ("Molecular & Cellular Proteomics", "BRM", "proteomics", "off_track", "MOLECULAR & CELLULAR PROTEOMICS",
     "细胞蛋白质组，不对轨。"),
    ("Molecular Imaging", "BRM", "imaging", "off_track", "Molecular Imaging",
     "分子成像，不对轨。"),
    ("Nature Methods", "BRM", "multidisciplinary", "q1_off", "NATURE METHODS",
     "中科院 1 区 Top 口径。要成为领域默认方法。TxPert 已在更高轨道；SafeConf 单独排序弱于幅度，无 E204/E205。现在不能冲。"),
    ("Nature Protocols", "BRM", "review", "review_only", "Nature Protocols",
     "逐步实验方案，不是本研究类型。"),
    ("New Biotechnology", "BRM", "wet_lab", "off_track", "New Biotechnology",
     "生物技术，不对轨。"),
    ("Phytochemical Analysis", "BRM", "chromatography", "off_track", "PHYTOCHEMICAL ANALYSIS",
     "植物化学分析，不对轨。"),
    ("Plant Methods", "BRM", "wet_lab", "off_track", "Plant Methods",
     "植物实验方法，不对轨。"),
    ("Plant Molecular Biology Reporter", "BRM", "wet_lab", "off_track", "PLANT MOLECULAR BIOLOGY REPORTER",
     "植物分子，不对轨。"),
    ("Preparative Biochemistry & Biotechnology", "BRM", "wet_lab", "off_track", "PREPARATIVE BIOCHEMISTRY & BIOTECHNOLOGY",
     "制备生化，不对轨。"),
    ("Protein Expression and Purification", "BRM", "wet_lab", "off_track", "PROTEIN EXPRESSION AND PURIFICATION",
     "蛋白表达纯化，不对轨。"),
    ("Proteomics Clinical Applications", "BRM", "proteomics", "off_track", "Proteomics Clinical Applications",
     "临床蛋白质组，不对轨。"),
    ("Proteome Science", "BRM", "proteomics", "off_track", "Proteome Science",
     "蛋白质组，不对轨。"),
    ("Proteomics", "BRM", "proteomics", "off_track", "PROTEOMICS",
     "蛋白质组，不对轨。"),
    ("Rapid Communications in Mass Spectrometry", "BRM", "mass_spec", "off_track",
     "RAPID COMMUNICATIONS IN MASS SPECTROMETRY",
     "质谱快报，不对轨。"),
    ("SLAS Discovery", "BRM", "wet_lab", "off_track", "SLAS Discovery",
     "高通量筛选，不对轨。"),
    ("SLAS Technology", "BRM", "engineering", "off_track", "SLAS Technology",
     "实验室自动化，不对轨。"),
    ("Synthetic Biology", "BRM", "wet_lab", "off_track", "Synthetic Biology",
     "合成生物学，不对轨。"),
    ("Transgenic Research", "BRM", "wet_lab", "off_track", "TRANSGENIC RESEARCH",
     "转基因，不对轨。"),
    ("Journal of Biomolecular Screening", "BRM", "wet_lab", "off_track", "JOURNAL OF BIOMOLECULAR SCREENING",
     "分子筛选（旧名/低影响因子残留），不对轨。"),
    ("JALA", "BRM", "engineering", "off_track", "JALA",
     "实验室自动化，不对轨。"),
    ("Methods in Enzymology", "BRM", "review", "review_only", "Methods in Enzymology",
     "酶学方法丛书，不对轨。"),
    ("Methods in Microbiology", "BRM", "review", "review_only", "Methods in Microbiology",
     "微生物方法丛书，不对轨。"),
    # ---- Nature / Science / Cell / genomics extras (not in the two WoS lists) ----
    ("Cell Reports Methods", "GENOMICS", "multidisciplinary", "q1_off", "",
     "Cell 子刊方法。要新实验方法被实验室采用。当前是计算审计，不够。"),
    ("Nature", "MULTI", "multidisciplinary", "q1_off", "",
     "综合顶刊。当前是预测后风险审计现象，不对轨。"),
    ("Science", "MULTI", "multidisciplinary", "q1_off", "",
     "综合顶刊，不对轨。"),
    ("Cell", "MULTI", "multidisciplinary", "q1_off", "",
     "综合顶刊，不对轨。"),
    ("Nature Biotechnology", "MULTI", "multidisciplinary", "q1_off", "",
     "TxPert 本家轨道。SafeConf 尚不是可部署、能改实验预算的方法。"),
    ("Nature Machine Intelligence", "MULTI", "ml", "q1_off", "",
     "机器学习顶刊。缺方法新意与跨架构。"),
    ("Nature Computational Science", "MULTI", "ml", "q1_off", "",
     "计算科学综合。当前增量是评价协议，不是新计算原理。"),
    ("Nature Communications", "MULTI", "multidisciplinary", "q1_off", "",
     "中科院 1 区口径。缺完整故事和训练收益。现在不能冲。"),
    ("Nature Cell Biology", "MULTI", "wet_lab", "off_track", "",
     "细胞生物学机制，不对轨。"),
    ("Genome Biology", "GENOMICS", "genomics", "q1_off", "",
     "中科院常见 1 区。要生物学问题和机制，不只是排序相关。"),
    ("Genome Research", "GENOMICS", "genomics", "q1_off", "",
     "基因组研究，要新生物学发现。"),
    ("Nucleic Acids Research", "GENOMICS", "database", "q1_off", "",
     "资源 / Web server 或广泛工具。当前没有对外服务。"),
    ("NAR Genomics and Bioinformatics", "GENOMICS", "bioinfo_methods", "discuss_narrow", "",
     "NAR 开源计算刊。收窄协议稿可讨论。"),
    ("Bioinformatics Advances", "GENOMICS", "bioinfo_methods", "discuss_narrow", "",
     "OUP 新兴方法刊，口味接近 Bioinformatics，分区未稳。可收窄，仍缺安装包和独立复现。"),
    ("GigaScience", "GENOMICS", "bioinfo_methods", "discuss_narrow", "",
     "大数据 / 可复现。可讨论，需要完整数据包。"),
    ("Genomics, Proteomics & Bioinformatics", "GENOMICS", "bioinfo_methods", "discuss_narrow", "",
     "GPB。可收窄，不是 CAS 保证二区。"),
    ("Molecular Systems Biology", "GENOMICS", "systems", "q1_off", "",
     "系统生物学顶刊，要对细胞决策有机制。"),
    ("Cell Systems", "GENOMICS", "systems", "q1_off", "",
     "Cell 系统生物学，不对当前审计稿。"),
    ("Cell Genomics", "GENOMICS", "genomics", "q1_off", "",
     "基因组与功能，要生物学。"),
    ("Science Advances", "MULTI", "multidisciplinary", "q1_off", "",
     "Science 开源。故事不够完整。"),
    ("PNAS", "MULTI", "multidisciplinary", "q1_off", "",
     "综合。缺广泛兴趣和机制。"),
    ("eLife", "MULTI", "multidisciplinary", "q1_off", "",
     "要完整生物学问题。"),
    ("iScience", "MULTI", "multidisciplinary", "discuss_narrow", "",
     "Cell 开源综合。可讨论但定位散，不是二区主目标。"),
    ("Scientific Reports", "MULTI", "multidisciplinary", "discuss_narrow", "",
     "Nature 开源。可发技术报告，不是二区主目标。"),
    ("Communications Biology", "MULTI", "multidisciplinary", "q1_off", "",
     "Nature 开源生物学。仍要生物学问题和完整故事。"),
    ("PeerJ", "MULTI", "bioinfo_methods", "discuss_narrow", "",
     "OA 综合。可收窄，分区低。"),
    ("Frontiers in Bioinformatics", "GENOMICS", "bioinfo_methods", "discuss_narrow", "",
     "OA 前沿。可讨论，注意年发文与口碑。"),
    ("Frontiers in Genetics", "GENOMICS", "genomics", "off_track", "",
     "遗传主题过宽，不是方法主刊。"),
    ("Briefings in Functional Genomics", "GENOMICS", "review", "review_only", "",
     "功能基因组综述。"),
    ("Human Genomics", "GENOMICS", "genomics", "off_track", "",
     "人类基因组临床，不对轨。"),
    ("Molecular Biology of the Cell", "MULTI", "wet_lab", "off_track", "",
     "细胞生物学，不对轨。"),
    ("Genome Medicine", "GENOMICS", "medical", "off_track", "",
     "基因组医学，不对轨。"),
    ("Nature Genetics", "GENOMICS", "genomics", "q1_off", "",
     "遗传发现顶刊，不对轨。"),
    ("Nature Reviews Genetics", "GENOMICS", "review", "review_only", "",
     "综述约稿。"),
    ("Nature Reviews Methods Primers", "MULTI", "review", "review_only", "",
     "方法引物约稿。"),
    ("Patterns", "MULTI", "ml", "discuss_narrow", "",
     "Cell 数据科学。可讨论计算叙事，不是二区保证。"),
    ("Cell Reports", "MULTI", "wet_lab", "q1_off", "",
     "完整实验故事，不对轨。"),
    ("iMeta", "GENOMICS", "genomics", "discuss_narrow", "",
     "微生物组/组学新兴刊。轨道不完全匹配。"),
    ("aBIOTECH", "GENOMICS", "wet_lab", "off_track", "",
     "农业生物技术，不对轨。"),
    ("BMC Genomics", "GENOMICS", "genomics", "discuss_narrow", "",
     "基因组 OA。主题偏基因组发现；收窄成方法+数据可讨论，不是二区主投。"),
    # CCF / ML
    ("Proceedings of the IEEE", "CCF", "review", "review_only", "",
     "CCF-A 综述工程，不对轨。"),
    ("IEEE Transactions on Pattern Analysis and Machine Intelligence", "CCF", "ml", "q1_off", "",
     "CCF-A 视觉/模式识别，不对轨。"),
    ("NeurIPS", "CCF", "conference", "q1_off", "",
     "CCF-A 会议。要机器学习新方法，当前是应用审计。"),
    ("ICML", "CCF", "conference", "q1_off", "",
     "CCF-A 会议。同上。"),
    ("ICLR", "CCF", "conference", "q1_off", "",
     "机器学习会议，不对轨。"),
    ("RECOMB", "CCF", "conference", "q2_not_ready", "",
     "计算生物学会议。要算法新意。当前增量是评价协议，未闭合。"),
    ("ISMB / Bioinformatics Proceedings", "CCF", "conference", "q2_not_ready", "",
     "与 Bioinformatics 会议轨绑定，证据同样未闭合。"),
    ("AAAI", "CCF", "conference", "q1_off", "",
     "人工智能会议，不对轨。"),
    ("IJCAI", "CCF", "conference", "q1_off", "",
     "人工智能会议，不对轨。"),
    ("KDD", "CCF", "conference", "q1_off", "",
     "数据挖掘会议，不对轨。"),
]

OVERRIDES: dict[str, dict] = {
    "Bioinformatics": {
        "cas_note": "中科院生物大类常见 2 区；JCR Q1；CCF-A。投稿前以学院系统为准。",
        "care": (2, 2, 2, 1, 1, 1, 2, 0),
    },
    "Briefings in Bioinformatics": {
        "cas_note": "2026 新锐表常见：大类生物 2 区、小类计算生物学 1 区 Top；JCR Q1。",
        "care": (2, 2, 2, 1, 1, 1, 2, 0),
    },
    "BMC Bioinformatics": {
        "cas_note": "中科院常见 3 区；CCF-C。",
        "care": (2, 2, 1, 0, 0, 0, 1, 0),
    },
    "PLOS Computational Biology": {
        "cas_note": "CCF-B。分区以学院系统为准。",
        "care": (2, 2, 2, 1, 1, 1, 2, 1),
    },
    "Nature Methods": {
        "cas_note": "中科院 1 区 Top；JCR Q1。",
        "care": (2, 2, 2, 2, 2, 2, 2, 2),
    },
    "Nucleic Acids Research": {
        "cas_note": "中科院常见 1 区或 2 区（看当年大类表）。",
        "care": (2, 2, 2, 1, 1, 2, 2, 1),
    },
    "Genome Biology": {
        "cas_note": "中科院常见 1 区。",
        "care": (2, 2, 2, 2, 2, 2, 2, 2),
    },
    "Nature Communications": {
        "cas_note": "中科院 1 区。",
        "care": (2, 2, 2, 2, 2, 2, 2, 2),
    },
    "Bioinformatics Advances": {
        "cas_note": "较新刊，分区不稳定。",
        "care": (2, 2, 1, 0, 0, 0, 1, 0),
    },
}


def _fold(name: str) -> str:
    s = name.lower().replace("&", "and").replace("ieee-acm", "ieee acm").replace("ieee/acm", "ieee acm")
    s = s.replace("wiley interdisciplinary reviews", "wires")
    return re.sub(r"[^a-z0-9]+", "", s)


def names_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    fa, fb = _fold(a), _fold(b)
    if fa == fb:
        return True
    short, long = (fa, fb) if len(fa) <= len(fb) else (fb, fa)
    if long.startswith(short) and len(short) >= 22:
        return True
    stop = {"the", "of", "and", "in", "for", "a", "on", "journal", "section", "an"}
    ta = {t for t in re.findall(r"[a-z0-9]+", a.lower().replace("&", " and ")) if t not in stop}
    tb = {t for t in re.findall(r"[a-z0-9]+", b.lower().replace("&", " and ")) if t not in stop}
    if len(ta) >= 3 and ta <= tb:
        return True
    if len(tb) >= 3 and tb <= ta:
        return True
    return False


def care_for(track: str, verdict: str) -> tuple[int, ...]:
    if verdict == "off_track":
        if track in {"wet_lab", "chromatography", "crystallography", "proteomics", "mass_spec", "imaging"}:
            return (0, 0, 0, 0, 0, 0, 0, 2)
        return (0, 0, 0, 0, 0, 0, 0, 0)
    if verdict == "review_only":
        return (0, 0, 0, 0, 0, 0, 0, 0)
    if verdict == "q1_off":
        if track == "database":
            return (2, 2, 2, 1, 1, 2, 2, 1)
        if track in {"ml", "conference"}:
            return (2, 2, 2, 2, 2, 2, 1, 0)
        return (2, 2, 2, 2, 2, 2, 2, 2)
    if verdict == "q2_not_ready":
        return (2, 2, 2, 1, 1, 1, 2, 0)
    if track == "database":
        return (2, 2, 1, 0, 0, 0, 1, 0)
    return (2, 2, 1, 0, 0, 0, 1, 0)


def _issn_for(wos_name: str) -> str:
    for official, issn in MCB_OFFICIAL + BRM_OFFICIAL:
        if names_match(wos_name or "", official) or _fold(wos_name) == _fold(official):
            if issn:
                return issn
    return ""


def _source_opened(field: str) -> str:
    if field == "MCB":
        return f"{OPENED_ON} 打开 {MCB_SOURCE}"
    if field == "BRM":
        return f"{OPENED_ON} 打开 {BRM_SOURCE}"
    if field == "MCB+BRM":
        return f"{OPENED_ON} 打开 {MCB_SOURCE} 与 {BRM_SOURCE}（两学科交叉收录）"
    if field == "GENOMICS":
        return "不在上述两个 WoS 小类完整名单里，按基因组/方法交叉补入"
    if field == "CCF":
        return "不在上述两个 WoS 小类完整名单里，按 CCF 计算交叉补入"
    return "不在上述两个 WoS 小类完整名单里，按 Nature/Cell/Science 家族补入"


def _we_have(verdict: str) -> str:
    if verdict == "off_track":
        return "没有对口稿件。SafeConf 是单细胞扰动预测之后的任务风险排序（E201 Spearman 0.4082，幅度基线 0.6189），不是本刊学科对象。"
    if verdict == "review_only":
        return "没有约稿综述或逐步实验方案。现有材料是研究审计，不是本刊栏目。"
    return (
        "E201：1808 个主任务、四细胞系先封存再解封；SafeConf Spearman 0.4082，区间 [0.3506, 0.4621]；"
        "预测幅度 0.6189；控制幅度后偏 Spearman 0.2503；20% 效用 0.3200 vs 幅度 0.5943。"
        "封存哈希、CSV 和脚本链可核对。"
    )


def _we_lack(verdict: str) -> str:
    shared = (
        "单独排序没有超过幅度（效用差 -0.2743）；E204 只有工程验收，没有 80 轮正式效果；"
        "E205 未跑，四个种子仍是同一 GAT；没有冻结后的新外部数据；没有湿实验。"
    )
    if verdict == "off_track":
        return "学科不对轨。补 E204 也填不上这本杂志的栏目。"
    if verdict == "review_only":
        return "不是研究论文轨道。改写成综述需要编辑约稿，不是现在这条线。"
    if verdict == "q2_not_ready":
        return shared + " 以当前正式表，不能把二区写成一定能发。"
    if verdict == "q1_off":
        return shared + " 一区/顶刊还要机制、社区地位或完整故事，现在不能冲。"
    return shared + " 若把主张收成协议 + 四背景盲测 + 负结果，才有讨论空间。"


def catalog() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for name, field, track, verdict, wos_name, reason in ROWS:
        key = name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        over = OVERRIDES.get(name, {})
        care = over.get("care") or care_for(track, verdict)
        row = {
            "name": name,
            "field": field,
            "track": track,
            "track_cn": TRACK_META[track]["cn"],
            "verdict": verdict,
            "verdict_cn": VERDICT_CN[verdict],
            "wos_name": wos_name,
            "issn": _issn_for(wos_name) if wos_name else "",
            "reason": reason,
            "they_want": TRACK_META[track]["they_want"],
            "we_have": _we_have(verdict),
            "we_lack": _we_lack(verdict),
            "cas_note": over.get(
                "cas_note",
                "本表不把分区写成录用保证。投稿前以学院指定查询系统为准。",
            ),
            "care": list(care),
            "source_opened": _source_opened(field),
        }
        out.append(row)
    return out


def counts(rows: Iterable[dict] | None = None) -> dict[str, int]:
    rows = list(rows or catalog())
    tally: dict[str, int] = {key: 0 for key in VERDICT_ORDER}
    for row in rows:
        tally[row["verdict"]] = tally.get(row["verdict"], 0) + 1
    tally["total"] = len(rows)
    tally["mcb_official"] = len(MCB_OFFICIAL)
    tally["brm_official"] = len(BRM_OFFICIAL)
    return tally


def official_unmatched(rows: Iterable[dict] | None = None) -> list[str]:
    rows = list(rows or catalog())
    missing = []
    for official, _issn in MCB_OFFICIAL + BRM_OFFICIAL:
        if not any(
            names_match(official, row["name"]) or names_match(official, row.get("wos_name") or "")
            for row in rows
        ):
            missing.append(official)
    return sorted(set(missing))


def _lamp(care: int, have: int) -> str:
    if care == 0:
        return "灰：一般不拿这一点桌面拒"
    if have >= 2:
        return "绿：杂志在乎且我们有"
    if have == 1:
        return "橙：部分"
    return "红：杂志在乎但我们没有"


def _care_cn(care: int) -> str:
    return {2: "很在乎（缺了容易桌面拒）", 1: "审稿时会问", 0: "一般不拿这一点直接拒"}[care]


def _have_cn(have: int) -> str:
    return {2: "有正式结果", 1: "部分", 0: "没有或失败"}[have]


def _matrix_value(care: int, have: int) -> float:
    if care == 0:
        return 0.45
    if have >= 2:
        return 1.0
    if have == 1:
        return 0.65
    return 0.0


def evidence_rows(row: dict) -> list[dict]:
    out = []
    for i, (key, label) in enumerate(EVIDENCE_ITEMS):
        care = int(row["care"][i])
        have = int(EVIDENCE_HAVE[key])
        out.append(
            {
                "key": key,
                "label": label,
                "care": care,
                "have": have,
                "care_cn": _care_cn(care),
                "have_cn": _have_cn(have),
                "fact": HAVE_FACT[key],
                "lamp": _lamp(care, have),
                "value": _matrix_value(care, have),
            }
        )
    return out


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _journal_section(row: dict) -> str:
    ev = evidence_rows(row)
    identity = _md_table(
        ["字段", "内容"],
        [
            ["刊名", row["name"]],
            ["WoS 刊名（若在两个小类里）", row["wos_name"] or "不在 MCB 61 / BRM 82 完整名单（交叉补入）"],
            ["ISSN（学科页快照，可能为空）", row["issn"] or "—"],
            ["学科来源", row["source_opened"]],
            ["轨道", row["track_cn"]],
            ["常见分区口径", row["cas_note"]],
            ["这本杂志通常要什么", row["they_want"]],
            ["我们现在对得上什么", row["we_have"]],
            ["我们现在缺什么", row["we_lack"]],
            ["判定", row["verdict_cn"]],
            ["一句话原因", row["reason"]],
        ],
    )
    lamps = _md_table(
        ["审稿人会问", "这本杂志在不在乎", "我们现在有没有", "格子"],
        [[e["label"], e["care_cn"], e["fact"], e["lamp"]] for e in ev],
    )
    return f"### {row['name']}\n\n{identity}\n\n{lamps}\n"


def render_markdown(rows: list[dict] | None = None) -> str:
    rows = list(rows or catalog())
    tally = counts(rows)
    by_verdict: dict[str, list[dict]] = {key: [] for key in VERDICT_ORDER}
    for row in rows:
        by_verdict[row["verdict"]].append(row)
    master = _md_table(
        ["刊名", "学科", "轨道", "判定", "这本杂志通常要什么", "我们现在对得上吗", "缺什么才配讨论"],
        [
            [
                r["name"],
                r["field"],
                r["track_cn"],
                r["verdict_cn"],
                r["they_want"],
                "对得上一部分" if r["verdict"] in {"discuss_narrow", "q2_not_ready"} else (
                    "故事不够" if r["verdict"] == "q1_off" else "对不上"
                ),
                r["we_lack"],
            ]
            for r in rows
        ],
    )
    parts = [
        "# 全球相关期刊逐本评价",
        "",
        f"评价日期：{OPENED_ON}。数字锁在 `CLAIM_TABLE.json`：SafeConf Spearman 0.4082，预测幅度 0.6189，偏相关 0.2503，20% 效用 0.3200 vs 0.5943。E204 没有 80 轮正式效果。E205 未跑。",
        "",
        "**不能把二区写成一定能发。** 本页也不保证任何一本杂志会录用。",
        "",
        "## 0. 先回答两件事：是不是全世界所有期刊，以及是不是真的打开了",
        "",
        "不是地球上全部 SCI 期刊。Web of Science SCIE 有数千到上万本，绝大多数是物理、化学、临床、农学，和这篇稿无关。把 3 万本都抄一遍会假装勤奋，帮不了投稿。",
        "",
        "真正打开并逐本写入的是：",
        "",
        f"1. {MCB_SOURCE}。官方页写明期刊总数 61，本目录 61 本全部有评价。",
        f"2. {BRM_SOURCE}。官方页写明期刊总数 82，本目录 82 本全部有评价。两学科交叉收录的刊只写一次。",
        "3. 另外补入 Nature / Cell / Science 家族、基因组方法和 CCF 计算交叉。这些不在上面两个小类里，所以不是“漏了 WoS 名单”，是按会被问到的投稿对象补的。",
        "",
        "打开的是**学科完整刊名表**（sort-699 第 1–3 页，sort-615 第 1–3 页），不是只看影响因子排序的前几本。",
        "",
        "没有对这 185 本杂志逐一点进官网 Aims & Scope。色谱、晶体学、质谱、流式、植物方法凭刊名加 WoS 小类就可以判定不对轨；方法刊、基因组刊、顶刊按下表用同一套 E201/E204 证据灯打分。",
        "",
        f"本目录去重后共 **{tally['total']}** 本。判定计数：可讨论 {tally['discuss_narrow']}，二区候选但未就绪 {tally['q2_not_ready']}，一区/顶刊不够 {tally['q1_off']}，只收综述 {tally['review_only']}，学科不对轨 {tally['off_track']}。",
        "",
        "图 6 是全集计数。图 7 是每一本对八盏证据灯的格子：绿=杂志在乎且我们有，红=杂志在乎但没有，灰=一般不拿这一点桌面拒（学科不对轨时整行会灰/红在湿实验上）。",
        "",
        "![图 6 期刊全集计数](figures/fig6_journal_universe.png)",
        "",
        "**图 6a** 五类判定各有多少本。**图 6b** 不对轨的刊按轨道拆开，用来回答“色谱那些你是不是没看”。",
        "",
        "![图 7 每一本杂志的证据灯](figures/fig7_journal_heatmap.png)",
        "",
        "机器可读总表：[journal_universe.csv](./journal_universe.csv)。8 本放大讲解仍在 [05_期刊对照表.md](./05_期刊对照表.md)。",
        "",
        "## 1. 一览表（每一本一行）",
        "",
        master,
        "",
        "## 2. 怎么读后面的逐本表",
        "",
        "每一本都有两张表。第一张是身份+主张：他们通常要什么、我们对得上什么、缺什么、判定。第二张是八盏证据灯，和总图 7 同一套规则。",
        "",
        "「可讨论」不是「一定能发」。「二区候选但未就绪」是用户口头目标里最常被点名的本，但当前正式表还不能说稳。",
        "",
    ]
    headings = {
        "discuss_narrow": "## 3. 可讨论（必须收窄主张）——逐本",
        "q2_not_ready": "## 4. 二区候选但未就绪——逐本",
        "q1_off": "## 5. 一区 / 顶刊不对轨或故事不够——逐本",
        "review_only": "## 6. 只收综述或方案——逐本",
        "off_track": "## 7. 学科不对轨——逐本",
    }
    for verdict in VERDICT_ORDER:
        parts.append(headings[verdict])
        parts.append("")
        parts.append(f"这一类 {len(by_verdict[verdict])} 本。")
        parts.append("")
        for row in by_verdict[verdict]:
            parts.append(_journal_section(row))
    parts.extend(
        [
            "## 8. 和「二区一定能发、一区可以冲刺」怎么对齐",
            "",
            "用户要：二区一定能发，一区可以冲刺。",
            "",
            "对照全集：二区里最常被点名的是 Bioinformatics、Briefings in Bioinformatics、PLOS Computational Biology，以及会议轨 RECOMB / ISMB。它们全部落在「二区候选但未就绪」，同一处卡住：单独排序没有超过幅度 0.6189，E204 没有正式效果，没有冻结后的新外部。",
            "",
            "可讨论的是 BMC Bioinformatics、Bioinformatics Advances、NAR Genomics and Bioinformatics、GigaScience 等。主张必须收成协议 + 四背景盲测 + 负结果。",
            "",
            "一区 / Nature Methods / Genome Biology / Nature Communications 在当前证据下不能冲。",
            "",
            "Chromatographia、Journal of Chromatography A、晶体学、蛋白质组、流式、植物方法全部是学科不对轨。不是“再改一改摘要就能投”，是栏目就不收这篇。",
            "",
            "下一步仍是 E204 的 32 个正式模型，不是先改期刊名单。",
            "",
        ]
    )
    return "\n".join(parts) + "\n"


def write_csv(path: Path, rows: list[dict] | None = None) -> Path:
    rows = list(rows or catalog())
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "wos_name",
        "issn",
        "field",
        "track",
        "track_cn",
        "verdict",
        "verdict_cn",
        "cas_note",
        "they_want",
        "we_have",
        "we_lack",
        "reason",
        "source_opened",
        "care",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["care"] = ",".join(str(x) for x in row["care"])
            writer.writerow(payload)
    return path


def write_markdown(path: Path, rows: list[dict] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rows), encoding="utf-8")
    return path


def _style_axes(ax, fp: FontProperties) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.6)
    ax.tick_params(colors="#222222", labelsize=8)
    ax.yaxis.label.set_fontproperties(fp)
    ax.xaxis.label.set_fontproperties(fp)


def _panel_id(ax, letter: str, fp: FontProperties) -> None:
    # Latin panel letters stay as SVG text (not CJK glyph outlines) so checks can grep a/b.
    ax.text(
        -0.08,
        1.08,
        letter,
        transform=ax.transAxes,
        fontfamily="DejaVu Sans",
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _save_fig(repo: Path, fig, stem: str, plotted: dict) -> dict:
    fig_dir = repo / PACK_REL / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{stem}.png"
    svg = fig_dir / f"{stem}.svg"
    sidecar = fig_dir / f"{stem}.values.json"
    fig.savefig(png, dpi=180, facecolor="white", edgecolor="none")
    fig.savefig(svg, facecolor="white", edgecolor="none")
    plt.close(fig)
    payload = {
        "stem": stem,
        "plotted": plotted,
        "png": str(png.relative_to(repo)),
        "svg": str(svg.relative_to(repo)),
    }
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def generate_fig6_universe(repo: Path, fp: FontProperties, rows: list[dict] | None = None) -> dict:
    rows = list(rows or catalog())
    tally = counts(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), facecolor="white",
                             gridspec_kw={"width_ratios": [1.05, 1.35]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    _style_axes(ax, fp)
    _panel_id(ax, "a", fp)
    labels = [VERDICT_CN[k] for k in VERDICT_ORDER]
    vals = [tally[k] for k in VERDICT_ORDER]
    colors = [OKABE["green"], OKABE["orange"], OKABE["sky"], OKABE["grey"], OKABE["vermillion"]]
    y = list(range(len(labels), 0, -1))
    ax.barh(y, vals, color=colors, height=0.62, linewidth=0)
    ax.set_yticks(y, labels, fontproperties=fp, fontsize=8)
    ax.set_xlabel("期刊本数", fontproperties=fp, fontsize=8)
    ax.set_title("全集判定（去重后）", fontproperties=fp, fontsize=9, loc="left")
    for yi, val in zip(y, vals):
        ax.text(val + 0.4, yi, str(val), va="center", fontproperties=fp, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.18)

    ax = axes[1]
    _style_axes(ax, fp)
    _panel_id(ax, "b", fp)
    off = [r for r in rows if r["verdict"] == "off_track"]
    track_counts: dict[str, int] = {}
    for row in off:
        track_counts[row["track_cn"]] = track_counts.get(row["track_cn"], 0) + 1
    items = sorted(track_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    y = list(range(len(items), 0, -1))
    ax.barh(y, [n for _, n in items], color=OKABE["vermillion"], height=0.62, linewidth=0)
    ax.set_yticks(y, [name for name, _ in items], fontproperties=fp, fontsize=7)
    ax.set_xlabel("不对轨的本数", fontproperties=fp, fontsize=8)
    ax.set_title("不对轨的刊按轨道拆开", fontproperties=fp, fontsize=9, loc="left")
    for yi, (_, n) in zip(y, items):
        ax.text(n + 0.15, yi, str(n), va="center", fontproperties=fp, fontsize=7)
    if items:
        ax.set_xlim(0, max(n for _, n in items) * 1.2)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.86, bottom=0.16, wspace=0.50)
    plotted = {
        "total": tally["total"],
        "verdict_counts": {k: tally[k] for k in VERDICT_ORDER},
        "off_track_by_track": dict(items),
        "mcb_official": len(MCB_OFFICIAL),
        "brm_official": len(BRM_OFFICIAL),
        "names": [r["name"] for r in rows],
    }
    return _save_fig(repo, fig, "fig6_journal_universe", plotted)


def generate_fig7_heatmap(repo: Path, fp: FontProperties, rows: list[dict] | None = None) -> dict:
    rows = list(rows or catalog())
    ranked: list[dict] = []
    for verdict in VERDICT_ORDER:
        ranked.extend([r for r in rows if r["verdict"] == verdict])
    on_topic = [r for r in ranked if r["verdict"] != "off_track"]
    off_topic = [r for r in ranked if r["verdict"] == "off_track"]
    item_labels = [lab for _, lab in EVIDENCE_ITEMS]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.6, 0.22 * len(on_topic) + 0.14 * len(off_topic) + 2.8),
        facecolor="white",
        gridspec_kw={"height_ratios": [max(len(on_topic), 1), max(len(off_topic), 1)]},
    )
    fig.patch.set_facecolor("white")
    cmap = ListedColormap(["#D55E00", "#C8C8C8", "#E69F00", "#009E73"])
    bounds = [-0.1, 0.2, 0.55, 0.8, 1.1]
    norm = BoundaryNorm(bounds, cmap.N)

    def _heat(ax, letter, subset, title, tick_size):
        _panel_id(ax, letter, fp)
        matrix = [[e["value"] for e in evidence_rows(row)] for row in subset]
        data = np.array(matrix) if matrix else np.zeros((1, len(EVIDENCE_ITEMS)))
        ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(item_labels)))
        ax.set_xticklabels(item_labels, fontproperties=fp, fontsize=6, rotation=32, ha="right")
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels([r["name"] for r in subset], fontproperties=fp, fontsize=tick_size)
        ax.set_title(title, fontproperties=fp, fontsize=9, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        return matrix

    m_on = _heat(axes[0], "a", on_topic, "可能被问到的刊（绿=在乎且有，红=在乎但缺，灰=不拿来拒）", 6.5)
    m_off = _heat(axes[1], "b", off_topic, "学科不对轨的刊也逐本打分，不是没看", 5.5)
    fig.subplots_adjust(left=0.38, right=0.99, top=0.985, bottom=0.03, hspace=0.22)
    plotted = {
        "have": EVIDENCE_HAVE,
        "item_keys": [k for k, _ in EVIDENCE_ITEMS],
        "on_topic": [r["name"] for r in on_topic],
        "off_topic": [r["name"] for r in off_topic],
        "all_names": [r["name"] for r in ranked],
        "matrix_on": m_on,
        "matrix_off": m_off,
        "verdicts": {r["name"]: r["verdict"] for r in ranked},
    }
    return _save_fig(repo, fig, "fig7_journal_heatmap", plotted)


def write_universe_pack(repo: Path, fp: FontProperties | None = None) -> dict:
    rows = catalog()
    pack = repo / PACK_REL
    md = write_markdown(pack / UNIVERSE_MD, rows)
    csv_path = write_csv(pack / UNIVERSE_CSV, rows)
    fp = fp or FontProperties(family="DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig6 = generate_fig6_universe(repo, fp, rows)
    fig7 = generate_fig7_heatmap(repo, fp, rows)
    return {
        "n": len(rows),
        "unmatched_official": official_unmatched(rows),
        "markdown": str(md.relative_to(repo)),
        "csv": str(csv_path.relative_to(repo)),
        "fig6": fig6["png"],
        "fig7": fig7["png"],
        "counts": counts(rows),
    }


def check_universe_doc(repo: Path) -> list[str]:
    pack = repo / PACK_REL
    missing: list[str] = []
    md_path = pack / UNIVERSE_MD
    csv_path = pack / UNIVERSE_CSV
    if not md_path.is_file():
        return [f"missing {UNIVERSE_MD}"]
    if not csv_path.is_file():
        missing.append(f"missing {UNIVERSE_CSV}")
    text = md_path.read_text(encoding="utf-8")
    rows = catalog()
    if len(rows) < 100:
        missing.append(f"catalog too small: {len(rows)}")
    by_name = {r["name"].lower(): r for r in rows}
    required = {
        "chromatographia": "off_track",
        "bioinformatics": "q2_not_ready",
        "nature methods": "q1_off",
        "bmc bioinformatics": "discuss_narrow",
    }
    for name, verdict in required.items():
        row = by_name.get(name)
        if row is None:
            missing.append(f"catalog missing {name}")
            continue
        if row["verdict"] != verdict:
            missing.append(f"{name} verdict {row['verdict']} != {verdict}")
        if row["name"] not in text:
            missing.append(f"{UNIVERSE_MD} missing journal {row['name']}")
    for row in rows:
        if row["name"] not in text:
            missing.append(f"{UNIVERSE_MD} missing catalog name {row['name']}")
            break
    for phrase in ("不能把二区写成一定能发", "0.4082", "0.6189", "Chromatographia", "sort-699", "sort-615"):
        if phrase not in text:
            missing.append(f"{UNIVERSE_MD} missing {phrase}")
    unmatched = official_unmatched(rows)
    if unmatched:
        missing.append("official titles not in catalog: " + "; ".join(unmatched[:8]))
    if len(MCB_OFFICIAL) != 61:
        missing.append(f"MCB official list is {len(MCB_OFFICIAL)} not 61")
    if len(BRM_OFFICIAL) != 82:
        missing.append(f"BRM official list is {len(BRM_OFFICIAL)} not 82")
    for stem in ("fig6_journal_universe", "fig7_journal_heatmap"):
        for suffix in (".png", ".svg", ".values.json"):
            if not (pack / "figures" / f"{stem}{suffix}").is_file():
                missing.append(f"missing {stem}{suffix}")
    sidecar6 = pack / "figures" / "fig6_journal_universe.values.json"
    if sidecar6.is_file():
        payload = json.loads(sidecar6.read_text())
        plotted = payload.get("plotted") or {}
        if plotted.get("mcb_official") != 61:
            missing.append("fig6 must record MCB official n=61")
        if plotted.get("brm_official") != 82:
            missing.append("fig6 must record BRM official n=82")
        if int(plotted.get("total") or 0) < 100:
            missing.append("fig6 total < 100")
    sidecar7 = pack / "figures" / "fig7_journal_heatmap.values.json"
    if sidecar7.is_file():
        payload = json.loads(sidecar7.read_text())
        plotted = payload.get("plotted") or {}
        names = plotted.get("all_names") or []
        if "Chromatographia" not in names:
            missing.append("fig7 missing Chromatographia")
        if "Bioinformatics" not in names:
            missing.append("fig7 missing Bioinformatics")
        if plotted.get("verdicts", {}).get("Nature Methods") != "q1_off":
            missing.append("fig7 Nature Methods not q1_off")
        xml = (pack / "figures" / "fig7_journal_heatmap.svg").read_text()
        if "drop-shadow" in xml.lower() or "feDropShadow" in xml:
            missing.append("fig7 has drop-shadow")
        for letter in ("a", "b"):
            if not re.search(rf">\s*{letter}\s*<", xml) and f">{letter}<" not in xml:
                if not re.search(rf">{letter}</", xml):
                    missing.append(f"fig7 missing panel letter {letter}")
    return missing
