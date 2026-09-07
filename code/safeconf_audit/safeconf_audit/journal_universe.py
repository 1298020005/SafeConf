#!/usr/bin/env python3
"""Score journals on two businesses, not one blended verdict.

业务A：预测后风险路由（SafeConf / E201 已有正式盲测）
业务B：风险引导训练（源域证据给 TxPert 加权 / E204 只有工程验收）

Opened complete WoS SCIE lists (impactfactor.cn, 2026-09-07):
MCB 61, BRM 82, Multidisciplinary 85, CS Interdisciplinary 115,
Genetics 186, Biotech 177/178, Medical Informatics 32, CS AI 153/154,
plus Nature/Cell/CCF extras. Each title gets a unique summary and two marks.
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

from safeconf_audit.journal_profiles import (
    A_HAVE,
    A_LACK,
    B_HAVE,
    B_LACK,
    HAND_PROFILES,
    fold_key,
    profile_for,
)

PACK_REL = Path("docs/学习导航/20260906_论文审核与从零教学")
UNIVERSE_MD = "06_全球相关期刊逐本评价.md"
UNIVERSE_CSV = "journal_universe.csv"
LISTS_PATH = Path(__file__).with_name("journal_lists.json")
OPENED_ON = "2026-09-07"

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

BIZ_A = "业务A：预测后风险路由（SafeConf）"
BIZ_B = "业务B：风险引导训练（E204 给 TxPert 加权）"

VERDICT_CN = {
    "discuss_narrow": "可讨论（必须收窄主张）",
    "q2_not_ready": "候选但未就绪",
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

CAT_CN = {
    "MCB": "数学与计算生物学",
    "BRM": "生化研究方法",
    "MULTI": "多学科科学",
    "CS_IA": "计算机科学跨学科应用",
    "GENETICS": "遗传学",
    "BIOTECH": "生物技术与应用微生物",
    "MEDINFO": "医学信息学",
    "CS_AI": "计算机科学人工智能",
    "GENOMICS": "基因组/方法交叉补入",
    "CCF": "CCF 会议交叉补入",
    "CELL": "Cell/Nature 家族交叉补入",
}

TRACK_META = {
    "bioinfo_methods": ("生物信息方法", "可复现计算方法或评价协议、公开代码、和最强简单基线比较。"),
    "theory_mathbio": ("数理生物学理论", "微分方程、随机过程或可证明的理论模型。"),
    "biostats": ("生物统计", "新的统计估计、检验或试验设计。"),
    "review": ("综述/约稿", "编辑约稿综述、方法引物或逐步实验方案。"),
    "medical": ("医学计算/临床", "临床决策、病历、医学影像或健康记录。"),
    "neuro": ("神经科学", "神经元、脑网络或神经实验方法。"),
    "database": ("数据库/Web 资源", "可公开点击的数据库或 Web server。"),
    "systems": ("系统生物学", "网络、通路或细胞决策的机制模型。"),
    "engineering": ("生物医学工程", "器件、数值求解或实验室自动化。"),
    "chemo": ("化学信息/分子模拟", "对接、QSAR 或小分子性质预测。"),
    "ontology": ("本体与语义", "生物医学本体或知识图谱。"),
    "imaging": ("生物医学成像", "光学、磁共振或医学影像。"),
    "wet_lab": ("湿实验方法", "实验室可执行的新实验方案、试剂或装置。"),
    "chromatography": ("色谱/电泳/分离分析", "分离方法、定量验证、检测限。"),
    "crystallography": ("晶体学", "蛋白质或核酸晶体结构。"),
    "proteomics": ("蛋白质组", "质谱蛋白质组鉴定定量。"),
    "mass_spec": ("质谱", "质谱仪器方法或谱图解析。"),
    "multidisciplinary": ("综合/方法顶刊", "完整故事：方法新、证据硬、最好成为默认用法。"),
    "genomics": ("基因组学", "基因组发现或能推动基因组实验的方法。"),
    "clinical_genetics": ("临床遗传学", "遗传病表型、综合征、遗传咨询。"),
    "ml": ("机器学习", "新的学习算法或理论。"),
    "conference": ("计算机会议", "算法新意和相对强基线的增量。"),
    "biotech_industrial": ("工业生物技术", "发酵、酶工程、生物燃料或菌种改造。"),
    "plant": ("植物", "植物基因组、育种或植物实验方法。"),
    "evolution": ("进化生物学", "系统发育、进化基因组或种群遗传。"),
    "other": ("其他学科", "刊名和栏目都不指向单细胞扰动预测或训练加权。"),
}

EXTRAS: list[tuple[str, str, str]] = [
    ("Cell", "CELL", "multidisciplinary"),
    ("eLife", "CELL", "multidisciplinary"),
    ("Cell Systems", "CELL", "systems"),
    ("Cell Reports", "CELL", "wet_lab"),
    ("Cell Genomics", "CELL", "genomics"),
    ("Cell Reports Methods", "CELL", "multidisciplinary"),
    ("Nature Cell Biology", "CELL", "wet_lab"),
    ("Nucleic Acids Research", "GENOMICS", "database"),
    ("NAR Genomics and Bioinformatics", "GENOMICS", "bioinfo_methods"),
    ("Bioinformatics Advances", "GENOMICS", "bioinfo_methods"),
    ("Patterns", "CELL", "ml"),
    ("Communications Biology", "CELL", "multidisciplinary"),
    ("PeerJ", "GENOMICS", "bioinfo_methods"),
    ("Frontiers in Bioinformatics", "GENOMICS", "bioinfo_methods"),
    ("Nature Computational Science", "CELL", "ml"),
    ("NeurIPS", "CCF", "conference"),
    ("ICML", "CCF", "conference"),
    ("ICLR", "CCF", "conference"),
    ("RECOMB", "CCF", "conference"),
    ("ISMB / Bioinformatics Proceedings", "CCF", "conference"),
    ("AAAI", "CCF", "conference"),
    ("IJCAI", "CCF", "conference"),
    ("KDD", "CCF", "conference"),
]


def load_lists() -> dict:
    return json.loads(LISTS_PATH.read_text(encoding="utf-8"))


def _official_pairs(code: str) -> list[tuple[str, str]]:
    titles = load_lists()["categories"][code]["titles"]
    return [(t, "") for t in titles]


MCB_OFFICIAL = None  # filled after module load to avoid double JSON read in import loops


def _init_official() -> None:
    global MCB_OFFICIAL, BRM_OFFICIAL, GENETICS_OFFICIAL
    data = load_lists()
    MCB_OFFICIAL = [(t, "") for t in data["categories"]["MCB"]["titles"]]
    BRM_OFFICIAL = [(t, "") for t in data["categories"]["BRM"]["titles"]]
    GENETICS_OFFICIAL = [(t, "") for t in data["categories"]["GENETICS"]["titles"]]


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
    aliases = {
        "pnas": "proceedingsofthenationalacademyofsciencesoftheunitedstatesofamerica",
        "proceedingsofthenationalacademyofsciencesoftheunitedstatesofamerica": "pnas",
        "wirescomputationalmolecularscience": "wileyinterdisciplinaryreviewscomputationalmolecularscience",
        "database": "databasethejournalofbiologicaldatabasesandcuration",
        "databasethejournalofbiologicaldatabasesandcuration": "database",
        "ismbbioinformaticsproceedings": "ismb",
        "nature": "nature",
    }
    if aliases.get(fa) == fb or aliases.get(fb) == fa:
        return True
    short, long = (fa, fb) if len(fa) <= len(fb) else (fb, fa)
    if long.startswith(short) and len(short) >= 22:
        return True
    stop = {"the", "of", "and", "in", "for", "a", "on", "journal", "section", "an", "part"}
    ta = {t for t in re.findall(r"[a-z0-9]+", a.lower().replace("&", " and ")) if t not in stop}
    tb = {t for t in re.findall(r"[a-z0-9]+", b.lower().replace("&", " and ")) if t not in stop}
    if len(ta) >= 3 and ta <= tb:
        return True
    if len(tb) >= 3 and tb <= ta:
        return True
    return False


def infer_track(name: str) -> str:
    n = name.lower()
    if re.search(r"annual review|current opinion|trends in |nature reviews|wires |expert review", n):
        return "review"
    if "chromatog" in n or "electrophoresis" in n:
        return "chromatography"
    if "crystallograph" in n:
        return "crystallography"
    if "proteom" in n:
        return "proteomics"
    if "mass spectrom" in n:
        return "mass_spec"
    if re.search(r"medical genetics|dysmorpholog|genetic counseling|syndromolog", n):
        return "clinical_genetics"
    if re.search(r"plant |crop |tree genetics|silvae", n):
        return "plant"
    if re.search(r"evolution|phylogen|systematics", n) and "bioinformatics" not in n:
        return "evolution"
    if re.search(r"biofuel|fermentation|enzyme|algal |food micro|biodegradation|biocontrol", n):
        return "biotech_industrial"
    if re.search(r"bioinformatics|computational biology|biodata mining|computational life", n):
        return "bioinfo_methods"
    if re.search(r"genome biology|genome research|genomics, proteomics", n):
        return "genomics"
    if "database" in n and "curation" in n:
        return "database"
    if re.search(r"neural network|machine intelligence|artificial intelligence|pattern analysis|fuzzy systems|knowledge-based", n):
        return "ml"
    if re.search(r"medical informatics|health informatics|digital health|medical internet|clinical informatics", n):
        return "medical"
    if re.search(r"mathematical biology|theoretical biology|biotheoretic|biomath", n):
        return "theory_mathbio"
    if re.search(r"biostatistic|statistics in medicine|biometrika|biometrics|biometrical", n):
        return "biostats"
    if re.search(r"neuro|brain ", n):
        return "neuro"
    if re.search(r"qsar|molecular graphics|cheminform|docking", n):
        return "chemo"
    if re.search(r"ontology|semantics", n):
        return "ontology"
    if re.search(r"optics|photonic|magnetic resonance|medical imaging", n):
        return "imaging"
    if re.search(r"transgenic|cytometry|microfluidic|lab on a chip|synthetic biology|protein expression", n):
        return "wet_lab"
    if re.search(r"systems biology|cell systems", n):
        return "systems"
    if re.search(
        r"nature methods|nature biotech|nature comm|scientific reports|science advances|"
        r"pnas|elife|iscience|plos one|communications biology|"
        r"proceedings of the national academy of sciences of the united states",
        n,
    ):
        return "multidisciplinary"
    if "genom" in n or n.startswith("gene ") or "genetic" in n:
        return "genomics"
    if "biotech" in n:
        return "biotech_industrial"
    return "other"


OFF_TRACKS = {
    "chromatography",
    "crystallography",
    "proteomics",
    "mass_spec",
    "clinical_genetics",
    "plant",
    "evolution",
    "biotech_industrial",
    "theory_mathbio",
    "biostats",
    "neuro",
    "chemo",
    "ontology",
    "imaging",
    "wet_lab",
    "engineering",
    "medical",
    "other",
}


def default_verdicts(track: str, name: str) -> tuple[str, str]:
    n = name.lower()
    if track == "review":
        return "review_only", "review_only"
    if track in OFF_TRACKS:
        return "off_track", "off_track"
    if track == "database":
        return "off_track", "off_track"
    if track == "conference":
        if re.search(r"recomb|ismb", n):
            return "q2_not_ready", "q2_not_ready"
        return "q1_off", "q1_off"
    if track == "ml":
        if re.search(r"nature machine|pattern analysis and machine intelligence|neurips|icml|iclr", n):
            return "q1_off", "q1_off"
        return "off_track", "off_track"
    if track == "systems":
        if re.search(r"cell systems|molecular systems biology|npj systems", n):
            return "q1_off", "q1_off"
        return "off_track", "off_track"
    if track == "genomics":
        if "bmc" in n:
            return "discuss_narrow", "q2_not_ready"
        if re.search(
            r"(^| )genome biology($| and|,)|(^| )genome research($| )|nature genetics|"
            r"human molecular genetics|american journal of human genetics|plos genetics|"
            r"genome medicine|cell genomics|nucleic acids research",
            n,
        ):
            return "q1_off", "q1_off"
        return "off_track", "off_track"
    if track == "bioinfo_methods":
        if any(k in n for k in ("briefings in bioinformatics", "plos computational", "bioinformatics")) and "advances" not in n and "bmc" not in n and "current" not in n:
            return "q2_not_ready", "q2_not_ready"
        return "discuss_narrow", "q2_not_ready"
    if track == "multidisciplinary":
        if any(k in n for k in ("scientific reports", "peerj", "iscience", "plos one")):
            return "discuss_narrow", "q2_not_ready"
        return "q1_off", "q1_off"
    return "off_track", "off_track"


def keyword_extra(name: str) -> str:
    n = name.lower()
    bits = []
    rules = (
        ("chromatography a", "A 辑发分离机理和柱技术，不是生物医学检测应用。"),
        ("chromatography b", "B 辑发生物基质定量，仍是分析化学。"),
        ("chromatographia", "这是色谱方法短文刊，栏目低于 Chromatogr A/B，对象仍是分离。"),
        ("electrophoresis", "电泳分离，不是单细胞预测。"),
        ("medical genetics part a", "人类遗传病临床表型和综合征，读者是医学遗传科医生。"),
        ("medical genetics part b", "神经精神遗传病，临床遗传，不是计算方法。"),
        ("medical genetics part c", "医学遗传综述/研讨会，不是研究算法文。"),
        ("plos one", "综合 OA，什么学科都收，但不是二区方法主刊。"),
        ("nature protocols", "逐步实验方案，不是研究论文。"),
        ("database", "要可访问数据库。当前没有对外库。"),
        ("ieee journal of biomedical and health informatics", "临床信号和健康信息，不是 K562 扰动。"),
        ("computers in biology and medicine", "医学数据计算，不是细胞系扰动风险。"),
        ("computer methods and programs in biomedicine", "生物医学程序，偏临床软件。"),
        ("statistics in medicine", "临床试验和生存分析，不是扰动预测。"),
        ("neurips", "机器学习主会。"),
        ("genome biology and evolution", "进化基因组，不是 CRISPRi 扰动预测。"),
        ("human genomics", "人类基因组临床，不对轨。"),
        ("plant methods", "植物实验方法。当前四个细胞系是人源。"),
        ("journal of chromatography a", "分离科学本身。"),
        ("journal of chromatography b", "生物样品色谱定量。"),
    )
    folded = _fold(name)
    for key, sent in rules:
        if _fold(key) in folded or key in n:
            bits.append(sent)
    if "k562" not in "".join(bits) and re.search(r"plant|crop|algal|fungal|tree ", n):
        bits.append("对象不是人源 K562 / RPE1 / HepG2 / Jurkat。")
    return "".join(bits)


def unique_publishes(name: str, track: str, cats: list[str]) -> str:
    hand = profile_for(name)
    if hand:
        return hand["publishes"]
    cn, want = TRACK_META.get(track, ("其他", "该学科常规稿件。"))
    cat_cn = "、".join(CAT_CN.get(c, c) for c in cats)
    extra = keyword_extra(name)
    return (
        f"{name} 出现在本次打开的完整名单里，学科来源：{cat_cn}。"
        f"按刊名和栏目，它日常刊登的是{cn}：{want}"
        f"{extra}"
        f"评价时必须把两条业务拆开：业务A是预测后给任务打风险分（SafeConf，E201 有 Spearman 0.4082），"
        f"业务B是用源域风险给 TxPert 重新加权训练（E204 没有 80 轮正式效果）。"
        f"不能用一句“生信刊都能投”或“都不对轨”代替逐本判断。"
    )


def care_for(track: str, verdict: str, which: str) -> list[int]:
    if verdict in {"off_track", "review_only"}:
        if track in {"wet_lab", "chromatography", "crystallography", "proteomics", "mass_spec", "imaging"}:
            return [0, 0, 0, 0, 0, 0, 0, 2]
        return [0, 0, 0, 0, 0, 0, 0, 0]
    if which == "a":
        if verdict == "q1_off":
            return [2, 2, 2, 2, 1, 2, 2, 2]
        if verdict == "q2_not_ready":
            return [2, 2, 2, 1, 0, 1, 2, 0]
        return [2, 2, 1, 0, 0, 0, 1, 0]
    if verdict == "q1_off":
        return [2, 2, 2, 1, 2, 2, 2, 2]
    return [2, 2, 1, 0, 2, 1, 2, 0]


def _display_name(wos_name: str) -> str:
    hand = profile_for(wos_name)
    if hand:
        return hand["name"]
    prefixes = (
        ("BMC ", "BMC "),
        ("IEEE ", "IEEE "),
        ("IEEE-", "IEEE-"),
        ("PLOS ", "PLOS "),
        ("PLoS ", "PLoS "),
        ("NAR ", "NAR "),
        ("PNAS ", "PNAS "),
        ("NPJ ", "npj "),
        ("ACM ", "ACM "),
        ("IET ", "IET "),
        ("AAAI", "AAAI"),
        ("IJCAI", "IJCAI"),
        ("ICML", "ICML"),
        ("ICLR", "ICLR"),
        ("KDD", "KDD"),
    )
    body = wos_name
    prefix = ""
    upper = wos_name.upper()
    for raw, keep in prefixes:
        if upper.startswith(raw.upper()):
            prefix = keep
            body = wos_name[len(raw) :]
            break
    if body.isupper() or (wos_name.isupper() and len(wos_name) > 4):
        small = {"AND", "OF", "THE", "IN", "FOR", "ON", "A", "&"}
        parts = []
        for tok in body.split(" "):
            if not tok:
                continue
            if tok in {"&"}:
                parts.append(tok)
            elif tok.upper() in small:
                parts.append(tok.lower())
            else:
                parts.append(tok.title())
        return prefix + " ".join(parts)
    return wos_name


def catalog() -> list[dict]:
    data = load_lists()
    buckets: dict[str, dict] = {}
    for code, payload in data["categories"].items():
        for title in payload["titles"]:
            key = _fold(title)
            rec = buckets.setdefault(key, {"wos_names": [], "cats": []})
            if title not in rec["wos_names"]:
                rec["wos_names"].append(title)
            if code not in rec["cats"]:
                rec["cats"].append(code)
    for name, field, track in EXTRAS:
        key = _fold(name)
        rec = buckets.setdefault(key, {"wos_names": [], "cats": []})
        if name not in rec["wos_names"]:
            rec["wos_names"].append(name)
        if field not in rec["cats"]:
            rec["cats"].append(field)
        rec.setdefault("forced_track", track)
    rows = []
    seen_display = set()
    for key, rec in buckets.items():
        wos_name = rec["wos_names"][0]
        display = _display_name(wos_name)
        # prefer hand-profile canonical name
        hand = None
        for candidate in rec["wos_names"] + [display]:
            hand = profile_for(candidate)
            if hand:
                display = hand["name"]
                break
        dkey = display.strip().lower()
        if dkey in seen_display:
            continue
        seen_display.add(dkey)
        track = (hand or {}).get("track") or rec.get("forced_track") or infer_track(display)
        verdict_a, verdict_b = default_verdicts(track, display)
        if hand:
            verdict_a = hand["verdict_a"]
            verdict_b = hand["verdict_b"]
            track = hand["track"]
        care_a = list(hand["care_a"]) if hand else care_for(track, verdict_a, "a")
        care_b = list(hand["care_b"]) if hand else care_for(track, verdict_b, "b")
        field = "+".join(rec["cats"])
        source = "；".join(
            f"{OPENED_ON} 打开 {CAT_CN.get(c, c)} 完整名单"
            if c in data["categories"]
            else f"{CAT_CN.get(c, c)}"
            for c in rec["cats"]
        )
        row = {
            "name": display,
            "wos_name": " | ".join(rec["wos_names"]),
            "issn": "",
            "field": field,
            "track": track,
            "track_cn": TRACK_META.get(track, ("其他", ""))[0],
            "verdict": verdict_a,
            "verdict_a": verdict_a,
            "verdict_b": verdict_b,
            "verdict_cn": VERDICT_CN[verdict_a],
            "verdict_a_cn": VERDICT_CN[verdict_a],
            "verdict_b_cn": VERDICT_CN[verdict_b],
            "they_want": TRACK_META.get(track, ("", "该刊常规栏目。"))[1],
            "publishes": unique_publishes(display, track, rec["cats"]),
            "audience": (hand or {}).get("audience", ""),
            "a_ask": (hand or {}).get("a_ask", "这本杂志会不会把业务A当成自己栏目里的方法/评价稿。"),
            "b_ask": (hand or {}).get("b_ask", "这本杂志会不会把业务B当成训练/算法增量稿。"),
            "a_why": (hand or {}).get("a_why", ""),
            "b_why": (hand or {}).get("b_why", ""),
            "we_have": A_HAVE if verdict_a not in {"off_track", "review_only"} else "没有对口的业务A稿件。",
            "we_lack": A_LACK if verdict_a not in {"off_track", "review_only"} else "学科不对轨。补 E204 也填不上业务A。",
            "b_have": B_HAVE if verdict_b not in {"off_track", "review_only"} else "没有对口的业务B稿件。",
            "b_lack": B_LACK if verdict_b not in {"off_track", "review_only"} else "学科不对轨。做出训练效果也填不上业务B。",
            "reason": (hand or {}).get("a_why") or f"{display}：A={VERDICT_CN[verdict_a]}；B={VERDICT_CN[verdict_b]}",
            "cas_note": (hand or {}).get(
                "cas_note",
                "本表不把分区写成录用保证。投稿前以学院指定查询系统为准。",
            ),
            "care": care_a,
            "care_a": care_a,
            "care_b": care_b,
            "source_opened": source,
        }
        if not row["a_why"]:
            row["a_why"] = row["reason"]
        if not row["b_why"]:
            row["b_why"] = f"业务B判定：{VERDICT_CN[verdict_b]}。"
        rows.append(row)
    rows.sort(key=lambda r: (VERDICT_ORDER.index(r["verdict_a"]), r["name"].lower()))
    return rows


def counts(rows: Iterable[dict] | None = None) -> dict[str, int]:
    rows = list(rows or catalog())
    data = load_lists()
    tally: dict[str, int] = {f"a_{k}": 0 for k in VERDICT_ORDER}
    tally.update({f"b_{k}": 0 for k in VERDICT_ORDER})
    for row in rows:
        tally[f"a_{row['verdict_a']}"] += 1
        tally[f"b_{row['verdict_b']}"] += 1
    tally["total"] = len(rows)
    tally["mcb_official"] = len(data["categories"]["MCB"]["titles"])
    tally["brm_official"] = len(data["categories"]["BRM"]["titles"])
    tally["genetics_official"] = len(data["categories"]["GENETICS"]["titles"])
    tally["multi_official"] = len(data["categories"]["MULTI"]["titles"])
    # backward key used by old fig6 test
    tally["discuss_narrow"] = tally["a_discuss_narrow"]
    tally["q2_not_ready"] = tally["a_q2_not_ready"]
    tally["q1_off"] = tally["a_q1_off"]
    tally["off_track"] = tally["a_off_track"]
    tally["review_only"] = tally["a_review_only"]
    return tally


def official_unmatched(rows: Iterable[dict] | None = None) -> list[str]:
    rows = list(rows or catalog())
    data = load_lists()
    missing = []
    for code in ("MCB", "BRM"):
        for official in data["categories"][code]["titles"]:
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


def evidence_rows(row: dict, which: str = "a") -> list[dict]:
    care_list = row["care_a"] if which == "a" else row["care_b"]
    out = []
    for i, (key, label) in enumerate(EVIDENCE_ITEMS):
        care = int(care_list[i])
        have = int(EVIDENCE_HAVE[key])
        out.append(
            {
                "key": key,
                "label": label,
                "care": care,
                "have": have,
                "care_cn": _care_cn(care),
                "fact": HAVE_FACT[key],
                "lamp": _lamp(care, have),
            }
        )
    return out


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in row) + " |")
    return "\n".join(lines)


def _journal_section(row: dict) -> str:
    identity = _md_table(
        ["字段", "内容"],
        [
            ["刊名", row["name"]],
            ["名单中的写法", row["wos_name"]],
            ["学科来源", row["source_opened"]],
            ["轨道", row["track_cn"]],
            ["常见分区口径", row["cas_note"]],
            ["这本杂志实际发什么", row["publishes"]],
            ["读者是谁", row["audience"] or "见上栏"],
        ],
    )
    dual = _md_table(
        ["标记", BIZ_A, BIZ_B],
        [
            ["判定", row["verdict_a_cn"], row["verdict_b_cn"]],
            ["他们会问", row["a_ask"], row["b_ask"]],
            ["我们现在有什么", row["we_have"], row["b_have"]],
            ["我们现在缺什么", row["we_lack"], row["b_lack"]],
            ["为什么这样判", row["a_why"], row["b_why"]],
        ],
    )
    lamps_a = _md_table(
        ["审稿人会问", "业务A在不在乎", "我们现在有没有", "格子"],
        [[e["label"], e["care_cn"], e["fact"], e["lamp"]] for e in evidence_rows(row, "a")],
    )
    lamps_b = _md_table(
        ["审稿人会问", "业务B在不在乎", "我们现在有没有", "格子"],
        [[e["label"], e["care_cn"], e["fact"], e["lamp"]] for e in evidence_rows(row, "b")],
    )
    return (
        f"### {row['name']}\n\n{identity}\n\n{dual}\n\n"
        f"**业务A 证据灯**\n\n{lamps_a}\n\n**业务B 证据灯**\n\n{lamps_b}\n"
    )


def render_markdown(rows: list[dict] | None = None) -> str:
    rows = list(rows or catalog())
    tally = counts(rows)
    master = _md_table(
        ["刊名", "学科来源", "轨道", "业务A 路由", "业务B 训练", "这本杂志实际发什么（摘要）"],
        [
            [
                r["name"],
                r["field"],
                r["track_cn"],
                r["verdict_a_cn"],
                r["verdict_b_cn"],
                r["publishes"][:180] + ("…" if len(r["publishes"]) > 180 else ""),
            ]
            for r in rows
        ],
    )
    by_a: dict[str, list[dict]] = {k: [] for k in VERDICT_ORDER}
    for row in rows:
        by_a[row["verdict_a"]].append(row)
    parts = [
        "# 全球相关期刊逐本评价（两条业务分开打标）",
        "",
        f"评价日期：{OPENED_ON}。数字锁在 `CLAIM_TABLE.json`：SafeConf Spearman 0.4082，预测幅度 0.6189，偏相关 0.2503，20% 效用 0.3200 vs 0.5943。E204 没有 80 轮正式效果。E205 未跑。",
        "",
        "**不能把二区写成一定能发。** 本页也不保证任何一本杂志会录用。",
        "",
        "## 0. 两条业务必须分开，不能糊成一张总表",
        "",
        f"**{BIZ_A}** 是给已经做完的预测打“这题可能错”的分。E201 有正式盲测。",
        f"**{BIZ_B}** 是用源域证据给 TxPert 重新加权再训练。E204 只有工程验收，没有正式效果。",
        "",
        "上一份把两条业务揉成一个判定，又用同一段“他们要什么”套在所有生信刊上，这是糊弄。下面每一本都有：它实际发什么、业务A判定、业务B判定、两套证据灯。",
        "",
        "## 0.1 打开了哪些完整名单（不是只看前几本）",
        "",
        "不是地球上全部 SCI。打开并写入的是：",
        "",
        f"1. SCIE 数学与计算生物学 61/61（sort-699）",
        f"2. SCIE 生化研究方法 82/82（sort-615）",
        f"3. SCIE 多学科科学 85/85（sort-716）",
        f"4. SCIE 计算机科学跨学科应用 115/115（sort-636）",
        f"5. SCIE 遗传学 186/186（sort-672）",
        f"6. SCIE 生物技术与应用微生物 177/178（sort-620；分页末页少 1 本）",
        f"7. SCIE 医学信息学 32/32（sort-705）",
        f"8. SCIE 计算机科学人工智能 153/154（sort-632；分页少 1 本）",
        "9. Nature / Cell / CCF 交叉补入（不在上述名单里的，例如 Cell Systems、NAR、NeurIPS）",
        "",
        f"去重后共 **{tally['total']}** 本。业务A：可讨论 {tally['a_discuss_narrow']}，未就绪 {tally['a_q2_not_ready']}，一区不够 {tally['a_q1_off']}，综述 {tally['a_review_only']}，不对轨 {tally['a_off_track']}。业务B：可讨论 {tally['b_discuss_narrow']}，未就绪 {tally['b_q2_not_ready']}，一区不够 {tally['b_q1_off']}，综述 {tally['b_review_only']}，不对轨 {tally['b_off_track']}。",
        "",
        "色谱、晶体、临床遗传凭刊名加小类判定不对轨，没有再点进每本 Aims & Scope。方法刊、基因组刊、顶刊按下表用 E201/E204 证据逐本写。",
        "",
        "![图 6 两条业务的判定计数](figures/fig6_journal_universe.png)",
        "",
        "**图 6a** 业务A和业务B的判定各有多少本。**图 6b** 业务A还可能被问到的刊，按轨道拆开。",
        "",
        "![图 7 每一本对两条业务的格子](figures/fig7_journal_heatmap.png)",
        "",
        "**图 7a** 业务A或业务B不是“学科不对轨”的刊。绿=可讨论，橙=未就绪，蓝=一区不够，灰=综述，红=不对轨。**图 7b** 其余不对轨的刊也逐本打了两个标记。",
        "",
        "机器可读总表：[journal_universe.csv](./journal_universe.csv)。",
        "",
        "## 1. 一览表（每一本一行，两列业务）",
        "",
        master,
        "",
        "## 2. 怎么读后面的逐本表",
        "",
        "每一本三块：它实际发什么；业务A vs 业务B对照表；两套八盏证据灯。业务A的灯里，E204 不是桌面拒项（那是另一篇稿）。业务B的灯里，E204 没有正式效果就是红灯。",
        "",
    ]
    headings = {
        "discuss_narrow": "## 3. 业务A可讨论——逐本（业务B仍可能未就绪）",
        "q2_not_ready": "## 4. 业务A候选但未就绪——逐本",
        "q1_off": "## 5. 业务A一区/顶刊不够——逐本",
        "review_only": "## 6. 只收综述或方案——逐本",
        "off_track": "## 7. 业务A学科不对轨——逐本（业务B同样逐本打标）",
    }
    for verdict in VERDICT_ORDER:
        parts.append(headings[verdict])
        parts.append("")
        parts.append(f"这一类以业务A计 {len(by_a[verdict])} 本。每一本仍然同时给出业务B。")
        parts.append("")
        for row in by_a[verdict]:
            parts.append(_journal_section(row))
    parts.extend(
        [
            "## 8. 和「二区一定能发、一区可以冲刺」怎么对齐",
            "",
            "用户要：二区一定能发，一区可以冲刺。两条业务分开看：",
            "",
            "- 业务A（路由）：二区里最常被点名的是 Bioinformatics、Briefings in Bioinformatics、PLOS Computational Biology。全部是未就绪。卡住的是幅度 0.6189 更强、没有新外部。BMC Bioinformatics 可以收窄讨论，不是一定能发。",
            "- 业务B（训练）：所有方法刊目前都未就绪，因为 E204 没有 80 轮正式效果。不存在“先投训练稿、路由稿后补”的捷径。",
            "- 一区 Nature Methods / Genome Biology / Nature Communications：两条业务现在都不能冲。",
            "- Chromatographia、色谱 A/B、临床遗传、发酵生物技术：两条业务都是学科不对轨。",
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
        "field",
        "track",
        "track_cn",
        "verdict_a",
        "verdict_b",
        "verdict_a_cn",
        "verdict_b_cn",
        "cas_note",
        "publishes",
        "a_ask",
        "b_ask",
        "a_why",
        "b_why",
        "we_have",
        "we_lack",
        "b_have",
        "b_lack",
        "source_opened",
        "care_a",
        "care_b",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["care_a"] = ",".join(str(x) for x in row["care_a"])
            payload["care_b"] = ",".join(str(x) for x in row["care_b"])
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


def _panel_id(ax, letter: str, fp: FontProperties) -> None:
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


VERDICT_COLOR = {
    "discuss_narrow": 1.0,
    "q2_not_ready": 0.65,
    "q1_off": 0.35,
    "review_only": 0.45,
    "off_track": 0.0,
}


def generate_fig6_universe(repo: Path, fp: FontProperties, rows: list[dict] | None = None) -> dict:
    rows = list(rows or catalog())
    tally = counts(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), facecolor="white",
                             gridspec_kw={"width_ratios": [1.15, 1.25]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    _style_axes(ax, fp)
    _panel_id(ax, "a", fp)
    labels = [VERDICT_CN[k] for k in VERDICT_ORDER]
    x = np.arange(len(labels))
    a_vals = [tally[f"a_{k}"] for k in VERDICT_ORDER]
    b_vals = [tally[f"b_{k}"] for k in VERDICT_ORDER]
    ax.bar(x - 0.18, a_vals, 0.36, color=OKABE["blue"], label="业务A 路由")
    ax.bar(x + 0.18, b_vals, 0.36, color=OKABE["orange"], label="业务B 训练")
    ax.set_xticks(x, labels, fontproperties=fp, fontsize=7, rotation=18, ha="right")
    ax.set_ylabel("期刊本数", fontproperties=fp, fontsize=8)
    ax.set_title("两条业务分开计数", fontproperties=fp, fontsize=9, loc="left")
    ax.legend(prop=fp, fontsize=7, loc="upper right")
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - 0.18, a + 1, str(a), ha="center", fontsize=6, fontproperties=fp)
        ax.text(i + 0.18, b + 1, str(b), ha="center", fontsize=6, fontproperties=fp)

    ax = axes[1]
    _style_axes(ax, fp)
    _panel_id(ax, "b", fp)
    on = [r for r in rows if r["verdict_a"] != "off_track" or r["verdict_b"] != "off_track"]
    track_counts: dict[str, int] = {}
    for row in on:
        track_counts[row["track_cn"]] = track_counts.get(row["track_cn"], 0) + 1
    items = sorted(track_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:16]
    y = list(range(len(items), 0, -1))
    ax.barh(y, [n for _, n in items], color=OKABE["sky"], height=0.62, linewidth=0)
    ax.set_yticks(y, [name for name, _ in items], fontproperties=fp, fontsize=7)
    ax.set_xlabel("至少一条业务不是完全不对轨", fontproperties=fp, fontsize=8)
    ax.set_title("还可能被问到的刊按轨道", fontproperties=fp, fontsize=9, loc="left")
    for yi, (_, n) in zip(y, items):
        ax.text(n + 0.15, yi, str(n), va="center", fontproperties=fp, fontsize=7)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.86, bottom=0.22, wspace=0.38)
    plotted = {
        "total": tally["total"],
        "verdict_counts": {k: tally[f"a_{k}"] for k in VERDICT_ORDER},
        "verdict_counts_a": {k: tally[f"a_{k}"] for k in VERDICT_ORDER},
        "verdict_counts_b": {k: tally[f"b_{k}"] for k in VERDICT_ORDER},
        "off_track_by_track": dict(items),
        "mcb_official": tally["mcb_official"],
        "brm_official": tally["brm_official"],
        "genetics_official": tally["genetics_official"],
        "names": [r["name"] for r in rows],
    }
    return _save_fig(repo, fig, "fig6_journal_universe", plotted)


def generate_fig7_heatmap(repo: Path, fp: FontProperties, rows: list[dict] | None = None) -> dict:
    rows = list(rows or catalog())
    on_topic = [r for r in rows if r["verdict_a"] != "off_track" or r["verdict_b"] != "off_track"]
    off_topic = [r for r in rows if r["verdict_a"] == "off_track" and r["verdict_b"] == "off_track"]
    on_topic.sort(key=lambda r: (VERDICT_ORDER.index(r["verdict_a"]), r["name"].lower()))
    off_topic.sort(key=lambda r: r["name"].lower())
    # heatmap of on-topic only in panel a (2 columns); panel b is a compact count not 600 rows
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, max(6.5, 0.18 * len(on_topic) + 1.8)),
        facecolor="white",
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    fig.patch.set_facecolor("white")
    cmap = ListedColormap(["#D55E00", "#56B4E9", "#C8C8C8", "#E69F00", "#009E73"])
    bounds = [-0.1, 0.2, 0.4, 0.55, 0.8, 1.1]
    norm = BoundaryNorm(bounds, cmap.N)

    ax = axes[0]
    _panel_id(ax, "a", fp)
    matrix = [[VERDICT_COLOR[r["verdict_a"]], VERDICT_COLOR[r["verdict_b"]]] for r in on_topic]
    ax.imshow(np.array(matrix), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["业务A 路由", "业务B 训练"], fontproperties=fp, fontsize=8)
    ax.set_yticks(range(len(on_topic)))
    ax.set_yticklabels([r["name"] for r in on_topic], fontproperties=fp, fontsize=6)
    ax.set_title("还可能被问到的刊：两条业务分开着色", fontproperties=fp, fontsize=9, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = axes[1]
    _style_axes(ax, fp)
    _panel_id(ax, "b", fp)
    track_counts: dict[str, int] = {}
    for row in off_topic:
        track_counts[row["track_cn"]] = track_counts.get(row["track_cn"], 0) + 1
    items = sorted(track_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    y = list(range(len(items), 0, -1))
    ax.barh(y, [n for _, n in items], color=OKABE["vermillion"], height=0.62, linewidth=0)
    ax.set_yticks(y, [name for name, _ in items], fontproperties=fp, fontsize=7)
    ax.set_xlabel("两条业务都不对轨的本数", fontproperties=fp, fontsize=8)
    ax.set_title("不对轨的刊也按轨道点过名", fontproperties=fp, fontsize=9, loc="left")
    for yi, (_, n) in zip(y, items):
        ax.text(n + 0.2, yi, str(n), va="center", fontproperties=fp, fontsize=7)
    fig.subplots_adjust(left=0.32, right=0.98, top=0.94, bottom=0.08, wspace=0.35)
    plotted = {
        "have": EVIDENCE_HAVE,
        "item_keys": [k for k, _ in EVIDENCE_ITEMS],
        "on_topic": [r["name"] for r in on_topic],
        "off_topic": [r["name"] for r in off_topic],
        "all_names": [r["name"] for r in rows],
        "matrix_on": matrix,
        "verdicts": {r["name"]: r["verdict_a"] for r in rows},
        "verdicts_a": {r["name"]: r["verdict_a"] for r in rows},
        "verdicts_b": {r["name"]: r["verdict_b"] for r in rows},
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
    blurbs = [r["publishes"] for r in rows]
    return {
        "n": len(rows),
        "unmatched_official": official_unmatched(rows),
        "unique_publishes": len(set(blurbs)) == len(blurbs),
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
    csv_text = csv_path.read_text(encoding="utf-8") if csv_path.is_file() else ""
    rows = catalog()
    if len(rows) < 400:
        missing.append(f"catalog too small: {len(rows)}")
    blurbs = [r["publishes"] for r in rows]
    if len(set(blurbs)) != len(blurbs):
        missing.append("publishes text is not unique per journal")
    by_name = {r["name"].lower(): r for r in rows}
    required = {
        "chromatographia": ("off_track", "off_track"),
        "bioinformatics": ("q2_not_ready", "q2_not_ready"),
        "nature methods": ("q1_off", "q1_off"),
        "bmc bioinformatics": ("discuss_narrow", "q2_not_ready"),
    }
    for name, (va, vb) in required.items():
        row = by_name.get(name)
        if row is None:
            missing.append(f"catalog missing {name}")
            continue
        if row["verdict_a"] != va:
            missing.append(f"{name} verdict_a {row['verdict_a']} != {va}")
        if row["verdict_b"] != vb:
            missing.append(f"{name} verdict_b {row['verdict_b']} != {vb}")
        if row["name"] not in text:
            missing.append(f"{UNIVERSE_MD} missing journal {row['name']}")
    for phrase in (
        "不能把二区写成一定能发",
        "0.4082",
        "0.6189",
        "Chromatographia",
        "业务A",
        "业务B",
        "sort-699",
        "sort-672",
    ):
        if phrase not in text:
            missing.append(f"{UNIVERSE_MD} missing {phrase}")
    if "verdict_a" not in csv_text or "verdict_b" not in csv_text:
        missing.append("csv missing dual-business columns")
    unmatched = official_unmatched(rows)
    if unmatched:
        missing.append("official titles not in catalog: " + "; ".join(unmatched[:8]))
    data = load_lists()
    if len(data["categories"]["MCB"]["titles"]) != 61:
        missing.append("MCB official list is not 61")
    if len(data["categories"]["BRM"]["titles"]) != 82:
        missing.append("BRM official list is not 82")
    if len(data["categories"]["GENETICS"]["titles"]) != 186:
        missing.append("GENETICS official list is not 186")
    for stem in ("fig6_journal_universe", "fig7_journal_heatmap"):
        for suffix in (".png", ".svg", ".values.json"):
            if not (pack / "figures" / f"{stem}{suffix}").is_file():
                missing.append(f"missing {stem}{suffix}")
    sidecar6 = pack / "figures" / "fig6_journal_universe.values.json"
    if sidecar6.is_file():
        plotted = json.loads(sidecar6.read_text()).get("plotted") or {}
        if plotted.get("mcb_official") != 61:
            missing.append("fig6 must record MCB official n=61")
        if plotted.get("brm_official") != 82:
            missing.append("fig6 must record BRM official n=82")
        if int(plotted.get("total") or 0) < 400:
            missing.append("fig6 total < 400")
        if "verdict_counts_b" not in plotted:
            missing.append("fig6 missing business-B counts")
    sidecar7 = pack / "figures" / "fig7_journal_heatmap.values.json"
    if sidecar7.is_file():
        plotted = json.loads(sidecar7.read_text()).get("plotted") or {}
        names = plotted.get("all_names") or []
        if "Chromatographia" not in names:
            missing.append("fig7 missing Chromatographia")
        if plotted.get("verdicts_b", {}).get("Bioinformatics") != "q2_not_ready":
            missing.append("fig7 Bioinformatics business B not q2_not_ready")
        xml = (pack / "figures" / "fig7_journal_heatmap.svg").read_text()
        if "drop-shadow" in xml.lower() or "feDropShadow" in xml:
            missing.append("fig7 has drop-shadow")
        for letter in ("a", "b"):
            if not re.search(rf">\s*{letter}\s*<", xml) and f">{letter}<" not in xml:
                if not re.search(rf">{letter}</", xml):
                    missing.append(f"fig7 missing panel letter {letter}")
    # every catalog name must appear in 06
    for row in rows:
        if row["name"] not in text:
            missing.append(f"{UNIVERSE_MD} missing catalog name {row['name']}")
            break
    return missing


_init_official()
