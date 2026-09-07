"""Hand-written dual-business profiles for journals that could actually receive a manuscript.

业务A = 预测后风险路由（SafeConf / E201 已有正式盲测）
业务B = 风险引导训练（源域证据给 TxPert 加权，E204 只有工程验收）

Every other title is still scored, but these get unique long descriptions instead of a track template.
"""
from __future__ import annotations

# care: 8 lamps = blind, magnitude, partial, beat_magnitude, e204, e205, new_external, wet
# 2 = desk-reject if missing, 1 = reviewers ask, 0 = not a usual desk item

CARE_A_METHODS = (2, 2, 2, 1, 0, 1, 2, 0)  # routing paper: E204 is another manuscript
CARE_A_Q1 = (2, 2, 2, 2, 1, 2, 2, 2)
CARE_B_METHODS = (2, 2, 1, 0, 2, 1, 2, 0)  # training paper: E204 is the result
CARE_B_Q1 = (2, 2, 2, 1, 2, 2, 2, 2)

A_HAVE = (
    "E201：1808 个主任务、K562/RPE1/HepG2/Jurkat 先封存再解封；"
    "SafeConf Spearman 0.4082，区间 [0.3506, 0.4621]；"
    "预测幅度 0.6189；偏相关 0.2503；20% 效用 0.3200 vs 幅度 0.5943。"
)
B_HAVE = (
    "E204 只有工程验收：权重覆盖正确、对照权重为 1、没有偷看目标扰动表达。"
    "没有 80 轮正式效果表，不能写模型变准。"
)
A_LACK = (
    "单独排序没有超过幅度（效用差 -0.2743）；E205 未跑，四个种子仍是同一 GAT；"
    "没有冻结后的新外部数据；没有湿实验。"
)
B_LACK = (
    "缺 16 个 risk_weighted + 16 个 dispersion_only 的 80 轮正式模型，"
    "以及与 E201 的 16 个 uniform 模型按同一 target、同一 seed 成对比较。"
    "主判断已冻结：高难任务 RMSE 是否下降，全体 RMSE 恶化是否不超过 5%。"
)

# fold(name) -> profile. name matching is done by journal_universe.names_match.
HAND_PROFILES: dict[str, dict] = {
    "bioinformatics": {
        "name": "Bioinformatics",
        "track": "bioinfo_methods",
        "cas_note": "中科院生物大类常见 2 区；JCR Q1；CCF-A。投稿前以学院系统为准。",
        "publishes": (
            "Bioinformatics（牛津）是计算生物学方法的标准件，不发色谱、不发临床病例、也不发纯湿实验方案。"
            "它要的是：新算法或新评价协议、别人能跑的软件、公开数据、和最强简单基线的增量、以及失败边界。"
            "近年单细胞和扰动预测稿不少，但审稿人几乎必问：你比最强基线强在哪、代码能否复现、主张有没有写过头。"
            "TxPert 已经作为预测器发在 Nature Biotechnology。把 SafeConf 再包装成“又一个扰动预测模型”会被直接打回。"
            "这本杂志可以收的是评价协议或训练协议，不是又一个 STRING-GAT。"
        ),
        "audience": "生物信息方法开发者、核心实验室里写工具的人",
        "verdict_a": "q2_not_ready",
        "verdict_b": "q2_not_ready",
        "a_ask": "新评价协议是否说清、是否打过最强简单基线、四背景是否盲测、软件能否复现、失败边界写没写。",
        "b_ask": "加权训练有没有让模型在困难任务上变准、简单任务有没有被训坏、对照是不是同一架构同一 seed。",
        "a_why": (
            "业务A对得上这本杂志的栏目，但证据未闭合：幅度 0.6189 强过 SafeConf 0.4082，"
            "缺新外部。不能把二区写成一定能发。"
        ),
        "b_why": (
            "业务B也是这本杂志会收的故事（用风险改训练），但 E204 没有正式 80 轮效果。"
            "现在投训练稿是空的。"
        ),
        "care_a": CARE_A_METHODS,
        "care_b": CARE_B_METHODS,
    },
    "briefingsinbioinformatics": {
        "name": "Briefings in Bioinformatics",
        "track": "bioinfo_methods",
        "cas_note": "2026 新锐表常见：大类生物 2 区、小类计算生物学 1 区 Top；JCR Q1。",
        "publishes": (
            "Briefings in Bioinformatics 年发文量大，栏目是可复用工作流、教程、工具评测和有案例的方法，"
            "不是数理证明，也不是湿实验。"
            "它比 Genome Biology 更吃“别人能跟着做”的叙事，比 BMC Bioinformatics 更看软件生态和讲解清晰度。"
            "学院口头里的“二区 Top / 小类一区”经常指它。方法稿仍要基线、多背景或至少多案例。"
        ),
        "audience": "要工具和教程的生信用户，不只是算法作者",
        "verdict_a": "q2_not_ready",
        "verdict_b": "q2_not_ready",
        "a_ask": "工作流是否可复用、案例是否讲得清、幅度基线是否保留、有没有面向外人的安装包。",
        "b_ask": "训练加权是不是一个别人能复用的训练协议，而不是一次内部调参。必须有正式效果。",
        "a_why": "栏目匹配业务A，但缺训练收益叙事、新外部和可运行软件。比 Genome Biology 现实，不等于现在能投就中。",
        "b_why": "业务B需要“加权训练可复用”的软件叙事，当前连正式 RMSE 表都没有。",
        "care_a": CARE_A_METHODS,
        "care_b": CARE_B_METHODS,
    },
    "bmcbioinformatics": {
        "name": "BMC Bioinformatics",
        "track": "bioinfo_methods",
        "cas_note": "中科院常见 3 区；CCF-C。",
        "publishes": (
            "BMC Bioinformatics 收可复现计算方法、公开数据和与简单基线的比较，通常不强制湿实验。"
            "口味比牛津 Bioinformatics 更宽，对“协议 + 诚实负结果”容忍度更高，但影响力也更低。"
            "它不是“一定能发”的垃圾桶：主张若写成已经省湿实验预算，一样会被拒。"
        ),
        "audience": "方法可复现即可的计算生物学作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "协议是否说清、封存哈希和 CSV 能否核、幅度是否当主基线写进摘要、四细胞系是否都报。",
        "b_ask": "有没有正式训练对照。没有效果表就不要投业务B。",
        "a_why": "业务A收窄成协议+四背景盲测+负结果，有讨论空间。不是一定能发。",
        "b_why": "业务B在这本杂志也要有效果数字。E204 未完成，不能混进业务A一起投。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "ploscomputationalbiology": {
        "name": "PLOS Computational Biology",
        "track": "bioinfo_methods",
        "cas_note": "CCF-B。分区以学院系统为准。",
        "publishes": (
            "PLOS Computational Biology 要计算方法背后的生物学洞察，不只是排序相关。"
            "审稿人会问：这个风险分有没有改变对细胞或扰动的理解，而不只是一张 Spearman 表。"
            "当前 SafeConf 是审计现象，还没有机制闭环，也没有证明复核决策真的变了。"
        ),
        "audience": "计算和生物学都要讲得通的读者",
        "verdict_a": "q2_not_ready",
        "verdict_b": "q2_not_ready",
        "a_ask": "风险分是否对应可解释的生物学困难（未见背景、效应离散），还是只有相关。",
        "b_ask": "加权训练有没有在生物学困难任务上把预测拉回来。",
        "a_why": "栏目能收业务A，但现在缺生物学洞察和训练收益，未就绪。",
        "b_why": "业务B若做出困难任务变准，会比纯审计更像这本杂志；现在没有正式效果。",
        "care_a": (2, 2, 2, 1, 1, 1, 2, 1),
        "care_b": CARE_B_Q1,
    },
    "naturemethods": {
        "name": "Nature Methods",
        "track": "multidisciplinary",
        "cas_note": "中科院 1 区 Top；JCR Q1。",
        "publishes": (
            "Nature Methods 要成为领域默认用法：新原理或新评价被社区接着用，而不是一次内部审计。"
            "TxPert 本身已在更高影响力的 Nature Biotechnology 轨道上。SafeConf 目前是 TxPert 预测之后的风险层，"
            "单独排序还弱于幅度，也没有证明实验室会改复核名单或改训练流程。"
            "业务A若投这里，会被问：为什么不是幅度就够了。业务B若投这里，会被问：训练到底有没有变准。"
        ),
        "audience": "方法会被很多实验室接着用的人",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "是不是领域默认评价、有没有独立实验室接着用、是否改变实验预算。",
        "b_ask": "加权是不是新训练原理、是否在多个预测器上成立、有没有硬收益。",
        "a_why": "业务A现在不能冲。把“一区可以冲刺”写进组会，会被问幅度和 E204。",
        "b_why": "业务B连正式效果都没有，更不能冲 Nature Methods。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "naturebiotechnology": {
        "name": "Nature Biotechnology",
        "track": "multidisciplinary",
        "cas_note": "中科院 1 区 Top 口径。",
        "publishes": (
            "Nature Biotechnology 是 TxPert 本家轨道：可部署的生物技术或预测方法，能改变实验或产业决策。"
            "SafeConf 不是新预测器。把审计层再投进 TxPert 的家，主张重复而且证据更弱。"
            "业务B若真的让 TxPert 在未见背景上稳定变准，才有资格重新讨论这本；现在没有。"
        ),
        "audience": "要可部署方法的生物技术读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "是不是可部署方法、是否改变实验预算。",
        "b_ask": "训练收益是否大到能改写 TxPert 的使用方式。",
        "a_why": "业务A不是这本杂志要的可部署预测器。",
        "b_why": "业务B没有正式效果，不能碰这本。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "naturecommunications": {
        "name": "Nature Communications",
        "track": "multidisciplinary",
        "cas_note": "中科院 1 区。",
        "publishes": (
            "Nature Communications 要完整故事：方法新、证据硬、边界清，最好有功能验证。"
            "当前主结果是审计现象，不是“方法已经改写训练和实验预算”。两条业务都还缺完整故事。"
        ),
        "audience": "跨领域但要完整证据链的读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "完整故事、独立确认、失败边界。",
        "b_ask": "正式训练收益加独立确认。",
        "a_why": "业务A现在不能冲。",
        "b_why": "业务B现在不能冲。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "genomebiology": {
        "name": "Genome Biology",
        "track": "genomics",
        "cas_note": "中科院常见 1 区。",
        "publishes": (
            "Genome Biology 要基因组尺度的生物学问题或能推动基因组实验的方法，不只是排序相关。"
            "四个细胞系上的风险相关，如果讲不成“哪些扰动/背景在生物学上更难预测”，就只是一张统计表。"
            "缺湿实验、缺跨实验室、缺改变生物学决策的硬收益时，桌面拒稿风险很高。"
        ),
        "audience": "基因组生物学和方法并重的读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "生物学问题、机制或至少能改变基因组实验决策。",
        "b_ask": "训练是否让生物学困难扰动变得可预测。",
        "a_why": "业务A现在不是冲刺对象。",
        "b_why": "业务B没有正式效果，更远。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "genomeresearch": {
        "name": "Genome Research",
        "track": "genomics",
        "cas_note": "中科院常见 1 区口径。",
        "publishes": (
            "Genome Research 要新的基因组生物学发现，或能直接推动基因组实验的资源/方法。"
            "纯预测后审计、又没有新数据资源，不对这本杂志的栏目。"
        ),
        "audience": "基因组发现导向的读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "新生物学发现或新基因组资源。",
        "b_ask": "训练是否带来新生物学可用的预测。",
        "a_why": "业务A不是发现文。",
        "b_why": "业务B没有效果，也不是发现文。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "nucleicacidsresearch": {
        "name": "Nucleic Acids Research",
        "track": "database",
        "cas_note": "中科院常见 1 区或 2 区（看当年大类表）。",
        "publishes": (
            "Nucleic Acids Research 的方法/资源轨道很强：数据库、Web server、被很多人用的工具。"
            "当前没有对外可点击的风险评价服务，也没有新资源库。"
            "业务A若改成“可点击的风险评价服务 + 冻结数据”，才进入讨论；那是另一条工程线。"
            "业务B不是 NAR 的资源栏目。"
        ),
        "audience": "用数据库和 Web 工具的分子生物学/生信用户",
        "verdict_a": "q1_off",
        "verdict_b": "off_track",
        "a_ask": "对外 Web 服务或广泛工具、使用说明、冻结数据。",
        "b_ask": "NAR 不靠一篇训练加权实验吃饭。",
        "a_why": "业务A现在没有对外服务，不对轨到资源栏目。",
        "b_why": "业务B不是 NAR 栏目。",
        "care_a": (2, 2, 2, 1, 1, 2, 2, 1),
        "care_b": (0, 0, 0, 0, 0, 0, 0, 0),
    },
    "nargenomicsandbioinformatics": {
        "name": "NAR Genomics and Bioinformatics",
        "track": "bioinfo_methods",
        "cas_note": "NAR 开源计算刊，分区未当主目标。",
        "publishes": (
            "NAR Genomics and Bioinformatics 是 NAR 的开源计算刊，收基因组计算和生物信息方法，"
            "比 NAR 主刊更不强制 Web 资源，但仍看可复现和数据开放。"
        ),
        "audience": "开源基因组计算方法作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "开放数据、可复现协议、幅度基线。",
        "b_ask": "正式训练效果。",
        "a_why": "业务A可收窄讨论。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "bioinformaticsadvances": {
        "name": "Bioinformatics Advances",
        "track": "bioinfo_methods",
        "cas_note": "较新刊，分区不稳定。",
        "publishes": (
            "Bioinformatics Advances 是牛津 Bioinformatics 的开放获取姊妹刊，口味接近主刊，但影响力和分区都还在变。"
            "更看软件/工作流是否别人能跑。适合收窄后的协议稿，不适合当“二区保证”。"
        ),
        "audience": "要发方法短文和软件的生信作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "安装包、独立复现、幅度基线。",
        "b_ask": "正式训练效果。",
        "a_why": "业务A可收窄，分区未稳。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "gigascience": {
        "name": "GigaScience",
        "track": "bioinfo_methods",
        "cas_note": "大数据/可复现刊，分区以学院系统为准。",
        "publishes": (
            "GigaScience 核心是大规模数据、FAIR 和可复现。哈希、封存、CSV 合同和这本杂志的口味接近。"
            "它要的是数据包能不能被别人完整重跑，不是新生物学故事。"
        ),
        "audience": "重视可复现和数据包的计算作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "完整数据包、封存哈希、复现脚本。",
        "b_ask": "训练实验的数据包和正式效果。",
        "a_why": "业务A可讨论，需把数据包做成别人能下的形态。",
        "b_why": "业务B没有正式训练结果，无包可交。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "genomicsproteomicsbioinformatics": {
        "name": "Genomics, Proteomics & Bioinformatics",
        "track": "bioinfo_methods",
        "cas_note": "GPB。不是 CAS 保证二区。",
        "publishes": (
            "GPB 收基因组、蛋白质组和生物信息交叉方法。可以收计算协议，但不是“二区一定能发”的替代刊。"
        ),
        "audience": "组学计算方法作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "组学计算叙事、可复现。",
        "b_ask": "正式训练效果。",
        "a_why": "业务A可收窄。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "ieeeacmtransactionsoncomputationalbiologyandbioinformatics": {
        "name": "IEEE/ACM Transactions on Computational Biology and Bioinformatics",
        "track": "bioinfo_methods",
        "cas_note": "CCF 交叉常见 B 档口径。",
        "publishes": (
            "IEEE/ACM TCBB 收算法、系统和可复现的计算生物学稿，工程味比牛津 Bioinformatics 更重。"
            "业务A可以收窄成评价系统；业务B要有训练算法增量，而不是验收脚本。"
        ),
        "audience": "计算系统/算法作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "系统或算法是否说清、复现是否完整。",
        "b_ask": "训练算法相对 uniform 的增量。",
        "a_why": "业务A可收窄，不是二区主投保证。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 1, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "journalofcomputationalbiology": {
        "name": "Journal of Computational Biology",
        "track": "bioinfo_methods",
        "cas_note": "算法刊，不是中科院二区主投。",
        "publishes": (
            "Journal of Computational Biology 偏算法和离散方法，不是湿实验，也不是临床。"
            "可讨论协议，但影响力低于 Bioinformatics / Briefings。"
        ),
        "audience": "计算生物学算法作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "算法/协议是否清楚。",
        "b_ask": "训练算法增量。",
        "a_why": "业务A可讨论，不是 CAS 二区主投。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "algorithmsformolecularbiology": {
        "name": "Algorithms for Molecular Biology",
        "track": "bioinfo_methods",
        "cas_note": "算法刊。",
        "publishes": (
            "Algorithms for Molecular Biology 收分子生物信息里的算法，篇幅和影响力都小于牛津 Bioinformatics。"
            "幅度必须当主基线。不能写成二区一定能发。"
        ),
        "audience": "算法作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "算法贡献和基线。",
        "b_ask": "训练算法是否新。",
        "a_why": "业务A可收窄。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "biodatamining": {
        "name": "BioData Mining",
        "track": "bioinfo_methods",
        "cas_note": "数据挖掘方法刊。",
        "publishes": (
            "BioData Mining 收生物数据上的挖掘方法。协议加负结果可讨论，不是二区主投。"
        ),
        "audience": "生物数据挖掘作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "挖掘协议和基线。",
        "b_ask": "训练效果。",
        "a_why": "业务A可讨论，缺新外部。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "currentbioinformatics": {
        "name": "Current Bioinformatics",
        "track": "bioinfo_methods",
        "cas_note": "影响力与二区主刊不同。",
        "publishes": (
            "Current Bioinformatics 收生物信息方法，影响力明显低于 Bioinformatics / Briefings。"
            "可收窄讨论，不能当二区主目标。"
        ),
        "audience": "一般生信方法作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "方法是否说清。",
        "b_ask": "训练效果。",
        "a_why": "业务A可收窄，不是二区主目标。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "scientificreports": {
        "name": "Scientific Reports",
        "track": "multidisciplinary",
        "cas_note": "Nature 开源综合，不是二区主目标。",
        "publishes": (
            "Scientific Reports 收技术报告和完整实验/计算记录，审稿看是否站得住，不看是不是“热点”。"
            "可以发业务A的技术报告，但不是二区主目标，也不能把负结果藏掉。"
        ),
        "audience": "综合 OA 读者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "技术是否完整可核。",
        "b_ask": "训练效果。",
        "a_why": "业务A可发技术报告，不是二区主目标。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "iscience": {
        "name": "iScience",
        "track": "multidisciplinary",
        "cas_note": "Cell 开源综合。",
        "publishes": (
            "iScience 是 Cell 开源综合刊，定位散，可收计算叙事，但不是方法主刊。"
        ),
        "audience": "Cell 开源综合读者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "故事是否完整。",
        "b_ask": "训练效果。",
        "a_why": "业务A可讨论但定位散。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "patterns": {
        "name": "Patterns",
        "track": "ml",
        "cas_note": "Cell 数据科学刊。",
        "publishes": (
            "Patterns 是 Cell 的数据科学刊，收可复用的数据方法，不一定要湿实验。"
            "业务A的计算叙事可以谈；业务B要有训练方法增量。"
        ),
        "audience": "数据科学读者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "数据方法是否可复用。",
        "b_ask": "训练方法是否新。",
        "a_why": "业务A可讨论计算叙事，不是二区保证。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 1, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "cellsystems": {
        "name": "Cell Systems",
        "track": "systems",
        "cas_note": "Cell 系统生物学。",
        "publishes": (
            "Cell Systems 要系统生物学：网络、通路、细胞决策，最好能改变对细胞如何工作的理解。"
            "当前审计稿不对这本杂志。"
        ),
        "audience": "系统生物学读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "细胞决策机制。",
        "b_ask": "训练是否改变系统行为预测。",
        "a_why": "业务A不是系统生物学故事。",
        "b_why": "业务B没有效果，也不是系统故事。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "molecularsystemsbiology": {
        "name": "Molecular Systems Biology",
        "track": "systems",
        "cas_note": "系统生物学顶刊口径。",
        "publishes": (
            "Molecular Systems Biology 要对细胞决策有机制，不是误差排序相关。"
        ),
        "audience": "系统生物学读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "机制。",
        "b_ask": "训练是否改善系统预测。",
        "a_why": "业务A不对轨到机制刊。",
        "b_why": "业务B未就绪且不对机制刊。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "cellreportsmethods": {
        "name": "Cell Reports Methods",
        "track": "multidisciplinary",
        "cas_note": "Cell 子刊方法。",
        "publishes": (
            "Cell Reports Methods 要新实验或计算方法被实验室采用。"
            "当前是计算审计，采用证据不够。"
        ),
        "audience": "方法会被实验室用的人",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "方法是否被采用。",
        "b_ask": "训练协议是否被采用。",
        "a_why": "业务A不够。",
        "b_why": "业务B没有正式效果。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "naturemachineintelligence": {
        "name": "Nature Machine Intelligence",
        "track": "ml",
        "cas_note": "机器学习顶刊口径。",
        "publishes": (
            "Nature Machine Intelligence 要新的学习算法或理论，不是把已有图网络接到事后审计协议上。"
            "业务A是应用审计，不对轨。业务B如果只是加权重，也不是新学习原理。"
        ),
        "audience": "机器学习研究者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "算法新意。",
        "b_ask": "学习原理新意加硬收益。",
        "a_why": "业务A缺方法新意与跨架构。",
        "b_why": "业务B没有效果，也未必有算法新意。",
        "care_a": (2, 2, 2, 2, 2, 2, 1, 0),
        "care_b": (2, 2, 2, 1, 2, 2, 1, 0),
    },
    "naturecomputationalscience": {
        "name": "Nature Computational Science",
        "track": "ml",
        "cas_note": "计算科学综合。",
        "publishes": (
            "Nature Computational Science 要新计算原理，不是一次领域内评价协议。"
        ),
        "audience": "计算科学读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "计算原理。",
        "b_ask": "训练算法原理。",
        "a_why": "业务A增量是评价协议，不是新计算原理。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 2, 2, 2, 2, 1, 0),
        "care_b": (2, 2, 2, 1, 2, 2, 1, 0),
    },
    "recomb": {
        "name": "RECOMB",
        "track": "conference",
        "cas_note": "计算生物学会议。",
        "publishes": (
            "RECOMB 要算法新意。当前业务A是评价协议，增量不足；业务B没有正式训练算法结果。"
        ),
        "audience": "计算生物学会议读者",
        "verdict_a": "q2_not_ready",
        "verdict_b": "q2_not_ready",
        "a_ask": "算法新意。",
        "b_ask": "训练算法新意加效果。",
        "a_why": "业务A当前增量不足。",
        "b_why": "业务B未闭合。",
        "care_a": CARE_A_METHODS,
        "care_b": CARE_B_METHODS,
    },
    "ismbbioinformaticsproceedings": {
        "name": "ISMB / Bioinformatics Proceedings",
        "track": "conference",
        "cas_note": "与 Bioinformatics 会议轨绑定。",
        "publishes": (
            "ISMB 会议轨常与 Bioinformatics 绑定，证据要求和主刊类似。"
        ),
        "audience": "ISMB 社区",
        "verdict_a": "q2_not_ready",
        "verdict_b": "q2_not_ready",
        "a_ask": "与 Bioinformatics 类似。",
        "b_ask": "正式训练效果。",
        "a_why": "业务A证据未闭合。",
        "b_why": "业务B未闭合。",
        "care_a": CARE_A_METHODS,
        "care_b": CARE_B_METHODS,
    },
    "neurips": {
        "name": "NeurIPS",
        "track": "conference",
        "cas_note": "CCF-A 会议。",
        "publishes": (
            "NeurIPS 要机器学习新方法。业务A是应用审计，不对轨。业务B若只是任务加权，通常也不够当主会方法。"
        ),
        "audience": "机器学习会议",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "ML 新方法。",
        "b_ask": "新学习算法加充分实验。",
        "a_why": "业务A是应用审计。",
        "b_why": "业务B没有效果，也未必有算法新意。",
        "care_a": (2, 2, 2, 2, 2, 2, 1, 0),
        "care_b": (2, 2, 2, 1, 2, 2, 1, 0),
    },
    "icml": {
        "name": "ICML",
        "track": "conference",
        "cas_note": "CCF-A 会议。",
        "publishes": "ICML 同样要机器学习新方法。业务A不对轨；业务B未就绪。",
        "audience": "机器学习会议",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "ML 新方法。",
        "b_ask": "新学习算法。",
        "a_why": "业务A不对轨。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 2, 2, 2, 2, 1, 0),
        "care_b": (2, 2, 2, 1, 2, 2, 1, 0),
    },
    "iclr": {
        "name": "ICLR",
        "track": "conference",
        "cas_note": "机器学习会议。",
        "publishes": "ICLR 要表征学习或新训练方法。当前两条业务都不够。",
        "audience": "机器学习会议",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "新方法。",
        "b_ask": "新训练方法。",
        "a_why": "业务A不对轨。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 2, 2, 2, 2, 1, 0),
        "care_b": (2, 2, 2, 1, 2, 2, 1, 0),
    },
    "frontiersinbioinformatics": {
        "name": "Frontiers in Bioinformatics",
        "track": "bioinfo_methods",
        "cas_note": "OA 前沿；注意年发文与口碑。",
        "publishes": (
            "Frontiers in Bioinformatics 是 OA 方法刊，可讨论收窄后的协议，但年发文和口碑需要自己评估。"
        ),
        "audience": "OA 生信作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "方法可复现。",
        "b_ask": "训练效果。",
        "a_why": "业务A可讨论。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "bmcgenomics": {
        "name": "BMC Genomics",
        "track": "genomics",
        "cas_note": "基因组 OA。",
        "publishes": (
            "BMC Genomics 主题偏基因组发现和组学数据。收窄成方法+数据可讨论，不是二区主投。"
        ),
        "audience": "基因组 OA 读者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "组学数据和方法。",
        "b_ask": "训练效果。",
        "a_why": "业务A可收窄，不是二区主投。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "journalofbiomedicalinformatics": {
        "name": "Journal of Biomedical Informatics",
        "track": "medical",
        "cas_note": "医学信息，不是扰动预测主刊。",
        "publishes": (
            "Journal of Biomedical Informatics 发电子病历、临床决策和生物医学信息学，不是 K562 扰动预测。"
            "两条业务都不对轨。"
        ),
        "audience": "医学信息学",
        "verdict_a": "off_track",
        "verdict_b": "off_track",
        "a_ask": "临床信息学问题。",
        "b_ask": "临床模型训练。",
        "a_why": "业务A不是临床信息学。",
        "b_why": "业务B不是临床模型。",
        "care_a": (0, 0, 0, 0, 0, 0, 0, 0),
        "care_b": (0, 0, 0, 0, 0, 0, 0, 0),
    },
    "chromatographia": {
        "name": "Chromatographia",
        "track": "chromatography",
        "cas_note": "色谱刊，分区与本稿无关。",
        "publishes": (
            "Chromatographia 发色谱分离和仪器应用短文，读者是分析化学家。"
            "SafeConf 没有保留时间、没有检测限、没有基质效应。"
            "业务A不是色谱方法，业务B也不是给色谱仪训练加权。完全不对轨。"
        ),
        "audience": "色谱分析化学家",
        "verdict_a": "off_track",
        "verdict_b": "off_track",
        "a_ask": "分离方法验证。",
        "b_ask": "不收深度学习训练加权。",
        "a_why": "业务A完全不对轨。",
        "b_why": "业务B完全不对轨。",
        "care_a": (0, 0, 0, 0, 0, 0, 0, 2),
        "care_b": (0, 0, 0, 0, 0, 0, 0, 2),
    },
    "journalofchromatographya": {
        "name": "Journal of Chromatography A",
        "track": "chromatography",
        "cas_note": "色谱 A 辑。",
        "publishes": (
            "Journal of Chromatography A 发分离科学本身：新固定相、保留机理、柱效。"
            "比 Chromatographia 更基础，仍然是分析化学，不是单细胞计算。"
        ),
        "audience": "分离科学家",
        "verdict_a": "off_track",
        "verdict_b": "off_track",
        "a_ask": "分离机理。",
        "b_ask": "不收训练加权。",
        "a_why": "业务A不对轨。",
        "b_why": "业务B不对轨。",
        "care_a": (0, 0, 0, 0, 0, 0, 0, 2),
        "care_b": (0, 0, 0, 0, 0, 0, 0, 2),
    },
    "journalofchromatographyb": {
        "name": "Journal of Chromatography B",
        "track": "chromatography",
        "cas_note": "色谱 B 辑。",
        "publishes": (
            "Journal of Chromatography B 发生物样品上的色谱定量：血药浓度、代谢物、生物基质。"
            "比 A 更偏生物医学检测，仍然不是扰动预测风险。"
        ),
        "audience": "生物分析化学家",
        "verdict_a": "off_track",
        "verdict_b": "off_track",
        "a_ask": "生物基质定量。",
        "b_ask": "不收训练加权。",
        "a_why": "业务A不对轨。",
        "b_why": "业务B不对轨。",
        "care_a": (0, 0, 0, 0, 0, 0, 0, 2),
        "care_b": (0, 0, 0, 0, 0, 0, 0, 2),
    },
    "nature": {
        "name": "Nature",
        "track": "multidisciplinary",
        "cas_note": "综合顶刊。",
        "publishes": "Nature 要改变一个领域看法的完整发现或方法。当前两条业务都不是。",
        "audience": "综合科学读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "领域级发现。",
        "b_ask": "领域级方法。",
        "a_why": "业务A不对轨。",
        "b_why": "业务B不对轨。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "science": {
        "name": "Science",
        "track": "multidisciplinary",
        "cas_note": "综合顶刊。",
        "publishes": "Science 同样是综合顶刊。当前两条业务都不是。",
        "audience": "综合科学读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "领域级发现。",
        "b_ask": "领域级方法。",
        "a_why": "业务A不对轨。",
        "b_why": "业务B不对轨。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "cell": {
        "name": "Cell",
        "track": "multidisciplinary",
        "cas_note": "综合顶刊。",
        "publishes": "Cell 要完整细胞生物学故事。当前两条业务都不是。",
        "audience": "细胞生物学读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "细胞生物学发现。",
        "b_ask": "能改写细胞实验的方法。",
        "a_why": "业务A不对轨。",
        "b_why": "业务B不对轨。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "peerj": {
        "name": "PeerJ",
        "track": "bioinfo_methods",
        "cas_note": "OA 综合，分区低。",
        "publishes": "PeerJ 是 OA 综合，可收窄发计算报告，分区低，不是二区主目标。",
        "audience": "OA 综合读者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "技术是否完整。",
        "b_ask": "训练效果。",
        "a_why": "业务A可收窄，分区低。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
    "communicationsbiology": {
        "name": "Communications Biology",
        "track": "multidisciplinary",
        "cas_note": "Nature 开源生物学。",
        "publishes": (
            "Communications Biology 仍要生物学问题和完整故事，不是纯审计表。"
        ),
        "audience": "生物学 OA 读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "生物学问题。",
        "b_ask": "训练是否改善生物学预测。",
        "a_why": "业务A故事不够。",
        "b_why": "业务B未就绪。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "elife": {
        "name": "eLife",
        "track": "multidisciplinary",
        "cas_note": "要完整生物学问题。",
        "publishes": "eLife 要完整生物学问题。当前审计稿不够。",
        "audience": "生物学读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "完整生物学问题。",
        "b_ask": "训练收益加生物学。",
        "a_why": "业务A不够。",
        "b_why": "业务B不够。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "pnas": {
        "name": "PNAS",
        "track": "multidisciplinary",
        "cas_note": "综合。",
        "publishes": (
            "PNAS（Proceedings of the National Academy of Sciences of the United States of America）"
            "要广泛兴趣和机制。当前两条业务都缺。"
        ),
        "audience": "综合科学读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "广泛兴趣。",
        "b_ask": "训练硬收益。",
        "a_why": "业务A缺广泛兴趣和机制。",
        "b_why": "业务B未就绪。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "scienceadvances": {
        "name": "Science Advances",
        "track": "multidisciplinary",
        "cas_note": "Science 开源。",
        "publishes": "Science Advances 故事不够完整时不要投。当前两条业务都不够。",
        "audience": "Science 开源读者",
        "verdict_a": "q1_off",
        "verdict_b": "q1_off",
        "a_ask": "完整故事。",
        "b_ask": "训练硬收益。",
        "a_why": "业务A故事不够完整。",
        "b_why": "业务B未就绪。",
        "care_a": CARE_A_Q1,
        "care_b": CARE_B_Q1,
    },
    "computationalbiologyandchemistry": {
        "name": "Computational Biology and Chemistry",
        "track": "bioinfo_methods",
        "cas_note": "计算生物学与化学交叉。",
        "publishes": (
            "Computational Biology and Chemistry 收计算生物学和化学信息交叉，影响力低于主刊。"
            "可收窄讨论协议，不是二区主投。"
        ),
        "audience": "计算交叉作者",
        "verdict_a": "discuss_narrow",
        "verdict_b": "q2_not_ready",
        "a_ask": "计算方法。",
        "b_ask": "训练效果。",
        "a_why": "业务A可收窄。",
        "b_why": "业务B未就绪。",
        "care_a": (2, 2, 1, 0, 0, 0, 1, 0),
        "care_b": CARE_B_METHODS,
    },
}


def fold_key(name: str) -> str:
    return "".join(ch for ch in name.lower().replace("&", "and") if ch.isalnum())


_PROFILE_ALIASES = {
    "proceedingsofthenationalacademyofsciencesoftheunitedstatesofamerica": "pnas",
    "ieeeacmtransactionsoncomputationalbiologyandbioinformatics": "ieeeacmtransactionsoncomputationalbiologyandbioinformatics",
    "ieee-acmtransactionsoncomputationalbiologyandbioinformatics": "ieeeacmtransactionsoncomputationalbiologyandbioinformatics",
    "database-thejournalofbiologicaldatabasesandcuration": "database",
    "nargenomicsandbioinformatics": "nargenomicsandbioinformatics",
}


def profile_for(name: str) -> dict | None:
    key = fold_key(name)
    key = _PROFILE_ALIASES.get(key, key)
    if key in HAND_PROFILES:
        return HAND_PROFILES[key]
    for prof in HAND_PROFILES.values():
        if fold_key(prof["name"]) == key:
            return prof
    return None
