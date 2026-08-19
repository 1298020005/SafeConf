# 期刊分区与投稿事实（GLM 联网核实，2026-08-17）

标注：**[V]** 官方页直接核实或多源一致；**[U]** 单源/冲突，仅供参考。
中科院分区口径 = **2025 年升级版（最后一版）**；IF = 2026-06-17 发布的 JCR 2025。

## 1. 中科院分区表停更——属实 [V]

- fenqubiao.com 官网声明原文："自 2026 年起，中国科学院文献情报中心将不再更新与
  发布期刊分区表。"最后一版 = 2025 年升级版（2025-03 发布）。
- 高校主流做法：沿用 2025 年版认定成果；第三方《新锐期刊分区表》（2026-03-24，
  xr-scholar.com）多所 985/211 **暂不认可**。
- **行动项（作者本人做，AI 不能代办）：向学院书面确认认哪一年、认大类还是小类。**
  这决定 Briefings 在"一区"口径下算不算数。

## 2. 期刊事实表

| 期刊 | IF(2025) | JCR | 中科院 2025 升级版 | 对本文契合度 |
|---|---|---|---|---|
| Nature Methods | 28.3 [V] | Q1 | 大类生物学 **1 区 Top** [V] | 口径最对口但难度极高；2025 已发线性基线 Brief Comm.（08-04）与 27×29 基准 |
| Genome Biology | 9.2 [V] | Q1 | 大类生物学 **1 区** [V]（Top 标记两源冲突 [U]） | 冲大类一区的主目标；要软件包+宽验证+生物用途 |
| Cell Systems | 7.5 [V] | Q1 | 大类生物学 **1 区** [V] | 偏生物学洞见，纯方法文非典型 |
| **Briefings in Bioinformatics** | 7.3 [V]（LetPub 页头 7.7 冲突 [U]） | Q1 [V] | 大类生物学 **2 区**；小类数学与计算生物学 **1 区**（个别抓取源渲染异常 [U]） | **主投推荐**：生信 ML 方法主场 |
| Bioinformatics | 5.5 [V] | Q1 [V] | 大类生物学 **3 区**（小类同步下调） [V] | 第一备选；Applications 轨道强制投稿时代码公开 [V-官方指南] |
| Communications Biology | 5.8 [V] | Q1 | 大类生物学 **1 区 Top** [V] | 覆盖广但方法文非典型，桌拒率高（~90%+ [U]） |
| Genome Medicine | 10.8 [V] | Q1 | 大类生物学 **1 区 Top** [V] | 偏临床，不对口 |
| NAR Genomics & Bioinformatics | 3.0 [V] | Q2 | 大类生物学 **4 区** [V] | 保底 |
| Bioinformatics Advances | 2.6 [V] | Q2 | 大类生物学 **4 区** [V] | 保底 |
| Patterns | 7.4 [U-冲突] | **ESCI，无 JCR Q** [V] | 计算机科学 2 区 [U-冲突] | 对口但 ESCI，国内认定需谨慎 |

## 3. 软件/代码政策（决定"最小可安装包"的必要性）

- **Bioinformatics**：Applications 类投稿时**必须**"provide the source code freely
  available at stable URLs… without request" [V-官方 author guidelines]。
- Nature Portfolio（NM/CommsBio）：代码须向编辑/审稿人提供，Code Availability 章节
  必填 [V]。
- Cell Press（Cell Systems/Patterns）：STAR Methods 强制 Data and code availability；
  Patterns 要求全部代码进公共仓库（GitHub/Zenodo DOI）[V]。
- 结论：**无论投哪一家，pip 可安装 + 一条复现命令都是硬需求**，现在就做。

## 4. 审稿周期（社区数据，[U] 近似）

Bioinformatics ≈1.8 个月首决；BIB ≈2–6 个月；Patterns ≈19 周；Genome Biology /
Nature Methods 全程典型 3–6 个月。对毕业时间敏感的话，先把 Briefings/Bioinformatics
轨道跑通。

来源清单（节选）：
fenqubiao.com（停更声明）；nature.com/nmeth/journal-impact（IF）；
academic.oup.com/bioinformatics/pages/author-guidelines（代码政策）；
nature.com/articles/s41592-025-02772-6（线性基线）；
nature.com/articles/s41592-025-02980-0（scPerturBench）；
iikx.com / JustScience / LetPub（分区，多源交叉）；完整链接见 GLM 会话记录。
