# Phase1 主表初版结果（2026-06-03）

这份表只记录 5 个可直接跑的数据集的初版结果。

重要：这不是最终论文结论，因为还没有完成 partial rho（控制效应幅度后的相关）、Bootstrap CI（自助法置信区间）和 McFarland drug-only adapter。

| 数据集 | 线 | test records | simple combined rho | learned risk rho | model disagreement rho | 说明 |
|---|---|---:|---:|---:|---:|---|
| SrivatsanTrapnell2020_sciplex3 | chem_robust | 1128 | 0.432 | 0.652 | 0.480 | 初版已跑，待幅度审计 |
| SantinhaPlatt2023 | chem_robust | 566 | 0.193 | 0.698 | 0.177 | 初版已跑，待幅度审计 |
| LaraAstiasoHuntly2023_invivo | gene_main | 780 | 0.466 | 0.776 | 0.466 | 初版已跑，待幅度审计 |
| LaraAstiasoHuntly2023_exvivo | gene_main | 662 | 0.689 | 0.764 | 0.276 | 初版已跑，待幅度审计 |
| Frangieh | gene_main | 1266 | 0.598 | 0.865 | 0.857 | 初版已跑，待幅度审计 |

一句话：5 个直接数据集都跑通了，simple combined 在 4/5 个数据集上 > 0.20，Santinha 约 0.19 接近门槛；但 learned risk 很高，必须警惕过拟合和效应幅度混杂。

下一步：补 McFarland drug-only adapter，然后对 7 个主表统一跑 partial rho / magnitude-only baseline / Bootstrap CI。
