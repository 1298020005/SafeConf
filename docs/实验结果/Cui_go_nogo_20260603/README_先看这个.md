# Cui go/no-go 结果先看这个

这份目录是 CuiHacohen2023（崔，免疫细胞基因扰动大数据集）的第一关结果。

## 一句话结论

**GO（继续）。** 冻结的 protocol v0.2（固定可信度打分规则）在 Cui 上达到 aligned rho（方向对齐相关）= **0.4454**，超过 0.30 门槛。

## 这说明什么

- test（测试）记录数：**2506**，不是旧小数据集那种几十条。
- partial rho（控制 effect magnitude，效应大小后的相关）= **0.3285**，说明分数不是完全靠“扰动效应大所以误差大”的假象。
- magnitude-only rho（只看效应大小的基线）= **0.7358**，说明效应大小混杂非常强，后续论文必须把这个作为重点审计项。
- per-fold（每折）rho 全部为正：0.409 到 0.502，没有只靠某一折撑起来。

## 怎么跟 Claude / Qoder 说

请他们重点审查三件事：

1. Cui 上 v0.2 原始结果是否可以作为进入下一批大数据集的 GO 证据。
2. partial rho=0.3285 是否足够说明 confidence score（可信度分数）有独立信号。
3. 下一批 blind（盲测）数据集应该优先跑 McFarland drug-only、Srivatsan sciplex3、Santinha、Lara invivo/exvivo 的哪几个。

## 文件

- `CUI_GO_NOGO_REPORT.md`：中文+表格报告。
- `tables/CUI_GO_NOGO_SUMMARY.csv`：各 score（分数）的总体结果。
- `tables/CUI_PER_FOLD_RHO.csv`：每折结果，避免只看 pooled（混合总数）。
- `tables/CUI_PER_PREDICTOR_RHO.csv`：V0 和 ContextSim 分开看。
- `tables/CUI_SINGLE_FEATURE_DIAG.csv`：单特征诊断。

## 注意不要夸大

这不是“一区稳了”。它只说明：Cui 这个大数据集第一关过了，值得继续跑更多大数据集和更严格的混杂校正。
