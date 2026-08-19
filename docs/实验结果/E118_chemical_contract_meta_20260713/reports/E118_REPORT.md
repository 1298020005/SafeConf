# E118｜化学扰动统一合同元审计

E118 不学习新权重，只把 E84、E87、E89 的 formal CPA 双预测器结果按同一指标汇总。三个来源分别代表 sciPlex3 内部四象限、sciPlex3→OpenProblems 跨数据集和 sciPlex3→sciPlex4 同族外部验证。

## 合同审计

| experiment | strict issues | truth used for score | pass |
|---|---:|---|---|
| E84_cpa_rdkit_cartesian_formal | 0 | False | True |
| E87_sciplex_to_openproblems_cpa | 0 | False | True |
| E89_sciplex3_to_sciplex4_cpa | 0 | False | True |

## 三来源等权宏平均

| score | Spearman | top-20% error enrichment | top-20% total error capture |
|---|---:|---:|---:|
| model_disagreement | 0.811 | 1.323 | 0.286 |
| predicted_magnitude | 0.842 | 1.327 | 0.287 |

## 分歧相对幅度

| metric | Δ | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| spearman | -0.0313 | [-0.0511, -0.0060] | 0.002 |
| top20_error_enrichment | -0.0039 | [-0.0085, 0.0069] | 0.309 |
| top20_total_error_capture | -0.0009 | [-0.0019, 0.0015] | 0.298 |

## 预设判定

- chemical 独立增量通过：**否**。
- 合同闭环与方法增量是两件事：strict contract 全部通过，只能证明预测—冻结—解封—评价流程可信；若分歧没有稳定超过 magnitude，就必须作为跨模态负边界。
