# E94｜CPA pinned 环境复现

E83 的 manifest、种子、细胞抽样、网络、剂量变换和 10 epochs 全部保持不变，只把运行环境从兼容环境切换到 CPA 0.8.8 官方依赖组合。该测试集已在 E83 查看，因此 E94 只检查环境再现性，不增加新的独立证据。

- tasks：59
- disagreement 排序 old vs pinned：ρ=1.000
- pair-mean error 排序 old vs pinned：ρ=1.000
- pinned 内 disagreement–error：ρ=0.901
- original 内 disagreement–error：ρ=0.901
- disagreement 中位绝对变化：0.000033
- pair-mean error 中位绝对变化：0.000024
