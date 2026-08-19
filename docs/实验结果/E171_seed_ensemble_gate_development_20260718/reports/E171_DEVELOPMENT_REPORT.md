# E171｜seed-ensemble gate 开发审计

E170 test truth 仍未读取，F3 目录数为 0。本审计只使用 pretruth 预测与已允许的 validation donor seen-target effects。

原 G4 比较三个单-seed scGPT–GEARS 配对，24 个 panel×state×stratum 单元通过 12/24。改为三组 leave-one-seed-out family means 后通过 24/24；最终部署分数仍使用三个 seeds 的完整 family mean，SafeConf 公式、目标和阈值没有变化。这个修正让 gate 检查的估计器更接近实际部署估计器，可以进入新的未读目标协议，但不能回头解封 E170。

validation donor 的 SafeConf 相对 magnitude 平均 Δ(AURC_magnitude−AURC_SafeConf)=0.000174154，仅 4/12 单元为正。固定开发网格中最好的候选是 `safeconf_only`，平均 Δ=0.000174154；这些结果没有形成稳定的 performance rescue。因此下一次 fresh-target 实验可修正 seed gate，但不得宣称已经解决 magnitude 基线，也不得以 validation 结果筛掉负面板。
