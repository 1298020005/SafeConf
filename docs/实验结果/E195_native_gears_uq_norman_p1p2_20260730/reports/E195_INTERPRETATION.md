# E195 结果如何使用

E195 支持“GEARS 原生 logvar 可作为候选风险信号”：两个面板的 family 相关都为正。它不支持“已校准方差”或“稳定优于简单 magnitude”。Magnitude的相关点估计更高，但 paired 差值区间跨 0；固定 20% 预算的相对表现又随面板改变。

Seed disagreement 的相关较弱，而且它参与 family RMS 恒等式，不能解释成独立保证。PRESCRIBE 的 combined confidence 与 magnitude 几乎同序，表现还依赖 effect-Pearson 或 RMSE 终点。

GEARS-UQ、GEARS–scGPT pair 和 PRESCRIBE 不共享预测值与误差终点，跨系统并列表只能描述各自内部排序，不能宣称共同结果变量上的全面胜负。本实验是看过真值后的 condition-holdout 复现，不改变 E192 的 ABSTAIN。
