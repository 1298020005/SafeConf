# E174 prior-data method development

E174 的 800 个目标 expression X 在本阶段读取数为 **0**。基础误差估计器只使用 E168+E172 已解封的 1,000 个目标、3,000 个任务。

20 次按 panel 与 seen/unseen 分层的 target-level 60/20/20 重采样中，复合候选必须同时达到平均上界至少 0.5% 的相对缩短，并在至少 75% 的重复中胜过 magnitude，才能取代 magnitude。冻结选择为：ensemble RMSE 使用 `magnitude`；pair-mean RMSE 使用 `magnitude`。

未通过的复合候选不会在 E174 校准或评价真值打开后复活。constant、magnitude 与复合模型都保留用于透明效率对照；最终主输出只使用此处冻结的选择。
