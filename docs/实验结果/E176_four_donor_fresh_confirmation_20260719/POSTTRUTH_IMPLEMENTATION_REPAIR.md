# E176 开真值后的实现修复记录

## 发生了什么

四个 F4 评价资产已按提交 2a6f2aba4039b410bc11b7c6f7fa36a72bc76685 的校准门控完成物理解封。第一次最终统计运行在生成任何结果文件之前停止，异常为 AttributeError：辅助模块没有 tie_aware_curve。

错误位于次要 AURC 诊断的辅助模块入口。代码导入了 E170 的联合执行模块；tie_aware_curve 实际定义在其调用的 E168 postgate 数学辅助模块中。

## 修复边界

- 只把 POSTGATE_HELPER 从 run_e170_primary_cd4_joint_postgate.py 改为 run_e168_primary_cd4_postgate.py。
- 不改变 target、donor、模型、seed、基础误差模型、conformal quantile、覆盖率目标、主要总体、证书公式或 bootstrap 方案。
- 第一次失败没有写出 final_evaluation 目录、统计表、图或摘要；操作者没有看到覆盖率、误差、相关性或 AURC 数值。
- 原始真值访问门控提交保持为 2a6f2aba4039b410bc11b7c6f7fa36a72bc76685；修复代码另行提交并双远程冻结。最终状态同时记录两个提交。

该修复属于结果未知时的确定性接口纠正，不能用于重新选择方法或改善结果。
