# E189 正式笛卡尔缺失实验

状态：**PASS**。共评估 13,440 个任务实例；family-RMS 下界违例 0，family-worst 下界违例 0。

同一 scGPT–GEARS 六成员模型族同时覆盖随机缺一格、未见细胞背景整行、未见扰动整列和二者同时未见。训练支持量固定为每个已见扰动 1、2、3、5 个背景。

## 汇总

| support | e189_setting | n_tasks | family_rms_error_mean | family_worst_error_mean | fraction_members_beating_zero_mean |
| --- | --- | --- | --- | --- | --- |
| 1 | double_unseen | 480 | 0.118536 | 0.128184 | 0.346181 |
| 1 | random_missing_pair | 480 | 0.119106 | 0.128596 | 0.367014 |
| 1 | unseen_context_row | 1440 | 0.116507 | 0.126509 | 0.364120 |
| 1 | unseen_perturbation_column | 960 | 0.121206 | 0.130249 | 0.365972 |
| 2 | double_unseen | 480 | 0.116119 | 0.123153 | 0.433333 |
| 2 | random_missing_pair | 480 | 0.116351 | 0.122963 | 0.478819 |
| 2 | unseen_context_row | 1440 | 0.113979 | 0.120815 | 0.458333 |
| 2 | unseen_perturbation_column | 960 | 0.118618 | 0.125275 | 0.472049 |
| 3 | double_unseen | 480 | 0.115483 | 0.121900 | 0.487847 |
| 3 | random_missing_pair | 480 | 0.116079 | 0.122583 | 0.514236 |
| 3 | unseen_context_row | 1440 | 0.113526 | 0.120433 | 0.501968 |
| 3 | unseen_perturbation_column | 960 | 0.118219 | 0.124227 | 0.515972 |
| 5 | double_unseen | 480 | 0.115207 | 0.121177 | 0.478125 |
| 5 | random_missing_pair | 480 | 0.115966 | 0.122478 | 0.517014 |
| 5 | unseen_context_row | 1440 | 0.113699 | 0.121230 | 0.488079 |
| 5 | unseen_perturbation_column | 960 | 0.118279 | 0.123918 | 0.496007 |

PASS 只表示数据隔离、任务合同和确定性下界成立。预测性能、相关性以及是否优于 magnitude 均按原值报告，不作为事后改写的通过条件。
