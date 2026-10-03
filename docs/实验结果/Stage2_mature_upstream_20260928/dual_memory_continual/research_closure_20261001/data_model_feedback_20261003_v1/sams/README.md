# SAMS-VAE：已启动的独立上游与跨家族验证

## 当前状态入口

动态状态以 `TRAINING_STATUS.json` 为准；`TRAINING_START.json` 保存实际进程、参数量和每轮步数。原始日志、checkpoint及大数组在：

`/home/yyf/runtime_artifacts/safeconf_research_20261003/sams_v1`

当前训练服务：`safeconf-sams-v1-stream-20261003.service`，GPU0。SAMS-VAE是本轮唯一新增上游，使用PerturBench `c84038bc1ea409aa54f3832cfa6f34f5059adf0c` 的官方McFaline配方。输入15009基因，259890081个参数，训练738901细胞、验证70300细胞；69028个原上游TEST细胞不进入训练读取路径。

## 已完成的输入与恢复验收

- 生成路径固定随机数，表达改为另一组值后预测最大差为0。
- 重建路径负对照有非零差异，证明该检查能识别错误推理路径。
- 预测入口只接受任务元数据与模型，不接收真实扰动表达。
- HDF5 X读取资格守卫在读取前拒绝TEST行；不读取counts层。
- 官方小模型实际训练2步，保存优化器和checkpoint，重载后的生成预测一致。
- 逐项结果见 `PREDICTION_INPUT_TESTS.json`、`READER_ISOLATION_TESTS.json`。

## 三项工程修复

1. `read_h5ad(backed='r')`会急读counts层。准备进程在模型初始化前停止；该层未用于拟合、选择或评分。改用X-only资格守卫，保留访问事件。
2. 官方全部训练行CSR大切片触发systemd-oomd，仍为0训练步。改成2048行流式生成float32 CSR memmap，逐行顺序和最终模型输入值保持一致。
3. Lightning CSV日志无法遍历官方transform字典子类。保留CSV指标，复杂hparams由冻结JSON与官方checkpoint保存；通过训练/保存/重载检查。

以上没有改变模型结构、ELBO、优化器或训练划分，属于同一个上游尝试。原始记录分别保留在 `ENGINEERING_REPAIR_001/002/003.json`，未覆盖历史实验资产。

## 能力与后续比较

- checkpoint仍按官方验证重建ELBO选择。
- 独立生成每任务1000个样本，先落盘预测及哈希，再计算任务误差。
- 本轮主能力合同为2840共同基因；原512轴仅作同checkpoint、同样本的历史可比附表，不按成绩切换轴。
- 主门为相对强简单基线误差差距≤2%、至少60%分层满足该界限，沿用原5000基因簇bootstrap判据。
- 同时保存生成样本方差，作为无额外训练的分歧对照；不解释为校准UQ、不加入主13项特征。
- 主轴过门后，脚本自动继续DecoderOnly→SAMS与SAMS→DecoderOnly，来源风险训练331任务/228基因，评价212任务/152基因。
- 对照包括幅度、公共直接距离、加权历史距离、固定Ridge/HGB及五个源标签置乱；5000次配对基因簇bootstrap。
- 已见McFaline结果按回顾性SEEN证据报告，不重新命名为独立确认。

## 恢复与运行

解释器：`/home/yyf/.venvs/perturbench-c84038bc/bin/python`。

脚本：`tools/scripts/run_safeconf_sams_crossfamily_v1.py`。

入口：`preflight`、`train`、`evaluate`、`followup`。训练结束会自动进入`followup`；如需独立恢复后处理，可指定`followup --checkpoint <已登记best checkpoint>`。

训练恢复使用`train --resume <recovery/last/final_resumable checkpoint> --gpu-hours <该段剩余额度>`，先核总账再设置该段时间；所有段累计服从本候选48GPUh及全局96GPUh预算。原始训练24小时初始拨款，400轮上限可能高于本轮预算，预算停止与充分训练后能力不足分别记录。

使用`systemctl --user show safeconf-sams-v1-stream-20261003.service`查看服务；输入日志与状态JSON给出真实进度，GPU空闲或验证ELBO下降均不代替能力结论。

若能力暂未合格，保存可恢复训练状态，由负责人检查收敛和已授权修复；不会把尚未充分训练直接解释成源错误经验无价值。
