# SafeConf v2.1：实际结果与接续

更新UTC：2026-10-09 02:43:20。正文、PDF暂停。

## 已完成

- 两种预测器的标签效率曲线、严格20次匹配内容置乱、Source108次固定消融、Adamson两个24任务面板均已实际执行。
- 公平PertEMA配方同时保留P6、Native61、Native61+Public，以及CDF/原始RMSE两个适配。未完成完整conformal流程，不把适配称为完整官方区间复现。
- 登记标签等效成本以背景宏平均U20为主；Target-only没有预算通过非劣门，保持右删失。单一全局排序另作事先要求的实际收益及敏感性分析，两者分列。
- 严格内容置乱主端点：Public=0.8612，null中位数=0.8426，名义p=0.2857。历史支持量和历史效应能量为必报强对照。
- 已完成支持字段修复诊断、原始误差单位诊断。单位修正DEV增量0.0073，但CI较宽且分层未过门，未采用，不继续扫描。

## 真实20%复核收益

- DecoderOnly：同样复核43/212项，Public发现22个真实高误差任务，幅度发现4个；剩余平均误差相对幅度降低7.07%。实际入选历史支持量发现14个。
  全部登记反馈下，Native+Public平均发现31.00个，global U20=0.8714。该收益不能写成每个背景内部排序都改善。
- SAMS_VAE：同样复核43/212项，Public发现24个真实高误差任务，幅度发现5个；剩余平均误差相对幅度降低6.84%。实际入选历史支持量发现14个。
  全部登记反馈下，Native+Public平均发现30.67个，global U20=0.8603。该收益不能写成每个背景内部排序都改善。

## 已落实的全量排序诊断

- 充分反馈Native+Public相对Public的全局U20增量：DecoderOnly +0.1982，95%CI[0.0807,0.3023]；SAMS +0.1406，CI[0.0406,0.2738]。
- 25%反馈时两个方向仍为正增量：+0.1728/[0.0369,0.2639]及+0.1068/[0.0016,0.2131]。这属于全局实际排序收益，登记宏平均主端点不替换。
- 只学习各背景平均误差的同预算规则，global U20约0.51/0.50，没有达到Public水平；更完整反馈模型的收益并非仅由背景平均值决定。
- 本次结果支持继续验证“无当前错误时启动审核，少量反馈改善跨背景风险尺度”；当前服务配置不由SEEN表里选最高值自动替换。

## 当前采用与正在运行

- 完整系统明确采用PublicRule；Source和Target候选与采用系统分列。MC整批212任务全部有历史；外部队列另验证自然混合覆盖和固定CDF回退。
- 当前外部作业状态：RUNNING；pipeline PID=3300037，监督PID=3300038。实际接续目录=/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21/external_kolf_panel1400_v1；最终确认是否开启=False。
- 原累计下载/GPU预算继续扣减；E208两个受保护进程保留。外部worker通过能力门后自动冻结分数、读取确认真值、统计；科学门失败保持确认封存并登记备用资产。
- 备用CM4AI作者文件清单与两个pilot元数据已核准；pilot仅98/108个目标名称，不能把guide数当作≥150个确认基因。大文件公开下载端TLS故障记录在资产回执中。
- 独立确认未完成时，本轮研究保持未完成；自动监督保存真实PID、日志、失败回执和恢复点。

## 外部预测器诊断与实际处理

- 原始预测器的能力检查：ridge：FAIL，相对均值RMSE差距5.11%；mlp：FAIL，相对均值RMSE差距211.18%。
- MLP批大小造成的float32重载误差已修复：原权重不变，统一float64推理后一次保存float32输出，原容差未放宽；180个跨背景/角色查询的重载差为0，加入伪造答案列输出严格不变。
- 另立一个训练修复版本，原始负结果、预测和权重保留；相同50维控制特征在上游训练区标准化，MLP学习中心化响应、从零残差输出开始，仅用上游训练区内部留出决定步数。风险层不据确认结果调参。

## 已接续的备用资产

- 作者网站已核到10,167个扰动、38,606个输出基因；其展示矩阵是int8量化NTC z-score，未读取响应行，不用它替代高精度确认真值。
- 作者原始basic-QC HDF5已接通。完整2839共有轴需约27.8GB存储块，超过剩余额度；在任何KOLF响应读取前，用Source基因固定哈希选择1400输出面板，保留旧结果和完整轴审计。新研究的所有方法、误差和能力门使用同一登记面板，不宣称全转录组确认。
- 新面板预算估计14.25GB，加10%余量15.67GB；实际保留600上游训练、300开发、300确认基因，累计预算不重置。
- 实际备用作业PID=3300037，仍运行=True；元数据和成本门=RAW_ASSET_METADATA_AND_COST_QUALIFIED；数值处理只允许训练/开发与NTC，确认先冻结风险分数再打开。
- 独立任务数量和资源满足门后接入精确计数协议，先生成训练/开发预测并验能力门；不因换资产而放宽能力、误差或确认隔离规则。

## 复现与事实入口

- 运行目录：/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21
- 状态：/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21/SUPERVISOR_STATE.json
- 逐任务采用排序：/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21/SYSTEM_RANKING.parquet
- 代码：run_safeconf_submission_evidence_v21.py / run_safeconf_content_matched_v21.py / run_safeconf_source_explicit_public_v21.py / run_safeconf_adamson_replication_v21.py / run_safeconf_gladstone_v21.py。
