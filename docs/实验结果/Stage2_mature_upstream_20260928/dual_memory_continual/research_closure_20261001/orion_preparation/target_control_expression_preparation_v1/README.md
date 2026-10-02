# 固定 target-control expression 代理：准备完成

实际 `COMPLETE_FIXED232_TRAIN_CONTROL_PROXY_PREPARATION`，PID844452／session64141，1.046706秒。全部232 tasks／144 genes，HCT116107、HEK293T125，原行序不变且分数全部finite。

唯一规则：`TargetControlExpression_CP4000 = own-context TRAIN NTC mean_cell_logCP4000[target_gene_id]`。高表达固定为高风险；无拟合、变换、反号选择或校准。Root已明确更正初始消息中的CP10000为原科学合同CP4000；更正保存在执行registration，CP10000未执行，未对已平均log值做换算。

**实际控制一致性：**两context完整38606侧car与原已拟合`MODEL.rds$baseline`逐位一致；原3285控制投影亦逐位一致。R进程仅读取原RDS并导出baseline，没有predict／fit调用。Canonical Ensembl/symbol逐项一致。保留全部232任务，其中145个target不在3285 readout：HCT11670、HEK293T75；没有缩到87个readout-supported任务。

## 固定后续接口

- runtime目录：`/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_target_control_expression_preparation_20261002_v1`。
- `FIXED_PRIMARY_TARGET_CONTROL_RISKS.parquet`：5个原query identity字段＋float64列`TargetControlExpression_CP4000`，232行。SHA `115dac13a4e452e495052279b445edbca3f86e83d29678ed96d3475978537fc7`。
- `RISK_FLOAT64.npy`：原query行序的232个float64分数。SHA `fda2d28f021973df28ec074704ae0e141b281d2a89a75e7f544888e85f6d3d0f`。
- `PREPARATION_RESULT.json`：SHA `f9ec4f0910d3f2a87557ab3ade380034388263b63d80cc09400cef5e0e025441`；包含全部原input/code/model/control绑定、逐位检查和runtime输出绑定。另见`CONTROL_IDENTITY_AND_BITWISE_PROJECTION_CHECKS.csv`、`INFORMATION_SCOPE_LEDGER.json`。
- 执行脚本：`tools/scripts/prepare_safeconf_orion_target_control_expression_agent.py`，SHA `b30169c4b2ff821464f0ec76c3ecb463d1a4fac57cb97e4a90d85969ca94d7fe`。

控制缓存和原模型在执行前后SHA/size完全相同；输出已只读冻结。分数Parquet重载后identity与数值逐位保持。原NC、Source、LM和13主方法未修改。大向量／R导出和日志只在上述独立runtime保存。

## 信息与解释边界

这是利用现有完整38606 TRAIN control输入域的生物信息强对照；它使用原P/PUBLIC13-feature readout以外的target坐标，不宣称相同feature信息预算、算法优势或新主方法。控制匹配层级为own-context TRAIN NTC，不是query-specific plate/batch。Source／MC缺轴全队列版本保持停止。

本阶段仅准备和封存分数：0cached-error numeric reads、0raw expression reads、0new labels、0model prediction calls、0upstream calls／fits、0statistics／bootstrap、0GPU、新下载0。RDS读取只是既有控制baseline核验，不是新上游模型尝试。此为SEEN辅助对照，不产生新确认资格。后续配对评价由Root另行固定执行；本阶段没有查看效果。
