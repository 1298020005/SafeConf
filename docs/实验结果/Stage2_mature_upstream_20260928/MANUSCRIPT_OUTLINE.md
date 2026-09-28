# SafeConf v4 稿件大纲

## 暂定题目

**SafeConf: Evidence-Aware Post-hoc Risk Auditing for Single-Cell Perturbation Predictions**

## 1. Introduction

- 扰动预测准确率不足以告诉实验人员先复核哪些任务。
- 预测幅度是强而被低估的风险基线。
- 历史实验有帮助潜力，但会稀疏、无关、冲突；直接拼接会失效。
- 贡献：统一输出合同、能力门、历史证据分层、nested shrinkage、同预算 selective review。

## 2. Methods

1. PredictionRecord 与 effect/error contract。
2. History Eligibility 与 internal/external history。
3. Universal P、Support、Relevance、Conflict、Missingness。
4. V1 与 V2 nested cross-fitting。
5. Utility@20、risk-coverage、AURC 和 paired cluster bootstrap。
6. PertEMA 同合同适配与能力门。

## 3. Results

1. TxPert 两架构均通过上游能力门。
2. Magnitude 强，但 Support/Relevance 提供开发增量。
3. V2 通过预注册开发门；复杂模型没有收益。
4. 历史 shuffle、within-context 与 matched-support 排除明显 shortcut。
5. E190 GEARS 跨 study 压力线：V1 信号与 V2 gate 失效边界。
6. 同合同 PertEMA 适配比较。
7. E208/E216/E247/E258 展示能力门和适用边界。

## 4. Discussion

- 当前证据限于基因扰动与 TxPert family。
- Quality/external history/chemical 仍是扩展，不包装为已完成。
- 上游 competence 是风险审计的前置条件。
- 部署流程：冻结上游→生成 PredictionRecord→冷启动风险→积累合法反馈→周期重训。

## 5. 必须在投稿前补齐

- 一个能力合格且此前未用于设计的 sealed family/context confirmation；E190 只能作为已开封开发证据。
- 对应 confirmation 的一次性 Gate A/B 表。
- 若无法取得，则把稿件转为 Gate C，并在独立 split/context 复现至少两条边界规律。
