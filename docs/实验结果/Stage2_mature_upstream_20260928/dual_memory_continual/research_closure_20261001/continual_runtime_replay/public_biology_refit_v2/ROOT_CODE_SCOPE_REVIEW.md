# ROOT代码范围复核（运行中，尚不替代实际结果验收）

审阅代码SHA-256: `6f5b98430ca0bc18986eacd0cbcd4e39c3bb78dceb03d3eb04b507240397c4d5`。

- 使用原Source DEV 1699task/563gene/3398prediction records，沿原5fold角色；同gene两架构同fold。初始bank1047、追加bank2008均为既有只读immutable快照。
- PublicBio训练只对fold2/3或2/3/4的任务调用原build_pairs生成transfer_rmse；不接入upstream错误。fold0/1只生成9个预测前pair字段，查询treated truth不用于训练或权重。
- 初始PublicBio保留2个OOF+final；更新保留3个OOF+final，共7个原HGB拟合。risk三个固定臂，原配置、source分组CDF及cluster权重，总10fits；历史ErrorMemory的shared residual不作为监督。
- B和C拥有相同bank、Source错误预算和风险器配置；差别是BiologicalLearner重训与其OOF参数。B只是诊断，C是唯一发布候选。
- 逐值检验unlabelled pair字段与原算法：已纠正math.log1p数值路径。V1在metadata行索引缺失时0fit停止；V2按task_id一对一绑定原始source_mean_delta_row，并核1808轴顺序，未改容差或删任务。
- 全部模型/先验/权重/风险先保存，再解码两个开发评价角色的错误；从保存Bio+bank+原预测/控制重建先验及risk做bitwise reload，不仅缓存feature推理。
- 仅新isolated ServingModelRegistry可发布或拒绝；原Public/Error版本和服务CURRENT有before-after hash保护，Orion/MC数值不访问。
- 原finite_release_gate用于C对A，B不参加选择。复用227gene×5000个原计数，全角色/模型/context同步抽样，未新建随机抽样；并列导致Spearman NA保留。

后续还需ROOT实际核验：真实fit计数、固定人群、B/C CDF和训练ID相等、门决定/实际服务bundle、全部42个CI的saved-draw复算、文件前后hash。通过代码范围不等于增量成立，也不构成外部确认或部署无损保证。
