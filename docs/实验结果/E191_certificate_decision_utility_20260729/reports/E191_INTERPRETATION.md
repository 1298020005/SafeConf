# E191 决策收益怎么解释

## 外部迁移中，证书能提高复核效率

E190 的 692 个跨研究任务中，复核风险最高的 20%：

- diversity lower bound 捕获 47.48% 的真实 top-error 任务；
- predicted magnitude 同样捕获 47.48%；
- source-effect magnitude 捕获 48.20%；
- 随机选择的期望只有 20.09%。

三种可用风险量都把有限复核资源集中到更容易出错的任务。diversity 的平均误差富集
为 1.150 倍，达到 oracle 收益的 44.3%；其下界平均覆盖所选任务真实 family error
的 41.7%。这说明证书并非只有“零违例”这一数学结果。

## 下界没有普遍超过 magnitude

E190 中 diversity 与 predicted magnitude 的捕获率完全相同，oracle-normalized
utility 分别为 0.443 和 0.441；source-effect magnitude 为 0.473。下界有严格含义，
但排序效果没有形成对 magnitude 的明确优势。

E189 的 16 个 support×setting 层中，diversity 相对 magnitude 正好 8 胜 8 负。
20% 预算按 setting 平均：

- random pair：diversity 0.310，magnitude 0.239；
- unseen column：diversity 0.117，magnitude -0.090；
- unseen row：diversity 0.046，magnitude 0.101；
- double unseen：diversity -0.127，magnitude -0.080。

双未见时两种排序都比随机期望更差，不能部署为高风险检索器。

## 方法应采用双层输出

第一层始终输出确定性 family-error lower certificate，其数学含义不随 setting 改变。
第二层是经验复核排序，只在验证过的 random-pair、column-unseen 和 E190 这类 setting
启用；在 double-unseen 中应标记为“排序未验证”，而不是根据小分歧放行。

这个定位比“一个通用 confidence score”窄，但与 E189/E190 的真实结果一致，也直接
回应老师对 model disagreement 的质疑。
