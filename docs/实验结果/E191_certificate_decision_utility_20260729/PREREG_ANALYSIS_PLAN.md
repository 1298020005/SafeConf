# E191｜有限复核预算下的证书决策收益

冻结日期：2026-07-29  
性质：基于已解封 E189/E190 的注册式二次分析，不冒充前瞻确认。

## 问题

如果只能人工复核 10%、20% 或 30% 的预测任务，确定性下界能否比 predicted
magnitude 找到更多真实高错误任务？这是证书从数学成立走向实际使用必须回答的
问题。

## 冻结对象

- E189：每个 `(support, setting)` 单独排序，四个 donor 面板合并；
- E190：692 个 Adamson→Replogle 任务单独排序；
- family RMS error：比较 diversity lower bound、predicted magnitude；E190 另加
  source-effect magnitude；
- worst-member error：比较 diameter/2 lower bound、predicted magnitude；E190
  另加 source-effect magnitude。

不拟合新权重，不把真实误差用于构造风险分数，不跨 setting 混排。

## 指标

每个预算固定选取风险量最高的 `ceil(budget × n)` 个任务：

1. high-error capture：真实误差最高的同等数量任务被捕获的比例；
2. error lift：被选任务平均误差 / 全部任务平均误差；
3. oracle-normalized utility：
   `(selected mean - overall mean) / (oracle mean - overall mean)`；
4. lower-bound tightness：确定性下界 / 对应真实 family error。

同分值按任务 ID 的 SHA-256 顺序打破，不人为挑选。oracle 只给性能上限，不作为可用
方法。E189 主要看四种 setting 是否方向一致；E190 看下界是否优于两个 magnitude
基线。

## 解释规则

- normalized utility > 0 表示优于随机期望，1 表示达到 oracle；
- 下界 0 违例不等于有良好排序收益；
- 任一 setting 中下界弱于 magnitude 必须保留；
- 只有多数困难 setting 和外部迁移都产生稳定增益，才能写成通用复核策略；
- 本分析不替代真实后续实验。
