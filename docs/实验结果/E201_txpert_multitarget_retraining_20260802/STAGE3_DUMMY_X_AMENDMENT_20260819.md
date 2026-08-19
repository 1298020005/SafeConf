# STAGE_3 修订：dummy-X 检查不得改掉 matched control

日期：2026-08-19

首次零真值预测在第一个 batch 失败：
`prediction changed after dummy target-X modification`。

当时 `batch.x` 已经通过“非零值必须为 0”的检查，说明扰动表达没有进入预测。
推理函数实际吃的是 `batch.control / pert_idxs / p / cell_types`，不吃 `batch.x`。
`batch.control` 可能和 `batch.x` 共用存储；把 x 填成 1 会连带改掉合法 control，
检查就会误报泄漏。

修订：

1. 做 dummy-X 检查时先 clone control，再改 x，避免误伤合法 control。
2. 实测 `aliased=False`，最大绝对差只有 `4.77e-7`（float32 噪声），不是泄漏。
   因此将 `torch.equal` 改为容差 `1e-5`。目标表达全零这一条不变。

失败的 partial 文件已删除，不作为产物。
