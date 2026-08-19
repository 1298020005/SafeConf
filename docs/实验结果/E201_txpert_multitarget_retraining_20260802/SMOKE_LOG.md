# E201 工程 smoke 记录

## RPE1 / seed 1 / attempt 1

- 时间：2026-08-02 06:09–06:10（Asia/Shanghai）；
- 代码提交：`336470c4bf7736882c5e4a3d460ac686ab0c27b8`，运行前已同时推送到
  GitHub 和 Gitee；
- 输入：`E201_blind_RPE1`，目标扰动细胞 0，目标对照 11,485；
- 计划：1 epoch 中只取前 20 个训练 batch，不运行 validation；
- 实际：datamodule 和模型已构造，optimizer step 为 0；上游
  `SequentialLR` 在初始化时拒绝“一个 warmup scheduler + 一个 milestone”的
  组合；
- 错误：`Sequential Schedulers expects number of schedulers ...`；
- 原始失败状态保存在
  `DATA/txpert_official_20260802/e201/smoke_RPE1_seed1_20b/E201_RUN_STATUS.json`；
- 未读取目标扰动表达，未生成目标预测或评价结果，GPU 显存已释放。

修正：smoke 只用于测量单步可运行性，改为固定学习率 `3e-4` 且不启用 scheduler。
正式 80-epoch 协议仍使用 checkpoint 反推的 5-epoch warmup + 75-epoch cosine，
没有发生改变。修正提交双远程推送后，以新目录运行 attempt 2。

## RPE1 / seed 1 / attempt 2

- 时间：2026-08-02 06:12–06:13（Asia/Shanghai）；
- 代码提交：`0c2fb0561c10db7c131512d50b7d8cdad3f1fe80`，运行前已同时推送到
  GitHub 和 Gitee；
- datamodule、4.1 M 参数模型和 AdamW 已成功初始化；
- 第一个训练 batch 在 `TxPert.forward()` 的 `z_p[p]` 处停止，optimizer step
  仍为 0；错误为 `IndexError: too many indices for tensor of dimension 2`；
- 原始失败状态保存在
  `DATA/txpert_official_20260802/e201/smoke_RPE1_seed1_20b_attempt2/E201_RUN_STATUS.json`；
- 未运行 validation，未读取目标扰动表达，GPU 显存已释放。

根因经源码逐行核对后确定：上游 `XcelltypePerturbDataset._extend_with_train_data()`
把追加到训练目标中的对照写成字符串列表 `["ctrl"]`，而原始扰动条件和推理路径
使用整数 ID，`pert2id["ctrl"]` 固定为 `-1`。字符串传入二维 tensor 后触发该
IndexError。E201 新增局部 dataset 子类，只把这些新增对照目标编码为 `[-1]`；
其他条件、表达值、批次匹配和模型代码不变。再次运行前扫描全部训练行，要求每个
扰动索引均为整数，并要求 `[-1]` 行数等于允许进入训练的对照数。

## RPE1 / seed 1 / attempt 3

- 时间：2026-08-02 06:27–06:28（Asia/Shanghai）；
- 代码提交：`f7e3664c8b0acbd075880bd16fbcdefaae748a7b`，运行前已同时推送到
  GitHub 和 Gitee，两端远程哈希与本地一致；
- 状态：`COMPLETE`，20/20 个真实优化步完成，`global_step=20`；
- 训练集 273,003 行（4,265 batches/epoch），验证集 76,950 行
  （1,203 batches/epoch）；
- 39,165 个允许进入训练的对照目标全部使用公开图索引 `-1`，
  其他训练扰动索引也全部为整数；
- 没有构造目标测试集，目标扰动细胞访问数为 0；
- 重建损失在第 1 步约为 0.254，第 20 步为 0.120146；
- 运行状态 SHA-256：`2b656183ff0f8216bbffc2b4a43d64dcbe0f363bb1ac8cca1f1cf9258a30f9ef`；
- smoke checkpoint SHA-256：`ed65e45b207d1cbaf40e3d3924c35ed614660585637a1c6fa577e6ef35d8e029`。

纯训练吞吐约为 5.27 batches/s，换算单个完整训练 epoch 约 13.5 分钟。
这个数字不包含每轮 1,203 个验证 batch、80 轮的累计 I/O 及 checkpoint 写入，
因此不用它直接承诺正式总时间。下一个资源门是完整 1 epoch 训练+验证测量。

## RPE1 / seed 1 / 完整 1-epoch 资源门

- 时间：2026-08-02 06:46–06:56（Asia/Shanghai）；
- 代码提交：`0d8e12dc885f9b9b63eac9f5ba36967594aa42fc`，运行前已双远程推送；
- 状态：`COMPLETE`，训练 4,265/4,265 batches，source validation 完整运行；
- `global_step=4265`，checkpoint 内的 `global_step=4265` 与状态文件独立一致；
- `Trainer.fit` 实测 535.97 秒，即 8 分 56 秒；
- PyTorch 峰值 allocated 显存 14,659,002,880 bytes（约 13.65 GiB），
  reserved 显存 15,183,380,480 bytes（约 14.14 GiB）；
- 1 epoch 的 source `val_pearson_delta=0.3382727504`，只作训练/验证通路检查；
- 目标测试集未构造，RPE1 扰动真值访问数仍为 0；
- 状态文件 SHA-256：`d2df860b22e79625a01d937d1d721b718005307ba602e56f4f3c777771c92749`；
- checkpoint SHA-256：`696f713dff0d2904dec1857ec17016b4ce2811086bda13c784742c0140440330`；
- 独立复核 11 项状态、隔离、step、epoch、时间和显存条件全部通过。

按这个实测速度线性估算，单个 80-epoch 模型约 11.9 GPU 小时；
4 个目标背景×4 个事先固定种子约 190 GPU 小时，还需加上多轮 checkpoint I/O。
正式运行必须分阶段排队，但不能根据目标真值决定是否补跑后续种子。

## 正式冻结后 provenance 预检

- 时间：2026-08-02 07:20–07:21（Asia/Shanghai）；
- 冻结提交：`f157a9ca899bd99035edb83b793cc28e89110719`；
- 1/1 个真实优化步完成，目标扰动访问数为 0；
- 状态文件自动记录的 SafeConf HEAD、Gitee 跟踪 HEAD 和 GitHub 跟踪 HEAD
  三者完全一致；
- 训练 adapter SHA-256：`274344472a2e19d91cf8e971e4e46cf08e39868a36d7892310281d1d847d5e7a`；
- 状态文件自动写入的 checkpoint SHA-256 为
  `9387ee7db9976c147ad9358733eecb25c300b85386dd0961b777bfc5ac6085a2`，与磁盘独立
  `sha256sum` 结果一致。

预检通过后，正式作业不再修改训练 adapter、种子、目标顺序或优化协议。
