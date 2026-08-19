# E172 pretruth 运行修正记录

代码冻结时尚无运行修正。

若出现环境、路径或原子写入故障，必须先确认未读取 test targeting X，再在此逐项记录：时间、面板、失败位置、已读取的数据类别、修正范围和是否改变模型/分数/门槛。修正代码提交并推送两个远程后才可继续。模型表现不佳、gate FAIL 或正式结果不显著均不属于可修正的运行故障。

## 2026-07-18 19:52 CST｜Q01/Q02 asset launcher Python 版本

- 触发命令错误地使用了 `scgpt_env` 的 Python 3.9.25；asset helper 使用 Python 3.10 起支持的 `zip(..., strict=True)`，因此两个进程均在 `consume_control_rows` 的第一个 batch 退出。
- 在异常前，Q01 和 Q02 各自完成源文件整文件 SHA-256，并各解码了首批 128 行 `PRETRUTH_CONTROL_X`。异常发生在 Python 建立 `zip` 迭代器时，尚未遍历该批；train/validation targeting X 读取数为 0，test targeting X 读取数为 0，column-unseen targeting X 读取数为 0。
- `build_pretruth` 的异常清理删除了 staging；检查确认 `isolated/E172/Q01` 和 `Q02` 下没有 F2 文件或残留 staging。
- 修正仅为：asset builder 使用服务器 Python 3.12.4；模型训练继续使用冻结的 `scgpt_env` Python 3.9.25。没有修改代码、目标、模型、SafeConf 分数、G4 估计器或门槛。
