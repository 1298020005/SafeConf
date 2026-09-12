# E208 Jiang24 下载与完整性记录

## 2026-09-12 第一次尝试：损坏，禁止使用

第一次下载先后混用了 `wget` 和 `aria2c` 的断点状态。任务以 HTTP 416 结束后，本地临时文件为 `15,236,380,670` 字节，而 Hugging Face 响应头声明正式对象为 `15,232,554,616` 字节，本地多出 `3,826,054` 字节。`gzip -t` 返回 `invalid compressed data--format violated`。

该文件已改名为：

`/home/yyf/data/external/perturbench_2025/jiang24/jiang24_processed.h5ad.gz.corrupt_attempt_20260912`

它不得进入 D0、训练或正式结果。保留到第二次下载通过完整性校验后再清理。

## 2026-09-12 第二次尝试：进行中

重新使用单一 `aria2c` 会话从零下载，输出：

`/home/yyf/data/external/perturbench_2025/jiang24/jiang24_processed.h5ad.gz`

会话：`tmux safeconf_e208_download`。完成后必须同时满足：

1. 文件大小为 `15,232,554,616` 字节；
2. `gzip -t` 通过；
3. 计算并记录本地 SHA-256；
4. 解压后再检查 H5AD 行列规模、`obs`/`var` 字段和官方 split 对齐。

## Split 文件

`jiang24_split.csv` 已下载，大小约 40 MiB，共 `1,628,476` 行。该文件没有表头，两列依次为细胞条码和 `train`/`val`/`test` 标记；读取时必须显式指定列名，不能把首个细胞误当表头。正式 D0 仍需在 H5AD 完整后核对条码一一对应。

2026-09-12 的独立流式检查得到：文件精确大小 `41,837,463` 字节，SHA-256 为 `5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d`；`train=1,216,453`、`val=189,264`、`test=222,759`；共 `1,628,476` 个唯一条码，无重复、无未知标签、无格式错误。

已新增 `tools/scripts/audit_e208_jiang24_contract.py`。它在不读取表达矩阵 `X` 的条件下执行压缩包大小与完整解压流检查、逐文件 SHA-256、split 全量审计、H5AD 行列与字段清单、条码集合一一对齐。脚本只列出可能的字段，不会猜测扰动、细胞背景或对照列；字段映射仍需在 D0 记录中人工确认并冻结。
