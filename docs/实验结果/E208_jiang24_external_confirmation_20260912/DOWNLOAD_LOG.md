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
