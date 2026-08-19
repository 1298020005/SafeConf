# Repository Slim Policy

本仓库的 Git 版本只保存代码、实验协议、轻量级汇总结果和图。训练数据、模型权重、预测矩阵、bootstrap 原始抽样和运行缓存留在服务器的数据目录，不放进 Git。

## 保留在 Git 中

- `code/`、`tools/` 下的源代码和测试；
- `docs/` 下的 Markdown、实验协议、审计报告和小型汇总表；
- `workspace/` 下的汇报图、制图脚本和说明；
- 可复现实验所需的 JSON、YAML、SHA-256 清单和小型 CSV。

## 不进入 Git

- `npz`、`npy`、`h5ad`、`pt`、`pth`、`ckpt`、`pkl` 等模型或数组文件；
- `arrays/`、`raw_*`、`checkpoints/`、`logs/`、`models/`、`runtime/` 等运行产物目录；
- `csv.gz` 和超过 100 KB 的任务级原始 CSV；
- `_archive/` 中的聊天转录、旧实验草稿和重复交付包；
- `node_modules`、Python 缓存、压缩包和本地构建产物。

这些文件没有从服务器删除。需要复现实验时，按各实验 README 中记录的数据路径和哈希，从服务器数据目录加载。GitHub/Gitee 版本只作为代码、协议和小型证据的同步入口，便于 Gemini、Codex 和远程主机读取。
