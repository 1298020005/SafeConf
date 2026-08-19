# E39 data acquisition: advisor multidimensional data request

生成日期：2026-07-09

这份记录对应用户要求：“按老师意思，多下更多维度、更多量的数据，趁流量够先抓下来。”

## 已确认的公开数据

| 来源 | 官方记录 | 本地状态 |
|---|---|---|
| scPerturBench cellular context generalization | Zenodo `14607156` | 12/12 文件已在本地 |
| scPerturb RNA/protein perturbation atlas | Zenodo `13350497` | 54/54 文件已在本地 |
| Tahoe-100M pseudobulk differential expression | HuggingFace `tahoebio/Tahoe-100M` | 1026/1026 分片已在本地，约 88.9GB |
| Tahoe-100M raw single-cell parquet | HuggingFace `tahoebio/Tahoe-100M` | 3388 分片已启动后台下载 |

## Tahoe raw 下载

启动脚本：

```bash
python3 tools/scripts/download_tahoe_100m_raw.py --mode start --concurrent 24 --split 4
```

后台进程：

```text
pid=1822925
```

数据目录：

```text
/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M/data
```

监控命令：

```bash
python3 tools/scripts/download_tahoe_100m_raw.py --mode status
ps -C aria2c -o pid,stat,etime,cmd
tail -n 40 /home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M/download_logs/tahoe_100m_raw_aria2.console.log
```

## 当前下载状态

截至 2026-07-09 21:08 左右：

- Tahoe raw 目标分片：3388
- 已进入下载队列：3388
- 当前 partial parquet：26
- 当前已写入约：2.15GB
- aria2 后台进程仍在运行
- 下载速度受 HuggingFace/CDN 限制，约 1.5MiB/s；已支持断点续传

## 为什么先抓这些

周老师要求的重点是不同 setting、不同数据类型、更多任务量。当前最匹配的是：

- gene perturbation：Haber、Parekh、kang、Shifrut、Lara、Norman、Adamson、Replogle 等；
- chemical perturbation：TCDD、sciplex3、Srivatsan、McFarland、Tahoe；
- cross-context / cross-patient：KaggleCrossCell、KaggleCrossPatient、crossPatient、kangCrossCell、kangCrossPatient；
- huge-scale chemical：Tahoe-100M pseudobulk + raw single-cell parquet。

scPerturBench 和 scPerturb 已经全量在本地，所以新增最大量来自 Tahoe raw。

## 后续动作

1. 让 Tahoe raw 后台持续下载；
2. 下载完成后更新 `metadata/h5ad_scan.tsv` 之外的 mega-external inventory；
3. 先用已完成的 Tahoe pseudobulk 继续 E34/E35/E36 formal；
4. raw parquet 完成后，再考虑细胞级抽样、跨 cell line / drug family / dose setting。
