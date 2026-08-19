# E168 byte acquisition

公开对象支持匿名 HTTP Range。VersionId 查询本身不允许匿名读取，因此下载未带 versionId；下载前后都必须重新 HEAD，并要求 Content-Length、ETag、VersionId、CRC64NVME 与 `SOURCE_LOCK.json` 完全一致。

```bash
mkdir -p /home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/source
aria2c --continue=true --max-connection-per-server=8 --split=8 --min-split-size=16M --file-allocation=none --dir=/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/source --out=GWCD4i.pseudobulk_merged.h5ad 'https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.pseudobulk_merged.h5ad'
sha256sum /home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/source/GWCD4i.pseudobulk_merged.h5ad
```

下载和哈希不解析 HDF5。全文件 SHA-256 写入新的 byte attestation；冻结的 SOURCE_LOCK 不回写。
