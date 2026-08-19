# E92｜PRESCRIBE 本地 scGPT 资产

基因嵌入直接从本机 whole-human scGPT 冻结权重与对应词表提取，并经过原模型的 `encoder.enc_norm`。两个 PRESCRIBE 所需文件为同一 inode，避免保存两份约百 MB 的重复 pickle。完整哈希见 `RUN_STATUS.json`。
