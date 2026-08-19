# E150｜Replogle 原始计数组合资产

## 完成状态

E149 的源文件、选择表和任务 manifest 哈希全部通过后，按固定的 128 个 perturbations 加 control 读取 K562/RPE1。两个源文件取共同 7226 个基因，合并得到 72008 个细胞。

- K562：36362 cells；RPE1：35646 cells。
- perturbation cells=49832，control cells=22176。
- X 为 CSR float32，nnz=230707877，density=0.443；非零值逐块审计均为非负整数计数。
- CSR 内存占用约 1.72 GiB；h5ad 磁盘占用约 0.47 GiB；构建过程 peak RSS=5074.2 MiB。
- 256 个 selected context × perturbation 组合的细胞数全部与 E149 selection/manifest 精确一致；另记录两个 control 组合。

## 信息边界

本步骤只作行过滤、共同基因列过滤、稀疏格式转换和两个细胞系拼接。没有 normalize_total、log1p、scale 或其他数值变换；没有计算 perturbation effect、模型预测或误差，也没有改变 E149 的任务选择与划分。
