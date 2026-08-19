# E161 首次预检失败与 RNA feature 轴勘误

时间：2026-07-15。性质：metadata-only preflight failure，发生在任何 Wessels `X` 索引、物化或转换之前。E160 的 train/validation/test 条件和细胞切分未改变。

## 失败原因

首次预检按 E161 初版假设要求 21,052 个 `var_names` 全部大写，实际在该门终止：

```text
RuntimeError: Raw Wessels gene axis is no longer uppercase
```

只读 `var_names`和 `var` 元数据后发现，问题不只是基因符号大小写。该 h5ad 的 `X` 列由连续的三块特征组成：

| block | raw column index | count | first | last | E161 policy |
|---|---:|---:|---|---|---|
| RNA features | 0–20,638 | 20,639 | `OR4F5` | `KRAB` | 允许 train/validation 读取 |
| guide/array features | 20,639–21,023 | 385 | `ATXN7L3_g1:NT_g2` | `SUPT5H_g1:SUPT6H_g2` | 排除，不物化 |
| additional feature barcodes | 21,024–21,051 | 28 | `NT` | `CD71-Mpknot` | 排除，不物化 |

后 413 列的名称直接编码 guide/扰动信息。将它们纳入 HVG、PCA 或 library-size 分母会产生标签泄漏。另一份本地官方 generalization Wessels 资产的 5,020 个预选 RNA features 全部属于前 20,639 列，也支持该边界。

## 冻结身份

- full 21,052-feature order SHA256: `dea725a87c973ca15590b08b309df3a926dc0233391cb2df76518c847229e780`
- RNA 20,639-feature order SHA256: `e01181f8f46d7e871d6d335a528e8861bff393dbb7b27ab81cdfe9c95b573371`
- excluded 413-feature order SHA256: `9088328f4ac6b2a1b109c254f0068504d25618478383fcbb3f43be8e59dd06d2`

## 修正规则

1. `read_allowed_expression` 只可读 train/validation rows 与前 20,639 个 RNA columns。
2. 后 413 列在 E161 不得读取，也不得用于归一化、HVG、PCA、graph 或 endpoint。
3. 归一化分母为前 20,639 个 RNA features 的总 raw count，并必须与 `obs[ncounts]` 逐细胞精确相符。
4. 发布资产同时保存 full、RNA 和 excluded 三份顺序文件及哈希。
5. E163 一次性解封后也只能在同一 RNA 轴上构造 test truth。

这是解封前的方法勘误，不是根据结果更换任务、指标或切分。
