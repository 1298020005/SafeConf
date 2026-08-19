# E199 预测封存记录

封存时间：2026-08-02 01:39（Asia/Shanghai）

## 已完成的预测

| 输出 | 官方配置 | 图信息 | 测试细胞 | 基因 |
|---|---|---|---:|---:|
| GAT | `config-gat` | STRING | 38,475 | 5,000 |
| Exphormer | `config-exphormer` | STRING | 38,475 | 5,000 |
| Exphormer-MG | `config-exphormer-mg` | STRING + GO | 38,475 | 5,000 |
| General baseline | `config-baseline` | 官方启发式基线 | 38,475 | 5,000 |

测试集包含 272 个扰动条件，细胞系均为 K562。三种神经网络读取官方
公开 checkpoint；general baseline 直接调用 TxPert 仓库的 `MeanBaseline`
实现。未修改 TxPert 源码，未训练新模型。

## 封存门

- 四套输出的细胞顺序、扰动标签、批次顺序和基因顺序完全一致；
- 三种 checkpoint 输出的 ground truth 和 control H5AD 文件逐字节一致；
- baseline H5AD 的存储编码不同，但逐行解码后的 ground truth 和 control
  浮点数组 SHA-256 分别与 checkpoint 输出完全一致；
- 所有原始文件已记录字节数和 SHA-256。原始预测仅保存在数据盘，不推送
  checkpoint、细胞表达矩阵或 TxPert 权重。

自此提交之后，评价脚本只能读取封存文件。如果任一哈希不匹配，E199 必须
fail closed，不生成风险排序结论。
