# 六个本地数据别名的资格复核

## 结论

**本固定六项中没有证明为尚未见数值、且可进入不变Source3285人类遗传合同的候选。** Wessels、Xu、Liang、Tian有直接历史结果/解封证据。Zhao和Chang的既往数值暴露只能记为UNKNOWN，但官方来源说明它们是药物/治疗及谱系追踪数据，不能因文件名存在就称为遗传候选。没有新增拟合、adapter、下载数值资产或第三attempt；在此结束本地六项排查。

## 角色、基因轴与判定分开报告

Source参照是原3285个gene symbols（GENE_IDS SHA `b7f80db0c0e46dd7667cb83f39153c4add31948ca00e5391c210b9c6a18d7c68`）。以下只读原H5AD `var`字符串；**这是测量轴，不是已发布prediction/control轴的证明**。没有按表达、效果或SafeConf得分筛选。

| 固定别名 | 本地数值暴露角色 | 原var rows；Source3285 symbol匹配 | 具体资格结论 |
|---|---|---|---|
| WesselsSatija2023 | **REJECT_SEEN** | 21052；3252，缺33 | E165不可逆TEST解封事件存在；不能fresh。缺失symbol名单保存在gene-axis JSON，也不能直接供完整3285参数核心使用。 |
| XuCao2023 | **REJECT_SEEN** | 59429；3285，缺0 | E180 final summary/report已完成正式结果评价；完整测量轴不恢复freshness，也不等于现成全轴预测。 |
| ZhaoSims2021 | **READ_EXPOSURE_UNKNOWN** | 60725；3285，缺0 | 已查本地只命中E10官方下载catalogue，未找到明确UNSEEN记录。官方说明是GBM组织切片的multiplexed **drug** responses，拒绝当前genetic-target合同。 |
| ChangYe2021 | **READ_EXPOSURE_UNKNOWN** | 45066；3284，缺UBE2M | 已查本地只有E10目录命中。TraCe-seq治疗/谱系条形码不是≥100gene扰动簇；Root核作者数据表为drug三条件。拒绝genetic合同，不补零/修轴。 |
| LiangWang2023 | **REJECT_SEEN** | 17444；0个human exact symbols | E123正式scGPT/GEARS评价complete；既有研究说明为mouse organoid hepatocyte，不能做人类Source基因合同。没有大小写/orthology映射。 |
| TianKampmann2019_day7neuron | **REJECT_SEEN** | 33752；3285，缺0 | E168候选选择说明明确E129/E153/SafeTrans已见，只能桥接重分析。不能把day7文件或另一个iPSC状态重新称为untouched研究。 |

所有六个原文件均存在；路径、size、mtime、var字段和gene-list SHA见 `GENE_AXIS_EVIDENCE.json`。同一var里的Ensembl字段与Source symbols是不同ID namespace，0直接字符串匹配不能解释为对应基因生物学缺失；上表使用symbol/index字段。针对Chang缺失UBE2M的一次冻结Source Ensembl核对也未找到其 `ENSG00000130725`，没有修复或再次追查；模态已独立否决这条路线。

本审计没有计算obs target/state分布，也没有把var数量、guide/barcode数、fold/seed或既有任务记录数当作独立genetic clusters。≥100合格gene/≥3真实生物strata的本次资格数保持未知或不适用；不能从全基因测量轴推出通过。

## 可复核的历史角色证据

- Wessels：`E165_wessels_truth_unseal_evaluation_20260715/TEST_TRUTH_UNSEAL_EVENT.json`，schema为 `safeconf_e165_irreversible_truth_unseal_v1`，SHA `7de6b6e7d207e5a01b296b65802f5c1d10ca8600b8254a5c2da24e7f4d915043`。
- Xu：`E180_xucao_fresh_guide_certificate_20260723/final_evaluation/E180_FINAL_SUMMARY.json`，schema `safeconf_e180_final_summary_v1`，SHA `c18635c9a54c8f2bc941bb37c29ee2c64c334a71bc9ed569778dff6b3dfbba75`。其成功/失败数值不用于入选，仅已存在最终评价即证明SEEN。
- Tian：`E168_primary_human_cd4_fresh_confirmation_20260716/CANDIDATE_SELECTION_NOTE.md`，SHA `c24d6a018a7492a3a29b1aeb4ebe056e8eb199444aee8a301ddddff6ba5b8786`，明确已有旧研究分析。
- Liang：E123 `RUN_STATUS.json` SHA `bd3e59345b028e7c3972fd5241ef2ffcaeec3aa38fae78a134d73b6b0ac27685`及 `E123_REPORT.md` SHA `a04f580fed80b00f34c649749d41aa2b9d97d32b4d191d8552180440fdb056ac`。612tasks/1224records是该历史曝光的范围，不是当前100独立gene门证据；对应legacy数值CSV/NPZ只核存在，不打开。
- Zhao/Chang：E10 `scperturb_zenodo_13350497.json` SHA `460868fe41a6f7cf9dc3023b72b89e826e7479097d626f7d743aaf9ab2074844`仅证明收录/下载。两个repo的指定文本、scripts、status/manifest/registry中未找到曝光结果或explicit UNSEEN断言；**未命中不证明未访问**。

官方[pertpy数据加载器说明](https://pertpy.scverse.org/en/stable/_modules/pertpy/data/_datasets.html)明确Zhao为drug perturbation，并链接原论文DOI `10.1186/s13073-021-00894-y`；Chang说明TraCe-seq条形码/治疗响应并链接 `10.1038/s41587-021-01005-3`。条形码转导不能被当成gene-target perturbation。Root的[作者dataset表](https://www.sanderlab.org/scPerturb/datavzrd/scPerturb_vzrd_v2/dataset_info/index_1.html)复核进一步指出Chang drug三条件；本审阅只取得该表HTML，未独立解码动态行，因此明确保留此来源层级，没有声称自己重算了条件数。Nature原始索引页访问被cookie身份端点阻断，未绕行，也不因此推测内容。

## 边界与小包

没有打开X/layers/raw数值、obs分布、Source error/prediction向量或新的真值。只读取gene ID字符串及其分类编码；`var/ncounts`、`var/ncells`只是字段名，未读值。原H5AD只stat，不为此资格问题扫描全文件/X字节做SHA；gene-list SHA及既有角色文档SHA足以复核当前判定，不能冒称已绑定整个raw文件内容。无新增算法、模型、拟合、抽样、参数、审批、正式访问事件或shared tracker变更。

这关闭了本固定六项的**资格可用性**问题，未关闭普遍科学问题，也不把UNKNOWN改写成SEEN或UNSEEN。要改变判定需另有合法遗传研究与明确未见角色及完整预测/控制/normalization合同；本次不扩大候选名单或申请新表达。

本目录仅新增本AUDIT、`GENE_AXIS_EVIDENCE.json`、`ROLE_QUALIFICATION_EVIDENCE.json`。Frangieh角色纠正另保存在 `../fresh_prediction_release_audit_v1/LOCAL_ROLE_RECHECK.md/.json`，原官方发布审计保持原bytes。
