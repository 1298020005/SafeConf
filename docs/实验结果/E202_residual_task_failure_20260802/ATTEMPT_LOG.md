# E202 正式运行尝试记录

## Attempt 1

- 开始：2026-08-02，冻结提交 `94b06ea53bbdb3c1be6d71f51202a8ffd491ec62` 已同时推送到 GitHub 和 Gitee；
- 输入哈希、566 个主任务及 8,490 行主层 scPertEval 合同均通过；
- 程序完成内存中的统计计算后，在生成 Markdown 报告时因当前 Python 环境未安装
  pandas 的可选依赖 `tabulate` 而失败；
- 终端没有打印任何统计结果；没有发布表格、报告、图片或状态文件；
- `.formal_evaluation.staging` 已由异常处理删除，`formal_evaluation/` 不存在；
- 修正仅将 `DataFrame.to_markdown()` 替换为脚本内的确定性 Markdown 表格格式化函数。
  任务集合、输入、风险量、结局、partial Spearman、bootstrap 次数、随机种子、
  次要分析和主判定门槛均未改变。

修正提交必须再次推送到两个远程后，才能开始 Attempt 2。

## Attempt 2

- 开始提交：`d1be8e02d96a40d1bf47c0b02918646f8042e8e0`，运行前已同时推送到
  GitHub 和 Gitee；
- 正式统计、报告、12 项输出哈希和独立主统计量重算全部通过；
- 主 partial Spearman 独立重算为 `-0.06795009555675122`，与正式表完全一致；
- 目视检查发现 B 面板标题在 PNG 右边界被截断。该问题不涉及表格或结论，但图片
  未达到汇报和归档标准；
- 整个 Attempt 2 发布目录已原样移到
  `DATA/txpert_official_20260802/e202/attempt_002_prefigure_correction/`，目录内
  `E202_OUTPUT_HASHES.csv` 的 12 项哈希再次验证全部通过；
- 唯一修正是把 B 面板标题从 `Magnitude-adjusted associations` 缩短为
  `Adjusted associations`。不改变绘图数据，更不改变任何统计分析。

制图修正提交再次双远程推送后，才开始 Attempt 3。

## Attempt 3

- 开始提交：`a7ac2685926a69a236f98a8e44baea56a1bce994`，运行前已同时推送到
  GitHub 和 Gitee；
- 运行完成，12 项发布文件哈希全部通过；
- Attempt 3 的全部统计表和报告与保留的 Attempt 2 逐字节一致；只有预期中的
  PNG/PDF 标题和状态文件的运行时间、Git 提交号发生变化；
- 新 PNG 以原始分辨率目视检查，四个面板、坐标、误差条和标题均完整，白色背景
  正常；
- 独立从 E200 主表重算得到 partial Spearman
  `-0.06795009555675122`，与正式状态文件完全一致。

Attempt 3 验收为 E202 唯一正式发布结果。
