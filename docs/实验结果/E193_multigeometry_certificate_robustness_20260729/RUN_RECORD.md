# E193 运行记录

- 分析冻结提交：`cf3b342`
- 实现提交：`326b919`
- 两次提交均在运行前推送到 GitHub 与 Gitee
- 执行命令：
  `python tools/scripts/run_e193_multigeometry_certificate_audit.py`
- 环境：`/home/yyf/.conda/envs/scgpt_env`
- 墙钟时间：3 分 59.73 秒
- 最大常驻内存：322,668 KB
- 退出状态：0
- 输出状态：`PASS`
- E190/E192 真值在 E193 设计前已经打开，因此证据类型固定为
  `POSTTRUTH_METRIC_ROBUSTNESS`

运行时 pandas 对布尔列的初始化产生 FutureWarning。警告来自先以 `NaN` 建列、再
写入布尔值的 dtype 转换，不影响数值或裁决。后续脚本将布尔列显式初始化为 nullable
boolean，输出结果不作重算或改判。
