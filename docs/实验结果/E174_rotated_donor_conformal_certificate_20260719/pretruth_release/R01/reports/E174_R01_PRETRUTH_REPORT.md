# E174 R01 pretruth gate

状态：**FAIL**。test donor targeting X 读取数为 0；所有模型输出都由不含 `y` 的 query graph 生成。

训练参考任务 960，validation query 600，test query 600。

本阶段没有真实 test error，也没有形成部署授权。只有将该 snapshot 作为不可变commit 同时推到 GitHub/Gitee 后，独立 postgate builder 才可尝试解封。
