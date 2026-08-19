# SafeConf 远程电脑 / 远程 Codex 初始化说明

更新时间：2026-08-12
当前材料基线：以远程 `exp/task-risk-audit-20260611` 的最新提交为准；拉取后用 `git rev-parse HEAD` 核对。
当前推荐分支：`exp/task-risk-audit-20260611`

> 当前有一项正在运行的 E201 四背景盲训练。它尚未打开目标真值，不能被当作已有
> 实验结论。第一次进入先读 `docs/项目交接_20260812.md`，其中写明当前队列、数据
> 是否在 Git、以及后续绝不能颠倒的步骤。

这份文件给远程电脑和新的 Codex 用。目标很简单：先把项目拉下来，再让 Codex 按正确顺序读材料，不要从旧草稿或旧实验里迷路。

## 1. 仓库地址

GitHub：

```text
git@github.com:1298020005/SafeConf.git
https://github.com/1298020005/SafeConf
https://github.com/1298020005/SafeConf/tree/exp/task-risk-audit-20260611
```

Gitee：

```text
https://gitee.com/librety/safe-conf.git
https://gitee.com/librety/safe-conf
https://gitee.com/librety/safe-conf/tree/exp/task-risk-audit-20260611
```

当前工作只以两端的 `exp/task-risk-audit-20260611` 为事实分支；不要假定
`main/master` 已同步到相同提交。

## 2. 远程电脑拉取方式

优先从 GitHub 拉：

```bash
git clone git@github.com:1298020005/SafeConf.git safeconf
cd safeconf
git fetch --all --prune
git checkout exp/task-risk-audit-20260611
git pull --ff-only
git rev-parse HEAD
```

如果 GitHub SSH 不方便，就用 HTTPS：

```bash
git clone https://github.com/1298020005/SafeConf.git safeconf
cd safeconf
git fetch --all --prune
git checkout exp/task-risk-audit-20260611
git pull --ff-only
git rev-parse HEAD
```

如果 GitHub 访问慢，就从 Gitee 拉：

```bash
git clone https://gitee.com/librety/safe-conf.git safeconf
cd safeconf
git fetch --all --prune
git checkout exp/task-risk-audit-20260611
git pull --ff-only
git rev-parse HEAD
```

最后一行应当是远程实验分支最新提交。仓库中应能看到
`docs/实验结果/E194_family_governance_stress_20260729/`。

## 3. 远程 Codex 第一轮必须读的文件

按这个顺序读，不要跳：

```text
REMOTE_CODEX_INIT.md
docs/项目交接_20260812.md
START_HERE_FOR_AGENTS.md
docs/学习导航/README.md
docs/学习导航/01_目录与权威等级.md
docs/学习导航/02_当前证据链与实验谱系.md
START_HERE_FOR_GPT.md
README.md
INDEX.md
docs/实验结果/GATE_STATUS_20260729.md
docs/投稿准备/期刊与文献定位_20260729/README.md
docs/实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md
docs/实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md
docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md
docs/实验结果/E191_certificate_decision_utility_20260729/reports/E191_INTERPRETATION.md
docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729/final_evaluation/reports/E190_INTERPRETATION.md
docs/实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md
docs/实验结果/E186_presubmission_integrity_audit_20260724/reports/E186_REPORT.md
docs/实验结果/E185_minimal_release_validation_20260724/reports/E185_REPORT.md
docs/实验结果/E184_direct_competitor_positioning_20260724/reports/E184_REPORT.md
docs/学习导航/05_E133_E140方向风险与七数据更新.md
docs/学习导航/06_E141_E144机制审计与前瞻湿实验.md
workspace/group_meeting_20260709_MAINLINE_WHITE/周老师聊天记录_要求拆解与实验设计_20260709.md
```

如果只想理解当前研究是否达到投稿水平，最重要的是：

```text
docs/实验结果/GATE_STATUS_20260729.md
docs/投稿准备/期刊与文献定位_20260729/README.md
docs/实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md
docs/实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md
docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md
```

## 4. 直接复制给远程 Codex 的提示词

```text
你现在在 SafeConf 仓库里。请先不要改代码，也不要新增实验。

请按顺序阅读：
1. docs/项目交接_20260812.md
2. REMOTE_CODEX_INIT.md
3. START_HERE_FOR_AGENTS.md
4. docs/学习导航/README.md
5. docs/学习导航/01_目录与权威等级.md
6. docs/实验结果/周老师问题_证据矩阵_20260801.md
7. docs/实验结果/E201_txpert_multitarget_retraining_20260802/FORMAL_TRAINING_FREEZE.md
8. docs/实验结果/E201_txpert_multitarget_retraining_20260802/EXECUTION_CHECKPOINT_20260812.md
9. docs/实验结果/E201_txpert_multitarget_retraining_20260802/TARGET_RELEASE_AND_EVALUATION_FREEZE.md
10. docs/实验结果/GATE_STATUS_20260729.md
11. docs/投稿准备/期刊与文献定位_20260729/README.md
12. docs/实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md
13. docs/实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md
14. docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md
15. docs/实验结果/E191_certificate_decision_utility_20260729/reports/E191_INTERPRETATION.md
16. docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729/final_evaluation/reports/E190_INTERPRETATION.md
17. docs/实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md

读完后请用中文给我输出五件事：
1. SafeConf 一句话定位；
2. 周老师的每个问题现在由什么证据回答；
3. E168–E194 为什么让主线收缩为预注册加权 family 证书；
4. 哪些结论不能夸大；
5. 为什么当前有投稿竞争力，却仍不能承诺期刊录用。
```

## 5. HTML 截图入口

当前证据和投稿判断直接打开：

```text
docs/投稿准备/录用判断与项目总账_20260713/index.html
```

这个页面是白底长页，左侧目录可跳转，适合阅读和分段截图。若要使用 7 月 9 日的 1440×810 组会单页，再打开：

```text
workspace/group_meeting_20260709_MAINLINE_WHITE/导师组会_问题驱动汇报逻辑_20260709.md
workspace/group_meeting_20260709_MAINLINE_WHITE/SafeConf_主线汇报_术语跳转增强版_20260709.html
```

截图模式会隐藏顶部导航栏。浏览器打开本地文件后，在地址后加：

```text
?capture=1#s01
```

示例：

```text
file:///你的本地路径/safeconf/workspace/group_meeting_20260709_MAINLINE_WHITE/SafeConf_主线汇报_术语跳转增强版_20260709.html?capture=1#s01
```

每页是 1440 × 810 白底，适合直接截图贴 PPT。

## 6. 当前组会材料怎么理解

主线：

```text
SafeConf 位于单细胞扰动预测之后，负责判断哪些预测更容易错、哪些最该先复核。
```

当前最稳结论：

- E176 在四个完整留出供体的 640 个评价靶点上得到模型对下界零违例、靶点簇同时覆盖 90.31%；
- E177 在独立研究的 50 个评价靶点上得到下界零违例、靶点簇覆盖 88%，属于外部边界结果；
- E178 显示 scGPT 与 GEARS 的误差相关为 0.975/0.992，主要是共享任务难度；模型分歧不能当作单模型置信度；
- E168/E172 两次全新靶点实验未确认 fixed SafeConf 相对 magnitude 的排序增量，旧主张已经停止；

- 七套数据已完成正式 scGPT–GEARS 验证，共 3,209 个测试任务和 6,418 条严格记录；
- E140 七数据 absolute 元分析支持 SafeConf 总体超过 disagreement；相对 magnitude 的区间跨 0；
- E139 冻结 Directional-SafeConf 在 Nadig 第七数据确认通过；
- E141 仅提供较弱的 PROGENy 通路误差联系；相对 magnitude 未闭环；
- E142 蛋白正交 gate 失败，E144 STRING/靶基因自身 gate 失败，不能包装成机制成功；
- E143 已完成 48 基因前瞻湿实验的功效、盲法、QC、候选模板和白底图，物理实验尚未执行；
- Santinha、Shifrut、Tian 和 Nadig absolute 的弱结果必须保留；
- E132 只支持 AURC 相对 disagreement 的稳定改善，固定 top-20% 捕获未闭环；
- E116 指向背景新颖度；E117 紧上界失败；E118 chemical 中 magnitude 更强；
- 当前具备投稿竞争力，不等于编辑和同行评审必然接收。

下一步建议：

```text
以 E173–E178 的“确定性下界 + 校准上界 + fail-closed 真值访问”作为当前主线；E140/E139 保留为 absolute/direction 历史证据，旧排序只作诊断，不再在已解封真值上调路由器。
```

## 7. 远程 Codex 不要做的事

- 不要把旧 archive 当成当前结论。
- 不要主动删除历史文件。
- 不要把 smoke 写成正式实验。
- 不要声称 SafeConf 已经证明对 GEARS、CPA、scGPT 全部普适。
- 不要在没有用户确认时新增大实验或改论文主张。

## 8. 快速自检命令

```bash
git status --short --branch
git log -1 --oneline
python3 - <<'PY'
from pathlib import Path
import re
p=Path('workspace/group_meeting_20260709_MAINLINE_WHITE/SafeConf_主线汇报_术语跳转增强版_20260709.html')
s=p.read_text(encoding='utf-8')
ids=set(re.findall(r'id="([^"]+)"', s))
hrefs=set(re.findall(r'href="#([^"]+)"', s))
print('slides:', s.count('class="slide"'))
print('term links:', s.count('class="term"'))
print('missing anchors:', sorted(hrefs-ids))
PY
```

期望：

```text
slides: 14
term links: 47
missing anchors: []
```
