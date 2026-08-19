# safeconf-audit（最小审计包，v0.1.0）

目的：投稿所需的"别人能否复现"门槛——**一条命令**独立复算 SafeConf 已冻结的
E199/E200 主数字，并核对结构性不变量。E201 释放后自动纳入。

## 安装与运行

```bash
cd code/safeconf_audit
pip install -e .
safeconf-audit --repo /home/yyf/proj
# 或不安装直接：
python3 -m safeconf_audit.verify --repo /home/yyf/proj   # 需在包目录下
```

## 验证内容（与冻结报告逐项对照）

- E199（K562 未见基因，263 主任务）：diversity 下界与 predicted magnitude 的
  Spearman 点估计（容差 5e-4）；恒等式残差 ≤1e-10；下界违反数=0；
  簇 bootstrap CI 的**符号一致性**（分歧 CI 排除 0、幅度 CI 跨 0）。
- E200（K562 整背景留出，566 主任务）：transfer_risk / predicted_magnitude /
  source dispersion 的 Spearman 点估计；transfer risk CI 排除 0。
- E201：盲测未释放时报 `pending(blind)`；释放后核验状态文件存在。

## 诚实性说明（写进论文 Methods 的口径）

点估计与结构不变量做**数值等值**检查；bootstrap CI 因原始 RNG 种子不属于冻结
内容，做**符号一致性**检查（区间是否排除 0 的结论必须一致），不做逐位相等。

## 状态

- 2026-08-17 GLM 编写并在本仓库通过全部检查（见 agents/glm/06_EVIDENCE_AUDIT.md）。
- **未提交 Git**——等作者确认后随 E201 封存物一起或单独提交。
