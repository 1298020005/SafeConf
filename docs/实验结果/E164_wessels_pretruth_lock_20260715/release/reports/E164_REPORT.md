# E164 Wessels test truth解封前锁

## 授权结果

- baseline arm：已冻结并授权；
- E163 validation gate：通过；
- PRESCRIBE label-only forward：已执行；
- seed3407 raw score gate：True；
- PRESCRIBE arm最终授权：True。

E163三seed validation rho为3407=0.16173913043478258、3408=0.042608695652173914、3409=-0.22608695652173913。E163只是validation-informed futility diagnostic，不是外部确认。

## 不改写的失败

E162正式phase仍是`failed_main_validation_nondegeneracy_gate_no_test_label_query`。三个validation prediction都只有一个exact vector。E164只在新的、test truth仍封存的合同下锁定raw score；没有把E162失败改写为通过。

## baseline与Systema

E162b四个预锁预测器原样进入五预测器层级；新增`condition_balanced_perturbed_mean`由71个train非control condition先分别求细胞均值，再对condition等权。它与E162b cell-weighted mean的2023基因RMS差为0.015071311910261734。该值只来自11,779个train rows。

## 访问边界

raw Wessels文件未打开；test、validation和excluded X rows均为0；test truth/effect/error/DE和test graph均为0。若执行PRESCRIBE，输入只有48个condition字符串及E161 train-control mean。E165评价规则已在`E165_EVALUATION_SPEC.json`冻结，本阶段没有任何test endpoint。
