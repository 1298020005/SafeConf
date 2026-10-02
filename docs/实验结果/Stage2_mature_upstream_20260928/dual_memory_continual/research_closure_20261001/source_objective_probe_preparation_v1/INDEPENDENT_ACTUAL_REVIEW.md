# 独立实际审阅：首轮技术失败，目标函数假设尚未测试

## 终态与范围

实际输出 `/home/yyf/runtime_artifacts/safeconf_research_20261001/source_objective_probe_v1/approved_20261002_v1` 的 `STATUS.json` 为 **FAILED**，耗时1.446413秒。只完成 GAT→Exphormer 外折0的一个rank HGB（训练1446行/460gene），raw_affine拟合0次；20-fit上限未突破。没有 `SCORE_FREEZE.json`、整队列预测文件或三个metric输出。因此不存在可审阅的20-fit收益/全七指标结论，不能判定该假设得到DEV支持或被否定。

代码和实际注册仅允许20个固定 Source parquet：10个Learned13输入、10个Manual身份核对输入；共享主CSV在数值转换前筛选Source两方向/原seed/Learned_hgb。Manual/Learned逐任务顺序、CDF分组及truth相等检查发生在首次拟合前，Source模型和gene外折隔离不变。未读取McFaline/Orion/E170数值，不新增家族/上游调用，不发布或替换正式分数。25个输入/代码SHA在执行前后及本次复核时全部相同。

已计算的四个训练组affine审计只来自该折1446个允许训练行：全部斜率为正，加权均值/标准差匹配差分别不超过2.22e-16/5.55e-17。原公式在每CDF组匹配rank均值/标准差，故各组不同斜率会改变pooled loss有效权重；这不是全pool原始RMSE MSE的同义实现。该变换没有query-error输入，但尚未训练raw模型。

## 实际失败原因已闭合

按Root追加的只读诊断授权，加载保存的首折rank模型、对原362条Source query执行一次预测，没有重新拟合：

- rank分数与原归档最大差 **1.110223e-16**，所有行均低于程序既定1e-14门限，范围及原HGB参数一致。不是风险分数/错误cache路径导致失败。
- 失败来自同一个if内的truth比较：缓存 `true_error_rmse` 为float32，而原CSV短十进制被解析为float64，最大差 **3.7037468e-9**，超过truth比较的既定门限。报错文字将它笼统称为score replay失败。
- 将归档十进制按其原float32 codec还原后，362条truth与缓存**逐字节相同**，gap=0，两个数组SHA均为 `02d99398b7c8b4fa995568adbc547b9c4885969785d86fa93b32cbe7f342b30e`。例如AAMP缓存的float32转float64为0.04179440066218376，CSV文本为0.0417944；它们是同一float32观测的不同显示精度。

最低技术续接是只在归档truth身份复核中恢复已证明的原float32表示，继续保持risk门限/标签/公式/特征不变；不是放宽误差容差。旧执行代码、失败目录和1次控制拟合必须保留，新注册须明示续接及实际fit账本。本审阅未修改代码/批准文件，也未自行重跑。后续原主metric点复核仍须披露cached float32与CSV解析值的评价精度；不能把两种表示无说明地称为逐点bit复现。

## 科学决定

**当前应停在技术恢复，不能停止或接受科学假设。**理论上CDF条件期望排序可能与U20原始误差选择不同，首轮却没有形成raw-arm数据。证据支持一次窄codec修复，不支持增加目标/尺度/L2变体。即使续接成功，完整两方向/五折/七指标只能给出Source DEV目标设计证据，不能解释已见外部失败、更新原Gate或声称新增跨研究确认。

精确绑定：执行代码 `f77968fb5537c850b9812e58cd8d69eda904255babcec1a210eeddd3185be496`；REGISTRATION `7959e8ff1d0123dd4c51cbeac0e39b8b0d6eb90082863dac2f48b89e4d03ca50`；FAILED STATUS `5fea8cc4c8fd32cb3034068f1ec7b18bf8a24ba6b3406d0bcc92ed6969e13119`；保存rank模型 `37e378db03fe7949a1ffc8ed400b480fb6c6f580be26bccc10f08e3b8f271097`。本审阅新增拟合0、RNG0、外部评分0。

## V2独立实际审阅：技术通过，停止本固定目标变体

V1以上记录保留不变。Root授权的窄codec续接输出 `source_objective_probe_v1/approved_20261002_v2` 为 **COMPLETE_DEV_DIAGNOSTIC**，耗时5.784282秒。独立核验没有实质阻碍：

- 29个输入/代码/旧版本保护SHA执行前后及审阅时全部一致。旧首折rank模型在V2复制后仍为上述 `37e378…`；20个模型评价、19次新拟合、1次旧拟合/复用，**累计20拟合**。10组配对的训练record hash/行数/gene数相同，gene外折隔离成立。
- 10组Manual/Learned身份及float32真值逐值相等；独立复核3616个query的归档decimal恢复float32 bits全部一致。全部六个计分列的评价truth都精确采用原归档decimal float64；训练CDF/affine仍用原缓存float32，未把codec修复混入训练标签。10折risk复现最大差1.11e-16，仍通过原1e-14门限。
- 40个affine训练组的斜率/截距按实际允许训练行独立复算，差0；每组正斜率和加权moment匹配，不含query误差。10组两臂preprocessor参数逐字节相同；20个保存模型的HGB结构/参数/seed一致。组间斜率造成loss权重差异的限制继续成立。
- 20个model SHA及21696行score文件SHA精确匹配 `SCORE_FREEZE`；程序顺序和文件时间均显示封存先于metric计算。每方法每方向均完整1808 tasks/575 genes/四context，无选择子集。48个context、12个macro、240个fold-context的全部七指标从保存score复算差0；独立counterpoint验证最大差1.11e-15。原四个参考方法的56个Source macro单元复现差0。

### 全七指标固定点差（raw_affine−rank_unbounded）

| 指标 | Exphormer→GAT | GAT→Exphormer |
|---|---:|---:|
| U20（越高越好） | -0.008287 | -0.003490 |
| Spearman（越高越好） | -0.002511 | -0.000913 |
| AURC（越低越好） | -0.000139 | -0.000104 |
| high-risk miss（越低越好） | +0.014800 | -0.003673 |
| error@10（越低越好） | -0.001117 | -0.000895 |
| error@20（越低越好） | -0.000329 | +0.000020 |
| error@50（越低越好） | +0.000069 | +0.000151 |

两方向rank U20为0.783134/0.783743，raw-affine为0.774848/0.780253；Magnitude仍为0.748808/0.746253，既有Source强基线不确定性不由这些新点值消除。40个fold-context U20差为14正/20负/6相同，仅作完整描述，不能把重复折/context当独立显著性检验。原rank有1条负输出，raw-affine有100条>1输出；原rank裁剪与不裁剪的全部七指标相同。新分数明确不具备自动部署的[0,1]合同。

**科学决定：该固定干预没有得到主U20一致收益的DEV支持，停止此目标变体和其尺度/损失/参数延伸。**低风险10%误差与AURC的小点改善保留，不能用secondary指标替代预定主判断。初轮按注册未做bootstrap，没有新CI，不能宣称统计显著退化、等价、理论错配不存在或已证明外部失败原因。这只是一个Source同家族两架构/四context、held-gene及held-predictor的DEV目标设计诊断，不是新的独立家族/研究确认；MC、Orion、E170以及正式模型/门限保持原证据定位。

V2绑定：执行代码 `6b6863f375bdf997f55fe320ae80ef11eeb6af907dd326b4089c042d14a95dc9`；REGISTRATION `929fea471afd514c1b69bb95ad1d9991b75dcc262537fa59fae1026164f16757`；COMPLETE STATUS `be2f23c848c97c65ac8ed94fe15ca8aff1733f619ebcca0ce88cc241165bc520`；SCORE_FREEZE `17514b553c28d56030677d2d9cfd60127adebfb73cab41fb37440f9d96f6769c`；保存score `9455199d03d832b390d86c33f86eebb340f1ff24e4d573fa91edea1402d66eaf`。独立代码复核子任务没有拟合/预测/写文件；本V2复核新增拟合0、RNG0、外部评分0。
