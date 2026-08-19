# E83/E84 CPA 环境记录

- CPA source：`/home/yyf/archive/external/cpa`
- source commit：`fbd7c0250edc23eff003a10c99655579c53afd63`
- package：CPA 0.8.8
- predictor environment：`/home/yyf/.conda/envs/cpa_runtime_env`
- PyTorch 2.1.2+cu118，scvi-tools 0.20.3，Scanpy 1.10.3，Anndata 0.10.8
- RDKit 2022.09.5，adjustText 1.3.0

CPA 的 `__init__.py` 默认导入 Ray tuner，核心训练本身不需要 Ray。本地只做了一个兼容补丁：Ray 未安装时跳过 `run_autotune` 导入；`CPA`、`CPAModule`、训练和预测代码未改。补丁保存在 `tools/patches/cpa_0_8_8_optional_ray.patch`。

兼容环境高于 CPA 0.8.8 的部分官方依赖上限，因此 E83–E89 先按兼容环境证据保留。官方依赖环境现已建立：

- pinned environment：`/home/yyf/.conda/envs/cpa_env`
- CPA 0.8.8，PyTorch 2.0.0+cu117，scvi-tools 0.20.3
- Lightning 2.2.5，Scanpy 1.10.3，Anndata 0.9.2
- NumPy 1.23.5，SciPy 1.12.0，scikit-learn 1.6.1
- Ray 2.9.3，PyArrow 14.0.2，CUDA 可用，两块 GPU 均可识别

首次按包声明安装得到 PyArrow 21.0.0，与 Ray 2.9.3 冲突：Ray 使用了已删除的 `PyExtensionType`。将 PyArrow 固定为 14.0.2 后，CPA、scvi、Ray 与 CUDA 均通过导入核验。

E94 随后在 pinned 环境复跑 E83 的同一 manifest、随机种子、细胞抽样、网络与 10 epochs。59 个任务的 disagreement 排序和 pair-mean error 排序与兼容环境均为ρ=1.000；disagreement 与 error 的内部相关两边同为ρ=.901。disagreement 与 pair-mean error 的中位绝对数值变化分别为 3.29e−5 与 2.36e−5。E94 只证明环境再现性，不作为新的独立测试证据。
