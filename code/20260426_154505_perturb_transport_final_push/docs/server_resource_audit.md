# 服务器资源审计

生成时间：2026-05-21 16:43:37

## 1. 当前服务器基本信息

- hostname: `csb-NF5280M5`
- 当前用户: `yyf`
- 当前工作目录: `/home/yyf`
- 操作系统: `Linux csb-NF5280M5 6.8.0-45-generic #45~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed Sep 11 15:25:05 UTC 2 x86_64 x86_64 x86_64 GNU/Linux`
- Python:

```text
Python 3.12.4
/home/miniconda/bin/python
```

- conda / mamba 环境:

```text
# conda environments:
#
base                  *  /home/miniconda
scgpt_env                /home/yyf/.conda/envs/scgpt_env

mamba:
/bin/sh: 1: mamba: not found
```

- CUDA / nvcc:

```text
/bin/sh: 1: nvcc: not found
```

说明：`nvidia-smi` 显示 CUDA runtime 12.2；`nvcc` 命令没找到，说明当前 PATH 下没有 CUDA 编译器。

## 2. CPU 信息

```text
架构：                                x86_64
CPU 运行模式：                        32-bit, 64-bit
Address sizes:                        46 bits physical, 48 bits virtual
字节序：                              Little Endian
CPU:                                  96
在线 CPU 列表：                       0-95
厂商 ID：                             GenuineIntel
型号名称：                            Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz
CPU 系列：                            6
型号：                                85
每个核的线程数：                      2
每个座的核数：                        24
座：                                  2
步进：                                7
CPU 最大 MHz：                        4000.0000
CPU 最小 MHz：                        1000.0000
BogoMIPS：                            4800.00
标记：                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req vnmi pku ospke avx512_vnni md_clear flush_l1d arch_capabilities
虚拟化：                              VT-x
L1d 缓存：                            1.5 MiB (48 instances)
L1i 缓存：                            1.5 MiB (48 instances)
L2 缓存：                             48 MiB (48 instances)
L3 缓存：                             71.5 MiB (2 instances)
NUMA 节点：                           2
NUMA 节点0 CPU：                      0-23,48-71
NUMA 节点1 CPU：                      24-47,72-95
Vulnerability Gather data sampling:   Mitigation; Microcode
Vulnerability Itlb multihit:          KVM: Mitigation: VMX disabled
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Mitigation; Clear CPU buffers; SMT vulnerable
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Mitigation; Enhanced IBRS
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI SW loop, KVM SW loop
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Mitigation; TSX disabled
```

简要判断：双路 Intel Xeon Gold 6240R，总计 96 logical CPU。适合 CPU 小实验、多 seed 轻量实验、CSV/AnnData metadata 审计。

## 3. 内存信息

```text
total        used        free      shared  buff/cache   available
内存：      125Gi        10Gi        44Gi        13Mi        70Gi       114Gi
交换：      8.0Gi       2.0Gi       6.0Gi
```

简要判断：总内存约 125GiB，可用约 114GiB。适合读取中小 h5ad 的表达矩阵子集；不建议一次性把所有大 h5ad 全部 dense 化。

## 4. GPU 信息

```text
Thu May 21 16:43:37 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.06             Driver Version: 535.183.06   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Quadro RTX 6000                On  | 00000000:3B:00.0 Off |                  Off |
| 33%   37C    P8               4W / 260W |      8MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Quadro RTX 6000                On  | 00000000:AF:00.0 Off |                  Off |
| 33%   30C    P8               4W / 260W |      8MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   2523053      G   /usr/lib/xorg/Xorg                            4MiB |
|    1   N/A  N/A   2523053      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```

简要判断：2 张 Quadro RTX 6000，每张 24GB 显存。审计时 GPU 基本空闲。当前任务要求“不训练”，所以这次没有占用 GPU。

## 5. 磁盘信息

```text
文件系统        大小  已用  可用 已用% 挂载点
tmpfs            13G  5.3M   13G    1% /run
efivarfs        512K   75K  433K   15% /sys/firmware/efi/efivars
/dev/sda2       1.8T  150G  1.5T    9% /
tmpfs            63G     0   63G    0% /dev/shm
tmpfs           5.0M  4.0K  5.0M    1% /run/lock
/dev/sda1       511M  6.1M  505M    2% /boot/efi
/dev/sdb1        22T   17T  3.8T   82% /home
tmpfs            13G   80K   13G    1% /run/user/128
tmpfs            13G  240K   13G    1% /run/user/1020
```

项目/数据目录大小：

```text
项目目录: 62M	/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push
/home/yyf/datasets: 132G	/home/yyf/datasets
/home/yyf/codex_cout: 77M	/home/yyf/codex_cout
/home/yyf/codex_archive: 20G	/home/yyf/codex_archive
```

项目所在 `/home` 盘约 22T，总体已用约 82%，剩余约 3.8T。空间还够做 confidence MVP，但不建议继续无计划下载大数据。

## 6. 当前后台任务

```text
no server running on /tmp/tmux-1020/default
```

没有检测到正在运行的 tmux session。也就是说当前不是“实验还挂在后台”的状态。

## 7. 运行成本判断

- CPU 小实验：适合。96 线程 + 充足内存，足够做 PredictionRecord、risk/error correlation、risk-coverage、bootstrap/多 seed 小规模验证。
- GPU 深度模型：硬件适合，2 × 24GB；但当前不建议一上来训练，因为导师建议是重新定义 confidence/risk task，而不是继续换模型。
- 多 seed / 多数据集：适合做轻量 CPU 版；GPU 多模型训练要谨慎排队。
- 资源有限时优先级：先跑 CPU confidence MVP，先证明“confidence/risk 和真实 error 有相关性”，再考虑 GPU 模型。
