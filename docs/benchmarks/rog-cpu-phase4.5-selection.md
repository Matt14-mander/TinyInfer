# ROG CPU — Phase 4.5 融合性能与内核选择收口

- 日期：2026-10-10（Asia/Shanghai）。
- 基线：`a43e1577eaa4e9255d1d840ef5e38d55b37555b5` 加本次实现；源码/二进制哈希见[完整明细](rog-cpu-phase4.5-selection-details.md)。
- 验收范围：ROG 上的静态 FP32 Prepared CPU、保守的 MSVC AVX2 自动选择。**不是任意形状、任意 ISA 的融合性能验收。**
- 默认模式由 Specialized 改为 Auto；显式 Legacy/Specialized/Fused 仍可用于比较。一般 Fused 仍为实验模式。

## 为什么采用选择策略

历史结果已经表明：支持融合并不代表融合更快。K=512、无 bias/ReLU、
SSE2 等路径不能由某些有 bias 场景的收益推导出收益。
本次先尝试把最终 K 块判断移出列循环，以编译期模板拆分最终/非最终块。
104 个 native 进程均通过数值检查，但多个场景性能下降，故撤回该改动。

| 场景 | 拆分实验 Fused / Specialized 配对 p50 speedup | 恢复原内核后同指标 |
| --- | ---: | ---: |
| m1_k128_n128_t0 | 0.937 | 1.040 |
| m16_k128_n128_t0 | 0.938 | 1.053 |
| m64_k128_n128_t0 | 0.931 | 1.033 |
| m16_k512_n512_t0 | 0.806 | 0.974 |

两轮是独立测量；绝对延迟不能直接归因于代码变化。表格是每轮内部的配对比值，
不据此断言特定寄存器/缓存停顿。失败实验及内核候选补丁保留，未进入最终实现。
最终算术内核、K 归约次序、打包格式和缓冲策略保持基线实现；优化落点是
**准备阶段选择合适的已有内核，并保留运行时回退**。

## 自动选择边界

共同要求：MSVC（非 clang-cl）AVX2、常量 packed RHS、FP32、K=N=128、
alpha=beta=1、无 transA，运行时 lhs 连续，bias 列步幅为 0 或 1。

| M | 激活 | bias 布局 | Auto 选择 |
| --- | --- | --- | --- |
| 1 / 64 | ReLU | row-vector | Fused |
| 16 | ReLU | scalar / row-vector / column-vector / full | Fused |
| 16 | None | row-vector | Fused |
| 其他组合 | 任意 | 任意 | Specialized / 原有动态 RHS 路径 |

transB 两种形式已测量。None bias、identity、非单位系数、零 K、尾部形状、
大 K、未测量 M、转置 lhs 和其他编译器/ISA 不自动融合。
运行时 bias 步幅变化或非连续 lhs 会按当前绑定回退，不缓存首次输入的布局。
步骤暴露 `epilogue_mode`、`automatic_epilogue` 和
`epilogue_selection_reason`；上下文计数记录实际派发，而非仅记录准备阶段候选。
其他 CPU 型号的收益未在本机结果中得到验证；该白名单不是通用最优或自动调优器。

## 验证与计时口径

- native 与 portable Release 完整测试各 **41/41**，最终耗时 3.65 / 5.30 秒。
- 默认 Auto、显式模式、转置、尾部、零尺寸、特殊值、非单位系数、动态 B、
  改变 bias 步幅及非连续 lhs 的回退均有覆盖。Release 保留断言。
- 26 场景 × 4 进程 × 2 构建 = **208 个最终基准进程**。
- 14 个选中目标/回退控制 × 4 进程 × 2 构建 = **112 个独立复测进程**。
- 三个 native p95 波动场景再独立检查 4 进程，共 **12 个进程**。
- 上述 **332 个最终/复测进程**全部通过数值、打包和派发计数检查。
- MSVC 19.43.34808，Intel Core i9-13980HX，Windows 11 10.0.26200；
  native SIMD width=8，portable width=1。portable 不是 SSE2 验证。
- 不宣称本次运行了 Mac、ARM NEON 或 ROG sanitizer；历史结果保留但不替代本次实机证据。

四种模式有独立计划/上下文，但图、packing、激活缓冲策略相同。
每进程 warmup=20、samples=100、repeats=10，先构建/测试，再顺序测量。
每 sample 轮换计时次序；四进程分别用 allocation-order=0/1/2/3。
准备、输入绑定、输出复制及检查在计时外；没有修改电源模式或固定 CPU 亲和性。
进程间时钟、调度与热状态波动仍存在，不能把分配顺序控制当作内核加速。

## 性能门槛

p50 使用每进程样本配对 speedup 的四进程中位数；大于 1 表示 Auto 更快。
5% 延迟回退门槛对应 speedup >= 1/1.05（约 0.952381）。
最终全场景最低中位数：native **0.954751**，portable **0.964262**；两种构建所有场景中位数通过。

独立复测的所有场景 p50 中位数也通过。native 选中目标在完整测量及复测中的
中位数均大于 1：speedup 增幅在完整测量约为 2.5%–13.7%，复测约为 0.7%–7.8%。
**不保证每个进程都加速或每个进程都小于 5% 回退**；小批量 transB=1
复测收益仅约 0.7%，四进程中只有两次 speedup >1，收益较弱，不能夸大。

历史三种回退控制（无 bias/ReLU、identity、动态 B）也没有重新触发门槛失败：
在两构建的完整/独立复测中，Specialized / Legacy 配对 p50 最低中位数
为 0.992962，Legacy p50 / Auto p50 quotient 最低中位数为 0.990745。
后者是各进程延迟分位数相除的聚合值，不伪称为新增的样本配对指标。

p95 单独使用 Specialized p95 / Auto p95 的四进程中位数，不是配对 p50，
也不能由 p50 比值推导。初次 native 的大矩阵、aligned KN 和 actor fixture
出现超过 5% 的 p95 中位数回退；独立复查如下，未重复出现中位数门槛失败。

| p95 复查场景 | 原全场景配对 p50 | 原 p95 quotient | 独立复查配对 p50 | 独立复查 p95 quotient |
| --- | ---: | ---: | ---: | ---: |
| m64_k512_n512_t0 | 0.988 | 0.923 | 0.990 | 1.094 |
| m3_k128_n128_aligned_kn | 0.955 | 0.942 | 1.020 | 0.985 |
| rl_actor_mlp_tanh | 1.003 | 0.943 | 1.003 | 1.045 |

所有 14 场景独立复测的 p95 中位数也通过 5% 门槛。
进程级 p50/p95 离群点没有删除；见完整明细，不能把中位数通过表述为“所有进程无回退”。

## 26 场景最终结果

延迟为四进程 p50 中位数（微秒）；speedup 独立聚合每进程样本配对比值，
所以显示延迟相除不一定等于 speedup。

| 场景 | native Auto fusion | native Spec p50 | native Auto p50 | native paired speedup | portable Spec p50 | portable Auto p50 | portable paired speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | yes | 1.325 | 1.210 | 1.137 | 6.293 | 6.258 | 0.994 |
| m1_k128_n128_t1 | yes | 1.330 | 1.290 | 1.026 | 5.393 | 5.478 | 0.964 |
| m16_k128_n128_t0 | yes | 13.675 | 13.137 | 1.047 | 80.765 | 80.920 | 1.001 |
| m16_k128_n128_t1 | yes | 12.900 | 12.495 | 1.050 | 87.100 | 88.295 | 0.994 |
| m64_k128_n128_t0 | yes | 52.927 | 50.523 | 1.047 | 341.087 | 344.283 | 1.000 |
| m64_k128_n128_t1 | yes | 52.850 | 50.472 | 1.052 | 330.377 | 329.880 | 1.002 |
| m1_k512_n512_t0 | no | 19.405 | 19.172 | 1.005 | 100.740 | 101.453 | 0.999 |
| m1_k512_n512_t1 | no | 22.017 | 22.238 | 0.996 | 97.740 | 97.588 | 1.000 |
| m16_k512_n512_t0 | no | 225.053 | 228.113 | 0.993 | 2576.882 | 2545.870 | 0.998 |
| m16_k512_n512_t1 | no | 230.397 | 227.545 | 1.004 | 2575.895 | 2619.360 | 1.002 |
| m64_k512_n512_t0 | no | 983.783 | 996.875 | 0.988 | 9847.222 | 9923.185 | 1.006 |
| m64_k512_n512_t1 | no | 993.290 | 1010.010 | 1.006 | 7569.190 | 7329.780 | 1.012 |
| m3_k127_n131_t0 | no | 2.975 | 2.970 | 1.000 | 15.230 | 15.330 | 0.999 |
| m3_k127_n131_t1 | no | 3.415 | 3.407 | 1.002 | 15.058 | 14.992 | 0.999 |
| m16_k128_n128_bias_none | no | 14.070 | 14.282 | 0.993 | 79.737 | 79.955 | 1.002 |
| m16_k128_n128_bias_scalar | yes | 12.930 | 12.740 | 1.035 | 88.360 | 88.000 | 1.003 |
| m16_k128_n128_bias_row | yes | 13.328 | 13.085 | 1.025 | 85.708 | 85.982 | 1.000 |
| m16_k128_n128_bias_column | yes | 13.352 | 12.740 | 1.048 | 92.570 | 92.713 | 1.003 |
| m16_k128_n128_bias_full | yes | 13.878 | 13.692 | 1.035 | 100.558 | 99.695 | 1.004 |
| m16_k128_n128_no_relu | yes | 13.512 | 12.838 | 1.061 | 100.282 | 102.377 | 1.000 |
| m16_k128_n128_no_epilogue | no | 12.040 | 12.363 | 0.987 | 94.435 | 92.523 | 0.999 |
| m16_k128_n128_dynamic | no | 197.410 | 198.058 | 1.003 | 263.315 | 263.560 | 1.001 |
| m3_k127_n128_aligned_n | no | 2.882 | 2.942 | 0.977 | 16.697 | 16.750 | 1.002 |
| m3_k128_n128_aligned_kn | no | 3.063 | 3.175 | 0.955 | 17.192 | 17.023 | 1.008 |
| phase3_mlp_gemm | no | 0.935 | 0.935 | 1.000 | 1.095 | 1.085 | 1.004 |
| rl_actor_mlp_tanh | no | 1.710 | 1.710 | 1.003 | 2.365 | 2.350 | 1.004 |

## 独立目标/控制复测

| 场景 | native paired p50 speedup | native p95 quotient | portable paired p50 speedup | portable p95 quotient |
| --- | ---: | ---: | ---: | ---: |
| m1_k128_n128_t0 | 1.078 | 1.048 | 1.034 | 1.039 |
| m1_k128_n128_t1 | 1.007 | 1.000 | 1.034 | 1.029 |
| m16_k128_n128_t0 | 1.043 | 1.176 | 1.001 | 0.988 |
| m16_k128_n128_t1 | 1.055 | 1.034 | 1.001 | 1.001 |
| m64_k128_n128_t0 | 1.030 | 1.050 | 0.996 | 0.976 |
| m64_k128_n128_t1 | 1.031 | 0.978 | 0.998 | 0.991 |
| m16_k128_n128_bias_none | 0.999 | 1.003 | 0.998 | 0.990 |
| m16_k128_n128_bias_scalar | 1.048 | 1.005 | 0.999 | 1.023 |
| m16_k128_n128_bias_row | 1.038 | 1.081 | 1.002 | 1.021 |
| m16_k128_n128_bias_column | 1.049 | 1.101 | 0.998 | 1.003 |
| m16_k128_n128_bias_full | 1.023 | 1.043 | 1.001 | 1.003 |
| m16_k128_n128_no_relu | 1.064 | 1.086 | 0.998 | 0.974 |
| m16_k128_n128_no_epilogue | 0.991 | 0.998 | 0.999 | 0.982 |
| m16_k128_n128_dynamic | 1.002 | 1.001 | 0.999 | 1.011 |

## 内存和未采用的实验

packing 数量、packed-weight payload 和 activation payload 未因 Auto 改变；
这不是新的内存优化。完整明细逐场景列出这些数字，不等同于进程 RSS 或
完整准备状态的对象/allocator 开销；四模式实验同时保留四套独立状态。

本次总共保留 **583 个已完成且数值检查通过的进程**：
失败循环实验 native 104、恢复原内核的首轮 native 104、
首轮 portable 已完成 43（在最终回退保护加入后中止，被最终全量替代），
最终/独立复测 332。中止轮不是完整 portable 验收，不对未完成进程作成功声明。
只有最终 332 个进程用于本次验收。原有 Phase 4.5 修复报告与失败历史未覆盖。

## 复现

使用 Visual Studio x64 开发环境，按[先前构建说明](rog-cpu-phase4.5-fusion.md#reproduction)
创建 native/portable Release，改用 `build-phase45-selection-native` /
`build-phase45-selection-portable`。已有目录建议 clean-first，避免本机已观察到的
头文件依赖识别问题；本次完成干净重建并在最终增量更新后重跑完整测试。

```powershell
python benchmarks/run_phase45_fusion.py --executable build-phase45-selection-native/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5-selection-final/rog-native --warmup 20 --samples 100 --repeats 10 --runs 4
python benchmarks/run_phase45_fusion.py --executable build-phase45-selection-portable/benchmarks/tinyinfer_cpu_performance_analysis_benchmark.exe --output benchmark-results/phase4.5-selection-final/rog-portable --warmup 20 --samples 100 --repeats 10 --runs 4
```

分别加 `--recheck-only`，输出到独立目录，得到每构建 56 个目标/控制复测进程。
`--controls-only` 只复测三种回退控制。进程数必须是四的倍数；
CLI allocation-order=4 和 runner runs=3 均被拒绝。

额外 p95 复查分别使用 `--m 64 --k 512 --n 512`、
`--m 3 --k 128 --n 128` 和 `--model tests/fixtures/rl_actor_mlp_tanh.onnx`，
每场景顺序执行 allocation-order=0/1/2/3，其余参数同上。

原始 JSON/环境/哈希、最终实现补丁：`benchmark-results/phase4.5-selection-final`；
诊断轮：`benchmark-results/phase4.5-selection-provisional`、
`benchmark-results/phase4.5-selection`（均为 ignored 本机产物）。
最终实现补丁以 a43e157 为基线。Markdown 明细提交后可长期保留全部过程统计；
诊断轮源码在后续开发中变化，不把其元数据当作最终源码验收证据。

## 收口结论

Phase 4.5 的已测量 FP32 CPU 优化与保守内核选择完成 ROG 验收。
未接受的通用 Fused 不自动启用；SSE2/NEON/其他形状与编译器的融合调优
转为后续扩展，不标记为已解决。多行 tiling、线程、Int8 不属于本次实现。
