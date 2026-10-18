# ROG CUDA Phase 5.4 — 性能与内存验证

日期：2026-10-18（Asia/Shanghai）；基线 `6698ccc` 加本轮基准/统计脚本。
**结论：本轮固定目标的性能门槛两轮均通过；仅接受初始静态 FP32 ROG 范围。**
微型 ONNX 模型仍明显 CPU 更快；本结果不是全模型 GPU 加速或自动选后端策略。
Mac Metal Phase 5.3 未达标状态不变。

## 协议与环境

开始测量前冻结[ROG 协议](../plans/phase5.4-performance.md)：同一五个跨后端 workload，
两轮、每轮三个顺序独立进程，每个 warm mode 5 次预热/31 次采样。
交替 backend 和 process/model 起始顺序，全部样本保留；nearest-rank ceil(n*p)，不插值。
CPU 与 GPU 相同 host lvalue 输入快照及独立 host 输出所有权。
每个 E2E 输出在计时外对照 CPU，保持 `1e-5*(1+abs(reference))`，没有放宽阈值。

ROG：i9-13980HX、RTX 4070 Laptop GPU 8188 MiB，Windows 11 10.0.26200、驱动 555.97。
Release native Prepared CPU Auto（MSVC 19.29 / AVX2）对照 CUDA 12.5.1 / cuBLAS 12.5.3.2，
sm_89、pedantic FP32、无 sanitizer/profiler。算子、Prepared 执行和等待策略未修改。
只添加 Windows CUDA 基准与独立统计/内存诊断工具。

电池 100%、BatteryStatus=2，当前 Performance 电源方案，未更改系统设置。
这不是空闲专用机：进程前 CPU LoadPercentage 为 8–39%，GPU 利用率 8–18%，
可用物理内存约 1.03–1.23 GiB；进程前后 GPU 温度 55–59°C。
这些是边界快照，不是运行期间的峰值。后台负载、WDDM、内存压力和调度会影响尾延迟；
全部环境记录保留，未停止用户应用或筛除受影响进程。

## 固定门槛

| 门槛（逐轮 process 比值中位数） | 第 1 轮 | 第 2 轮 | 结果 |
| --- | ---: | ---: | --- |
| Target E2E p50 CPU/GPU >=1.2 | 10.616 | 10.262 | 通过 |
| Target E2E p95 GPU/CPU <=1.05 | 0.099 | 0.102 | 通过 |
| Fallback p50 wrapper/CPU <=1.05 | 0.995 | 1.006 | 通过 |

Target 为 M=128/K=N=1024 Gemm+bias+ReLU，fallback 为 64x512 GELU 整模型 Prepared CPU。
加速比基于每进程比值的中位数，不是下面独立聚合 latency 再相除。

## 完整 E2E 结果

| 轮次 | 模型 | CPU E2E p50 (ms) | CUDA/wrapper E2E p50 (ms) | p50 加速比 | p95 GPU/CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | phase3_mlp_gemm | 0.0022 | 0.1164 | 0.020 | 52.042 |
| 1 | rl_actor_mlp_tanh | 0.0023 | 0.1146 | 0.020 | 50.826 |
| 1 | control | 1.2736 | 0.2844 | 4.578 | 0.269 |
| 1 | target | 10.7914 | 1.0130 | 10.616 | 0.099 |
| 1 | fallback | 0.4093 | 0.4150 | 1.005 | 0.969 |
| 2 | phase3_mlp_gemm | 0.0024 | 0.1158 | 0.021 | 62.435 |
| 2 | rl_actor_mlp_tanh | 0.0024 | 0.1183 | 0.020 | 55.174 |
| 2 | control | 1.2670 | 0.2974 | 4.221 | 0.300 |
| 2 | target | 10.8580 | 1.0547 | 10.262 | 0.102 |
| 2 | fallback | 0.4779 | 0.4810 | 0.994 | 0.987 |

E2E 包含输入 Tensor 快照、H2D、完成推理、D2H、独立输出和临时输出析构。
小模型仅约 2–2.4 µs CPU 计算/复制路径，CUDA/wait/传输固定开销使 E2E 约 110–120 µs；
不适合强制 GPU。control（M=64/K=N=512）本轮已有约 4.2–4.6 倍加速。
这两个合成规模不能拟合成通用交叉点，不能据此新增自动 GPU 选择策略。

## Target 阶段诊断

以下均为三进程 p50 的中位数，单位 ms；prepare/context/first 各进程只有一次观测，
不是独立冷启动百分位估计。

| 轮次 | 后端 | Prepare | Context | First E2E | Bind | Completed run | Host output | Resident-only | 独立 E2E |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | cpu_auto | 6.2099 | 4.9874 | 10.9061 | 0.5670 | 9.2974 | 0.5744 | 9.2173 | 10.7914 |
| 1 | cuda | 6.7497 | 0.0170 | 1.5201 | 0.6893 | 0.1830 | 0.1015 | 0.1966 | 1.0130 |
| 2 | cpu_auto | 6.9447 | 5.5045 | 11.6003 | 0.5689 | 9.4495 | 0.5662 | 9.2502 | 10.8580 |
| 2 | cuda | 7.8875 | 0.0341 | 38.5804 | 0.7267 | 0.1874 | 0.1033 | 0.2025 | 1.0547 |

Resident-only 包含 host dispatch/driver/event wait，不是纯 GPU kernel timestamp。
其约 0.20 ms 明显低于约 1.0 ms 的应用 E2E；主要剩余成本在 host 输入快照及上传边界。
Bind 与输出边界并非纯 PCIe 吞吐测量。各阶段在独立采样循环，**不能把阶段中位数相加替代 E2E**。

First E2E 的轮次差异保留：GPU-first 进程以反向模型顺序运行，target 是首个 cuBLAS 乘积，
而 CPU-first 顺序在 target 前已运行其他 CUDA 模型。驱动/库懒加载和缓存状态不同；
约 38.58 ms 的第 2 轮 first median 不应与约 1.05 ms warm E2E 混用。
这解释了测试顺序差异，但没有通过专项 profiling 将全部差异归因于某一库调用。
“Prepare”仅为该进程/模型的首次准备，未清空系统驱动缓存。

另一个明确的启动代价：当 fallback 是进程第一个模型时，wrapper prepare 为
75.70/78.53/102.77 ms，而直接 CPU prepare 约 0.007–0.011 ms；它作为后续模型时
wrapper prepare 约 0.81–0.87 ms。代码目前先创建 CUDA runtime/handle 再做模型能力选择，
即使 GELU 最终整体回 CPU，也会付出这项准备成本。本轮只冻结了 fallback **warm E2E**
门槛，不能把 warm gate 通过解读为 CPU fallback 冷启动无回退。下一步可单独评估
“已知静态不支持模型先做契约筛选”，本轮没有修改该路径。

后续已完成[静态 CPU 回退冷启动修复](../acceptance/phase5.4-rog-cold-fallback.md)：
先做 SDK-free 否定检查，六个隐藏设备的独立进程 prepare 为 28.1–36.3 µs；
primary-context probe 确认不激活 CUDA context。原固定 gate 另跑六进程仍全部通过，
target p50 加速 10.49x/10.34x。上面的旧值作为历史证据保留，不代表当前未修复状态。

## 资源与内存

Target CPU packed weights 为 4,194,304 B，arena 524,288 B。
CUDA 常量 4,198,400 B，input 524,288 B，arena 524,288 B；
四次逻辑 allocation 累计 5,246,976 B（约 5.004 MiB），pinned cache 4,194,304 B。
这些不包括源模型和快照、host 输入/输出、event、cuBLAS/驱动内部资源，不是总显存。

所有正式进程的 resident runs 没有新增 TinyInfer device allocations/uploads/downloads/copies，
warm 模式也没有新增 pinned staging 分配。每个合成 CUDA 模型 99 次真实 cuBLAS + 99 次后处理
kernel；小模型各 198 次 cuBLAS、396 次 kernel。fallback 为 0 CUDA 分配/传输/dispatch，
有效后端为 CPU，其 CPU activation 不包含在 CUDA 计数中。

六个 timing 进程的 Windows peak working set 范围 368.19–375.11 MiB，
包括 CPU 与 CUDA 两条路径及多个模型的历史高水位，不能据此比较单后端 RSS。
独立 memory-only 进程先初始化 CUDA context，再依次观测五个模型；
下表只展示 target，与前面模型共享驱动/库缓存，不能当作全新进程 target 峰值：

| 阶段 | 相对本模型初始设备 used 增量 (MiB) | 进程 working set (MiB) | 进程 private usage (MiB) |
| --- | ---: | ---: | ---: |
| initialized_baseline | 0.00 | 347.64 | 466.74 |
| prepared | 14.00 | 359.64 | 478.77 |
| one_context | 14.00 | 359.64 | 478.77 |
| first_completed | 14.00 | 360.52 | 479.78 |
| two_contexts_completed | 16.00 | 360.52 | 479.78 |
| contexts_destroyed | 14.00 | 360.52 | 479.78 |
| plan_destroyed | 0.00 | 348.51 | 467.75 |

设备级 free/total 来自 cudaMemGetInfo，含其他进程/WDDM 和分配粒度影响；
target 两 context 时相对阶段基线约 +16 MiB，销毁 plan 后该快照差值回到 0。
这不是精确每进程 VRAM 峰值，也不能单独证明无泄漏。已有
[Prepared memcheck/full leak 验收](../acceptance/phase5.4-rog-prepared.md)提供独立安全性证据。
内存诊断进程整体峰值约 360.52 MiB working set；private usage 约 479.78 MiB
仅是 target 完成阶段快照，不能等同物理 RAM 或 GPU 显存。

## 验证、复现与证据

- GPU require-device 正确性回归：49/49，0 skip。
- CPU-only 回归：45/45。
- 统计协议单元测试：3/3（nearest rank、逐轮 gate、拒绝坏样本/错误 placement/缺失计时）。
- 六个正式 latency 进程和一个独立 memory 进程均 exit=0；全部数值/placement/resource gate 通过。
- 冒烟（1 sample/1 warmup）仅验证 harness，不计入正式 gate；未做算子调优或选择性重测。

在与 `tools/build_cuda_rog.ps1` 相同的私有 SDK / MSVC 14.29 环境中：

```powershell
cmake -S . -B build-phase54-cuda -DTINYINFER_BUILD_BENCHMARKS=ON
cmake --build build-phase54-cuda --target tinyinfer_cuda_model_benchmark --parallel 8
E:/anaconda3/python.exe -B tools/run_phase54_cuda.py --binary build-phase54-cuda/benchmarks/tinyinfer_cuda_model_benchmark.exe --output benchmark-results/phase5.4-rog-new --samples 31 --warmup 5
```

Runner 不覆盖已有输出目录。基准使用 Windows GetProcessMemoryInfo，当前仅提供 WIN32 target。
非 Windows CUDA 后端不受影响，但尚未移植此内存测量 harness。

[全部样本与统计](../acceptance/phase5.4-results-rog/summary.json)、
[源码/二进制哈希、Git/build cache/环境](../acceptance/phase5.4-results-rog/metadata.json)、
[内存快照](../acceptance/phase5.4-results-rog/memory.json)、
[GPU 回归](../acceptance/phase5.4-rog-performance-correctness.log)、
[CPU-only 回归](../acceptance/phase5.4-rog-performance-cpu-tests.log)。
每进程的原始 JSON、stderr 与开始/结束环境均保留在同一 results 目录。

## 下一步

本轮 ROG 固定模型范围通过，可据此收口初始 CUDA 正确性与固定性能目标。
更广的模型/动态形状、并发 contexts、多 stream、低精度、自有优化 Softmax/矩阵 kernel、
精确每进程显存峰值均不包含在本次结论。若继续优化，应优先分析输入快照/上传边界，
并在更空闲、内存充足的环境补充尾延迟稳定性，而不是放宽目标。
Phase 5 整体仍有 Mac Metal Phase 5.3 性能门槛未通过。
