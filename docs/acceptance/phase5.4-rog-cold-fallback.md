# Phase 5.4 — CUDA 静态 CPU 回退冷启动修复

日期：2026-10-18（Asia/Shanghai）；基线 `23dcc59` 加本轮工作区实现。
结论：已知静态不支持的模型提前拒绝/回退，不再创建设备 runtime/stream/cuBLAS handle；
ROG 的独立回退 prepare 中位数 **34.15 µs**，六进程范围 **28.1–36.3 µs**。
原始固定性能门槛复测仍全部通过。Mac Metal Phase 5.3 状态不变。

## 根因与修复

旧路径为 Model 快照/MemoryPlan → 枚举 CUDA → 创建 runtime/handle →
模型能力选择 → 发现 GELU 不支持 → 销毁 CUDA 资源 → Prepared CPU。
[原始报告](../benchmarks/rog-cuda-phase5.4.md)中，fallback 是进程首个模型时
prepare 为 75.70 / 78.53 / 102.77 ms；即使最终全模型在 CPU，仍付出首次 CUDA 初始化开销。

新路径先执行 SDK-free 静态否定检查：按节点调用现有 operator contract，
同时检查零节点 identity/constant 等端点的 dtype/rank/int32 尺寸约束。
静态检查不调用 CUDA，不进行设备枚举，也不产生可执行 CUDA placement。

- 静态不支持 + Strict：直接带节点/值原因拒绝。
- 静态不支持 + WholeModelCpu：先验证整个 CPU 模型，创建 Prepared CPU，保留 requested
  device、静态节点诊断和 fallback reason，不伪装成设备不可用。
- 静态检查通过：仍进行原来的真实设备发现、runtime 创建、native capability/物理容量
  检查、常量上传与 context 分配。正向静态结果不代表 GPU 可用。
- runtime 输入布局回退、动态设备条件、资源分配与 CUDA 执行错误行为不变；
  后者仍不自动 CPU 重跑。没有修改 kernel、数值阈值、warm 计时范围或 arena 等待策略。

## 新增契约证明

`cuda_static_fallback_test.cpp` 在 fresh process 中验证：

1. `CUDA_VISIBLE_DEVICES=-1` 下，Add→GELU 混合图仍整体正确走 CPU，无 GPU skip；
   Strict 和无效 ordinal 都优先报告静态不支持节点，保留两个节点的能力诊断。
2. 使用 Driver API 的 cuInit + cuDevicePrimaryCtxGetState，先确认 primary context inactive；
   完成静态拒绝、回退准备/多 context 推理、rank5/空维超 int32 端点回退后仍 inactive。
3. 同进程的受支持 Add→ReLU 正向控制实际激活 primary context，执行真实 GPU 推理；
   防止将整个 CUDA 路径无条件跳过而让回退测试虚假通过。

Driver API probe 会初始化驱动，本身不代表“没有任何驱动开销”；它仅证明没有创建 primary
context。独立冷启动计时进程不调用这个 probe，也不在 timer 前做 CUDA 枚举。
测试只证明无可见 GPU 的回退行为，不宣称已经验证无 NVIDIA 驱动/DLL 安装的所有部署情形。

## 独立冷启动诊断

六个顺序 fresh processes，全部隐藏 CUDA 设备，交替 wrapper-first / CPU-first；
仅测 64x512 GELU fallback，同一 harness，5 warmups /31 warm samples，
输入/输出所有权与原始协议一致。单进程 prepare 只有一次观测，因此报告 min/median/max，
不将六次 prepare 描述为稳定的 p95。

| 进程 | wrapper 先准备 | CPU prepare (µs) | wrapper prepare (µs) | warm E2E wrapper/CPU |
| --- | --- | ---: | ---: | ---: |
| 1 | 否 | 14.8 | 34.1 | 0.951 |
| 2 | 是 | 9.1 | 34.2 | 1.001 |
| 3 | 否 | 12.4 | 28.1 | 0.988 |
| 4 | 是 | 5.2 | 34.6 | 0.987 |
| 5 | 否 | 26.0 | 31.6 | 1.001 |
| 6 | 是 | 5.4 | 36.3 | 0.998 |

wrapper prepare 中位数 34.15 µs；仅 wrapper-first 的三进程为 34.2 /34.6 /36.3 µs。
本地预设诊断检查 <5 ms 全通过；这不是通用启动 SLA。
wrapper 仍有模型快照、MemoryPlan、静态检查和 CPU 备份准备成本，
不声称完全等于直接 CpuExecutionPlan 的 prepare。

旧值来自修复前保留的进程记录，新值来自本轮独立诊断，不是同时同负载的严格 A/B。
primary-context probe 和隐藏设备测试提供不依赖“单次时间变快”的机制证据。
API timer 不含 Windows 进程启动、CUDA/cuBLAS DLL 装载；修复不是移除 native target 的动态库依赖。

## 原固定性能门槛复测

沿用[冻结协议](../plans/phase5.4-performance.md)，另跑两轮各三个独立进程，
保留全部五模型、31 warm samples，不以 cold-only 子集替代正常 GPU 验收。

| 轮次 | Target p50 CPU/GPU | Target p95 GPU/CPU | Fallback p50 wrapper/CPU |
| --- | ---: | ---: | ---: |
| 1 | 10.494 | 0.106 | 1.001 |
| 2 | 10.337 | 0.099 | 0.996 |

Target p50 >=1.2、p95 <=1.05、fallback warm p50 <=1.05，两轮均通过。
完整运行中 fallback prepare 六次为 21.2–37.9 µs。
GPU-first 顺序的 target 现在成为首个真正 CUDA 模型，其 prepare 为约
84–114 ms（其他顺序约 6–7 ms）：初始化成本由实际需要 GPU 的模型支付，
没有移到 CPU 回退或计时外隐藏。现有小 ONNX fixture 仍 CPU 更快。

## 验证

- CUDA ON / require-device 全套：51/51，0 skip；包含两个新增 fallback 测试。
- CPU-only / CUDA OFF：45/45。
- 统计脚本单元测试：4/4，新增 isolated fallback 的 schema/placement 检查。
- Prepared CUDA Compute Sanitizer memcheck/full leak：0 errors，0 leaked bytes。
- 六个 cold-only、六个完整 latency、一个独立 memory 进程均 exit=0，未剔除失败或慢样本。

[独立 cold 原始数据](phase5.4-cold-fallback-fixed/summary.json)、
[完整复测](phase5.4-results-rog-fallback-fixed/summary.json)、
[CUDA/协议回归](phase5.4-rog-cold-fallback-correctness.log)、
[CPU-only 回归](phase5.4-rog-cold-fallback-cpu-tests.log)、
[memcheck](phase5.4-rog-cold-fallback-memcheck.log)、
[验证指纹](phase5.4-rog-cold-fallback-verification.json)。
两组结果目录分别保留 metadata、二进制/源码哈希及原始 JSON/stderr；
完整复测还包含逐进程环境、build cache 和 memory-only 快照。

## 复现

与 `tools/build_cuda_rog.ps1` 相同的私有 SDK bin PATH / MSVC 14.29 环境中：

```powershell
cmake --build build-phase54-cuda --parallel 8
ctest --test-dir build-phase54-cuda --output-on-failure
E:/anaconda3/python.exe -B tools/run_phase54_cuda_cold_fallback.py --binary build-phase54-cuda/benchmarks/tinyinfer_cuda_model_benchmark.exe --output benchmark-results/phase5.4-cold-new
E:/anaconda3/python.exe -B tools/run_phase54_cuda.py --binary build-phase54-cuda/benchmarks/tinyinfer_cuda_model_benchmark.exe --output benchmark-results/phase5.4-postfix-new
```

基准需预先启用 TINYINFER_BUILD_BENCHMARKS=ON；runner 拒绝覆盖既有目录。
本轮仍是桌面共享负载，不是清空缓存/后台程序的专用实验。无设备丢失/OOM/hang 注入，
也未新增并行 context 或精确每进程 VRAM 峰值测量。
