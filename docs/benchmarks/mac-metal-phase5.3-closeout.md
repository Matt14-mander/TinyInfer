# Mac Metal Phase 5.3 收尾报告

日期：2026-10-20（Asia/Shanghai）。**原冻结目标全部性能门槛通过。**

## 实现与测量范围

使用已有传输优化版本，不修改 shader、执行计划、同步或精度。本次工具修正仅使记录日期取实际上海日期，并补充 `kernels.h` 的源码哈希。测试时源码基线和构建缓存均保存在原始元数据中。

硬件：Intel Iris Plus Graphics 645，Intel x86_64 Mac，macOS 15.8.1，SDK 15.5，Apple Clang 17。Release，native CPU Auto，Metal 必须可见。

固定目标：静态 FP32 Gemm + vector bias + ReLU，M=128、K=N=1024，alpha=beta=1，transA=transB=0。输入在主机 → 完成推理 → 独立主机输出的 warm E2E；包括绑定、复制、同步及临时输出销毁。每次测量结果在计时外核对，门槛保持 `1e-5*(1+abs(reference))`。

两轮各三个独立进程顺序执行；每进程五次预热、31 个采样，CPU/GPU 次序交替。性能运行关闭 Metal 调试/分析开关；正确性测试另开 API Validation。六进程全部保留，没有剔除慢样本或重选轮次。本次只有这一组正式性能复测。

## 冻结门槛

下表每项均为三个进程各自比值的中位数；不是先汇总延迟再相除。

| 轮次 | 目标 p50 加速比 ≥1.2 | 目标 GPU/CPU p95 ≤1.05 | CPU 回退 p50 比 ≤1.05 |
|---|---:|---:|---:|
| 1 | 1.2258 | 0.7906 | 1.0032 |
| 2 | 1.2338 | 0.7786 | 1.0023 |

## 目标实际延迟

单位 ms，各列分别为三个进程对应分位数的中位数。

| 轮次 | CPU p50 | Metal p50 | CPU p95 | Metal p95 |
|---|---:|---:|---:|---:|
| 1 | 25.482 | 20.178 | 34.817 | 27.528 |
| 2 | 23.786 | 19.278 | 35.615 | 29.031 |

## 控制工作负载

| 工作负载 | 第一轮 p50 加速比 | 第二轮 p50 加速比 |
|---|---:|---:|
| phase3_mlp_gemm | 0.0092 | 0.0098 |
| rl_actor_mlp_tanh | 0.0087 | 0.0073 |
| control | 0.8905 | 0.9531 |

两个微型 ONNX fixture 明显更适合 CPU；M=64/K=N=512 控制也未获得 p50 加速。通过的是预先冻结的目标，不能据此自动将所有模型切换到 Metal。

## 环境与历史比较

六进程均接通电源，进程结束时一分钟 load average 为 7.35–9.01。复测前已完成构建，正式采样期间本工作流不并行启动编译、测试或其他基准。系统仍有桌面任务；负载快照不能证明完全空闲，也不能证明任一尾部样本的成因。没有终止外部应用。

本次较早前记录得到更低的 CPU 和 Metal 延迟，原门槛通过；跨日期环境并未严格控制，因此不能把差值表述为新的代码优化收益，或认为之前 p95 根因已经查明。结果接近 p50 门槛，应继续保留环境、原始分布和硬件适用范围。

此前失败记录继续保留：[首次报告](mac-metal-phase5.3.md)、[传输优化及 p95 调查](mac-metal-phase5.3-transfer.md)。

## 验收与复现

完整正确性、CPU-only、宿主内存检查及收口边界见[验收说明](../acceptance/phase5.3-closeout.md)。ASan/UBSan 不检测 shader 内存，当前未覆盖 Apple Silicon 和设备故障注入。

构建配置沿用[Phase 5.3 协议](../plans/phase5.3.md)，本机 SDK include 处理见[5.1 验收](../acceptance/phase5.1.md)。

```bash
python3 tools/run_phase53.py \
  --binary build-bench/benchmarks/tinyinfer_metal_model_benchmark \
  --output phase53-results-new --samples 31 --warmup 5
```

输出目录必须不存在；在可访问 GPU 的本机终端运行。

- [六进程原始汇总和全部门槛](../acceptance/phase5.3-results-closeout-baseline/summary.json)
- [硬件、源码/二进制哈希、构建缓存](../acceptance/phase5.3-results-closeout-baseline/metadata.json)
- 每进程 JSON 包含原始样本、独立阶段计时、驻留诊断及内存计数；同目录保留 stderr、供电和负载记录。
- 逻辑设备 payload、staging 缓存与进程 peak RSS 口径沿用原协议，不将其解释为精确进程显存峰值。
