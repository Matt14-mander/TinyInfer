# Phase 5.3 — Mac Metal 收尾验收

日期：2026-10-20（Asia/Shanghai）。状态：**初始静态 FP32 Metal 范围完成，原冻结目标性能验收通过。**

## 收尾交付

- 复用已完成的 Metal 算子、Prepared Model/context、常量驻留、独立激活 arena、完成安全复用和显式整模型 CPU 回退。
- 真实 Intel Iris Plus Graphics 645 上按原两轮六进程协议复测；工作负载、数值容差和性能阈值全部不变。
- 更新基准脚本日期记录，并将 shader 源码纳入哈希；没有新增算法、修改 shader 或改变计时口径。
- 保留旧失败报告、当前六进程原始样本及环境；更新 README、架构、路线图和阶段状态。

## 性能门槛

| 轮次 | target p50 CPU/Metal ≥1.2 | target p95 Metal/CPU ≤1.05 | fallback p50 wrapper/CPU ≤1.05 |
|---|---:|---:|---:|
| 1 | 1.2258 | 0.7906 | 1.0032 |
| 2 | 1.2338 | 0.7786 | 1.0023 |

两轮每轮三个进程，均取进程比值中位数。固定 target 为 M=128、K=N=1024 的 FP32 Gemm/vector bias/ReLU。每进程预热 5 次、采样 31 次。所有测量输出均通过 CPU 数值对照，驻留/分配/传输资源检查通过。没有丢弃进程、样本或改选工作负载。

完整阶段延迟、控制模型和环境局限见[性能报告](../benchmarks/mac-metal-phase5.3-closeout.md)。小型 ONNX fixture 仍优先使用 CPU，当前没有根据本次测量新增自动 GPU 选择规则。

## 当前源码验证

| 检查 | 结果 | 证据 |
|---|---:|---|
| Release native CPU + Metal，API Validation 开启 | 48/48，无跳过 | [日志](phase5.3-closeout-release.log) |
| Metal Debug，宿主 ASan/UBSan，API Validation 开启 | 3/3，无跳过 | [日志](phase5.3-closeout-asan.log) |
| CPU-only Release，Metal/CUDA 均关闭 | 45/45，无跳过 | [日志](phase5.3-closeout-cpu.log) |
| 性能，两轮各三个独立实际 GPU 进程 | 三项门槛均通过 | [原始汇总](phase5.3-results-closeout-baseline/summary.json) |

初次沙箱测试因 GPU 不可见失败；随后在可访问 GPU 的主机环境运行并得到以上实际结果。不可见设备的失败不计入通过结果。测试数量与历史记录不同是因为当前仓库包含 CUDA SDK-free 契约和 serving 示例的回归项目。

性能采样先完成，再运行本次全套测试与 CPU-only 构建，避免本工作流并发负载干扰。测试命令：

```bash
MTL_DEBUG_LAYER=1 ctest --test-dir build-bench --output-on-failure
MTL_DEBUG_LAYER=1 ctest --test-dir build-phase51-metal-asan -R metal --output-on-failure
ctest --test-dir build-phase53-closeout-cpu --output-on-failure
```

实机 GPU 验收必须设置 `TINYINFER_METAL_REQUIRE_DEVICE=ON`。本机 SDK include 处理见[5.1](phase5.1.md)；性能重现及采样约定见[报告](../benchmarks/mac-metal-phase5.3-closeout.md)。

## 完成边界

5.0 设备契约、5.1 算子、5.2 Prepared 模型和 5.3 本机固定目标性能均收口。结合已经单独验收的 ROG 5.4，Phase 5 初始静态 FP32 双后端范围完成。

本次未扩展到 Apple Silicon、任意 ONNX、动态 shape、Metal Transformer/Attention/KV、FP16/Int8、多队列或并发 context、逐节点异构图分区。宿主 sanitizer 不检测 shader 内存；设备 hang/loss 注入仍未覆盖。无通用 GPU 加速或精确进程显存峰值声明。

当前测试环境仍有桌面负载；不能声称完全空闲，也不能将较历史结果改善归因于单一因素。旧 p95 波动根因调查保留其未能隔离单因的结论，本次通过仅收口原有数值门槛。

[结构化验收与文件哈希](phase5.3-closeout-verification.json) · [测量时环境与构建缓存](phase5.3-results-closeout-baseline/metadata.json)
