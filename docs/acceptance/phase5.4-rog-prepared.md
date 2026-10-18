# Phase 5.4 — ROG Prepared CUDA 模型与独立上下文验收

- 日期：2026-10-18（Asia/Shanghai）。基线：`4c83854` 加本轮工作区实现。
- 结论：初始静态 FP32 Prepared CUDA 模型正确性/所有权范围通过；Phase 5.4 性能尚未验收。
- 环境沿用[算子验收](phase5.4-rog-operators.md)：RTX 4070 Laptop GPU，sm_89，驱动 555.97，
  私有 CUDA 12.5.1 / NVCC 12.5.82、cuBLAS 12.5.3.2、MSVC 14.29，Release native。
  本轮未安装软件、更改驱动或持久环境变量。

## API 与资源归属

入口为 `tinyinfer/backend/cuda/execution_plan.h`，链接可选 `TinyInfer::cuda`。

| 对象/阶段 | 所有权与行为 |
| --- | --- |
| CudaExecutionPlan | 深拷贝 Model 快照、执行顺序、MemoryPlan 和 placement；实际使用的常量只上传一次 |
| 共享 plan | 共享只读设备常量及同一 runtime/stream/cuBLAS handle；复制 plan 不重复上传 |
| CudaExecutionContext | 独立输入设备分配、独立 activation arena、绑定状态/统计、可选独立 CPU context；可移动不可复制 |
| bind_input | 同步快照/上传；lvalue Tensor 的按值副本会连续化，移动非连续 view 则保留布局用于 Strict/回退判断 |
| run | 同步；按拓扑执行，arena slot 覆盖前等待前值最后消费者的 Completion，返回前完成全部工作 |
| output | 每次返回独立 host Tensor；GPU 路径在这里 D2H，已有结果副本不被后续绑定/运行覆盖 |

共享同一 plan 的所有 contexts 必须外部串行调用，**不支持并发调用/多 stream 并行推理**。
独立指可写输入和 activation 不共享，不表示线程安全。context 持有 plan 状态，plan wrapper
销毁后仍能运行；移动后的源 context 对操作报错。run 失败或成功重新绑定后不能读取旧结果；
在参数校验阶段拒绝的非法绑定不改变既有成功输出。

```cpp
#include "tinyinfer/backend/cuda/execution_plan.h"
tinyinfer::CudaExecutionPlan plan(model); // 默认 Strict
auto first = plan.create_context();
auto second = plan.create_context();
first.bind_input(plan.inputs()[0].first, input_tensor);
first.run();
auto independent_output = first.output(plan.outputs()[0].first);
```

初始模型范围沿用七类算子、FP32、连续设备布局、rank <=4、int32 尺寸界限。
零节点 identity、直接常量输出、空输出也检查值级契约，不能借没有计算节点绕过限制。
实际设备 allocation/transfer/launch 的所有权和错误语义仍由 CUDA runtime 管理。

## 整模型回退

`CudaExecutionPlan(model, {0, FallbackPolicy::WholeModelCpu})` 显式允许回退，CPU 为
`CpuExecutionPlan` 默认 Auto 策略；默认 Strict 不回退。

- 准备时设备不可用或能力不支持：验证整个 CPU 模型后选择 CPU，不上传 CUDA 模型常量。
  设备发现/runtime 初始化错误仅在显式允许回退时转为 unavailable 诊断。
- GPU plan 收到移动的非连续输入：本次 run 整体走 CPU，placement 含原因；重新绑定为连续输入后可回到 GPU。
- 不做逐节点混合回退。常量上传、context 分配、CUDA 提交/等待等资源或执行错误直接传播，
  不自动 CPU 重跑。没有注入真实 driver loss/OOM/hang，本轮不宣称验证了这些故障恢复。
- CPU context 直接接收已拥有的按值 Tensor：lvalue 一次快照，move 不额外复制。

## ROG 实测结果

| 检查 | 结果 |
| --- | --- |
| CUDA ON 完整 require-device suite | 49/49，0 skip；2.70 s（测试耗时，非推理性能） |
| CPU-only，CUDA OFF，MSVC 19.43 | 45/45；2.30 s，未引入 CUDA SDK 依赖 |
| Prepared CUDA memcheck/full leak | exit=0，0 errors，0 bytes leaked |
| 既有算子回归 | 包含原 1032 数值/模型链用例，继续通过 |
| 命令行模型示例 | 两个 ONNX fixture 均完成 3 次 GPU run |

专项测试分七组，均通过：

1. 两个 ONNX fixture：各 8 轮变化输入、双 context 交替、每轮再 warm run；CPU 对照、独立 golden、
   输出副本修改后重读、另一 context 的输出不变。阈值未放宽：`1e-5*(1+abs(reference))`。
2. 生命周期/契约：未绑定/未运行/非法 ID、shape、非法 fallback policy；Strict 拒绝非连续 view；
   arena 确实复用并等待 Completion；context move construction/assignment、plan wrapper 销毁后存活。
3. 端点：空输出、直接输入/常量输出、带 byte offset 的 narrow 输入、rank5/空维超 int32 界限拒绝与 CPU 回退。
4. 动态 Gemm：A/RHS/bias 连续 5 轮改变；绑定后修改原 Tensor 不改变结果；设备分配数稳定；
   非连续 RHS 导致整个 run 不提交 GPU 节点，重新绑定后回 GPU。
5. 分支多输出：共享中间值、提前输出、输入和常量端点输出；快照隔离源模型常量修改；
   copied plan 常量不重传；未使用常量不上传；保存的输出跨 rebind/run 不变。
6. K=0 Gemm/FusedGemmActivation：NaN 偏置、beta=0、signed zero 对照 CPU，确认 cuBLAS 调用数为零。
7. CPU 回退输入所有权：CountingAllocator 验证 lvalue 只快照一次、move 不额外复制。

warm run 前后比较 TinyInfer runtime 的 allocations/uploads/downloads/copies，均不增加；
input upload 位于 bind，output download 位于 output。此项不代表没有主机 bookkeeping 分配、
CUDA event 或 cuBLAS 内部分配，也不等于 RSS/显存峰值测量。

命令行示例使用确定性的 `input[i]=i/4`，准备计数如下；不是性能基准：

| 模型 | 常量上传数/字节 | 每 context activation arena | GPU runs |
| --- | --- | --- | --- |
| phase3_mlp_gemm.onnx | 4 / 68 B | 28 B | 3 |
| rl_actor_mlp_tanh.onnx | 4 / 148 B | 44 B | 3 |

activation 字节不含输入缓冲、共享常量、host snapshot、pinned cache、event 和库内部资源。
`CudaPlanStatistics.device_allocations` 是准备时快照，后续 context 分配见动态 device_statistics。

## 复现与证据

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_cuda_rog.ps1
```

构建后，在同样的私有 SDK bin PATH 中运行：

```powershell
build-phase54-cuda/tests/tinyinfer_cuda_execution_plan_tests.exe --require-device
build-phase54-cuda/examples/tinyinfer_cuda_model.exe tests/fixtures/phase3_mlp_gemm.onnx
build-phase54-cuda/examples/tinyinfer_cuda_model.exe tests/fixtures/rl_actor_mlp_tanh.onnx
compute-sanitizer.exe --tool memcheck --leak-check full --error-exitcode 1 build-phase54-cuda/tests/tinyinfer_cuda_execution_plan_tests.exe --require-device
```

Compute Sanitizer 位于私有组件的
`packages/cuda_sanitizer_api-windows-x86_64-12.5.81-archive/compute-sanitizer/`。
工具链部署与 CPU-only 构建参考[运行时记录](phase5.4-rog-runtime.md)。

原始证据：[CUDA suite](phase5.4-rog-prepared-ctest.log)、[CPU-only](phase5.4-rog-prepared-cpu-tests.log)、
[memcheck](phase5.4-rog-prepared-memcheck.log)、[examples](phase5.4-rog-prepared-examples.log)、
[源码/二进制指纹](phase5.4-rog-prepared-verification.json)。工作区源码哈希对 Git 换行转换敏感。

## 剩余阶段边界

下一步冻结 ROG 性能目标，再测冷准备/首跑、H2D/D2H、已完成驻留执行、独立 host-to-host E2E、
CPU Auto 对照、p50/p95 和实际内存。当前没有 GPU 加速比或性能达标结论；未跑 ROG ASan/UBSan、
racecheck、并行 context 测试或硬件故障注入。Mac Phase 5.3 未达标状态不变，Phase 5.4/Phase 5
不得仅因这次正确性验收标记整体完成。
