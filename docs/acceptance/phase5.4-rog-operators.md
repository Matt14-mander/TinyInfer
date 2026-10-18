# Phase 5.4 — ROG 正式 CUDA 算子首版验收

- 日期：2026-10-18（Asia/Shanghai）。基线：`cb08b3b` 加本轮工作区实现。
- 范围：初始七类 FP32 device 算子及 capability bridge；不是 Prepared CUDA 模型或性能收口。
- 工具链沿用[资源运行时验收](phase5.4-rog-runtime.md)：RTX 4070 Laptop GPU（sm_89）、
  驱动 555.97、隔离 CUDA 12.5.1 / NVCC 12.5.82、cuBLAS 12.5.3.2、MSVC 14.29、Release native。
  本轮未更改驱动、系统 SDK 或持久环境变量。

## 实现与语义

| 路径 | 实现及边界 |
| --- | --- |
| MatMul / Gemm / FusedGemmActivation | cuBLAS SGEMM，row-major 转 column-major 的 B/A 交换及转置映射；融合激活仅 ReLU |
| Gemm 后处理 | CUDA kernel 处理 alpha/beta、标量/行/列/完整偏置和 ReLU；独立乘加舍入，不启用 fast math |
| Add | rank <=4 广播及标量，自有 kernel |
| ReLU / Tanh | 自有逐元素 kernel；ReLU 对 NaN/零的行为对齐现有 CPU |
| Softmax | 支持负轴及非末轴；每个 reduction group 一个线程的正确性基线 |
| 执行入口 | CudaTensorView + CudaDeviceRuntime::execute；CudaBackend 提供能力桥接，拒绝 host Executor |

每 runtime 创建一个 cuBLAS handle 并复用，绑定已有独立 stream；设置 pedantic FP32、
host pointer mode、禁用 atomics，不请求 TF32/降精度。库乘积固定 alpha=1/beta=0，
用户系数在后处理应用，因此 alpha=0 不跳过 NaN 乘积，beta=0 不跳过 NaN 偏置。
K=0 不调用 cuBLAS，使用 +0 乘积后仍执行系数/偏置；空输出先完成校验，再无提交返回。

真机测试发现 cuBLAS 对 `1 * -0` 可返回 -0，而 CPU 从 +0 累加得到 +0。
现对库乘积的零进行 +0 规范化，再执行用户 alpha/beta，保持包括 alpha=-0 在内的契约。
这也是普通 MatMul 后保留轻量后处理 kernel 的原因，不隐瞒额外 launch 成本。

实际 span 必须属于同一 runtime（同 ordinal 也不允许跨实例），FP32 偏移按 4 字节对齐、
字节数与 spec 精确相等、物理布局连续、输入输出非重叠。未知布局 capability 标记需要运行时检查；
非法属性/类型/形状/布局/设备在提交前拒绝，不改变 dispatch 成功计数。
Completion 保活 operands 和 queue；提交/等待异常沿用 sticky failure，无 CPU 重放。
外部仍须串行访问单个 runtime。

## 实机结果

| 检查 | 结果 |
| --- | --- |
| CUDA ON 全套，require-device=ON | 48/48，0 skip；2.49 s（测试时长，不是推理性能） |
| CPU-only，CUDA OFF，MSVC 19.43 | 45/45；3.41 s，无 CUDA 构建依赖 |
| 原 773 个矩阵映射用例 | 全部通过真实 CUDA 上传、cuBLAS/kernel、下载，与 CPU 对照；不是 CPU 解释器代跑 |
| 总数值/模型链用例 | 1032，通过；包含两个 ONNX fixture 的三个输入，fixture 参考值不变 |
| 主算子测试计数 | dispatches=1033，cuBLAS calls=686，kernel launches=1033；不含独立 lifecycle runtime |
| Compute Sanitizer memcheck + full leak | runtime / toolchain / kernel 三项分别 exit=0，0 errors，0 leaked bytes |

数值阈值保持 `1e-5*(1+abs(reference))`，另检查 NaN 分类、Inf 精确相等及 signed zero。
覆盖四组转置、各类偏置、负/零系数、K=0、空维、矩阵尾部、513 元素跨 block、
四维广播/Softmax、257 个 Softmax groups、正负轴和特殊值。输出前后各 4 字节哨兵
保持不变；拒绝测试覆盖跨 runtime、错 byte size、非对齐 offset、布局/类型等。
另有输入 view 销毁、runtime wrapper 销毁、输出 view 释放后再消费 moved Completion
和下载的保活检查；两个 fixture 使用设备节点链，并非 Prepared 模型实现。

证据：[CUDA 日志](phase5.4-rog-operators-ctest.log)、[CPU-only 日志](phase5.4-rog-operators-cpu-tests.log)、
[memcheck 原始摘要](phase5.4-rog-operators-memcheck.log)、[源码与二进制指纹](phase5.4-rog-operators-verification.json)。
源码哈希针对本次测试时的工作区字节；Git 换行转换后可不同。

## 复现

在项目根目录启动独立进程（不永久修改当前 shell 的 CUDA_PATH/PATH）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_cuda_rog.ps1
```

该脚本选择已验收私有工具链，构建并运行完整 require-device suite。
memcheck 需同样的 SDK bin PATH；工具为私有组件下的
`packages/cuda_sanitizer_api-windows-x86_64-12.5.81-archive/compute-sanitizer/compute-sanitizer.exe`：

```powershell
compute-sanitizer.exe --tool memcheck --leak-check full --error-exitcode 1 build-phase54-cuda/tests/tinyinfer_cuda_kernel_tests.exe --require-device
```

## 未收口内容

下一步是 Prepared CUDA 模型：常量仅上传一次、独立 context 的可写输入/activation、
completion-safe arena、变化绑定、输出副本所有权、整模型 Strict/CPU Auto 策略。
之后冻结 ROG 性能目标，测冷准备/首跑/H2D/D2H/驻留执行/host-to-host E2E、p50/p95 和内存。
本轮没有速度提升、RSS/显存峰值或模型性能验收结论；Softmax 未做并行归约优化。
未执行 ROG ASan/UBSan、racecheck 或 device-loss/hang 注入，不宣称覆盖所有硬件故障恢复。
Mac Phase 5.3 性能门槛未达标状态保持不变。
