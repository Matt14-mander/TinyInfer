# Phase 5.1 — Metal 设备与算子本地验收

- 日期：2026-10-12（Asia/Shanghai）。
- 基线：`ca37d22` 加本次实现，未提交；[源码与测试证据](phase5.1-verification.json)。
- 状态：**Phase 5.1 初始 Metal FP32 算子范围在 Mac 完成验收**。
- 边界：独立设备算子与手工算子链；Prepared GPU 模型 API、权重/激活计划及性能验收分别留到 5.2/5.3。

## 实现

- 可选 Objective-C++ `TinyInfer::metal` target，默认 OFF，CPU-only 无 Apple/GPU SDK 依赖。
- MetalDeviceRuntime：单队列、private 设备分配、staging upload/download、blit 复制、
  重叠 scratch 复制、Completion 保活、等待与主机结果发布、终态错误重复报告。
- MetalTensorView：FP32 spec、DeviceSpan、实际 strides；检查范围、布局、所属 runtime，
  禁止输出与输入重叠。未知布局能力返回条件支持，每次派发重新校验。
- 实现 Add、ReLU、Tanh、MatMul、Gemm、FusedGemmActivation、Softmax。
  详见[支持范围](../architecture/metal_backend.md)。矩阵/Softmax 使用初始正确性 kernel，未做性能验收。
- library 和五条 pipeline 在 runtime 创建时准备一次，派发中不重复创建。
- `MetalBackend` 提供准备能力查询，拒绝把 GPU 资源放入主机 Executor。
- [独立示例](../../examples/metal_ops.cpp)：设备 `Gemm → ReLU`，中间结果不回传主机。

## 实机与数值证据

GPU：**Intel Iris Plus Graphics 645**，macOS 15.8.1（24H32），x86_64，
Apple Clang 17.0.0，CMake 4.4.3。测试在可访问 GPU 的主机环境运行；
`MTL_DEBUG_LAYER=1`，日志确认 Metal API Validation Enabled。REQUIRE_DEVICE=ON，
本次没有 GPU skip。Metal 运行时源码编译，无需离线 metal 编译器。

计算测试完成 **248 个数值/边界用例，251 次实际 compute 派发**：

- 248 包含零输出用例；零输出不派发。三个 fixture 输入算子链每条含多个节点，
  因此用例数不等于派发数。
- transA/transB 四种组合、K=0/1/33/65、列尾部、alpha/beta、所有常见 bias 布局。
- scalar/rank-1/3/4、广播 Add、Softmax 所有 rank-3 轴及空输出。
- NaN、Inf、signed zero、ReLU、Tanh、Softmax，以及非有限 Gemm 输出/激活。
- FP32 对照 `1e-5 * (1 + abs(reference))`；非有限分类和零符号单独检查，未放宽门槛。
- `phase3_mlp_gemm.onnx` 一组输入、`rl_actor_mlp_tanh.onnx` 两组输入的手工
  GPU 算子链；与 CPU 执行和已有独立 golden 输出同时比较。中间值保留在设备。
  这是算子集成证据，不是新的自动 GPU 模型执行入口。

复制测试覆盖非对齐字节偏移、容量错误、重叠、零字节、单队列次序，
ready/wait/synchronize 主机发布，输入/输出与分配存活、wrapper 销毁、不同 runtime
拒绝和失败请求计数。未注入 GPU hang/device-loss，也未宣称已验证这种硬件故障恢复。

| 构建/检查 | 最终结果 | 耗时 |
| --- | ---: | ---: |
| CPU-only Release，Metal OFF | 43/43 | 0.86 秒 |
| Metal ON Release，portable CPU 参考，全套实机 | 45/45 | 39.09 秒 |
| Metal ON Release，native AVX2/FMA CPU 参考，GPU 定向 | 2/2 | 3.67 秒 |
| Metal ON Debug，ASan + UBSan，GPU 定向 | 2/2 | 12.32 秒 |
| Gemm → ReLU 示例 | 与固定输出一致 | 不作为延迟基准 |

ASan/UBSan 检查的是宿主 C++/Objective-C++ 代码；不检测 shader 的 GPU 地址访问。
耗时包含启动、编译/测试及主机活动；没有新的 CPU/GPU 加速比、吞吐量或稳定延迟宣称。
构建日志无最终编译 warning/error。CUDA/ROG 与其他 Metal GPU/ARM 尚未执行此阶段测试。

示例实测输出：

```text
Intel(R) Iris(TM) Plus Graphics 645
3 0 7
7 0 15
Checked GPU Gemm -> ReLU; pipelines=5, dispatches=2
```

## 发现并修复的问题

1. 首次计算测试由 Metal API Validation 捕获零 K 空输入视图的偏移等于 native
   buffer 长度。改为空输入绑定 dummy buffer；零 K 仍运行 epilogue。修复后通过。
2. 随后测试中的 host-Executor 拒绝检查创建了未注册输出的空 Graph，先被 Graph
   校验拒绝。修正测试 context 为合法 Graph，仍验证 GPU/host 接口拒绝。
3. 资源复查采用 exception_ptr 保留终态失败，保证等待/队列错误不会被后续成功
   状态掩盖；下载仅在成功完成后发布，运行时不自动重试 CPU。

中间失败日志与最终结果摘要保留在 [验证证据](phase5.1-verification.json)，
没有把失败/跳过记录作为通过证据。后续阶段依然需要实测原生执行错误和设备生命周期。

## 复现

普通主机开发终端执行；本 workspace 的 CommandLineTools/libc++ 需要显式 include：

```bash
cmake -S . -B build-phase51-metal -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_ENABLE_METAL=ON -DTINYINFER_METAL_REQUIRE_DEVICE=ON \
  -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=ON \
  -DTINYINFER_BUILD_BENCHMARKS=OFF \
  -DCMAKE_CXX_FLAGS='-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1' \
  -DCMAKE_OBJCXX_FLAGS='-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1'
cmake --build build-phase51-metal -j 4
MTL_DEBUG_LAYER=1 ctest --test-dir build-phase51-metal --output-on-failure
./build-phase51-metal/examples/tinyinfer_metal_ops
```

通常 SDK 正常的机器不需要手工 libc++ include。native 参考构建加
`-DTINYINFER_ENABLE_NATIVE_ARCH=ON`。Debug sanitizer 对 CXX/OBJCXX 同时加
`-fsanitize=address,undefined -fno-omit-frame-pointer`，链接加同一 sanitizer 参数。
默认 CPU-only 则 `-DTINYINFER_ENABLE_METAL=OFF`，无需启用 Objective-C++。

## 下一步：Phase 5.2

- 设备 Model 快照、准备步骤、常量权重上传一次。
- 每 context 独立设备激活，绑定/复制输出语义与 CPU 对齐。
- 在当前布局能力边界内完成严格拒绝/整模型 CPU 回退，运行时条件逐次验证。
- 两个 ONNX 模型通过正式 GPU Prepared API，多次更换输入和上下文隔离测试。
- GPU 计划需尊重队列/资源完成与复用关系；不能复用主机 MemoryPlan 的释放时刻
  来推断 GPU 任务已结束。性能门槛继续沿用 Phase 5.0 冻结目标，留到 Phase 5.3。

技术接口依据：[Apple 完成等待](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted%28%29?language=objc)、
[buffer blit](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy%28from%3Asourceoffset%3Ato%3Adestinationoffset%3Asize%3A%29)、
[编译选项](https://developer.apple.com/documentation/metal/mtlcompileoptions)。
