# Phase 5.0 — 设备运行时契约与 Mac 验收

- 日期：2026-10-11（Asia/Shanghai）。
- 基线：`692bf0c` 加本次实现，未提交；[验证指纹](phase5.0-verification.json)。
- 状态：**Phase 5.0 在 Mac 收口**。Phase 5.1 GPU 算子与设备执行计划尚未实现。

## 已交付

| 检查项 | 实现与验收 |
| --- | --- |
| 设备身份 | DeviceKind/DeviceId/DeviceInfo，CPU ordinal 校验、不可用原因 |
| 节点能力 | Node + dtype/shape/可选 strides/属性/设备查询；CPU Schema 与输出布局检查 |
| 回退策略 | Strict 准备阶段拒绝；WholeModelCpu 校验全图后选择 CPU，并保留原因/节点诊断 |
| 内存隔离 | DeviceBuffer/DeviceSpan 与主机 Tensor/Storage 分离；GPU host_data 返回 null |
| 复制契约 | owned Buffer 的 upload/download、同设备等长 copy，检查偏移与容量，零字节安全 |
| CPU 适配 | 同步分配、memmove 复制、重叠处理、成功请求与累计 payload 统计 |
| 生命周期 | Completion 移动、wait、析构和移动赋值排空；资源保活、终态失败重复报告 |
| 兼容性 | 原 Tensor 拷贝语义、Executor、CPU Prepared Auto 和主机 output 入口保持原行为 |
| 设计文档 | [ADR-0007](../decisions/0007-separate-device-resources-and-completion.md)、[运行时架构](../architecture/device_runtime.md) |
| 目标环境 | 实际枚举 Mac Metal GPU/queue，临时 shader library 编译成功；CUDA 条件待 ROG 检查 |

ModelPlacement 是可检查的准备元数据，不是 GPU 执行计划。未来后端必须在运行时
落实 layout 条件，拥有独立 context，遵循完成/复制错误契约。测试中的 GPU mock
只验证选择和错误处理，不能作为实机 GPU 计算或异步执行验收。

## 测试

| 配置 | 最终结果 | 耗时 |
| --- | ---: | ---: |
| Release，native OFF | 43/43 | 26.05 秒 |
| Release，native ON | 43/43 | 25.90 秒 |
| Debug，ASan + UBSan，设备/能力定向测试 | 2/2 | 2.36 秒 |

新增两个测试目标，覆盖复制偏移、边界/溢出、零字节、重叠、错误请求不增加计数、
设备隔离、分配存活、完成资源保活、移动赋值排空、析构排空、sticky failure，
以及 CPU dtype/shape/layout/属性检查、严格拒绝、不可用设备、整图回退和真实
ONNX fixture 的 CPU 输出/独立拷贝。现有全部测试继续通过。

CPU-only library/test/example/benchmark 构建不需要 GPU SDK，没有引入 CUDA/Metal
链接依赖。此次没有新增数值 kernel，也不宣称新的 CPU/GPU 性能收益。

复测（已有构建目录）：

```bash
cmake --build build-coverage-release -j 4
ctest --test-dir build-coverage-release --output-on-failure
cmake --build build-bench -j 4
ctest --test-dir build-bench --output-on-failure
cmake --build build-phase44-asan --target tinyinfer_device_runtime_tests tinyinfer_backend_capability_tests -j 4
ctest --test-dir build-phase44-asan -R 'device_runtime|backend_capability' --output-on-failure
```

构建缓存使用 Apple Clang 17、macOS SDK 的 libc++ include 路径。Release 保留测试断言。
耗时含进程启动及主机负载，两个 Release 配置并行构建/测试；不是性能基准。
未在 ROG/ARM 运行这些新增测试，未宣称其跨主机验收。

## Mac 开发条件与首个后端

[完整硬件/工具链记录](phase5.0-mac-environment.json)：

| 项目 | 结果 |
| --- | --- |
| 主机/OS | x86_64，macOS 15.8.1（24H32） |
| 编译器 | Apple Clang 17.0.0，CMake 4.4.3 |
| Developer directory | CommandLineTools |
| macOS SDK | 可用 |
| GPU | Intel Iris Plus Graphics 645 |
| Metal command queue | 创建成功 |
| Metal runtime shader library | 临时源代码编译成功，未执行 kernel |
| 离线 metal 编译器 | `xcrun --find metal` 未找到 |
| CUDA | 本机 PATH 未找到 nvidia-smi/nvcc；不据此推断 ROG 条件 |

沙箱内最初枚举结果为空；沙箱外重复检查实际设备和 runtime library 成功。
因此不能把隔离环境里的空列表等同于物理 Mac 无 GPU。下一阶段实机运行也须
确保进程可访问 Metal 设备。此次环境采集没有安装工具或修改系统设置。

继续在 Mac 开发，**Metal 作为 Phase 5.1 首个后端**。运行时源码编译可作为
最初入口，并在准备阶段缓存 library/pipeline、单独记录编译成本；无需为了
Phase 5.0 立即转到 ROG。CUDA 后端的 GPU 型号、驱动和 nvcc 版本在后续 ROG
环境采集时核实。Metal 准备/完成接口依据 [Apple 官方库编译 API](https://developer.apple.com/documentation/metal/mtldevice/makelibrary%28source%3Aoptions%3A%29?language=objc)。

通用采集命令（Mac/Windows，普通主机开发终端）：

```bash
python3 tools/inspect_gpu_environment.py --output benchmark-results/phase5.0/environment.json
```

脚本仅采集设备/SDK信息；Mac 上编译一个临时 Objective-C 枚举程序及 Metal
library，均不提交 GPU kernel。设备不可见/编译失败会记录原因，不伪称 GPU 就绪。
Windows 采集 nvidia-smi、nvcc 和显示适配器；缺少命令不会安装软件。

## Phase 5.1 接口入口

1. 可选 Objective-C++ Metal 模块，CPU-only 构建继续独立。
2. Metal DeviceBuffer/DeviceRuntime/CompletionState 真机适配，先测试传输与完成。
3. 逐元素算子、矩阵及模型所需算子，数值与 CPU 参考对照。
4. Phase 5.2 加入不可变设备权重、每 context 激活及运行时 layout 检查。
5. Phase 5.3 按 [已冻结目标](../plans/phase5.md#phase-50-收口2026-10-11)测量端到端收益。

Phase 5.0 不包括 GPU kernel、模型执行、自动设备性能选择、跨设备图分区、
Transformer/KV cache，也不提供这些能力的性能验收。
