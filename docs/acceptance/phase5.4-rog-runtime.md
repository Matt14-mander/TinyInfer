# Phase 5.4 — ROG 原生 CUDA 运行时首段验收

- 日期：2026-10-16（Asia/Shanghai）。
- 基线：`bfca554` 加本轮实现；[验证指纹](phase5.4-rog-runtime-verification.json)。
- 状态：兼容工具链与设备资源/传输首段已通过真机检查；**不是完整 CUDA 算子或模型后端验收**。

## 工具链部署

CUDA 12.5.1 的必要 NVIDIA 官方 redistributable 组件隔离解压到
`build-cuda-sdk-12.5.1/toolkit`。这不是系统级安装器安装；目录由 Git 忽略。
旧 CUDA 10.2 与现有两个 MSVC 工具集未删除/覆盖，驱动仍为 **555.97**。
没有更新驱动、修改注册表、持久 PATH/CUDA_PATH 或系统执行策略。
构建脚本的环境变量只用于其进程；用独立 PowerShell 进程执行。

| 项目 | 已验证值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB，WDDM，sm_89 |
| Toolkit / NVCC | 12.5.1 / 12.5.82 |
| cuBLAS | 12.5.3.2；API version 120503 |
| 主机编译器 | 已有 MSVC 14.29.30133 / cl 19.29.30158 |
| CMake / generator | 3.30.5-msvc23 / Ninja |
| 编译架构 | 89-real，生成本机 sm_89 cubin，不绕过编译器版本检查 |
| Runtime / driver API version | 12050 / 12050 |

CUDA 12.5.1 文档支持 MSVC 192x，Windows 对应驱动版本为 >=555.85，本机
555.97 满足。依据：[Windows 安装指南](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-installation-guide-microsoft-windows/index.html)、
[版本说明](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-toolkit-release-notes/index.html)。
没有依赖较新 Toolkit 的 minor-version compatibility 例外。

官方下载根为 `https://developer.download.nvidia.com/compute/cuda/redist/`。
组件来自 [12.5.1 官方清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.5.1.json)，
下载后逐一校验 SHA-256；归档及各组件 LICENSE 保留在 ignored SDK 目录。
这是一套构建/运行本项目所需的组件，不宣称完整安装所有 CUDA 工具。

| 组件 | 版本 | SHA-256 |
| --- | --- | --- |
| cuda_nvcc | 12.5.82 | `51cec40087b524478c4d204a102ad5de86cd30d0bd2f9e87339032b55296448b` |
| cuda_cudart | 12.5.82 | `079dd4349ee536a80308bfeb745a75b729d3957a430dd1cd0051bbd26fe4007e` |
| cuda_cccl | 12.5.39 | `748c07d466556b1562cc46a0389948af1e3fd92cc667ebf697bf6a9a30e736de` |
| libcublas | 12.5.3.2 | `d3628039eb3d39ba9fe6a5350447b10483dd13d5973b5e4cde1d3e917d457d01` |
| cuda_sanitizer_api | 12.5.81 | `09a8e3438cae3a2f391f3befe7d092bfd2c2157ec4383e5494a2904a2d708ca3` |

## 本轮实现

- 默认 OFF 的 `TINYINFER_ENABLE_CUDA` 与 `TinyInfer::cuda`，当前仅包含资源运行时。
- `CudaDeviceRuntime`：每实例独立 non-blocking stream，原生 cudaMalloc 设备资源，
  同 ordinal 的不同 runtime 资源也不能混用；不暴露可解引用的 host_data。
- pinned host staging 的 H2D/D2H，支持字节偏移、等长 D2D、重叠 scratch 复制、零字节请求。
- cudaEvent 完成状态；wait/ready/synchronize 在成功完成后发布下载结果。
  Completion 保活输入、输出与设备分配；wrapper 销毁、完成对象移动/析构遵循既有契约。
- 独立 queue 所有权避免 runtime/pending/allocation 的强引用循环。设备分配可在 wrapper
  销毁后存活；每次原生调用选择目标 device，离开时恢复调用线程原 device。
- 最多两项 pinned cache；busy lease 不复用，可临时分配额外 staging。
  限制的是缓存条目数，不是固定总字节；资源计数不等于 RSS 或完整 GPU 驻留量。
- 原生提交/等待错误保留终态异常，不自动在 CPU 重跑；非法参数不增加成功请求计数。
  未注入 device-loss/hang 等硬件故障，不宣称已验收所有致命 CUDA 错误恢复。
- `tools/build_cuda_rog.ps1` 固定本机已验证组合、启用 require-device，提供可重复构建/测试入口。

独立 toolchain 探针由 NVCC 编译一个真实 kernel，再执行 cuBLAS SGEMM。
它设置并读回 `CUBLAS_PEDANTIC_MATH`、禁用 atomics，检查 row-major 2×2 结果
`[38,44,86,100]`。这是安装与精度模式烟雾检查，**不是 TinyInfer MatMul/Gemm
生产算子的支持声明**，也不能替代 773 用例的原生 GPU 对照。

## 测试结果

| 检查 | 最终结果 |
| --- | --- |
| CUDA ON Release，MSVC 19.29，全套含真实 GPU 测试 | **47/47**，2.07 秒；require-device，零 skip |
| CUDA OFF Release，原 MSVC 19.43，全套 | **45/45**，4.29 秒；无 CUDA SDK 链接依赖 |
| Compute Sanitizer memcheck + full leak check，运行时 | **0 errors，0 bytes leaked** |
| 同上，kernel/cuBLAS 探针 | **0 errors，0 bytes leaked** |

设备测试复用与 Metal 对齐的传输、容量/偏移、重叠、零字节、发布、不同 runtime
拒绝、缓存 lease、wrapper/Completion 资源存活与非法请求计数场景。
工具链首次真机测试 47/47、12.84 秒；复查 47/47、1.78 秒，最终容量预检修改后
47/47、1.91 秒；补充真实 Completion 移动和非法序号用例后最终 47/47、2.07 秒。
耗时包含进程启动，不用于性能比较。

发现并处理的构建/检查问题：

1. 原 core 的 GELU lambda 在 MSVC 19.29 要求显式捕获局部 constexpr。
   只增加常量捕获，不改公式；两套编译器的全量测试通过。
2. NVCC 探针补齐显式 `<string>` include，不依赖宿主间接包含。
3. 首次 memcheck 将测试刻意请求 SIZE_MAX 的 cudaMalloc OOM 计为一次 API error，
   泄漏为零。运行时现先拒绝超过设备物理容量的请求，原测试未删除，未屏蔽检查；
   最终两项 memcheck 均零错误、零泄漏。

既有 C4819 和 /UNDEBUG 覆盖警告不在此轮清理范围。没有执行 ROG ASan/UBSan、
racecheck、完整算子验证、两个 fixture 的 GPU 模型执行或性能验收；不改变 Mac
Phase 5.3 未通过性能门槛的结论。

## 复现与证据

项目根目录，使用独立进程（不需要持久更改执行策略）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_cuda_rog.ps1
```

已安装在其他位置时传 `-ToolkitRoot <path>`。切换编译器必须使用独立构建目录；
默认输出为 `build-phase54-cuda`。本机 helper 在子进程内导入 MSVC 14.29 开发环境，
将私有 Toolkit 的 bin 放到该进程 PATH，以便 CTest 加载对应 DLL。
若只在普通终端直接运行 exe，不能依赖系统 PATH 里的旧 SDK。

memcheck 使用保留的官方组件：

```powershell
# 在与 helper 相同的进程环境中：
./build-cuda-sdk-12.5.1/packages/cuda_sanitizer_api-windows-x86_64-12.5.81-archive/compute-sanitizer/compute-sanitizer.exe --tool memcheck --leak-check full --error-exitcode 1 ./build-phase54-cuda/tests/tinyinfer_cuda_runtime_tests.exe --require-device
```

将目标替换为 `tinyinfer_cuda_toolchain_tests.exe` 复现探针检查。
已提交范围内的[测试日志](phase5.4-rog-runtime-ctest.log)、[memcheck 日志](phase5.4-rog-runtime-memcheck.log)
及[源码/二进制指纹](phase5.4-rog-runtime-verification.json)保留最终证据。
本机 ignored 目录另保留原始环境 `benchmark-results/phase5.4/rog-environment-cuda125.json`
和 `benchmark-results/phase5.4/runtime` 中的原始日志。
关闭 CUDA 的完整[CPU-only 测试日志](phase5.4-rog-runtime-cpu-tests.log)也保留。

## 剩余工作

1. 把 CUDA 契约接入真正的算子派发与 cuBLAS/逐元素 kernel，原生复跑 773 语义用例。
2. Prepared CUDA 模型、权重与独立激活、整模型回退，以及两个 ONNX fixture 真机验证。
3. 冻结 ROG 目标，再进行准备/传输/驻留/端到端性能测量与跨后端报告。

Phase 5.4 只完成资源运行时首段，不标记完整后端或整个 Phase 5 完成。
