# Phase 5.4 — ROG CUDA 开发前检查

- 日期：2026-10-16（Asia/Shanghai）。
- 基线：`dd9deb1`；本轮不包含原生 CUDA 执行实现或性能验收。
- 状态：ROG 硬件采集和 SDK-free 基线完成；原生 CUDA 工作等待兼容 Toolkit 配置。

后续同日更新：[原生运行时首段](phase5.4-rog-runtime.md)已在用户授权后部署
隔离 CUDA 12.5.1、复用 MSVC 14.29 并通过真机测试。以下保留配置前的原始状态，
不代表当前仍被工具链阻塞。

## 硬件与工具链

| 项目 | 实测 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB，WDDM |
| 驱动 | 555.97；nvidia-smi 显示驱动 CUDA 支持版本 12.5 |
| 实际 Toolkit | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2`，nvcc 10.2.89 |
| nvcc 查找 | PATH 未找到，通过 CUDA_PATH 找到；不是没有安装任何 Toolkit |
| CPU | Intel Core i9-13980HX，24 核 / 32 逻辑处理器 |
| MSVC | 默认 14.43.34808 / 编译器 19.43；另有 14.29.30133 工具集 |
| CMake | Visual Studio bundled 3.30.5-msvc23，Ninja |
| Python | 本次使用 `E:/anaconda3/python.exe`；PATH 的 python 指向 WindowsApps 别名 |

驱动显示的 CUDA 版本不代表已经安装相同版本的 Toolkit。现有 CUDA 10.2
`include/crt/host_config.h` 拒绝 `_MSC_VER >= 1930`，与当前默认 19.43
不兼容；其 `cublas_api.h` 没有本阶段计划使用的 `CUBLAS_PEDANTIC_MATH`。
不能通过忽略编译器检查、静默降低精度要求或把 CPU 契约测试计为 GPU 测试来收口。
本轮没有修改驱动、安装 SDK、持久修改 PATH 或切换旧工具集。

## 已执行验证

| 检查 | 结果 |
| --- | --- |
| 干净 Release native CPU-only 构建，含示例 | 成功；Metal OFF，无 CUDA SDK 链接依赖 |
| 完整 CTest | **45/45**，8.43 秒；包含新增 serving scheduler 测试 |
| CUDA 契约独立执行 | **773** 个 column-major CPU 解释用例通过；不代表 GPU 执行 |
| 两个 ONNX fixture 静态检查 | 通过；未执行 CUDA kernel |
| 环境采集脚本测试 | **4/4**：PATH 优先、CUDA_PATH 回退、未安装、无效 CUDA_PATH |
| CUDA 真机测试 / 性能测量 | **未执行** |

现有 C4819（代码页）与 /UNDEBUG 覆盖警告保留，未在此轮修改无关源码。
ASan/UBSan 未在 ROG 本轮执行。测试耗时不是性能基准。

本轮修复 `tools/inspect_gpu_environment.py`：PATH 缺少 nvcc 时检查
CUDA_PATH 下的实际编译器，并记录解析来源。没有把驱动版本当作 Toolkit。

## 复现

在 Visual Studio x64 开发终端，用有效 Python 执行：

```powershell
E:/anaconda3/python.exe tools/inspect_gpu_environment.py --output benchmark-results/phase5.4/rog-environment.json
cmake -S . -B build-phase54-contract -G Ninja -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_TESTS=ON -DTINYINFER_BUILD_EXAMPLES=ON -DTINYINFER_BUILD_BENCHMARKS=OFF "-DCMAKE_CXX_FLAGS=/EHsc /permissive-" -DTINYINFER_ENABLE_NATIVE_ARCH=ON
cmake --build build-phase54-contract --parallel 8
ctest --test-dir build-phase54-contract --output-on-failure
E:/anaconda3/python.exe -m unittest discover -s tools -p test_inspect_gpu_environment.py -v
./build-phase54-contract/tests/tinyinfer_cuda_contract_tests.exe
./build-phase54-contract/examples/tinyinfer_cuda_contract_inspect.exe tests/fixtures/phase3_mlp_gemm.onnx
./build-phase54-contract/examples/tinyinfer_cuda_contract_inspect.exe tests/fixtures/rl_actor_mlp_tanh.onnx
```

原始环境记录在 ignored 的 `benchmark-results/phase5.4/rog-environment.json`；
完整测试日志在 `build-phase54-contract/Testing/Temporary/LastTest.log`。

## 下一步与边界

先由用户确认可安装/使用的兼容 CUDA Toolkit 与 MSVC 组合，保留旧 10.2，
不默认更新显示驱动。之后按 [5.4 计划](../plans/phase5.4.md)依次实现：

1. 可选原生 CUDA target、设备资源、单 stream、pinned staging、event completion。
2. cuBLAS FP32 矩阵与必要逐元素 kernel，真实 GPU 对照现有 773 用例。
3. Prepared CUDA 模型、两个 fixture、资源生命周期和整模型回退。
4. 单独冻结 ROG 目标与性能门槛，再进行端到端测量。

Phase 5.4 没有因此标为完成；Mac Phase 5.3 未通过性能门槛的状态不变。
