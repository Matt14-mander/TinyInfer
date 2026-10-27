# MiniPrefix

MiniPrefix 是建立在 TinyInfer 之上的最小 LLM 生成引擎，目标是在单机单 GPU 上运行
一个真实的小型预训练模型，以 GPU 常驻 KV、持续批处理和前缀复用获得可验证的端到端表现。

**当前状态：已运行真实 SmolLM2-135M-Instruct 的原生 CPU/CUDA 推理闭环，并建立首个
FP32 端到端性能基线。** 入口、权重准备与数值验收见 [真实模型](docs/real_model.md)，
测量口径见 [基线协议](docs/baseline.md)，结果见 [历史 dense 报告](docs/acceptance/baseline-smollm2-fp32.md)
与 [分页验收报告](docs/acceptance/paged-kv-smollm2-fp32.md)。2026-10-25 新增
[最小精确前缀缓存](docs/prefix_cache.md)，默认关闭、可显式配置页面容量。
2026-10-26 补齐 [命中后的增量 prefill](docs/incremental_prefill.md) 范围校验、逐轮计数与长 suffix 验收。
共享/混合/零共享八请求流的 off/cold/warm 独立进程对照，见 [真实收益测量](docs/prefix_benefit.md)。
2026-10-27 从 CPU/CUDA runner 提取 [统一 KVManager](include/miniprefix/kv_manager.h)，
公共接口保留在 include，实现在 src/kv；行为与回归证据见 [重构验收](docs/acceptance/kv-manager-refactor.md)。

当前 CPU/CUDA KV 已改为共享 block 池和序列 block table，CUDA 权重/KV 常驻，详见
[分页 KV](docs/paged_kv.md)；runner 按 sequence 顺序执行。原
[serving_scheduler](../examples/serving_scheduler/README.md) 的重算、部分页 COW 和
跨请求矩阵 batching 仍是独立可运行参考，尚未全部迁入真实模型路径。

## 定位与边界

- TinyInfer：Tensor、内存、算子、设备执行与模型基础设施。
- MiniPrefix：生成请求生命周期、调度、ModelRunner、KV 管理、前缀索引与采样。
- examples/serving_scheduler：保留教学基线、模拟时钟、历史轨迹和迁移回归入口。

开发顺序是：接口与可配置参考模型 → 真实小模型 CPU 正确性闭环 → GPU 完整执行与
KV 常驻 → 可信端到端基线 → 前缀索引和复用优化。已有 serving 能力逐项迁入复用，
不把算法名称覆盖当成真实模型或性能验收。

## 目录

```text
miniprefix/
├── CMakeLists.txt                 # 默认关闭；contracts、kv、engine、CPU/可选 CUDA worker
├── include/miniprefix/           # 对外数据契约；引擎 API 随实现增加
├── src/
│   ├── engine/                   # 请求生命周期、迭代协调、事件与采样
│   ├── scheduler/                # token 预算、准入、重算、调度计划
│   ├── model/                    # 配置、权重、tokenizer 适配、ModelRunner
│   ├── kv/                       # KVManager、页池、精确索引、引用与追加事务
│   ├── prefix/                   # 前缀策略说明，当前实现由 kv 模块编译
│   └── backend/
│       ├── cpu/                  # 数值 oracle 与小规模参考执行
│       └── cuda/                 # GPU 常驻模型/KV、attention 与设备更新
├── bindings/python/              # 后续 Python SDK，复用 C++ 引擎
├── apps/                         # 后续本地 CLI；网络服务后置
├── configs/                      # 模型/引擎配置约定
├── tests/                        # 单元、差分、生命周期、真实设备验收
├── benchmarks/                   # 负载与端到端性能协议
├── tools/                        # 权重/参考 fixture 准备工具
└── docs/
    ├── architecture.md           # 模块依赖、数据流与所有权
    ├── contracts.md              # 接口契约与不变量
    ├── migration.md              # 从 serving_scheduler 增量迁移
    └── roadmap.md                # 阶段交付与验收门槛
```

每个模块 README 说明职责。engine、model、backend 已有实现；独立 scheduler/kv/prefix
提取和高级策略迁移仍在路线中。真实推理由原生 worker 执行，不在 Python 中调用模型代跑。

## 构建入口

仓库根目录配置 `-DTINYINFER_BUILD_MINIPREFIX=ON` 提供 `MiniPrefix::contracts`、
`MiniPrefix::engine` 和 `miniprefix_worker`。启用既有 `TINYINFER_ENABLE_CUDA` 时增加
`miniprefix_cuda_worker`。文本 CLI 为 `python -m miniprefix.apps.generate`，需显式
提供本地模型目录与 worker；尚无 miniprefix_cli 原生 tokenizer 或 HTTP server。

此开关默认 OFF，不要求 CUDA、Python、模型下载或其他第三方依赖。CUDA 实现后继续
受现有 `TINYINFER_ENABLE_CUDA` 约束，不创建第二套 CUDA SDK 配置。

下一步按 [路线图](docs/roadmap.md) 扩展前缀负载、索引效率、部分页 COW 与容量准入，
用缓存关闭/冷/热对照评估改动。

M0 检查：MSVC 19.43 / C++17 / Release / CUDA OFF 的独立 CMake consumer 已成功链接
`MiniPrefix::contracts` 并编译初始请求类型；18 份 Markdown 的本地链接检查通过。
这只验证骨架与契约可接入构建，不代表模型或引擎已实现。
