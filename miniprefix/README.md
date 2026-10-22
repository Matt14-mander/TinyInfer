# MiniPrefix

MiniPrefix 是建立在 TinyInfer 之上的最小 LLM 生成引擎，目标是在单机单 GPU 上运行
一个真实的小型预训练模型，以 GPU 常驻 KV、持续批处理和前缀复用获得可验证的端到端表现。

**当前状态：只有架构骨架、初始数据契约和开发路线，尚无可运行的 MiniPrefix 引擎。**
现有可运行实现仍在 [serving_scheduler](../examples/serving_scheduler/README.md)，
包括 batching、分页 KV、重算、共享/COW 和混合 CPU/CUDA attention 的参考实现。
本次建目录不移动或复制这些算法，不改变其入口和验收记录。

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
├── CMakeLists.txt                 # 默认关闭，仅提供 contracts INTERFACE target
├── include/miniprefix/           # 对外数据契约；引擎 API 随实现增加
├── src/
│   ├── engine/                   # 请求生命周期、迭代协调、事件与采样
│   ├── scheduler/                # token 预算、准入、重算、调度计划
│   ├── model/                    # 配置、权重、tokenizer 适配、ModelRunner
│   ├── kv/                       # block pool/table、引用、COW、资源事务
│   ├── prefix/                   # cache identity、最长前缀查找与驱逐策略
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

每个模块 README 说明职责和首批交付，空实现目录通过这些文件被 Git 跟踪。
规划中的类/target 不是已实现 API；没有占位 forward、伪输出或假的服务入口。

## 构建入口

仓库根目录配置 `-DTINYINFER_BUILD_MINIPREFIX=ON` 可暴露 `MiniPrefix::contracts`，
供后续 target 引用初始请求类型。当前它是 header-only INTERFACE library，不能通过
`cmake --build --target miniprefix_cli` 运行；CLI 尚未实现。

此开关默认 OFF，不要求 CUDA、Python、模型下载或其他第三方依赖。CUDA 实现后继续
受现有 `TINYINFER_ENABLE_CUDA` 约束，不创建第二套 CUDA SDK 配置。

开发从 [路线图 M1](docs/roadmap.md) 开始，按 [迁移规则](docs/migration.md) 保留数值参考。

M0 检查：MSVC 19.43 / C++17 / Release / CUDA OFF 的独立 CMake consumer 已成功链接
`MiniPrefix::contracts` 并编译初始请求类型；18 份 Markdown 的本地链接检查通过。
这只验证骨架与契约可接入构建，不代表模型或引擎已实现。
