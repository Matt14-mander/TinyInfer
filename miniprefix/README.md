# MiniPrefix

MiniPrefix 是建立在 TinyInfer 之上的原生 LLM 生成引擎，面向单机单 GPU 的学习与实验。
当前已运行 SmolLM2-135M-Instruct FP32 的 CPU/CUDA 推理闭环，并实现分页 KV、
非整页精确前缀复用、事务式写时复制（COW）及可复核的成本测量。

## 当前开发阶段

以下状态按 **2026-11-05 的 main 分支**核对。当前处于 **M5 前缀复用优化及执行路径优化**；
完整路线和验收门槛见 [开发路线](docs/roadmap.md)。

| 阶段 | 当前进展 |
| --- | --- |
| M0 架构骨架 | 已完成目录、数据契约与可选 CMake 入口 |
| M1 引擎与参考路径 | 已有请求生命周期、独立 Scheduler、统一 KVManager、容量预约与请求级容量错误隔离；抢占重算待实现 |
| M2 真实模型正确性 | SmolLM2-135M-Instruct FP32 的 CPU/CUDA 输出已通过独立 Transformers 对照 |
| M3 GPU 完整执行 | 已有完整 forward、权重/KV 常驻、增量页表更新与事务式部分页 COW |
| M4 端到端基线 | 已有本地 CLI、事件流、dense/paged 基线及 off/cold/hot 缓存收益实验 |
| M5 前缀复用优化 | 已有精确 trie/LRU、非整页命中、多轮历史快照、成本观测、CUDA 合并复制及发布元数据优化；持续完善中 |

## 已实现能力

- **模型与入口：**原生 CPU/CUDA runner、严格的 FP32 权重加载、本地 worker 协议、
  Python `NativeEngine` IPC 客户端和文本 CLI。权重准备及运行见 [真实模型指南](docs/real_model.md)。
- **调度与生命周期：**轮间补位、decode 优先、chunked prefill、token budget、Greedy、
  EOS/长度完成、取消、错误终态与清理。独立 [Scheduler](src/scheduler/README.md) 管理顺序及预算，
  Engine 协调请求与前缀恢复。
- **KV 与容量：**CPU/CUDA 共用 [KVManager](include/miniprefix/kv_manager.h)，管理页池、
  序列页表、引用及追加事务；backend 管理 payload、设备页表与同步。
  [容量准入](docs/capacity_admission.md) 按完整生成的最坏页数预约，不抵扣潜在共享收益。
- **非整页精确复用：**[前缀缓存](docs/partial_prefix_cache.md) 支持完整页和部分尾页快照，
  沿完整因果历史查找最长 token 前缀，并用 LRU 管理缓存引用。缓存默认关闭。
  prefill 完成和每次成功 decode 后发布已提交历史；新采样 token 在实际 forward 前不缓存。
- **共享尾页 COW：**[部分页事务](docs/partial_page_cow.md) 在写入共享尾页前复制有效 KV，
  失败时恢复旧映射并回收新页。CUDA 已将逐层 K/V 复制合并为两次二维复制，见
  [跨层合并复制](docs/merged_cuda_cow.md)。
- **成本与发布优化：**独立 `COST_STATS` / `COST_TIMING` 观测查找、发布及复制成本，
  保持既有 `STATS` 布局；耗时采集默认关闭。已接入
  [发布路径提示和快照元数据复用](docs/prefix_publication.md)，保留逐步发布与历史快照语义。

## 验证与性能证据

| 范围 | 入口与证据 |
| --- | --- |
| 真实模型数值与多轮生命周期 | [当前多轮验收工具](tools/verify_generation_history.py)、[CUDA 合并复制回归报告](docs/acceptance/merged-cow-review.md) |
| 非整页缓存正确性 | [验收报告](docs/acceptance/partial-prefix-cache.md)：独立索引参考、CPU/CUDA COW、容量与资源回收 |
| 缓存收益及成本 | [测量协议](docs/partial_prefix_costs.md)、[成本验收](docs/acceptance/partial-prefix-cost-review.md)：off/cold/hot 对照与单独 profiling |
| CUDA COW 优化 | [同机旧/新对比](docs/acceptance/merged-cow-review.md)：复制阶段显著下降，端到端收益随负载波动 |
| 发布元数据优化 | [实现与测量方法](docs/prefix_publication.md)、[原始对比](docs/acceptance/prefix-publication-comparison.json)、[CPU](docs/acceptance/prefix-publication-cpu-tests.log) / [CUDA](docs/acceptance/prefix-publication-cuda-tests.log) 回归日志 |
| 持续压力与瓶颈 | [原生 Engine 压测](docs/long_running_stress.md)，与 worker IPC 端到端测量分别报告 |

2026-11-04 的 CUDA 合并复制报告记录 CPU 56/56、CUDA 65/65 测试通过，CUDA memcheck
零错误/零泄漏，以及 44 个真实多轮用例和 196,608 个 logits 的独立对照。
这些是对应版本与环境的验收记录；旧 [dense](docs/acceptance/baseline-smollm2-fp32.md)、
[paged](docs/acceptance/paged-kv-smollm2-fp32.md) 和 [整页缓存收益](docs/prefix_benefit.md)
作为历史基线保留。当前非整页路径的性能应使用上述新协议测量。

缓存收益实验报告单请求输出吞吐，热缓存准备成本不计入热请求延迟。
元数据微基准和 CUDA event 复制计时各有独立范围，不能直接换算成端到端加速。

## 当前边界与下一步

- 同轮可以有多个活跃请求，但 runner 按 sequence 顺序执行；**跨请求矩阵批量计算是下一项计划**。
- 多轮复用由调用方提交完整历史 token 和新的 request ID；尚无自动会话存储或历史拼接。
- 最多复用 `prompt.size() - 1` 个 token，至少留下一个 token 计算 logits；没有 cached-logits 路径。
- 共享感知容量估算、抢占重算、HTTP、量化、CUDA Graph、异步流水与多 GPU 仍待实现。
- [serving_scheduler](../examples/serving_scheduler/README.md) 保留教学和高级算法参考，
  其跨请求 batching、抢占重算等能力尚未全部迁入真实模型路径。

## 目录

```text
miniprefix/
├── CMakeLists.txt                 # 可选 contracts、kv、scheduler、engine、workers 和基准
├── include/miniprefix/            # 请求、模型、Engine、KV、调度与统计 API
├── src/
│   ├── engine/                    # 请求生命周期、迭代协调、事件与采样
│   ├── scheduler/                 # 轮次补位、稳定 decode 优先顺序与输入预算
│   ├── model/                     # 配置与权重加载
│   ├── kv/                        # 页池、精确前缀索引、LRU、容量与 COW 事务
│   ├── prefix/                    # 前缀策略说明；实现位于 kv/
│   └── backend/
│       ├── cpu/                   # 原生 FP32 模型执行与参考路径
│       └── cuda/                  # GPU 模型/KV、设备页表及复制/计算
├── bindings/python/               # NativeEngine 本地 IPC 客户端
├── apps/                          # 原生 worker、协议组件与 Python 文本 CLI
├── configs/                       # 模型/引擎配置约定
├── tests/                         # 单元、差分、生命周期与设备回归
├── benchmarks/                    # 压力、索引/发布微基准和缓存收益实验
├── tools/                         # 模型准备与独立数值验收
└── docs/                          # 架构、契约、路线、测量协议与历史证据
```

## 构建与运行

从仓库根目录构建 CPU worker 和测试：

```bash
cmake -S . -B build-miniprefix -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_BUILD_MINIPREFIX=ON -DTINYINFER_BUILD_TESTS=ON
cmake --build build-miniprefix
ctest --test-dir build-miniprefix --output-on-failure
```

`TINYINFER_BUILD_MINIPREFIX` 默认 OFF。启用后提供 `MiniPrefix::contracts`、`MiniPrefix::kv`、
`MiniPrefix::scheduler`、`MiniPrefix::engine`、`miniprefix_worker` 及基准程序。
在已配置 CUDA 工具链的环境中另加 `-DTINYINFER_ENABLE_CUDA=ON`，可构建 `miniprefix_cuda_worker`。

构建原生 worker 不需要下载模型或安装 Python 推理框架。文本 CLI
`python -m miniprefix.apps.generate` 需要显式提供本地模型目录与 worker，准备命令见
[真实模型指南](docs/real_model.md)。Python 负责 tokenizer、IPC 和展示，模型推理在原生 worker 中执行。
