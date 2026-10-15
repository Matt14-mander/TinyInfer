# Serving Scheduler 学习实验

**第一阶段已实现：Static Batching 与 Continuous Batching。**
使用 Python 3.7+ 标准库、确定性模拟后端和虚拟时钟，无需下载模型、GPU 或安装依赖。
本例独立于 TinyInfer C++ Runtime，不改变 Phase 5.4 CUDA 开发与验收状态。

需求实际列出了四个阶段，因此按四阶段推进：

| 阶段 | 内容 | 当前状态／验收边界 |
|---|---|---|
| 1 | Static / Continuous Batching | 已实现；输出一致、FIFO、边界补位、批大小上限、可重现 trace 与测试 |
| 2 | KV Cache 容量约束、请求拒绝 | 待实现；先规定 token/block 单位、prompt+输出预算预留、等待与拒绝原因、完成后回收；验证无超分配或泄漏 |
| 3 | Chunked Prefill | 待实现；引入每轮 token budget 与 prefill 进度，避免长 prompt 一次阻塞所有 decode；验证 chunk 前后 token 一致、预算与公平性 |
| 4 | 真实推理后端及性能测量 | 待实现；真实 prefill/decode/logits/KV、完成同步、真实到达时间与流式输出；独立测 TTFT、TPOT、吞吐量、请求 P95 |

第二阶段的 KV 容量模拟不等于真实分配显存。第四阶段需要具备自回归 prefill/decode
与 KV 生命周期的模型接口；当前静态 MLP/ONNX Prepared 入口不能直接充当 LLM serving
后端。届时还需将虚拟事件驱动替换为真实时钟/等待/到达队列，并处理 EOS 和请求失败；
不能只把模型调用塞进本例 `simulate()` 就宣称得到生产服务。

## 运行

在仓库根目录执行：

```bash
python3 -m unittest discover -s examples/serving_scheduler -v
python3 examples/serving_scheduler/benchmark.py --scenario mixed --batch-size 2
python3 examples/serving_scheduler/benchmark.py --scenario staggered --batch-size 4 --trace
python3 examples/serving_scheduler/benchmark.py --scenario long_prefill --format json
# 模块方式同样可用
python3 -m examples.serving_scheduler.benchmark --scenario mixed --batch-size 2
# 自定义到达序列
python3 examples/serving_scheduler/benchmark.py --requests examples/serving_scheduler/requests.json --format json
```

`requests.json` 为请求列表。ID 必须唯一；arrival_ms 必须有限、非负；prompt_tokens、
output_tokens 和 batch-size 必须为正整数。非法输入报错退出，不属于阶段二的容量拒绝。

## 调度语义

- **Static**：取当前已到达的 FIFO 请求形成 cohort；请求完成后留下空位，整组结束后
  才启动下一组。不等待凑满，不加入人为 batching timeout。
- **Continuous**：每轮开始时，从已到达的 FIFO 请求补满空位。执行期间到达的请求
  等到下一轮边界；同到达时间保持输入顺序。不抢占运行中的请求。
- 新请求本轮执行完整 prefill，并在结束时产出第一个 token；已有请求执行一次 decode。
  同一轮两者可以混合，但所有 token 都在该轮结束时交付。
- 运行队列为空时，虚拟时钟直接跳到下一次到达；请求不会在到达之前执行。
- 每个请求生成固定数量 token，不支持 EOS 提前结束、取消、优先级、抢占或 KV 容量。
  `max_batch_size` 仅限制活跃请求数，不限制 token 或显存。

### 可手算例子

B=2，A/B/C 同时到达，输出长度为 3/1/1，假设每轮固定 1 ms：

| 模式 | 轮 1 | 轮 2 | 轮 3 | 轮 4 |
|---|---|---|---|---|
| Static | A,B | A | A | C |
| Continuous | A,B | A,C | A | — |

C 的首 token 从 4 ms 提前到 2 ms。此例在单元测试中逐步校验，不只是检查统计值。

## 模拟后端与限制

每个非空 step 的虚拟耗时：

```
1.0 ms 固定开销
+ 0.1 ms × 本轮所有新请求的 prompt token 数
+ 0.2 ms × 本轮所有已有请求数
```

它是用于观察调度的成本假设，没有 GPU 并行、真实 kernel、KV 访存、带宽、padding 或
batch 性能曲线。输出 token 仅由请求 ID、prompt 长度和生成位置决定，不受调度顺序影响。
benchmark 会断言两种策略输出完全相同；测试还与逐请求独立运行对照。

Continuous 在混合输出长度下通常减少空位与轮数；不承诺每种输入都更快或每个请求
延迟都更低。`long_prefill` 用例刻意保留长 prefill 阻塞 decode 的问题，供第三阶段
观察和改进。本阶段不偷偷切分 prompt 或模拟 KV 分配。

## 指标口径（本阶段全部为模拟值）

- **Queue**：admitted − arrival。
- **TTFT**：第一个输出 token 交付时刻 − arrival，包含排队和 prefill。
- **TPOT**：每请求 `(最后 token 时刻 − 首 token 时刻)/(输出 token 数 − 1)`。
  单 token 请求为 null，不伪装成 0；汇总为有效请求 TPOT 的算术平均，不是 token 加权值。
- **请求延迟**：最后一个输出 token 交付 − arrival；P95 使用 nearest rank，索引 ceil(0.95×N)。
- **吞吐量**：输出 token／完成请求数除以“首次到达至全部完成”的窗口，窗口包含中间空闲，
  不包含首次到达前的空闲；仅输出 token，不计 prompt token。
- **slot_utilization**：按 step 时长加权的活跃 slot 数除以窗口×批上限；另报去掉空闲
  时间的 service_slot_utilization。它不是 GPU 利用率或有效 token 计算利用率。
- 空工作负载吞吐为 0，分位数为 null。结果含全部请求时间、输出序列，`--trace` 加入每轮明细。

当前 mixed/B=2 的默认模拟结果：Static 18 轮、Continuous 14 轮；输出 tokens/s 约
1031.75／1226.42；TTFT P95 23.8／20.0 ms。只说明该成本模型中的差异，**不是 TinyInfer
实机加速数据**。可复现数据见 `results/`。

## 文件结构

| 文件 | 职责 |
|---|---|
| backend.py | 不可变请求、step 接口、确定性虚拟后端 |
| scheduler.py | FIFO、准入、固定 cohort／边界补位、完成与 trace |
| metrics.py | 模拟请求级与汇总指标，明确窗口与分位数口径 |
| benchmark.py | 内置／JSON 工作负载、输出一致性、表格和 JSON |
| test_scheduler.py | 12 项测试，包含手算轨迹与错误输入/后端输出 |
| requests.json | 可编辑的输入样例 |
| results/ | 第一阶段测试日志与三种场景的模拟报告 |

真实后端阶段应另存环境、模型/精度、batch/token/KV 预算、输入长度/输出长度/到达分布、
warmup、独立运行次数、失败/拒绝请求口径及每请求时间戳。模型完成和输出交付后才能计时，
不能拿本阶段成本公式或仅提交 GPU kernel 的时间当作真实 TTFT/TPOT。
