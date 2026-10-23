# Serving Scheduler 学习实验

**四个学习阶段已完成：前三阶段模拟调度，以及微型 Transformer 的真实 TinyInfer CPU 工程验收。**
前三阶段使用 Python 3.7+ 标准库、确定性模拟后端和虚拟时钟；第四阶段编译 C++ worker，
执行实际数值推理并使用墙钟。后续已补齐跨请求 CPU 张量 batching，详见 [BATCHING.md](BATCHING.md)。无需下载模型或 GPU。
可选物理分页 KV 与直接 CPU PagedAttention 已实现，运行与验收见 [PAGED_ATTENTION.md](PAGED_ATTENTION.md)。
可选动态准入/抢占重算、精确前缀共享/COW 与 CUDA 分页 kernel 见 [ADVANCED_SERVING.md](ADVANCED_SERVING.md)。
本例独立于 TinyInfer C++ Runtime，不改变 Phase 5.4 CUDA 开发状态。
后续引擎的独立目录与迁移路线已建立于 [MiniPrefix](../../miniprefix/README.md)；
MiniPrefix 已有真实 SmolLM2 CPU/CUDA 推理闭环、独占分页 KV 与 FP32 性能基线；
本例继续作为 paged/prefix 等算法的可运行参考与回归入口。

## 四阶段计划

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | Static / Continuous Batching | 已实现；FIFO、边界补位、输出一致性、手算 trace |
| 2 | KV 容量约束、请求拒绝 | 已实现；block 预留、等待／拒绝、完成／异常回收 |
| 3 | Chunked Prefill | 已实现；每轮 token 预算、decode 优先、prefill 轮转、进度与首 token 边界 |
| 4 | 真实后端及性能测量 | 已完成微型随机权重 Transformer CPU 验收：真实 logits/KV、墙钟、44 项测试与三轮测量 |

前三阶段 KV 是账本模拟；第四阶段在独立 TinyInfer C++ worker 中实际分配 CPU K/V Tensor。
真实入口是 real_benchmark.py，模型未训练；当前 Q/K/V、输出投影、FFN 与词表投影按打包
token 行合并计算，变长 attention 仍逐 query 执行。历史第四阶段构建、计时边界与验收详见
[STAGE4.md](STAGE4.md)，其中原始性能数据属于旧串行 worker。本例不提供生产网络服务。

## 运行

在仓库根目录执行：

```bash
python3 -m unittest discover -s examples/serving_scheduler -v
# 第一阶段默认行为保持：不限制 KV 容量
python3 examples/serving_scheduler/benchmark.py --scenario mixed --batch-size 2
python3 examples/serving_scheduler/benchmark.py --scenario staggered --trace
# 第二阶段：4 blocks，每 block 4 token slots
python3 examples/serving_scheduler/benchmark.py --scenario kv_pressure --batch-size 3 --kv-capacity-blocks 4 --kv-block-size 4 --kv-policy wait --trace
python3 examples/serving_scheduler/benchmark.py --scenario kv_pressure --batch-size 3 --kv-capacity-blocks 4 --kv-block-size 4 --kv-policy reject --format json
# 零容量：全拒绝，不执行后端
python3 examples/serving_scheduler/benchmark.py --kv-capacity-blocks 0 --format json
# 第三阶段：每轮最多 16 个输入 token，每请求 prefill chunk 最多 8
python3 examples/serving_scheduler/benchmark.py --scenario long_prefill --batch-size 3 --token-budget 16 --prefill-chunk-size 8 --trace
# 不传 token-budget 即保留完整 prefill，便于对照
python3 examples/serving_scheduler/benchmark.py --scenario long_prefill --batch-size 3 --format json
# 模块入口和自定义 JSON 同样可用
python3 -m examples.serving_scheduler.benchmark --scenario long_prefill --format json
python3 examples/serving_scheduler/benchmark.py --requests examples/serving_scheduler/requests.json --format json
```

请求 JSON 是含 request_id、arrival_ms、prompt_tokens、output_tokens 的对象列表。
ID 唯一；arrival_ms 有限且非负；token 数和 batch-size 为正整数。
非法输入直接报错退出，不属于容量拒绝。

## 调度语义

- **Static**：选取已到达 FIFO 请求形成 cohort。完成请求留下空位，整组完成后才启动下一组。
  不人为等待凑满，也不设置 batching timeout。
- **Continuous**：每轮边界补空位；step 执行期间到达的请求等下轮处理。同到达时间保持输入顺序。
- 默认新请求一次完成全部 prefill 并产出首 token；开启 token-budget 后分块处理。
  只有最后一个 prefill chunk 产出首 token，已有请求每轮 decode 一个 token；token 在 step 结束时交付。
- 没有可运行请求时，虚拟时钟跳到下一次到达；不提前执行未到达请求。
- 固定输出长度，无 EOS、取消、优先级、抢占、KV 驱逐或队列超时。batch-size 限制请求数，
  KV 预算另外限制预留容量；等待队列长度当前不受限制。

B=2，A/B/C 同时到达，输出长度 3/1/1，每轮固定 1 ms 的手算例子：

| 模式 | 轮 1 | 轮 2 | 轮 3 | 轮 4 |
|---|---|---|---|---|
| Static | A,B | A | A | C |
| Continuous | A,B | A,C | A | — |

C 的首 token 从 4 ms 提前到 2 ms；测试逐步校验这条轨迹。

## 第二阶段：KV 预算与准入

`--kv-capacity-blocks N`，`--kv-block-size S`，总预算 N×S token slots。
省略 N 表示无限容量，N=0 合法并拒绝所有请求；S 必须为正整数。
这是抽象 token 预算，不对应真实 K/V 张量或 GPU 字节数。

每个请求在准入时一次预留完整预算：

```
token_budget = prompt_tokens + output_tokens
required_blocks = ceil(token_budget / S)
```

保守地包含最终输出 token 的预算，不做 decode 动态扩容；因此已准入请求不会中途因
预算增长 OOM。每请求独立向上取整，block 尾部空位不能分给其他请求。

| 情况 | 处理 |
|---|---|
| 单请求超过总容量 | 在首次观察到到达时拒绝，原因 `request_exceeds_kv_capacity`；即使没有 batch 空位也会处理 |
| 有 batch 空位且 KV 足够 | 一次预留完整预算后准入；失败的 reserve 不改变账本 |
| 临时不足，默认 `wait` | FIFO 队头等待，后面小请求不绕行；等待请求不占 KV |
| 临时不足，`reject` | 当次准入尝试拒绝，原因 `kv_capacity_unavailable`；继续尝试后续请求 |
| batch 满，或 Static cohort 未结束 | 能装入总容量的请求排队，不把尚未尝试准入视为 KV 压力拒绝 |
| 请求完成 | 在 step 完成时释放，下一轮可复用 |
| 后端异常、非法结果或时钟错误 | 终止整次模拟，清理所有仍活跃预留；不返回伪造成功结果 |

所有到达在调度边界被观察，不中断正在运行的 step。超大请求在 Static cohort 运行中
也能于下一边界被拒绝，不会永久挡住队列。FIFO 临时阻塞会等待有限长度的活跃请求完成。
真实 GPU 的物理资源释放必须等待终止完成；这里异常清理的对象仅为虚拟账本。

`kv_reserved_blocks` 是 step 开始时的预留量；`kv_events` 记录 reserve/release 后的占用，
`admission_events` 记录 wait/reject。报告峰值、最终占用和剩余预留数；完成或异常清理后归零。
`kv_waited_requests` 统计至少因 KV 等待一次的请求数，不把每轮重试重复计数。

默认 kv_pressure/B=3/N=4/S=4，两种调度均得到：

| 策略 | 完成 | 拒绝 | 峰值 blocks | 最终 blocks |
|---|---:|---:|---:|---:|
| wait | 3 | 1（超容量） | 3/4 | 0 |
| reject | 2 | 2（超容量、临时不足） | 4/4 | 0 |

long 需要 ceil((4+6)/4)=3 blocks，peer 需要 2，small 需要 1。wait 保持 peer 在队头，
small 不绕行；reject 删除 peer 后能准入 small。另有测试覆盖不同调度产生不同完成集合。

## 模拟成本和正确性

每个非空 step 的虚拟耗时：

```
1.0 ms 固定开销
+ 0.1 ms × 本轮所有新请求的 prompt token 数
+ 0.2 ms × 本轮所有已有请求数
```

没有真实 GPU 并行、kernel、KV 访存、带宽、padding 或 batch 性能曲线。
输出由请求 ID、prompt 长度和 token 位置确定，与调度顺序无关。
benchmark 检查共同完成 ID 的输出相同，并把每个完成请求与无限容量独立运行对照。
`same_output_sequences` 表示逐请求输出正确；`same_completed_request_ids` 另行报告集合
是否一致。零完成请求时参考校验是空集，`reference_verified_requests` 为 0。

Continuous 不保证每种输入都更快，也不保证每个请求延迟都更低。`long_prefill` 刻意保留
长 prompt 阻塞 decode 的情况，用于对照 Chunked Prefill 的效果。

## 第三阶段：Chunked Prefill

`--token-budget T` 限制每轮处理的输入 token 总数：prefill chunk token 数之和，加每个
已有 decode 请求的 1 token。它不限制本轮产生的输出 token 数，也不改变 KV 总容量。
T 必须是正整数；省略时保持前两阶段的完整 prefill 行为。
`--prefill-chunk-size C` 为每请求每轮的额外 prefill 上限，必须为正整数且同时设置 T；
省略 C 时单请求可以使用本轮剩余预算。C 大于 T 也合法，实际仍受 T 限制。

调度边界的顺序：

1. 观察到达与拒绝，按原 Static/Continuous 和 KV 规则准入；FIFO 指准入顺序。
2. 优先给所有已有 decode 请求各分配 1 token。
3. 用剩余预算轮转已准入的 prefill 请求。每请求最多一个 chunk；处理量为剩余 prompt、
   剩余 T 和 C（若设置）的最小值。轮转从上轮最后服务的仍在队列中的 prefill 之后开始。
4. 分块范围用 `[prefill_start, prefill_end)` 记录。未处理的活跃请求列在 deferred_requests，
   继续占 batch slot 与完整 KV 预留。预算可有剩余，不为了凑满而给同一请求追加第二个 chunk。
5. 后端对未完成 prefill 返回 None；最后一个 chunk 才返回首 token。过早产生 token 或
   最后 chunk/decode 缺失 token 都会报错并清理预留。

decode 数量不会超过 T：初始没有 decode，新 decode 只能由本轮预算内完成的 prefill
产生，且每个至少消耗 1 个输入 token。每轮已有 decode 全部服务，因此小于 batch-size
的 T 也可以运行，无需跳过已有 decode。T 被 decode 用完时 prefill 等待；有限固定输出
长度的请求最终会完成，但本例不承诺在线负载下的等待时间上界或 SLA。

KV 仍在准入时预留完整 prompt+输出预算，分块不会提前释放、动态缩减或增加预留。
时钟/后端错误时依旧终止模拟并清理所有活跃预留。

### 手算：减少 decode 阻塞，可能延后首 token

A 在 0 ms 到达，prompt=1/output=4；B 在 2 ms 到达，prompt=6/output=1。
每轮固定开销 1 ms、prefill 每 token 1 ms、decode 每请求额外开销 0 ms；B=2：

| 策略 | A 输出时刻（ms） | B 首 token（ms） | 最大相邻 token 间隔 |
|---|---|---:|---:|
| 完整 prefill | 2,9,10,11 | 9 | 7 ms |
| T=3、C=2 | 2,5,8,11 | 11 | 3 ms |

这条轨迹在测试中精确验证。输出序列一致，但 B 的 TTFT 变差，说明分块不是无条件加速。

默认 long_prefill/B=3 的模拟对照见 results/stage3：

| Continuous 策略 | 最大 token 间隔 | TTFT P95 | 输出吞吐 |
|---|---:|---:|---:|
| 完整 prefill | 21.2 ms | 22.2 ms | 457.14 tokens/s |
| T=16、C=8 | 2.2 ms | 47.8 ms | 313.73 tokens/s |

分块减少一次长 prompt 对 decode 的阻塞，但每轮固定开销会累积。Static cohort 本来就
不在旧 cohort decode 期间准入新 prompt，其分块结果另行报告，不能套用 Continuous 的收益。
全部仍是成本模型模拟，不代表真实 GPU TTFT、TPOT 或吞吐提升。

后端接口增加 WorkItem.prefill_start/prefill_tokens 与可选 token 结果；这是模拟协议变化。
旧自定义后端若默认每项都返回 token，必须先支持未完成 chunk 返回 None 才能开启分块。

## 共享指标口径（benchmark.py 是模拟值；第四阶段 real_benchmark.py 是墙钟值）

- Queue = admitted − arrival。
- TTFT = 首 token 交付 − arrival，包含排队和 prefill。
- 每请求 TPOT = (最后 token − 首 token 时刻)/(输出数−1)；单 token 为 null。
  汇总是有效请求的算术平均，不是 token 加权平均。
- 完成请求延迟 = 最后 token 交付 − arrival；P95 为 nearest rank，ceil(0.95×N)。
- 吞吐量 = 输出 token／完成请求数除以“首次到达至所有请求完成或拒绝”的窗口。
  包含中间空闲，不含首次到达前空闲；prompt token 不计入输出吞吐。
- 拒绝请求单列 decision 时间、拒绝延迟、原因，不填充为零 TTFT，不计入完成吞吐或延迟分位数。
  同时报 submitted/completed/rejected、拒绝率和原因分布。拒绝困难请求会改变样本分布，
  不能仅凭更低延迟宣称策略更优。
- slot_utilization = 时间加权活跃 slot 数/(窗口×批上限)，包括已准入但本轮没有预算的 prefill；
  service_slot_utilization 去掉空闲时间。scheduled_slot_utilization 仅统计本轮真正执行的请求。
  三者都不是 GPU 利用率或有效计算利用率。
- inter_token_p95_ms/max_inter_token_ms 统计所有完成请求的相邻输出 token 间隔；P95 按
  token 间隔样本计算，与按请求取平均的 TPOT 不同。无间隔样本时为 null。
- 空工作负载或全拒绝的完成吞吐为 0，完成延迟分位数为 null；零时长窗口也返回 0 吞吐。
- JSON 包含完成与拒绝列表；`--trace` 额外输出每轮工作、KV 事件及等待/拒绝事件。

无限容量 mixed/B=2 保持第一阶段结果：Static 18 轮、Continuous 14 轮，模拟输出
吞吐约 1031.75／1226.42 tokens/s，TTFT P95 23.8／20.0 ms。不是 TinyInfer 实机加速数据。

## 文件与验收

| 文件 | 职责 |
|---|---|
| backend.py | 请求、step 接口、确定性虚拟后端 |
| scheduler.py | FIFO、KV 准入/拒绝、cohort/补位、完成/异常清理 |
| kv_cache.py | token/block 取整、预留、释放、峰值和事件账本 |
| metrics.py | 完成/拒绝分开统计，模拟延迟/吞吐/KV 指标 |
| benchmark.py | 内置/JSON 工作负载、独立参考对照、表格/JSON |
| test_scheduler.py | 第一阶段 12 项回归 |
| test_kv_cache.py | 第二阶段 17 项容量、拒绝、回收及异常测试 |
| test_chunked_prefill.py | 第三阶段 10 项进度、预算、decode 优先、输出/回收与手算取舍测试 |
| requests.json | 可编辑输入样例 |
| realtime.py | 真实到达等待、完成后 token 交付与墙钟驱动 |
| tinyinfer_worker.cpp / tinyinfer_backend.py | 真实 attention/logits/KV 数值后端及持久进程适配 |
| real_benchmark.py / test_real_backend.py | 真实计时、完整前缀参考、5 项集成测试 |
| test_tensor_batching.py | 5 项真实张量 batching 测试：logits、因果边界、混合阶段、请求映射、算子计数与异常清理 |
| native/paged_kv_cache.h / test_paged_attention.py | KV pool、block table、直接分页 online softmax、gather 对照与 7 项集成测试 |
| dynamic_scheduler.py / test_advanced_serving.py | 按需容量与分块重算、共享/COW、5 项 CPU 和 2 项真实 CUDA 集成测试 |
| native/cuda_paged_attention.h/.cu | 可选真实 GPU 分页 attention、stream/缓冲复用与传输/launch 计数 |
| results/ | 保留模拟历史；stage4/ 含 44 项测试、ASan、真实指标与构建环境 |

第四阶段已保留模型/精度、环境、固定到达/长度分布、warmup、独立 worker 次数、
每请求时间戳、拒绝及资源计数。详见 [第四阶段验收与限制](STAGE4.md)。
