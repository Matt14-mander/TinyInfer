# Serving Scheduler 学习实验

**第一、二阶段已实现：Static / Continuous Batching，以及 KV 容量约束与请求拒绝。**
Python 3.7+ 标准库、确定性模拟后端、虚拟时钟，无需模型或 GPU。
本例独立于 TinyInfer C++ Runtime，不改变 Phase 5.4 CUDA 开发状态。

## 四阶段计划

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | Static / Continuous Batching | 已实现；FIFO、边界补位、输出一致性、手算 trace |
| 2 | KV 容量约束、请求拒绝 | 已实现；block 预留、等待／拒绝、完成／异常回收 |
| 3 | Chunked Prefill | 待实现；token budget、prefill 进度、decode 公平性及输出一致性 |
| 4 | 真实后端及性能测量 | 待实现；真实 prefill/decode/logits/KV、完成同步、真实 TTFT/TPOT/吞吐/P95 |

KV 账本不等于真实显存分配。第四阶段需要自回归模型接口，当前静态 MLP/ONNX Prepared
入口不能直接充当 LLM serving 后端；还需替换虚拟时钟为真实到达队列和等待逻辑，并处理
EOS、取消和请求失败，不能只替换 `Backend.step()` 就宣称得到生产服务。

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
- 新请求一次完成全部 prefill，并产出首 token；已有请求 decode 一个 token。
  同一轮可以混合两种工作，所有 token 在 step 结束时交付。
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
长 prompt 阻塞 decode 的情况，等待第三阶段 Chunked Prefill 改进；当前不切分 prompt。

## 指标口径：全部是模拟值

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
- slot_utilization = 时间加权活跃 slot 数/(窗口×批上限)；另报去掉空闲时间的
  service_slot_utilization。它们不是 GPU 利用率或有效计算利用率。
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
| requests.json | 可编辑输入样例 |
| results/ | 保留第一阶段历史结果；stage2/ 含当前 29 项测试与容量实验 |

第四阶段应另存模型/精度、环境、batch/token/KV 预算、长度/到达分布、warmup、独立运行次数、
每请求时间戳及失败/拒绝口径。模型完成并交付 token 后才能计时，不能用成本公式或只提交
kernel 的时间代替真实 TTFT/TPOT。
