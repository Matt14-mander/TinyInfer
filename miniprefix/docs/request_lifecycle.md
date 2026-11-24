# 请求队列、超时与历史 ID 生命周期

Engine 与 Python 客户端不再永久保存全部请求 ID。请求容量不足在 submit 边界同步拒绝，不将 Engine 标记为 failed，
不生成终态事件、不占用请求 ID，也不进行 KV 预约。调用方可在推进请求、消费事件之后重试。

## 默认配置

`EngineConfig` 原有四个字段之后增加以下字段，原来的聚合初始化仍可使用：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| max_waiting_requests | 1024 | 已接受、尚未准入的等待请求上限，不包含 active 请求；必须大于零 |
| max_event_slots | 65536 | 未消费事件和活跃/等待请求未来事件预约的合计上限；必须大于零 |
| history_capacity | 4096 | 已消费终态后的 ID 去重窗口容量；零表示不保留 |
| history_ttl_ms | 300000 | 从终态被 poll_events 取走起算的保留时间；零表示不保留 |

每次提交保守预约 `max_new_tokens + 1` 个事件槽。产生事件消耗该请求的预约；终态释放未用额度，poll_events 释放已取走事件的槽。
已接受请求不需要等客户端消费事件才能完成计算或发送终态。事件额度不够时抛 `QueueFullError`，与等待队列满使用同一异常类型、不同错误信息。
单个请求的输出预算超过事件容量时也会立即拒绝，不存在等待后自行成功的承诺。

这些是请求/事件/ID 数量限制，不是进程 RSS 或设备显存限制。KV 容量仍由现有共享感知准入管理。
submit 不自动运行超时清理；队列中有过期请求时，调用方先 step 再重试。事件消费是显式的，不静默丢弃事件。

## 协作式超时

`Request::timeout_ms` 默认为 0，表示禁用；正值从成功提交时的单调时钟起算，包含排队、prefill 和 decode。
每次 step 开始取得一个时钟快照，先清理所有过期 active 请求，再清理所有过期等待请求，包括被队首挡住的请求。
清理使用现有 terminal 路径：释放序列和预约，发出一个不带 token 的 `FinishReason::Timeout`，正常超时不使 Engine 失败。
缓存已经发布的历史仍按原引用规则保留。后端释放失败则沿用原有致命错误清理路径，并报告 Error。

一个 step 内的超时快照不变：不打断 CUDA kernel、不丢弃已经返回的整批结果。计算期间跨过 deadline，当前 step 仍可产生 token；
未完成请求在下一次 step 超时。如果当前 step 已经完成 Length/EOS，则该终态优先。显式 cancel/shutdown 在超时检查之前发生时，结果为 Cancelled。
因此这不是硬实时截止时间；调用方不调用 step 时，也没有后台线程主动发出超时事件。

测试可通过 Engine 构造函数的第三个 `Now` 参数注入单调时钟，不依赖 sleep 或 GPU 执行速度。

## ID 保留与复用

等待、活跃以及终态事件尚未取走的 ID 始终禁止重复提交，不受历史容量或 TTL 淘汰影响。
poll_events 相当于把整批事件交给调用方：终态 ID 此时进入 FIFO 历史窗口，容量超限淘汰最早消费的 ID，TTL 到期也可淘汰。
到期清理在 submit、step、poll_events 时惰性执行；不要求空闲时运行定时器。

只有离开窗口后，ID 才能用于新请求。窗口内重复提交只拒绝，不重放旧结果；这不是永久幂等性协议。
若应用需要跨会话永久去重，应自行生成唯一业务 ID 或保留业务记录。原有“整个 Engine 生命周期禁止重用 ID”的行为已改为有限窗口。

Python 客户端对用户 ID 使用同样的容量/TTL 窗口，在收到终态后开始计时；只保留待完成请求和窗口内记录。
wire handle 单调递增、不复用，避免旧句柄对应新请求；超过协议有符号 32 位范围时要求新建会话。
客户端先完整解码一批事件，再淘汰映射，因此同批 token 与终态可以正确关联。

## Worker 与 Python

原 SUBMIT、STATS 以及协议版本 3 的既有响应保持格式兼容。新操作为：

```text
SUBMIT_TIMEOUT ID MAX_NEW EOS TIMEOUT_MS TOKEN_COUNT TOKENS...
LIFECYCLE_STATS
```

Timeout 的终态编号为 4，原 EOS/Length/Cancelled/Error 的 0–3 不变。新客户端识别 `finish_reason="timeout"`。
LIFECYCLE_STATS 返回 waiting、active、buffered_events、event_slots、tracked_ids、history_ids 六个计数；不会为了读取统计修改历史。
旧客户端不主动申请超时时，不会收到新的 Timeout 终态。

worker 启动位置参数在 PREFIX_BLOCKS 后增加四个可选参数，顺序为 MAX_WAITING、EVENT_SLOTS、HISTORY_CAPACITY、HISTORY_TTL_MS。
Python 默认限制使用原启动参数格式，保留连接历史 worker 的能力；非默认限制、新命令需要本次更新后的 worker。

```python
with NativeEngine(worker, model, "cuda", max_waiting_requests=128,
                  max_event_slots=8192, history_capacity=1024,
                  history_ttl_ms=60000) as engine:
    engine.submit("turn-1", prompt, max_new_tokens=64, timeout_ms=5000)
    while engine.pending:
        for event in engine.step():
            consume(event)
    print(engine.lifecycle_stats())
```

Engine 的时长字段是 uint32 毫秒；worker/Python 参数限定为非负有符号 32 位整数。Python 会在启动子进程前验证容量配置。
当前限制覆盖生成请求；已有调试用 FORWARD/REUSE 诊断句柄仍由调用方显式释放。

## 验证

CPU 56/56、CUDA 65/65、Python 29/29 回归通过。确定性测试覆盖队列拒绝与重试、未消费事件背压、超时跨 step 边界、
非队首过期请求、Length 优先、释放失败、ID 消费前固定/消费后 TTL 与容量淘汰，以及 10,000 次请求周转的 ID/事件计数上限。
CPU/CUDA runner 各执行 20 波两请求共享前缀生成后超时，检查预约归零、无重复 token/终态及最后 KV/页表释放。

SmolLM2-135M 的 CPU/CUDA worker 与 Python 客户端端到端验证覆盖过期前不执行 forward、超时编码、队列满拒绝后重试、
取消、TTL 后复用、24 次历史轮换、事件预算拒绝与缓存释放。
此轮是正确性与生命周期验收，不据短程检查宣称完成无限期运行或新的性能基线。

[真实模型结果](acceptance/lifecycle-resources.json) · [CPU 回归](acceptance/lifecycle-cpu-tests.log) · [CUDA 回归](acceptance/lifecycle-cuda-tests.log)

```text
python miniprefix/tools/verify_lifecycle.py --model MODEL.mpf --cpu-worker CPU_WORKER --cuda-worker CUDA_WORKER --output lifecycle-resources.json
```
