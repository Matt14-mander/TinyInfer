# Scheduler

当前实现为独立 C++17 策略组件，接口在 `include/miniprefix/scheduler.h`，
实现位于此目录，链接目标 `MiniPrefix::scheduler`，仅依赖公共 contracts。

- `admission_count` 在每轮开始决定 FIFO waiting 队列可补入多少请求。
- `plan` 固定本轮候选顺序：已有 decoder 优先，各组保持 active 顺序。
- `SchedulePlan::take_input` 分配一个 decode token，或受 suffix、chunk 和剩余预算
  三者约束的 prefill chunk。每个候选每轮只执行一次，剩余预算不重访请求。
- Engine 在候选有预算时才创建序列并恢复前缀，然后确定 chunk。前面的请求可在
  同一步发布缓存，供后面的请求使用；复用 token 不消耗输入预算。
- 完成/取消后的空位只在下一次 step 开始补入；保留现有小预算下的固定优先顺序，
  不额外引入轮转、公平性、抢占或动态 KV 容量准入。

Engine 继续拥有 waiting/active 状态与生命周期、调用 runner、采样并提交事件；
Scheduler 不持有请求、KV lease、设备地址，也不执行 kernel。计划中的 active 索引
只在本轮有效，Engine 执行计划期间不删除或重排 active，最后统一移除已完成请求。

回归：`miniprefix_scheduler_tests` 比较原有两遍排序/预算策略的 756 种组合；
`miniprefix_engine_prefill_tests` 覆盖跨步轨迹、同一步前缀发布/复用、延迟创建、
边界补入、取消、终态和统计。更完整的容量试算与抢占属于后续扩展。
