# Engine

状态：engine.cpp 已有 submit/step/poll_events/cancel/shutdown、decode 优先与分块 prefill、
EOS/长度/错误终态。active 请求的 runner 调用仍逐个执行；高级调度与重算迁移待完成。
已校验命中范围并从命中位置分块执行剩余 prompt；last_step_stats 分开报告复用、实际
prefill/decode token。长 suffix 与 decode 混合验证见 ../../docs/incremental_prefill.md。

终态统一由 `terminal` 提交：正常完成可携带最后一个 token，取消/失败不携带 token；
EOS 仍优先于长度限制。正常完成/取消先释放 resident sequence，执行失败路径在 runner
全局 clear 后使用 `RunnerCleared` 模式，禁止再次按请求释放。只有终态事件成功入队后
才标记 finished，`remove_finished` 统一移除队列中的已完成请求，保留其他请求的顺序。

同一轮中已经完成的请求不会被后续失败重复终结，已经发出的非终态 token 保持不变。
shutdown 按 active、waiting 顺序终结；若某个 release 抛异常，已完成项会先移除再
传播异常，重试 shutdown 不重复发事件或释放。失败的 release 不被标记为已完成。
独立的 `engine_terminal_test.cpp` 覆盖这些行为；runner 全局清理自身抛异常的修复仍
由独立 PR 处理，本次仅统一终态提交与移除，不调整全局失败清理策略。

负责 RequestState、事件队列、生成迭代、Greedy 采样、EOS/长度结束、取消、失败与
shutdown。持有请求已提交 token 与时间戳，协调 Scheduler、KVManager 和 ModelRunner。
恢复 KV 不改变已提交输出；内存回收等待设备 completion。

首批交付：tokenized submit/step/poll_events、显式状态机、CPU runner 调用和生命周期
回归。不要将队列逻辑放进模型 forward，或为未实现路径返回伪 token。
