# Engine

状态：engine.cpp 已有 submit/step/poll_events/cancel/shutdown、decode 优先与分块 prefill、
EOS/长度/错误终态。active 请求的 runner 调用仍逐个执行；高级调度与重算迁移待完成。
已校验命中范围并从命中位置分块执行剩余 prompt；last_step_stats 分开报告复用、实际
prefill/decode token。长 suffix 与 decode 混合验证见 ../../docs/incremental_prefill.md。

负责 RequestState、事件队列、生成迭代、Greedy 采样、EOS/长度结束、取消、失败与
shutdown。持有请求已提交 token 与时间戳，协调 Scheduler、KVManager 和 ModelRunner。
恢复 KV 不改变已提交输出；内存回收等待设备 completion。

首批交付：tokenized submit/step/poll_events、显式状态机、CPU runner 调用和生命周期
回归。不要将队列逻辑放进模型 forward，或为未实现路径返回伪 token。
