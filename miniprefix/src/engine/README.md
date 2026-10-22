# Engine

状态：规划目录，尚无执行实现。

负责 RequestState、事件队列、生成迭代、Greedy 采样、EOS/长度结束、取消、失败与
shutdown。持有请求已提交 token 与时间戳，协调 Scheduler、KVManager 和 ModelRunner。
恢复 KV 不改变已提交输出；内存回收等待设备 completion。

首批交付：tokenized submit/step/poll_events、显式状态机、CPU runner 调用和生命周期
回归。不要将队列逻辑放进模型 forward，或为未实现路径返回伪 token。
