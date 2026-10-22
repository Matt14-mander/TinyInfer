# 验证布局

状态：验收规划，当前没有新的引擎测试 target。

后续按 unit / integration / differential / cuda 分类组织：

- unit：请求状态机、计划预算、KV 事务/引用/COW、prefix identity、取消与终态。
- integration：打包 prefill/decode、共享、抢占恢复、事件只交付一次、失败回收。
- differential：真实权重对独立框架 logits/Greedy；原 worker 仅作迁移参考，不能作为
  真实预训练模型的唯一数值 oracle。
- cuda：真实设备数值、GPU 常驻、同步/异步资源生命周期与传输计数。

不把 skip、模拟值、空测试或 CPU fallback 当 GPU 验收。随机性固定 seed、容差明确；
仅为实际行为编写测试，不为文档目录或空实现堆测试。
