# 验证布局

状态：test_baseline.py 验证手算计时、单 token TPOT、分位数与错误样本拒绝；
tools/verify_real_model.py 提供实际 checkpoint 的 CPU/CUDA 数值与生命周期验收。
新增 C++ paged_kv_test.cpp 验证页面所有权、跨页地址、容量失败原子性与回收复用，
CTest 入口为 miniprefix_paged_kv_tests。真实分页测试和配置见 ../docs/paged_kv.md。
后续原生引擎更细的单元与故障隔离覆盖继续按下述布局补齐。

后续按 unit / integration / differential / cuda 分类组织：

- unit：请求状态机、计划预算、KV 事务/引用/COW、prefix identity、取消与终态。
- integration：打包 prefill/decode、共享、抢占恢复、事件只交付一次、失败回收。
- differential：真实权重对独立框架 logits/Greedy；原 worker 仅作迁移参考，不能作为
  真实预训练模型的唯一数值 oracle。
- cuda：真实设备数值、GPU 常驻、同步/异步资源生命周期与传输计数。

不把 skip、模拟值、空测试或 CPU fallback 当 GPU 验收。随机性固定 seed、容差明确；
仅为实际行为编写测试，不为文档目录或空实现堆测试。
