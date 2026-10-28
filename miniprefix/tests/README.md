# 验证布局

状态：test_baseline.py 验证手算计时、单 token TPOT、分位数与错误样本拒绝；
tools/verify_real_model.py 提供实际 checkpoint 的 CPU/CUDA 数值与生命周期验收。
新增 C++ paged_kv_test.cpp 验证页面所有权、跨页地址、容量失败原子性与回收复用，
CTest 入口为 miniprefix_paged_kv_tests。真实分页测试和配置见 ../docs/paged_kv.md。
prefix_cache_test.cpp / miniprefix_prefix_cache_tests 验证精确因果 key、共享引用、对齐
logits 边界、LRU 和活跃 lease。tools/verify_prefix_cache.py 验证真实 CPU/CUDA 共享
与 cold/warm compute counter；口径见 ../docs/prefix_cache.md。
后续原生引擎更细的单元与故障隔离覆盖继续按下述布局补齐。
engine_prefill_test.cpp / miniprefix_engine_prefill_tests 验证命中后逐 chunk token/绝对位置、
budget、首事件边界、取消与失败清理；verify_incremental_prefill.py 提供真实 CPU/CUDA 对照。
scheduler_test.cpp / miniprefix_scheduler_tests 独立链接 MiniPrefix::scheduler，验证
准入数、稳定 decode 优先顺序，以及 756 种 phase/budget/chunk 组合与原策略一致。
engine_prefill_test.cpp 同时检查同一步缓存发布/复用、预算耗尽时不提前创建序列、
完成后下一轮 FIFO 补位、取消及生成事件顺序；这些轨迹也在提取前 Engine 上验证过。
engine_terminal_test.cpp / miniprefix_engine_terminal_tests 验证 EOS/长度优先级、最终 token、
waiting/未创建/resident 请求取消、已有完成和流式事件后的全局失败、release 异常、
shutdown/析构与重试时的事件顺序和资源所有权。默认运行全部测试；
`--legacy-compatible` 仅供对照旧 Engine，跳过新增的部分 shutdown 失败后重试不重复终态保证。
test_prefix_benefit.py 验证每个负载的模式顺序独立轮换、命中必须减少实际 forward/H2D，
以及同轮配对速度比和 seed 成本摊销，避免性能报告的口径错误。
kv_manager_test.cpp 验证统一追加事务、输入快照、提交/异常回滚、共享 lease、重置、
压力淘汰及累计统计；engine_prefill_test.cpp 同时固定 RunnerStats 布局并检查设备计数不被覆盖。

后续按 unit / integration / differential / cuda 分类组织：

- unit：请求状态机、计划预算、KV 事务/引用/COW、prefix identity、取消与终态。
- integration：打包 prefill/decode、共享、抢占恢复、事件只交付一次、失败回收。
- differential：真实权重对独立框架 logits/Greedy；原 worker 仅作迁移参考，不能作为
  真实预训练模型的唯一数值 oracle。
- cuda：真实设备数值、GPU 常驻、同步/异步资源生命周期与传输计数。

不把 skip、模拟值、空测试或 CPU fallback 当 GPU 验收。随机性固定 seed、容差明确；
仅为实际行为编写测试，不为文档目录或空实现堆测试。
