# KVManager

状态：CPU/CUDA runner 已统一使用 KVManager。include/miniprefix/ 中的 kv_manager.h、
paged_kv.h、prefix_cache.h 保留接口，实现在本目录的三个 .cpp 中，链接 MiniPrefix::kv。
该库只依赖 C++17 契约，不依赖 Tensor、CUDA SDK 或 Python。

负责布局描述、物理页 metadata、sequence block table、引用/lease 和追加事务。
布局需表达 layers、KV heads、head dimension、block size、dtype 与设备，消除单层/D=16
假设。区分请求引用、缓存引用和在途设备访问；metadata 与 payload 的所有权一致。

KVManager 持有页池、ExactPrefixCache 和所有序列 metadata，统一校验、恢复、提交、回滚、
释放与统计。backend 持有实际 K/V payload、设备页表、stream/completion，并在回收前同步。
Append 事务提交前不推进 used/token history/forward 计数，异常退出归还新增页面；
backend 必须先结束在途设备访问，再 rollback/销毁事务。事务不能超过 manager 生命周期。
部分页面共享/COW 仍待实现；当前只共享完整不可变页，尾页私有。
前缀搜索与 LRU 选择属于 ExactPrefixCache。KVManager 的 reserve_pages 负责计算追加
所需的新页数、按实际空闲页触发逐次淘汰，最后调用 KvBlockPool::reserve。缓存提供
evict_one()，仅移除一个缓存引用；活跃请求仍持有的页不会变为空闲页，因此每次都需
重新检查物理容量。缓存为空时停止，保持原有 pool exhausted 错误及请求映射不变。
失败申请或 Append 回滚不会恢复已经淘汰的缓存条目，淘汰计数仍累计；只有新分配的
请求页面会回滚。输入/context 校验仍在任何压力淘汰前完成。

2026-10-28 容量协调重构：CPU CTest 51/51、CUDA CTest 58/58 通过；新增容量测试也在
原 KVManager/ExactPrefixCache 实现上通过。真实 CPU/CUDA worker 差分的 94 个检查点、
491,520 个 FP32 logits、事件/逐轮计数与全部 STATS 字段一致，证据见
[kv-capacity-differential.json](../../docs/acceptance/kv-capacity-differential.json)。

KVManager 另持有请求最大页数的逻辑预约，协调 Engine 准入和未预约诊断操作，
不提前分配 payload，不按预期前缀命中打折。见 [容量准入](../../docs/capacity_admission.md)。
