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
前缀搜索与驱逐策略属于 PrefixIndex，KVManager 只执行可证明安全的 retain/release。
