# KVManager

状态：分页池已在 include/miniprefix/paged_kv.h 实现，CPU/CUDA runner 共用
页面所有权与事务回滚，直接寻址 backend 自有的物理 payload。详见 ../../docs/paged_kv.md。

负责布局描述、每层物理 pool、sequence block table、引用/lease、COW 和资源事务。
布局需表达 layers、KV heads、head dimension、block size、dtype 与设备，消除单层/D=16
假设。区分请求引用、缓存引用和在途设备访问；metadata 与 payload 的所有权一致。

已交付参数化多层/GQA 布局、设备常驻追加、增量 table 更新和完整页面的请求/缓存引用。
部分页面共享/COW 仍待实现。
前缀搜索与驱逐策略属于 PrefixIndex，KVManager 只执行可证明安全的 retain/release。
