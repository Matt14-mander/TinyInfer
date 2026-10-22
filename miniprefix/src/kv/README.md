# KVManager

状态：规划目录；现有参考为 serving_scheduler/native/paged_kv_cache.h。

负责布局描述、每层物理 pool、sequence block table、引用/lease、COW 和资源事务。
布局需表达 layers、KV heads、head dimension、block size、dtype 与设备，消除单层/D=16
假设。区分请求引用、缓存引用和在途设备访问；metadata 与 payload 的所有权一致。

首批交付：参数化布局与原始 block 行为回归，再实现设备常驻的追加/COW/table 更新。
前缀搜索与驱逐策略属于 PrefixIndex，KVManager 只执行可证明安全的 retain/release。
