# PrefixIndex

状态：include/miniprefix/prefix_cache.h 声明 ExactPrefixCache，算法实现在 src/kv/prefix_cache.cpp，
由统一 KVManager 持有。CPU/CUDA runner 通过 manager 调用，不直接操作缓存或页池。
完整 token 历史的 vector key、完整页面匹配、请求/缓存引用、页面 LRU 和压力淘汰已接入。
默认关闭，跨同一 runner 的请求保留。边界与实际验收见 ../../docs/prefix_cache.md。

负责 cache namespace、最长因果前缀匹配、索引更新、访问记录和驱逐候选。
索引返回可复用范围与 KV lease，不持有独立物理 allocator，不直接释放或写入共享块。
跨 generate 保留缓存；flush_cache 与 reset_stats 分离。

首版精确匹配基线已接入，后续依据测量选择块哈希或 radix 等索引结构。
不同模型/权重/位置编码/布局不能误命中，未来 adapter 也必须进入 cache identity。
命中率、跳过的 token、查找成本与保留内存分别计数。
