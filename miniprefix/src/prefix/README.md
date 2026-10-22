# PrefixIndex

状态：规划目录；已有精确 vector 快照实现留在原 serving 示例中。

负责 cache namespace、最长因果前缀匹配、索引更新、访问记录和驱逐候选。
索引返回可复用范围与 KV lease，不持有独立物理 allocator，不直接释放或写入共享块。
跨 generate 保留缓存；flush_cache 与 reset_stats 分离。

首版先接入可验证的精确匹配基线，M5 再依据测量选择块哈希或 radix 等索引结构。
不同模型/权重/位置编码/布局不能误命中，未来 adapter 也必须进入 cache identity。
命中率、跳过的 token、查找成本与保留内存分别计数。
