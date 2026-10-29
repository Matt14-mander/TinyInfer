# PrefixIndex

状态：include/miniprefix/prefix_cache.h 声明 ExactPrefixCache，算法实现在 src/kv/prefix_cache.cpp，
由统一 KVManager 持有。CPU/CUDA runner 通过 manager 调用，不直接操作缓存或页池。
精确页 token trie、完整页面匹配、请求/缓存引用、双向链表 LRU 和压力淘汰已接入。
默认关闭，跨同一 runner 的请求保留。边界与实际验收见 ../../docs/prefix_cache.md。

ExactPrefixCache 不再提供 reserve，也不计算追加页数或检查 pool 空闲容量。
公开 evict_one() 按原 LRU 策略移除一个缓存引用并返回 true，空缓存返回 false 且
不增加淘汰计数。条目上限触发的淘汰仍由 publish 处理；物理池压力下的协调由
KVManager 执行。调用方追加数据应使用 KVManager::prepare_append，而非缓存分配接口。

负责 cache namespace、最长因果前缀匹配、索引更新、访问记录和驱逐候选。
索引返回可复用范围与 KV lease，不持有独立物理 allocator，不直接释放或写入共享块。
跨 generate 保留缓存；flush_cache 与 reset_stats 分离。

索引节点只存一页 token；缺失缓存祖先可保留为结构路径，空路径会剪枝。
匹配/发布使用无临时 key 分配的透明查找，LRU touch/选择为 O(1)，剪枝另按回收节点数计。
不同模型/权重/位置编码/布局不能误命中，未来 adapter 也必须进入 cache identity。
命中率、跳过的 token、查找成本与保留内存分别计数。
