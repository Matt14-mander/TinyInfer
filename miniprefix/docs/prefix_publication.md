# 前缀发布与快照元数据优化

Engine 仍在每次成功 append、logits 校验之后且采样之前发布全部已提交历史。
旧的部分页快照继续独立存在，缓存引用、逐页 LRU touch、容量淘汰次序、最长前缀
选择及 COW 行为不变。不把延迟发布或删除旧快照当作纯开销优化。

ExactPrefixCache 内部新增两项优化，公共接口与 worker 协议不变：

- 保存上一次发布的非拥有路径提示。每页仍比较完整 token 边和父节点，匹配才省去
  map 查找；切换请求、分叉和缩短历史都会重新核对。路径不持有 KV 引用，不能使
  已淘汰条目命中。结构节点被删除前截断对应提示后缀，避免节点地址复用造成误命中。
- 通过 C++17 map node handle 保留一个已删除的空叶节点（含其 key 存储）和一个
  available 索引节点。新快照可以复用这些分配；不是合并快照或共享不同 token 的 key。
  空闲节点不持有 KV 引用，最多各一个，clear 时销毁。路径容量最多为缓存条目上限，
  clear 后逻辑路径为空，vector 存储保留用于后续请求。

新 key/available 元数据仍在获取缓存引用与淘汰前准备。retain 失败时撤销 lookup
并剪枝；已有条目保持原顺序逐页 touch。清空仍按原有字典序释放，保持物理页 free-list
顺序。部分页仍只发布有效 token，至少一个 prompt token 留给 logits 的约束未变。

## 验证与测量

差分回归使用独立全历史键 oracle，逐操作比较命中长度、物理页、引用数、LRU 淘汰
与清理结果；新增连续发布、长短切换、跨请求分叉、淘汰后重发、节点复用后的失败恢复。
CPU/CUDA 设备回归覆盖不断增长的快照、容量轮换、源请求释放后命中、模型失败回滚、
重试 logits 与资源归零。原有 8192 层索引深度及随机差分回归保留。

`miniprefix_prefix_publication_bench` 单独测量主机元数据。页长 16、完整前缀 32 页，
缓存 64 条目；重复同一历史、96 个共享前缀尾快照轮换、96 个不同根分支轮换。
每种先运行 10000 次 warmup，再记录 10 组、每组 10000 次发布的平均 ns 和普通
operator new 分配次数。全部源序列预先构造并保持存活，不执行模型或 payload 复制。
它用于定位元数据收益，不代表真实生成延迟。

真实模型沿用上一轮 `compare_cuda_cow.py`，脚本和输入未改，仍检查输出、复用量、
forward 数、COW 次数/字节数与资源释放。基线 worker 为 d670742，并与上一轮记录
的 SHA256 核对。旧/新交替执行，off/cold/hot 分别测量；6 轮正式测量不插桩，
3 轮单独 profiling 用已有 publish_ns 观察发布开销，不从端到端延迟中直接相减。

```text
python miniprefix/benchmarks/compare_cuda_cow.py --model-dir MODEL_DIR --cpu-worker CPU_WORKER --before OLD_CUDA_WORKER --after NEW_CUDA_WORKER --repeats 6 --profiles 3 --output RESULT.json
```

吞吐为单请求输出 tokens/s。加载、分词、warmup、热缓存 prime 不在计时区间；
IPC 和事件交付在内。原始数据保留所有样本和异常值，不以一次均值或比值宣称稳定收益。
结果及环境见[本次验收](acceptance/prefix-publication-review.md)。
