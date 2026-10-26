# 最小精确前缀缓存

2026-10-25：真实 CPU/CUDA ModelRunner 已支持完整页面的精确前缀复用。
默认关闭（`prefix_cache_blocks=0`），打开后跨同一 NativeEngine 的请求保留；
进程退出、Engine shutdown 或失败时全部释放。缓存不开启跨进程持久化。

## 匹配与所有权

- `ExactPrefixCache` 位于 `include/miniprefix/prefix_cache.h`；与 KvBlockPool 共用页面
  引用计数。每个索引项持有一个完整页面，活跃请求也各持有引用；只有引用归零才回收。
- 每页的 key 是从位置 0 到该页末尾的**全部 token ID**，用 vector 精确比较，包含
  所有先前位置的因果上下文。后续页面 token 相同但此前 prefix 不同，不能误命中。
- 索引限定在一个固定权重、配置、FP32 布局和 RoPE 的 runner 内，不与另一个模型或
  runner 交换页面。运行时更换模型/adapter/位置编码不受支持，必须新建 runner。
- 按顺序匹配连续完整页面，遇到缺页或不一致即停止；只复用完全相同的前缀。
  不匹配文本、语义、非零起始位置，也不依赖可能碰撞的 hash 值。
- 命中最多复用 `floor((prompt_tokens-1)/block_tokens)*block_tokens` 个 token。
  因为首版不缓存 logits，始终保留至少一个 prompt token 重新计算；prompt 恰好页对齐
  时会重算最后一整页，避免拿不到首个输出 logits 或改写共享页。
- 只在 prompt prefill 成功结束时发布完整页面；生成阶段不自动缓存。未满尾页私有，
  后续输入写新页面或请求自己的尾页，永不改写共享 KV，因此本版本无需 COW。
  这不等于已实现任意位置 fork 或部分页面共享。
- Engine 在第一次 forward 前恢复命中页和 used/位置，prefilled 直接跳过命中 token。
  命中 token 不占本轮 forward token budget。CPU/CUDA attention 都直接访问同一物理页；
  CUDA 只上传恢复的页号，不复制 KV payload，层内 RoPE/causal position 从恢复位置继续。

## 容量与 LRU

容量单位为缓存页面条目，取值 `0..kv_blocks`，不另分配 KV pool。发布新条目时按页面
LRU 淘汰；申请物理页不足时也先淘汰缓存引用。淘汰不会撤销活跃请求引用，清空缓存
也不影响正在 decode 的请求。若活跃引用仍占满物理池，则维持现有明确失败语义：
Engine 结束所有未完成请求，clear 同时回收请求与缓存引用，不做抢占或 CPU fallback。

为让长 prompt 在小缓存中仍能保留根前缀，一次发布最多前 `prefix_cache_blocks` 页。
页面独立 LRU 可能留下缺祖先的后续条目；此时查找停在缺页处，不会不安全地复用它们。
首版采用完整 vector key 和有界 map，尚无 radix/tree、块哈希或查找时间指标。

## 使用与统计

```python
from miniprefix.bindings.python.client import NativeEngine

# prompt 是 tokenizer 产生的同一组 token ID；复用发生在同一 worker 生命周期内。
with NativeEngine(worker, model, backend="cuda", context=256,
                  block_tokens=16, kv_blocks=128, prefix_cache_blocks=32) as engine:
    for key in ("cold", "warm"):
        engine.submit(key, prompt, max_new_tokens=8)
        events = list(engine.events())
    stats = engine.stats()
    engine.clear_prefix_cache()  # 不重置累计计数；活跃请求可以继续持有已命中页面
```

C++ 使用 `KvCacheConfig{16,128,32}`。生成 CLI 支持 `--prefix-cache-blocks 32`，
但单次 CLI 只生成一个请求，进程结束会释放缓存；展示复用应使用同一个 NativeEngine。
worker 协议版本为 `READY MINIPREFIX 3`，追加参数 `PREFIX_BLOCKS`，新增 `CLEAR_PREFIX`，
diagnostic `CACHE`/`REUSE`；client 和 worker 必须同步更新。

| STATS 字段 | 语义 |
|---|---|
| prefix_cache_capacity | 配置的缓存页面容量，0 为关闭 |
| kv_cached_blocks | 当前索引持有的完整页面数 |
| kv_cache_only_blocks | 只有缓存引用、没有请求引用的物理页面数 |
| prefix_hits / prefix_misses | 开启缓存后的请求恢复次数，命中需跳过至少一页 |
| prefix_reused_tokens | 实际恢复并跳过 forward 的累计 token 数 |
| prefix_evictions | LRU/容量压力造成的累计淘汰次数，不包括显式 clear |

`kv_live_blocks`/`live_kv_bytes` 现在包含请求和缓存持有的物理页面，唯一页面只算一次；
开启缓存时 idle 后非零是预期行为。设备 block table 仍随请求释放，物理池预留量不变。
显式 clear 后且所有请求结束时，live KV 必须归零。热请求的 `forward_tokens` 和输入
H2D 应减少 `prefix_reused_tokens`，不能只报告命中却仍然重算命中部分。

## 重现验证

沿用 [真实模型准备与构建](real_model.md)，使用已存在的模型与依赖：

```powershell
$py = 'build-miniprefix-env/Scripts/python.exe'
& $py miniprefix/tools/verify_prefix_cache.py --model-dir build-miniprefix-models/smollm2-135m-instruct --cpu-worker build-phase54-cuda/miniprefix/miniprefix_worker.exe --cuda-worker build-phase54-cuda/miniprefix/miniprefix_cuda_worker.exe --runs 5 --output benchmark-results/miniprefix/exact-prefix-verification.json
```

独立 Transformers CPU FP32 full-prefix oracle 在计时前准备。验证关闭/冷/热缓存输出、
raw logits、对齐边界、同页不同因果历史、分支、并发共享、取消、clear、LRU、压力淘汰
与不可抢占活跃引用时的失败清理。另保留原无缓存真实模型和 CTest 回归。
结果见 [精确前缀验收](acceptance/exact-prefix-smollm2-fp32.md)。
2026-10-26 补齐长 suffix 的分块计数、调度边界和专门验收，见 [增量 prefill](incremental_prefill.md)。

该实验是单个暖进程内的五组 paired flows，参考计算、tokenizer、clear、stats、启动和
warmup 在窗口外；不是原八请求持续到达负载，也不取代历史无缓存 dense/paged 基线。
不声称生产尾延迟或跨进程统计置信度。更大共享 system prefix、多轮对话和饱和流量
仍需独立负载扩展，COW/抢占/跨请求张量 batching 仍是后续工作。
