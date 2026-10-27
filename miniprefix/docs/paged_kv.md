# 真实模型分页 KV

CPU 和 CUDA runner 已将每序列的整段 context KV 替换为共享物理 block 池。
默认每页 16 token、共 128 页。每页容纳模型全部层的 K/V，FP32 布局为
`[physical_block, layer, token_in_block, kv_head, head_dimension]`，K/V 分开存储。
逻辑 token 通过序列的 block table 转换为物理页地址。CPU attention 与 CUDA
append/attention kernel 直接寻址页面，不先 gather 成连续序列，也不回传 KV 到 CPU。

## 所有权与失败边界

- `include/miniprefix/paged_kv.h` 声明 KvBlockPool，src/kv/paged_kv.cpp 实现 free list、
  页面申请、回滚、回收和占用峰值。统一 KVManager 持有页池、缓存和所有序列 metadata。
- runner 持有物理 payload，manager 的序列持有逻辑页面表和已提交 token 数。create 只创建序列
  metadata，append 按 `ceil((used+input)/block_tokens)` 申请页面，release 归还页面。
- 默认缓存关闭、页面独占；开启 [精确前缀缓存](prefix_cache.md) 后，完整页面通过
  引用计数共享，未满尾页私有。部分页共享/COW 尚未实现，不能自行复制页表来共享。
- 申请所需页面不足时，在写入前失败，旧页面表和已提交 token 不变。
  forward 异常由 Append 事务归还本次新增页面；只写未提交位置，已有有效 prefix 不被覆盖。
- CUDA pool 在 runner 初始化时一次分配两块 K/V，生命周期内不迁移页面。序列 create
  只分配一张设备 block table；只有新增页号 H2D，已有表项与 KV 不重复上传。
  append 返回前同步完成，release/clear/异常回滚前也同步，避免在途访问使用已回收页。
- 这是固定容量池，开启缓存时先淘汰缓存引用，但尚无活跃请求的容量准入或抢占。
  Engine 遇到池耗尽会进入失败状态，结束全部
  未完成请求并归还页面；需要新建 Engine。容量不足不会静默回退或驱逐其他序列。

## 配置与统计

Python `NativeEngine(..., block_tokens=16, kv_blocks=128)`，C++ factory 可传
`KvCacheConfig{block_tokens,max_blocks}`。生成入口支持：

```powershell
python -m miniprefix.apps.generate --worker build-phase54-cuda/miniprefix/miniprefix_cuda_worker.exe --model-dir build-miniprefix-models/smollm2-135m-instruct --backend cuda --prompt 'What is the capital of France?' --context 128 --block-tokens 16 --kv-blocks 32
```

worker 参数为 `MODEL cpu|cuda CONTEXT [BATCH TOKENS CHUNK BLOCK_TOKENS KV_BLOCKS PREFIX_BLOCKS]`。
当前握手为 `READY MINIPREFIX 3`，STATS 共 25 字段（前缀版新增 7 字段）；Python client
与 worker 需同时更新，旧二进制会被新 client 拒绝。

| 字段 | 含义 |
|---|---|
| kv_block_tokens / kv_total_blocks | 页 token 数、池容量 |
| kv_live_blocks / kv_peak_blocks | 请求或缓存持有的物理页面当前/峰值数，共享页只计一次 |
| live_kv_bytes / peak_kv_bytes | 页数 × 全层 K/V 的每页 payload 字节数，包含尾页空位 |
| kv_reserved_bytes | 整个物理池的已预留 payload，空闲页也包含在内 |
| block_table_uploaded_bytes | CUDA 页号累计 H2D 字节数，不是 KV 数据传输 |
| block_table_device_bytes | 当前存活设备页表的 payload，释放序列后归零 |

SmolLM2 每页 737,280 字节，默认物理池 94,371,840 字节（90 MiB）。
分页减少的是序列占用及尾部浪费，不自动降低固定池预留量；池空闲后仍保留显存供复用。
权重、activation workspace、页表和 cuBLAS/驱动内部开销应分别计算。

## 验证与性能

`miniprefix_paged_kv_tests` 验证非连续页、边界地址、容量不足的原子性、回滚、复用和
配置溢出。`tools/verify_real_model.py` 用独立 Transformers full-prefix oracle 验证
CPU/CUDA 完整/分块 prefill、增量 decode，以及三 token 页大小下的交错分配、跨页读写、
孔洞复用、失败重试和 Engine 耗尽回收；CUDA decode 要求零 KV/权重上传、零新分配，
新增表项只在跨页时上传。性能采样使用相同 worker 指纹的通过验收结果。

旧 [dense 基线](acceptance/baseline-smollm2-fp32.md) 和原始 JSON 保持不变；
新 [paged 验收](acceptance/paged-kv-smollm2-fp32.md) 记录同口径结果。
上述分页报告是缓存关闭的历史对照，当前前缀实现与验收见 [精确前缀缓存](prefix_cache.md)。
跨请求张量 batching 尚未实现。
