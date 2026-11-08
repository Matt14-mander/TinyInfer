# CUDA 跨请求 prefill 与混合批次

MiniPrefix 将同一个调度步中相邻请求的输入按 token 行打包。每行携带所属请求的绝对位置和 GPU 页表，RoPE、分页 KV 写入和 causal attention 都使用该请求的上下文。各层投影和 MLP 共享一次批量矩阵计算。最终归一化与词表投影只处理每个 chunk 的末行。

引擎沿用 decode 优先顺序、token 预算和 chunk 限制。启用前缀缓存时，遇到尚待查询缓存的请求会先完成前面的批次，确保新请求看得到同一步内刚发布的历史，LRU 也按原顺序更新。KVManager 在打包前检查全部新页和共享尾页的 COW 用量。空闲物理页不足以容纳整组时，引擎按原顺序逐请求追加。CPU 继续提供逐请求数值参考。

### 如何复测

对照为上一版已支持 batched decode 的 CUDA worker。使用 SmolLM2-135M-Instruct FP32、上下文 256、预算 32、chunk 16、每页 16 token、128 个物理页，开启缓存时容量为 48。每轮固定八请求同时到达，每请求生成八个 token。每种配置独立 warmup，热缓存另行 prime，再交替运行两版。脚本核对 Transformers full-prefix greedy 输出、逐步事件、实际前向和复用 token、命中、淘汰及资源释放。

```text
python miniprefix/benchmarks/batched_decode.py --compare-prefill --model-dir MODEL_DIR --before DECODE_ONLY_WORKER --after PACKED_WORKER --repeats 5 --output comparison.json
```

`DECODE_STATS` 统计纯 decode；`PREFILL_STATS` 依次返回纯 prefill 批数、混合批数、两类批次中的 prefill token 数、混合批次中的 decode token 数、包含 prefill 的回退次数。

### 同机测量（2026-11-08）

RTX 4070 Laptop GPU，驱动 555.97，CUDA 12.5.1。五轮合计 60 个样本、480 个请求、3,840 个输出 token。下表为各版本五轮中位数，单元格按旧版 / 新版排列。TTFT 取每轮八请求从到达到首 token 的 p50，包含排队；TPOT 取每轮请求间隔均值。最后一列先在同轮计算吞吐比再取中位数，可与两组中位数之比不同。

| 并发上限 | 缓存 | TTFT (ms) | TPOT (ms) | 输出吞吐 (tok/s) | 配对吞吐比 |
|---:|---|---:|---:|---:|---:|
| 1 | 关闭 | 374.77 / 289.74 | 10.43 / 8.16 | 74.87 / 95.73 | 0.912× |
| 1 | 冷 | 318.36 / 258.20 | 10.91 / 8.05 | 85.45 / 115.48 | 0.964× |
| 1 | 热 | 213.03 / 229.43 | 11.31 / 8.70 | 88.57 / 115.22 | 1.000× |
| 4 | 关闭 | 167.66 / 88.95 | 18.10 / 11.77 | 128.91 / 219.75 | 1.564× |
| 4 | 冷 | 111.06 / 88.65 | 14.88 / 15.89 | 184.89 / 194.07 | 1.188× |
| 4 | 热 | 30.67 / 29.19 | 12.41 / 9.01 | 271.97 / 347.51 | 1.140× |

并发 4、关闭缓存时，每次运行执行 3 个纯 prefill 批与 12 个混合批，覆盖 346 个 prefill token 和 29 个 decode token。五对样本的吞吐比从 0.74× 到 2.23×。冷缓存每次执行 2 个纯 prefill 批和 2 个混合批，TTFT 下降，TPOT 中位数增加 1.01 ms。热缓存每次只有 14 个纯 decode 批；它与并发 1 都是观察系统波动的对照。固定八请求数据不能估计持续到达下的饱和吞吐与尾延迟。[完整配对样本、计数与二进制指纹](packed-prefill-comparison.json.gz)

### 正确性与资源

- CTest 全量通过 CPU 56/56、CUDA 65/65；Python 回归 16/16。
- 混合批次逐请求与 CPU logits 对照，覆盖异长位置、跨页追加、两个分支共享部分尾页、重排、非有限 logits 回滚与重试。页池不足时检查缓存引用和淘汰计数。
- Engine 用相同输入分别执行批量与串行路径，对照每步事件、token 预算、发布历史以及同一步内发布后命中。另行注入前缀恢复失败，校验已成功请求与故障请求的事件顺序和资源释放。
- CUDA Compute Sanitizer memcheck 执行完整 COW、decode 与变长混合批次用例，报告 0 个错误、0 字节泄漏。
- 独立 Transformers FP32 多轮生成对照通过 44 个用例、196,608 个 logits。真实性能实验另行对照八请求的全部 greedy 输出。
