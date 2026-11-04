# CUDA batched decode 验收（2026-11-04）

首版合并跨请求 decode 的各层 GEMM，并使用每行位置与页表执行 RoPE、分页 KV 写入和 attention。Prefill 保持逐请求执行。
Scheduler、准入、预算及发布顺序不变；物理页不够容纳整批追加/COW 时回退串行，不驱逐缓存来强行组成批次。

## 旧/新同并发性能

旧 worker 为 e21b1da，SHA256 与上轮前缀发布验收一致；新 worker 及脚本/模型指纹记录在原始 JSON。
复用 baseline.prepare_workload 的四个真实 prompt 和 Transformers FP32 full-prefix 输出参考，将八请求固定为同时到达。
每个并发/缓存组合五轮旧新进程交替，共 60 个样本（480 个请求、3840 个输出 tokens）全部通过参考校验。
每对样本的逐步事件轨迹、forward、命中、复用和淘汰计数一致。热缓存 prime 与 warmup 不在计时区间。
表中 TTFT 为每次运行八请求 p50 的跨运行中位数；TPOT 为每次运行请求均值的跨运行中位数；吞吐是八请求总输出除以总窗口。

|并发上限|模式|旧/新 TTFT ms|旧/新 TPOT ms|旧/新 tok/s|配对吞吐比中位数|
|---:|---|---:|---:|---:|---:|
|1|off|409.25 / 337.03|11.79 / 11.05|65.09 / 69.75|1.072|
|1|cold|284.57 / 332.99|10.18 / 10.89|92.64 / 84.75|0.898|
|1|hot|262.48 / 268.51|11.34 / 10.98|87.90 / 91.77|0.969|
|4|off|213.51 / 272.48|39.50 / 21.56|73.24 / 101.01|1.504|
|4|cold|160.72 / 199.69|47.63 / 23.83|68.57 / 124.08|1.682|
|4|hot|49.88 / 39.35|41.17 / 12.61|96.64 / 247.97|2.540|

配对比先在同轮相除，再跨轮取中位数，不等于两列中位数相除。完整逐请求延迟、step 事件、min/max 和配对比均保留。

并发 4 的 off/cold/hot 配对吞吐比约为 1.504× / 1.682× / 2.540×；TPOT 在三个模式均下降。
同时，off/cold 的 TTFT 中位数变差（213.51→272.48 ms、160.72→199.69 ms）；不把 decode 收益宣称为全阶段延迟改善。
并发 1 不合批，是控制组：配对吞吐比为 1.072× / 0.898× / 0.969×。存在系统波动及新路径开销，不能将所有旧新差异解释为合批效果。
热缓存更容易形成满批 decode；冷/关闭缓存还包含逐请求 prefill、排队和混合阶段时间。表格展示的是测量结果，不给出未经隔离的 TTFT 退化归因。

## 实际合批与资源

|模式（并发 4）|每轮成功批次|每轮合批 decode tokens / 56|回退次数|
|---|---:|---:|---:|
|off|18|52 / 56|0|
|cold|16|55 / 56|0|
|hot|14|56 / 56|0|

所有并发 1 样本的合批计数为零。正式窗口无权重/KV pool 重传和 workspace 增长；退出后活动序列与设备页表归零，清缓存后 KV 页归零。
批处理同时占用多行事务，瞬时页数及 COW 次数在一般容量/共享场景不要求与串行相同；小池与中途预约失败通过回退测试，不纳入这里的充足容量吞吐数字。

## 回归

- CPU 56/56、CUDA 65/65 全量通过；CPU 继续串行，保留参考行为。
- 批量事务：不驱逐容量检查、重复/非法输入、中途预约失败后的整组回滚、活动事务锁及资源释放。
- CUDA 数值：异长位置、整页边界、共享尾页双分支、COW、非有限 logits 回滚/重试及 dirty 页表修复，对照 CPU logits 和 greedy。
- Engine：预算 1/2/6，取消/EOS/长度、同轮 decode 发布后 prefill 命中，逐 step 事件与发布历史对照。
- CUDA memcheck 所有测试段完成，零错误、零泄漏。
- Python 指标及协议测试 15/15 通过，包含配对比统计与新增只读 DECODE_STATS。
- 独立 Transformers 多轮闭环：44 个生成用例、196608 个 logits 通过；真实 batched decode 的八请求输出另由性能基线中的 full-prefix oracle 校验。

## 环境与边界

Windows-11-10.0.26200-SP0；MSVC 14.29.30133、CUDA 12.5.1、FP32（cuBLAS PEDANTIC）。GPU 快照：`NVIDIA GeForce RTX 4070 Laptop GPU, 555.97, 8188 MiB, P0, 2430 MHz, 65, 26.07 W`。
模型为固定 SmolLM2-135M-Instruct；上下文 256、预算 32、prefill chunk 16、页长 16、KV 128 页，开启缓存时 48 条目。
未锁频或证明排除全部 WDDM/后台活动。五轮固定八请求实验不是持续到达饱和吞吐，也不足以声明 p99/SLA。
首版不含 prefill batching、混合 prefill/decode GEMM、CUDA Graph、异步执行重叠；这些能力需基于新的瓶颈单独验收。

[原始样本](batched-decode-comparison.json) · [CPU](batched-decode-cpu-tests.log) · [CUDA](batched-decode-cuda-tests.log) · [memcheck](batched-decode-memcheck.log) · [指标测试](batched-decode-metrics-tests.log)
[真实模型对照](batched-decode-generation-history.json) · [源码与验收哈希](batched-decode-source-sha256.json) · [实现和复现](../batched_decode.md)
