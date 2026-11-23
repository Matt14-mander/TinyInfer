# CUDA 残差加法与 RMSNorm 融合

2026-11-23，在 SmolLM2-135M-Instruct FP32 上对照 `f77110d` 与本次实现。融合覆盖 attention 输出接 post-attention RMSNorm、MLP 输出接下一层 input RMSNorm，以及末层 MLP 输出接模型最终 RMSNorm。首层输入单独归一化，后续归一化同时执行残差加法并写回 FP32 残差。归约顺序和编译选项 `--fmad=false` 沿用原实现。

末层根据各 chunk 最后一个 token 的行索引同时读取残差和 MLP 投影，完成归一化后进入词表投影。其他输入行的 KV 在此前已经写入分页缓存。GPU greedy、前缀发布、COW 事务和非有限值错误处理继续通过原调用路径执行。

测试扩充至隐藏维度 8、272、576 与 1、3 层的组合，对照 CPU FP32 logits 和 greedy ID，覆盖线程块内每 lane 多个元素、非 256 整倍数宽度、单层末归一化、跨层残差、异长 chunk 的末行采集、混合批次及共享尾页。原有 CPU/CUDA 全量回归验证生成历史、错误回滚和资源回收。

## 对照方法

持续到达测量复用现有 `sustained.py`。旧、新进程按配置交替运行，使用相同的到达时间、prompt、随机种子和独立 Transformers 输出参考。10、24 请求/s 每轮 96 请求，64 请求/s 每轮 512 请求，三种缓存模式各重复三轮。GPU 为 RTX 4070 Laptop GPU，CUDA 12.5.1，MSVC Release，并发上限 4、每步预算 32 token、chunk 16。正式窗口关闭 CUPTI 插桩。TTFT 从计划到达计时，包括控制器延迟与引擎内排队；TPOT 为单请求输出间隔均值，延迟分位数合并三轮，吞吐先按轮计算再取中位数。

CUPTI 对照分别使用旧、新 `miniprefix_launch_trace`，均执行 GPU greedy。三次独立进程交替采集，每进程预热一轮，再测五次四请求 prefill 和三十五次四请求 decode。只计算每次 forward 内从 embedding 到 greedy 的 kernel 区间。启动间隙是相邻 kernel 结束与开始的差值，包含驱动及插桩影响。另行运行关闭 activity tracing 的 control。

## CUPTI 结果

表中时间与间隙比例先在各进程内汇总，再取三次进程的中位数。每个进程都观测到 prefill 减少 60 个 kernel（8.6%），decode 减少 60 个 kernel（11.7%），对应 30 层各两次残差加法。

| 输入 | kernel/次 旧 / 新 | 间隙占比 旧 / 新 | 间隙 p95 µs 旧 / 新 | 残差加法与 RMSNorm 合计 µs/次 旧 / 新 |
|---|---:|---:|---:|---:|
| 4 请求 × 8 token prefill | 694 / 634 | 68.3% / 59.5% | 50.43 / 39.62 | 232.4 / 168.1 |
| 4 请求 × 1 token decode | 514 / 454 | 66.7% / 58.5% | 54.88 / 46.94 | 214.9 / 162.8 |

融合后的残差与归一化总 kernel 执行时间下降，三组进程的所有 kernel 都匹配到 launch API 时间戳。CUPTI 下 append 墙钟中位数为 prefill 12.91 / 10.46 ms、decode 11.93 / 10.96 ms。独立无 activity tracing 的单组 control 却为 prefill 12.89 / 18.00 ms、decode 9.25 / 13.42 ms，说明运行状态和插桩影响仍然明显。不能将 trace 中的差值直接当作端到端收益。

每条原始记录含 API、kernel 起止时间、调用跨度及 D2H 字节。逐进程指标见[汇总](residual-rms-trace-summary.json)，其原始文件为同目录 `residual-trace-{0,1,2}-{before,after}.tsv.gz`；可用 `launch_gaps.py` 重新计算。[旧版 control](residual-control-before.tsv) · [新版 control](residual-control-after.tsv)

## 持续到达延迟

每格按旧版 / 新版排列。延迟单位为 ms；到达窗口吞吐单位为输出 token/s。

| 请求/s | 缓存 | TTFT p50 | TTFT p99 | TPOT p50 | E2E p99 | 到达窗口吞吐 |
|---:|---|---:|---:|---:|---:|---:|
| 10 | off | 43.7 / 40.9 | 107.0 / 309.7 | 11.0 / 10.1 | 284.4 / 492.5 | 79.7 / 78.9 |
| 10 | cold | 10.8 / 8.6 | 85.5 / 55.9 | 9.6 / 7.2 | 358.2 / 233.1 | 79.4 / 79.8 |
| 10 | hot | 11.3 / 12.3 | 55.8 / 59.4 | 10.0 / 11.1 | 253.5 / 239.6 | 79.9 / 80.1 |
| 24 | off | 65.1 / 62.1 | 1682.6 / 333.4 | 14.2 / 12.8 | 1851.3 / 435.8 | 186.0 / 189.8 |
| 24 | cold | 25.8 / 23.6 | 382.2 / 474.9 | 12.5 / 11.3 | 496.3 / 611.2 | 193.5 / 191.0 |
| 24 | hot | 26.7 / 18.9 | 458.8 / 127.8 | 12.8 / 9.4 | 588.7 / 268.7 | 193.4 / 194.3 |

24 请求/s 下三种缓存模式的 TTFT 与 TPOT 中位数均下降，但冷缓存的 E2E p99 上升；10 请求/s 的关闭缓存 p99 也上升。低负载吞吐受请求到达速率限制，不能代表饱和容量。三轮结果保留全部波动样本。[旧版原始样本](residual-rms-arrivals-before.jsonl.gz) · [新版原始样本](residual-rms-arrivals.jsonl.gz)

## 饱和吞吐与尾延迟

64 请求/s、每轮 512 请求时，两版在到达窗口均持续积压。下表每格为旧 / 新，TTFT 与 E2E 单位为秒，TPOT 单位为 ms。配对吞吐比先在相同轮次计算，再取中位数。

| 缓存 | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p99 | 饱和吞吐 tok/s | 配对吞吐比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 关闭 | 4.85 / 5.44 | 12.64 / 15.70 | 11.7 / 13.0 | 52.8 / 30.8 | 12.75 / 15.82 | 236.7 / 218.9 | 0.92× |
| 冷 | 3.98 / 4.24 | 9.61 / 11.73 | 13.8 / 13.7 | 31.6 / 35.9 | 9.71 / 11.88 | 272.7 / 249.4 | 0.91× |
| 热 | 7.13 / 4.08 | 14.67 / 12.46 | 16.8 / 14.9 | 35.0 / 36.2 | 14.80 / 12.58 | 183.5 / 265.0 | 1.49× |

关闭、冷、热缓存三轮配对吞吐比范围分别为 0.66–1.15×、0.82–1.01×、1.02–1.50×。GPU 未锁频，样本只有三轮；当前结果支持启动数量与局部 kernel 成本降低，端到端收益取决于负载和运行状态。关闭和冷缓存的饱和吞吐与 E2E p99 回退必须纳入后续验收，本次没有测得三种模式一致的端到端改善。[旧版原始样本](residual-rms-capacity-before.jsonl.gz) · [新版原始样本](residual-rms-capacity.jsonl.gz)

## 正确性与资源

CPU 56/56、CUDA 65/65、Python 21/21 回归全部通过。独立 Transformers 验证通过 44 个生成历史用例和 196,608 个完整 logits，包含 EOS、取消后复用和并发分叉。CUDA memcheck 报告零错误、零泄漏。[源码、二进制哈希与验证摘要](residual-rms-verification.json)

持续到达两档扫描与饱和对照共 54 次运行、12,672 个请求，所有生成输出与独立参考一致。旧、新生成路径的 logits 回传均为 0，各版本分别验证实际 forward 与跳过的前缀 token 之和及 KV 清理。末层只收集输出行的路径覆盖单请求、批量 decode、变长 prefill 和混合批次。

## 复现

```text
python miniprefix/benchmarks/sustained.py --worker NEW --reference-worker OLD --model-dir MODEL_DIR --rates 10 24 --requests 96 --output arrivals.gz --reference-output arrivals-before.gz
python miniprefix/benchmarks/sustained.py --worker NEW --reference-worker OLD --model-dir MODEL_DIR --rates 64 --requests 512 --output capacity.gz --reference-output capacity-before.gz
python miniprefix/benchmarks/sustained.py --summarize capacity.gz
OLD_TRACE MODEL.mpf greedy > before.tsv
NEW_TRACE MODEL.mpf greedy > after.tsv
python miniprefix/benchmarks/launch_gaps.py before.tsv after.tsv
```

CUPTI 构建与运行要求见 [GPU greedy 验收](gpu-greedy.md)。二进制和模型哈希保存在持续到达原始文件头部。
