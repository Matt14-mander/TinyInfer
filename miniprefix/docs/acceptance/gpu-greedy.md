# GPU greedy 采样与 CUDA forward 启动间隙

## 生成契约

CUDA 单请求、纯 decode、变长 prefill 和混合批次共用原有 forward 与 KV 追加事务。生成时每个输出行由一个 256 线程的 block 扫描词表，选择最大有限 FP32 logit。若有并列，选择最小 token ID；若任意 logit 非有限，整行报告失败。已完成 embedding 的设备输入 ID 缓冲可容纳每个 chunk 的结果，生成路径每行只回传一个 32 位整数。中间 prefill chunk 仍检查整行有限性，其候选 ID 不产生输出事件。

单请求遇到非有限值会回滚；合批遇到非有限值先回滚整组，再交回已有的顺序执行路径，保留错误与前序事件顺序。发布的是已经提交的输入 KV；本轮选中的输出 token 在下一次 append 前不进入可复用历史。CPU 和自定义参考 runner 可以继续提供 logits。诊断命令 `FORWARD` 继续返回完整 FP32 词表。原有 `STATS` 布局保留，新增 `SAMPLE_BYTES` 单独统计 CUDA 生成结果的 D2H 字节，包括因非有限值返回的失败标记。

## 测量方法

使用同一台 RTX 4070 Laptop GPU、CUDA 12.5.1、SmolLM2-135M-Instruct FP32，页长 16，执行并发上限 4，token 预算 32。沿用[持续到达协议](sustained-arrivals.md)，将 `b7c3792` worker 与本次 worker 按配置交替运行。每对样本使用相同的模型、prompt、计划到达时刻与 Transformers greedy 参考；同步控制器在 STEP 之间提交请求，TTFT 和 E2E 从计划到达计时。每次运行独立预热，冷缓存允许随负载填充，热缓存提前 prime。正式端到端测量关闭 CUPTI 和 CUDA event 插桩。

CUPTI 12.5 的独立 runner 微基准对照完整 logits 和 token ID 两种输出方式。每轮先为 4 个请求各 prefill 8 个 token，再执行 7 个四请求 decode 步。先预热一轮，采集五轮。固定输入 ID 使两路执行形状可复现。对每次同步调用，仅计算从 embedding 到最后一个 kernel 的同 stream 区间；两次调用间的空闲时间排除在外。按 correlation ID 匹配 driver/runtime API 与 GPU kernel。若下一个 launch API 在前一个 kernel 结束之后才进入，二者之间的时间是可归到主机尚未提交的间隙下界。其余间隙可能包含驱动排队、同 stream 其他工作及 tracing 开销。

## Kernel 时间线

下表的间隙占比为全部同形状调用的间隙之和除以 kernel 跨度之和，p50/p95 按每一对相邻 kernel 计算。“主机晚提交”列给出上述 API 进入时间下界占所有间隙的比例，全部 launch 均匹配到 API 时间戳。

| 路径 | 输入 | kernel/次 | 间隙占比 | 间隙 p50 / p95 (µs) | 主机晚提交占间隙 |
|---|---|---:|---:|---:|---:|
| 完整 logits | 4 × 8 prefill | 693 | 77.4% | 13.38 / 63.20 | 20.8% |
| 完整 logits | 4 × 1 decode | 513 | 70.0% | 7.30 / 60.26 | 17.4% |
| GPU greedy | 4 × 8 prefill | 694 | 53.2% | 1.79 / 31.90 | 3.0% |
| GPU greedy | 4 × 1 decode | 514 | 41.3% | 1.70 / 29.25 | 2.0% |

新路径在四行 decode 中耗时最多的间隙类别依次为 GEMM → GEMM、RMSNorm → GEMM、残差加法 → RMSNorm。prefill 的累计间隙主要集中在 GEMM 与 cuBLAS split-K reduction 之间。多数间隙开始时，主机已经进入下一个关联 API。减少算子边界或尝试 CUDA Graph 应优先在这些调用序列上做对照；直接把间隙占比当成可回收的加速空间会高估收益。

每批四行的实测 D2H 从 **786,432 B 降至 16 B**，与模型 49,152 维词表吻合。decode 的设备复制中位数从 71.0 µs 降至 1.1 µs，新增 greedy kernel 本身为 53.1 µs。生成同时消除了主机词表分配、行复制与扫描。独立进程间的 forward 间隙差值不能全归因于采样改动。以 decode 为例，开启 CUPTI 的两路 append 中位数为 15.71 / 7.24 ms，关闭 activity tracing 的独立对照为 13.02 / 10.70 ms。绝对性能以无插桩的持续到达实验为准。[logits 时间线](greedy-trace-logits.tsv.gz) · [greedy 时间线](greedy-trace-greedy.tsv.gz) · [logits 对照](greedy-control-logits.tsv) · [greedy 对照](greedy-control-greedy.tsv)

## 持续到达对照

下表每格按旧版 / GPU greedy 排列。延迟单位为 ms，吞吐为请求仍在到达时中间 60% 时间窗口实际交付的输出 token/s。延迟分位数合并三轮请求，吞吐取三轮中位数。

| 请求/s | 缓存 | TTFT p50 | TTFT p99 | TPOT p50 | E2E p99 | 到达窗口吞吐 |
|---:|---|---:|---:|---:|---:|---:|
| 10 | 关闭 | 36.8 / 27.3 | 105.9 / 125.3 | 9.5 / 7.3 | 285.6 / 348.6 | 80.1 / 79.8 |
| 10 | 冷 | 9.4 / 16.0 | 75.2 / 120.4 | 8.6 / 10.9 | 319.1 / 434.2 | 79.7 / 80.1 |
| 10 | 热 | 8.5 / 9.3 | 44.7 / 80.8 | 7.4 / 8.5 | 205.0 / 296.9 | 79.8 / 79.8 |
| 24 | 关闭 | 48.0 / 36.9 | 650.6 / 172.6 | 11.1 / 9.5 | 760.1 / 280.9 | 189.3 / 190.5 |
| 24 | 冷 | 25.1 / 22.2 | 184.8 / 309.3 | 13.3 / 10.7 | 328.2 / 393.6 | 191.0 / 193.9 |
| 24 | 热 | 40.2 / 22.2 | 352.9 / 244.6 | 17.8 / 11.5 | 509.0 / 448.4 | 195.6 / 192.7 |

在每轮 96 请求的轻载对照中，两版均能跟上 10 请求/s，但新版三种模式的 E2E p99 均变长。24 请求/s 时新版三种缓存模式的 TTFT 和 TPOT 中位数均下降，冷缓存的 E2E p99 仍上升。[旧版完整样本](greedy-arrivals-before.jsonl.gz) · [新版完整样本](greedy-arrivals.jsonl.gz)

### 64 请求/s 的饱和容量

每轮 512 个请求、每请求 8 个输出 token。两版在到达窗口均持续积压。TTFT 和 E2E 单位为秒，TPOT 单位为 ms。配对吞吐比先计算每一轮新版 / 旧版，再取中位数，因此可与两组吞吐中位数之比不同。

| 缓存 | TTFT p50 旧 / 新 | TTFT p99 旧 / 新 | TPOT p50 旧 / 新 | TPOT p99 旧 / 新 | E2E p99 旧 / 新 | 饱和吞吐 tok/s 旧 / 新 | 配对吞吐比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 关闭 | 6.60 / 3.79 | 15.21 / 12.89 | 14.3 / 12.2 | 44.1 / 25.0 | 15.33 / 12.99 | 207.9 / 240.2 | 1.39× |
| 冷 | 4.99 / 3.27 | 10.64 / 7.18 | 14.7 / 12.1 | 39.6 / 25.4 | 10.73 / 7.26 | 227.3 / 264.0 | 1.20× |
| 热 | 5.33 / 3.75 | 12.59 / 9.92 | 17.4 / 14.8 | 39.0 / 36.8 | 12.71 / 10.01 | 216.0 / 275.0 | 1.29× |

九对饱和样本的到达窗口吞吐比均大于 1。各模式三轮比值范围分别为 1.12–1.55×、1.16–1.41×、1.12–1.33×。GPU 未锁频，只有三轮配对，具体幅度随运行波动。过载时的秒级 p99 包含队列累积，只适用于当前到达速率与窗口长度。[旧版完整样本](greedy-capacity-before.jsonl.gz) · [新版完整样本](greedy-capacity.jsonl.gz)

## 正确性

CPU 56/56、CUDA 65/65、Python 21/21 回归通过。CPU 与 CUDA 对照覆盖单请求与混合批次、同值跨 lane 取较小 ID、513 维非整倍数词表、单列 NaN/负无穷、共享尾页 COW、整批失败回滚和串行重试。CUDA memcheck 报告 0 个错误、0 字节泄漏。独立 Transformers 多轮验证通过 44 个用例和 196,608 个诊断 logits，覆盖 EOS、取消后的历史复用和并发分叉。

两档扫描与长窗口对照共 54 次运行、12,672 个请求，逐请求复核初始到达时刻、生成输出、完成原因与独立参考。全部新版正式窗口的 logits 下载计数均为 0，实际 token ID 下载另行记录。开放到达下 worker 速度会影响一次 STEP 能看到哪些请求，因此每个版本独立验证实际前向与前缀复用 token 之和。

## 复现

```text
python miniprefix/benchmarks/sustained.py --worker NEW --reference-worker OLD --model-dir MODEL_DIR --rates 10 24 --requests 96 --output arrivals.gz --reference-output arrivals-before.gz
python miniprefix/benchmarks/sustained.py --worker NEW --reference-worker OLD --model-dir MODEL_DIR --rates 64 --requests 512 --output capacity.gz --reference-output capacity-before.gz
python miniprefix/benchmarks/sustained.py --summarize capacity.gz
```

使用 CUDA 12.5 构建并将 CUPTI 的动态库目录加入运行时搜索路径。对 CUDA 构建目录指定 `-DMINIPREFIX_CUPTI_ROOT=<CUPTI 解压目录>`，构建 `miniprefix_launch_trace`。普通 worker 构建无需 CUPTI。

```text
miniprefix_launch_trace MODEL.mpf logits > logits.tsv
miniprefix_launch_trace MODEL.mpf greedy > greedy.tsv
miniprefix_launch_trace MODEL.mpf logits --control > logits-control.tsv
miniprefix_launch_trace MODEL.mpf greedy --control > greedy-control.tsv
python miniprefix/benchmarks/launch_gaps.py logits.tsv greedy.tsv
```
