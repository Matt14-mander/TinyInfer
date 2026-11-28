# FP32 warp 分页 attention（2026-11-28）

基于 `89f59a0` 优化 MiniPrefix CUDA attention，继续使用 FP32，未引入低精度、序列分块并行、FlashAttention 或 CUDA Graph。
生产实现见 `src/backend/cuda/attention.cuh`；原 256-thread kernel 只保留在独立测试/基线程序中。

## 实现与兼容边界

原 kernel 每个 query/head 使用 256 个线程，每读一个历史 token 都进行一次共享内存写入、八层 block 归约及尾部同步。
新 kernel 每个 query/head 使用一个完整 warp，按 head dimension 选择每 lane 1/2/4/8 个元素的模板；高维分片先按原来的高位到低位顺序相加，
再使用 warp shuffle 归约。在线 softmax 保持历史 token 顺序、FP32 累加和 `--fmad=false`，没有改变 K/V 页布局、GQA head 映射或因果范围。

支持现有验证范围内的 head dimension（不超过 256）、非 32 倍数维度、单请求连续位置、批量变长位置与逐行页表。
不改变调度、共享容量准入、前缀发布、COW/失败回滚、greedy 采样或超时/ID 生命周期。
每层 attention kernel 启动次数仍为一次；本轮减少的是核内同步，不是 forward 的 kernel 数量。
历史 token 仍顺序遍历，长上下文的序列维并行尚未实现。

## 数值检查与微基线

新增 `miniprefix_attention_bench --check`，纳入 CUDA CTest。矩阵共 144 个参数组合：
head dimension 4/32/34/64/128/256，上下文 1/7/33/128/512/2048，请求数 1/4，每请求 chunk 1/16（长度不足时裁剪）。
包含 MHA/GQA、7/8-token 页、非连续物理页、跨请求共享首个整页、不同请求长度和三层池中非零层偏移。
全部输出与原 FP32 kernel 对比，每例的首尾 query 行另用独立 dense FP64 softmax 验证。

三次独立进程共 432 案例，新旧输出的最大绝对差值为 **0**。普通输入与 FP64 的最大差值为 **1.52e-7**。
128-context 压力例将 query 放大到普通输入幅度的 2000 倍：旧、新 FP32 路径在首轮测试都超过普通输入的 2e-5 FP64 阈值，
且旧、新输出仍完全相同。因此该压力例使用单独的 1e-3 FP64 上界，保留所有元素的新旧 2e-6 严格对比；最终观测最大 FP64 差值为 **3.92e-5**。
不把压力输入的 FP32 舍入误差解释成新 kernel 引入的差异。

微基线先验证数据并预热，再在同一进程内交错执行旧/新路径，各三组，每组 20 次 launch。
CUDA event 记录 stream 时间除以 20，排除数据上传、oracle 和下载；短 kernel 的结果可能包含主机提交间隙，不能当纯 kernel 执行时间。
硬件为 RTX 4070 Laptop 8 GiB，未锁频。表格为各进程中位数的中位数；配对倍数是先计算同进程旧/新之比再取中位数，
不一定等于表中两列中位数之比。总矩阵配对倍数范围约 0.74–9.93，不能声称每个形状都加速。

下面选取 head dimension 64、4 请求的中长上下文结果，时间单位 µs：

| 上下文 | 每请求 chunk | 旧 stream 时间 | 新 stream 时间 | 配对倍数 |
|---|---:|---:|---:|---:|
| 33 | 1 | 41.42 | 18.02 | 1.47× |
| 33 | 16 | 80.84 | 19.19 | 4.21× |
| 128 | 1 | 94.16 | 59.39 | 1.58× |
| 128 | 16 | 415.39 | 69.07 | 6.57× |
| 512 | 1 | 409.75 | 295.48 | 1.58× |
| 512 | 16 | 2047.74 | 263.32 | 8.17× |
| 2048 | 1 | 1586.12 | 1003.66 | 1.58× |
| 2048 | 16 | 8297.11 | 1167.26 | 7.05× |

[微基线原始数据](warp-attention-kernels.jsonl.gz) · [全部形状、范围与 GPU/二进制指纹](warp-attention-kernels.jsonl.summary.json)

这是单个 attention 算子的合成形状扫描，不包含真实模型的线性层、归一化、KV 管理和其他 kernel，不能将倍数直接外推为模型吞吐提升。

## 同机端到端配对对照

SmolLM2-135M FP32、共享感知准入、Full 批量策略、20 页物理池、页长 8、最大 batch 4、token budget 32、chunk 16、context 128。
输入长度 65/68/71/74、公共前缀 64、各输出 8 token，三种缓存模式，40/100 请求每秒，各三轮 × 96 请求，按轮交错旧/新和缓存模式。
修改前二进制冻结于 `89f59a0`，同一模型和到达 seed 配对，所有输出逐 token 对照同后端无缓存生成；独立 Transformers 对照另行执行。
36 案例、3,456 请求全部通过，decode/prefill 批量回退均为 0，清理后 KV 和预约均为 0。

TTFT/E2E 从计划到达计时，包含提交延迟与排队；TPOT 为单请求输出间隔均值。
表中延迟单位 ms，指标为轮次分位数/吞吐的中位数；吞吐范围保留全部三轮，不筛除慢样本。
正式端到端测量没有 CUPTI 或成本计时；加载、参考、预热/prime 排除。

| 到达 req/s | 缓存 | 版本 | TTFT p50/p99 | TPOT p50/p99 | E2E p99 | 服务窗口 tok/s（范围） |
|---|---|---|---:|---:|---:|---:|
| 40 | off | before | 2340.5/4736.7 | 10.4/27.2 | 4796.2 | 111.8（103.2–116.2） |
| 40 | off | after | 2325.1/4370.8 | 9.2/22.7 | 4475.9 | 112.8（105.2–113.5） |
| 40 | cold | before | 558.9/1149.1 | 13.7/22.4 | 1279.0 | 226.2（218.4–292.0） |
| 40 | cold | after | 810.7/1155.4 | 13.3/26.1 | 1239.8 | 250.0（187.6–277.4） |
| 40 | hot | before | 447.0/870.8 | 14.6/25.9 | 982.5 | 259.2（255.2–261.0） |
| 40 | hot | after | 209.1/931.6 | 14.3/64.7 | 1023.0 | 254.8（194.8–290.2） |
| 100 | off | before | 2569.1/5149.6 | 9.0/13.5 | 5200.9 | 124.1（111.0–127.9） |
| 100 | off | after | 3319.9/6715.4 | 9.7/30.3 | 6780.0 | 102.2（89.9–132.0） |
| 100 | cold | before | 1318.8/2315.7 | 13.3/22.7 | 2379.3 | 238.6（192.4–272.1） |
| 100 | cold | after | 1050.0/2393.5 | 11.8/41.6 | 2456.1 | 263.2（215.5–310.6） |
| 100 | hot | before | 1189.6/2100.6 | 13.8/27.0 | 2179.3 | 234.0（214.0–236.5） |
| 100 | hot | after | 1092.9/2074.4 | 13.1/26.2 | 2145.5 | 235.4（206.8–251.0） |

| 到达 req/s | 缓存 | 配对吞吐比中位数 | 三轮比值范围 |
|---|---|---:|---:|
| 40 | off | 0.98× | 0.94–1.09× |
| 40 | cold | 1.14× | 0.64–1.23× |
| 40 | hot | 1.00× | 0.75–1.11× |
| 100 | off | 0.92× | 0.70–1.06× |
| 100 | cold | 1.10× | 0.79–1.61× |
| 100 | hot | 1.00× | 0.88–1.17× |

**端到端收益不稳定。** 冷缓存配对吞吐比中位数约 1.14×/1.10×，但范围跨过 1；热缓存基本持平。
100 req/s 的 off 配对吞吐比中位数约 0.92×，吞吐中位数 124.1 → 102.2 token/s，E2E p99 5,200.9 → 6,780.0 ms，必须记录为退化。
未锁频且仅三轮，不能以局部算子更快推断所有缓存模式端到端更快，也不能把退化简单剔除为噪声。
本轮固定模型/负载的性能数据不能与不同日期的既有基线直接作前后比较；100 req/s 下有明显积压，报告该负载的服务吞吐，不声称找到全局饱和上限。

[端到端原始时间线](warp-attention-arrivals.jsonl.gz) · [完整汇总、指纹与环境](warp-attention-arrivals.jsonl.summary.json)

## CUPTI 时间线

性能主测量结束后，复用现有 `miniprefix_launch_trace MODEL greedy`，每版本三个独立进程交错运行。
每次进程预热后采集五次 4 请求 × 8 token prefill 和 35 次 4 请求 decode；上下文为 8–15 token，
这是短序列 forward 的归因，不是端到端基线中 65–74 token 的同一负载。
attention 时间按每次 call 中所有 attention kernel 的 duration 求和，随后取 call 中位数及进程中位数。

| 阶段 | 版本 | 全部 kernel/次 | attention kernel/次 | attention µs/次 | 间隙占比 | append ms 中位数 |
|---|---|---:|---:|---:|---:|---:|
| prefill32 | before | 634 | 30 | 449.2 | 72.4% | 16.06 |
| prefill32 | after | 634 | 30 | 145.6 | 75.6% | 13.39 |
| decode4 | before | 454 | 30 | 378.7 | 64.2% | 11.29 |
| decode4 | after | 454 | 30 | 292.3 | 65.3% | 11.82 |

CUPTI 下 prefill 的 attention 总耗时约 449 → 146 µs，decode 约 379 → 292 µs；kernel 数量不变。
大量时间仍在 kernel 间隙中。间隙包含主机提交、驱动和 CUPTI 插桩影响，不能当作 Scheduler 自身耗时；
占比增大也不代表绝对间隙必然增加。这里支持“核内同步成本下降”，不支持“全部 forward/端到端指标稳定改善”。
采样路径每次只下载四个 token ID（16 字节），没有恢复 logits 回传。

[时间线汇总与指纹](warp-attention-trace-summary.json)
· [before 0](warp-attention-trace-0-before.tsv.gz) / [after 0](warp-attention-trace-0-after.tsv.gz)
· [before 1](warp-attention-trace-1-before.tsv.gz) / [after 1](warp-attention-trace-1-after.tsv.gz)
· [before 2](warp-attention-trace-2-before.tsv.gz) / [after 2](warp-attention-trace-2-after.tsv.gz)

## 正确性和资源验收

- CPU CTest 56/56、CUDA CTest 66/66、Python 29/29 通过，覆盖原有前缀/COW、批量、失败回滚和生命周期路径。
- 真实 SmolLM2 与独立 Transformers CPU FP32 eager full-prefix 参考：44 个生成历史用例、196,608 个 logits 通过。
- 真实 CPU/CUDA worker 生命周期专项通过，队列拒绝、超时、取消、ID 轮换和最终资源释放保持正常。
- Compute Sanitizer 对 34/256 维、7/33 context、4 请求、chunk 1/16 的八个边界形状执行 memcheck（含 leak-check full）及 synccheck：零错误、零泄漏。
  首次沙箱内检查因进程附加超时未完成，结束该次检查后在沙箱外重跑成功；不把未附加的结果记作通过。

[CPU 日志](warp-attention-cpu-tests.log) · [CUDA 日志](warp-attention-cuda-tests.log)
· [独立参考](warp-attention-reference.json) · [生命周期](warp-attention-lifecycle.json)
· [memcheck](warp-attention-memcheck.log) · [synccheck](warp-attention-synccheck.log)

本轮交付的是保持 FP32 计算顺序的分页 attention 核内优化及可复核测量；长上下文的序列维并行、低精度矩阵乘法和 CUDA Graph 仍未实现。
后续应结合真实 batch/context 的 forward 分布继续优化，并把 off 模式的高负载退化纳入验收；不继续堆叠仅能提高局部倍数的改动。

## 复现

```text
miniprefix_attention_bench --check --require-device
python miniprefix/benchmarks/attention.py --binary ATTENTION_BENCH --repeats 3 --output kernels.jsonl.gz
python miniprefix/benchmarks/capacity.py --binary AFTER --before-binary BEFORE --model MODEL.mpf --backend cuda --policies shared --rates 40 100 --requests 96 --repeats 3 --output arrivals.jsonl.gz
python miniprefix/tools/verify_generation_history.py --model-dir MODEL_DIR --cpu-worker CPU_WORKER --cuda-worker CUDA_WORKER --output reference.json
miniprefix_launch_trace MODEL.mpf greedy > trace.tsv
python miniprefix/benchmarks/launch_gaps.py trace.tsv
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 ATTENTION_BENCH --check --quick --require-device
compute-sanitizer --tool synccheck --error-exitcode 1 ATTENTION_BENCH --check --quick --require-device
```
