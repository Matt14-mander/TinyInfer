# 批量空间准备与缓存保留验收（2026-11-24）

上一轮实现已提交为 `754f8bd`。本轮基于它增加只读空间计划、独立累计成本观测，以及可选的 Adaptive 连续前缀拆批。
**Full 继续作为默认；本轮数据不足以证明 Adaptive 有稳定的端到端收益。**

## 语义与回归

设计、阈值、接口和计数口径见[批量工作页与缓存保留](../batch_cache_balance.md)。不增加 KV 池、不改变共享准入额度或调度顺序，
不提前发布缓存，失败保留已发生的驱逐、回滚请求页映射和工作页。缓存统计允许因执行分组不同而变化。

CPU CTest 56/56、CUDA CTest 65/65、Python 指标测试 27/27 通过。新增覆盖：只读规划不触碰缓存、整批不可行时选择可行连续前缀、
无效后续行不被掩盖、无压力保持整批、物理释放与引用驱逐分别计数、取消 COW、回滚与计数保留、CPU/CUDA 真实载荷对照、
事件/发布/预算一致、选择器无效返回、拆批后发布失败清理。还修复了下一组探测跨过已记录恢复失败的边界：失败请求不得进入 forward。

[CPU 日志](batch-cache-balance-cpu-tests.log) · [CUDA 日志](batch-cache-balance-cuda-tests.log) · [真实模型容量专项](batch-cache-balance-capacity.json)

## 方法

RTX 4070 Laptop 8 GiB、SmolLM2-135M FP32、共享准入，页长 8，最大 batch 4，token budget 32，prefill chunk 16，context 128。
同一二进制中比较 Full/Adaptive，每轮打乱配置顺序，配对使用相同到达 seed；主测量关闭 host/CUDA 成本计时。
TTFT/E2E 从计划到达计时，包含排队与提交延迟；TPOT 为每请求平均输出间隔，吞吐按实际输出数量计。加载、oracle、预热、prime 排除。
所有输出逐 token 对照同后端无缓存参考；末尾检查 KV 和预约归零。未锁 GPU 频率，保留所有轮次及范围，不作显著性推断。

mixed 输入 17/33/65/89、输出 4/8/16/24；公共前缀为 0 或 16，池为 32/64 页，缓存为 off/cold/hot，40 请求/s，三轮各 48 请求。
零公共前缀仍重复四个模板，存在同模板复用；hot 表示先顺序生成/发布四个模板，并不保证小池里全部历史仍可命中。
uniform 沿用输入 65/68/71/74、各 8 输出、公共前缀 64、池 20 页，40 请求/s，三轮各 128 请求。

72 个 mixed 案例测量期间另补充了上述恢复失败边界检查；此矩阵使用修复前的性能候选二进制，正常请求均无该异常。
最终构建重新跑全量 CPU/CUDA 回归、完整 uniform 对比，并对 mixed 的 32 页/零公共前缀/hot 退化配置重跑三轮。
两批二进制 SHA256 均保存，不把候选与最终构建混作同一组样本。模型 SHA256 和 GPU 环境保存在各数据头部。

## 混合负载候选矩阵

72 案例 / 3,456 请求全部校验通过，decode/prefill 回退均为零。下表为 hot 配置各轮指标中位数；延迟单位 ms。
完整 off/cold/hot 数据见[原始时间线](batch-cache-balance-mixed.jsonl.gz)与[完整汇总](batch-cache-balance-mixed.jsonl.summary.json)。

| 页池/公共 token | 策略 | 拆批 | 复用 token | 全路径驱逐引用 | TTFT p50/p99 | TPOT p50/p99 | E2E p99 | 服务窗口 tok/s（范围） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 32/0 | full | 0 | 1720 | 1043 | 242.8/456.8 | 9.6/12.6 | 700.4 | 372.4（366.4–374.0） |
| 32/0 | adaptive | 58 | 1720 | 792 | 381.0/829.4 | 11.9/14.7 | 1015.0 | 304.6（230.9–305.8） |
| 32/16 | full | 0 | 2176 | 650 | 286.8/513.2 | 10.0/13.1 | 700.1 | 363.8（316.1–371.0） |
| 32/16 | adaptive | 0 | 2176 | 641 | 267.8/498.5 | 10.0/13.1 | 696.6 | 368.3（352.7–377.5） |
| 64/0 | full | 0 | 2272 | 591 | 171.5/385.1 | 9.5/10.4 | 561.2 | 388.8（384.2–394.8） |
| 64/0 | adaptive | 2 | 2320 | 583 | 170.1/483.9 | 9.7/12.3 | 674.7 | 361.0（347.3–382.6） |
| 64/16 | full | 0 | 2400 | 550 | 192.6/447.9 | 9.6/12.6 | 637.5 | 360.3（278.6–382.9） |
| 64/16 | adaptive | 0 | 2400 | 550 | 170.2/410.7 | 9.5/12.5 | 597.0 | 389.8（357.6–404.9） |

在 32 页、零公共前缀 hot 案例中，驱逐引用中位数由 1,043 降至 792，但复用仍为 1,720 token，forward 仍为 1,304 token；
服务吞吐由 372.4 降至 304.6 token/s（约 -18%）。`immediate_victims_avoided` 为 347，绝不等于净保留了 347 个物理页或节省了 347 页计算。
这里减少驱逐没有带来计算量下降，拆批并非免费。cold 同类配置也没有稳定获益。

64 页、16 公共 token 的 hot 吞吐中位数虽较高，但拆批次数中位数为 0，不能归因为策略收益。
off 全部无拆批却仍有明显轮间性能波动，说明频率/系统噪声不可忽略；不能据个别较快样本推广策略。
mixed 输入需求为 520 token/s，多数组存在积压；表中是该固定到达率下的服务窗口吞吐，并未完整扫描饱和点。

## 最终构建复测

uniform 18 案例 / 2,304 请求，全部逐 token 校验和资源检查通过，批量回退保持零。
Full 无拆批，Adaptive 每轮拆批 0–2 次、各缓存模式中位数均为零；不能把轮间指标变化直接解释为缓存保留收益。冷/热吞吐接近 320 token/s 的输入需求，
不能作为新的饱和吞吐上限，且尾延迟仍有明显波动。

| 缓存/页池 | 策略 | 拆批 | TTFT p50/p99 | TPOT p50/p99 | E2E p99 | 服务窗口 tok/s（范围） |
|---|---|---:|---:|---:|---:|---:|
| cold/20 | full | 0 | 24.4/94.4 | 10.7/15.0 | 171.2 | 319.8（305.5–326.2） |
| cold/20 | adaptive | 0 | 43.5/130.2 | 10.0/13.6 | 211.7 | 319.5（309.2–329.8） |
| hot/20 | full | 0 | 22.2/215.1 | 10.8/22.6 | 297.0 | 314.4（296.9–319.4） |
| hot/20 | adaptive | 0 | 38.6/126.3 | 11.1/17.0 | 211.2 | 321.6（314.9–326.9） |
| off/20 | full | 0 | 1892.1/3588.0 | 7.5/10.4 | 3639.0 | 153.6（145.9–156.5） |
| off/20 | adaptive | 0 | 1889.1/3448.4 | 7.5/10.7 | 3498.0 | 158.2（146.8–160.2） |

[uniform 时间线](batch-cache-balance-uniform.jsonl.gz) · [汇总与最终二进制指纹](batch-cache-balance-uniform.jsonl.summary.json)

mixed 的 32 页/零公共前缀/hot 再跑 6 案例 / 288 请求，输出及资源检查全部通过、批量回退零。
驱逐引用中位数 Full 1,055 → Adaptive 845（约 -20%）；复用 token 1,792 → 1,808，增加很少。
服务吞吐 345.4 → 272.1 token/s（约 -21%），TTFT/TPOT/尾延迟均退化。这个配置不适合启用当前启发式。

| 缓存/页池 | 策略 | 拆批 | TTFT p50/p99 | TPOT p50/p99 | E2E p99 | 服务窗口 tok/s（范围） |
|---|---|---:|---:|---:|---:|---:|
| hot/32 | full | 0 | 286.6/613.6 | 10.0/11.8 | 744.6 | 345.4（333.4–358.4） |
| hot/32 | adaptive | 56 | 501.9/1030.1 | 12.8/19.2 | 1229.1 | 272.1（254.2–287.5） |

[最终 mixed 复测时间线](batch-cache-balance-confirm.jsonl.gz) · [汇总](batch-cache-balance-confirm.jsonl.summary.json)

## 独立 profiling

另跑 mixed / hot / 零公共前缀、32/64 页、两策略各一轮 48 请求，启用成本计时；只用于成本归因，不与上述端到端主测量合并。
下面是整轮累计 host 时间；准备不包含 GPU 执行，选择和准备之和相对于 Engine step 总耗时作粗略占比。

| 页池 | 策略 | 选择 ms | 准备 ms | 两者占 step | 批量驱逐引用 | 真正释放页 | COW 工作页 |
|---|---|---:|---:|---:|---:|---:|---:|
| 32 | full | 0.000 | 2.213 | 0.108% | 982 | 538 | 448 |
| 32 | adaptive | 1.249 | 1.575 | 0.119% | 699 | 506 | 446 |
| 64 | full | 0.000 | 1.866 | 0.098% | 529 | 523 | 512 |
| 64 | adaptive | 1.060 | 1.517 | 0.139% | 521 | 519 | 510 |

选择成本约 1 ms/整轮，选择+准备低于 step 的 0.15%，无法解释约 20% 的吞吐下降。结合 forward 计算量未明显降低，
这些证据更符合拆批损失批量执行效率，而非主机预估成为主要瓶颈；尚未用此轮数据量化 GPU 每种批形的效率，不能视为完整 kernel 归因。

[profiling 原始数据](batch-cache-balance-profile.jsonl.gz) · [汇总](batch-cache-balance-profile.jsonl.summary.json)

## 验收决定与后续

- Full 保持默认：原有批量工作页准备继续使用；Adaptive 仅通过 C++ runner 或实验基线显式开启，不改变 worker 协议。
- 不以“即时少驱逐引用”作为优化成功指标。后续若继续改策略，需要同时估计保留前缀能节省的 forward token 和缩小批次的执行成本，
  并用独立负载验证，不能在当前合成矩阵上调阈值后宣称普遍提升。
- CPU 混合参考 smoke 两种策略各 8 请求全部通过，只验证策略接入与生成正确性，不作为 CPU 批量加速证据。
  [CPU 数据](batch-cache-balance-cpu.jsonl.gz) / [汇总](batch-cache-balance-cpu.jsonl.summary.json)；真实 CPU/CUDA 共享准入、取消、EOS 与无进展专项见上方资源检查报告。

## 复现

所有命令在同机顺序执行，CUDA 二进制运行时需要对应 CUDA/cuBLAS DLL 在 PATH。变量 BINARY / MODEL 替换为实际路径。

```text
python miniprefix/benchmarks/capacity.py --binary BINARY --model MODEL --backend cuda --policies shared --batch-policies full adaptive --workloads mixed --shared-prefixes 0 16 --pools 32 64 --rates 40 --requests 48 --repeats 3 --output batch-cache-balance-mixed.jsonl.gz
python miniprefix/benchmarks/capacity.py --binary BINARY --model MODEL --backend cuda --policies shared --batch-policies full adaptive --rates 40 --requests 128 --repeats 3 --output batch-cache-balance-uniform.jsonl.gz
python miniprefix/benchmarks/capacity.py --binary BINARY --model MODEL --backend cuda --policies shared --batch-policies full adaptive --workloads mixed --shared-prefixes 0 --pools 32 --modes hot --rates 40 --requests 48 --repeats 3 --output batch-cache-balance-confirm.jsonl.gz
python miniprefix/benchmarks/capacity.py --binary BINARY --model MODEL --backend cuda --policies shared --batch-policies full adaptive --workloads mixed --shared-prefixes 0 --pools 32 64 --modes hot --rates 40 --requests 48 --repeats 1 --profile --output batch-cache-balance-profile.jsonl.gz
```
