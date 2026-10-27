# 精确前缀缓存的真实收益测量协议

2026-10-26：在真实 SmolLM2-135M-Instruct FP32、分页 KV、GPU 常驻路径上比较关闭缓存、
冷缓存和预填充缓存。复用既有 baseline.measure 的计划到达→token 交付端点；
不修改模型实现、内核、精度或缓存算法来迎合测量结果。

## 固定模型、配置与负载

- 主实验 CUDA，RTX 4070 Laptop；模型 revision、native model、worker 二进制指纹必须
  与通过的 incremental-prefill 数值验收一致。采样前及结束后检查指纹。
- context 256、活跃上限 2、token budget 32、prefill chunk 16、页长 16、KV pool 128 页。
  on cache 容量 48 页；off 为 0。runner 仍逐 sequence 执行，没有跨请求矩阵合并。
- 每条 prompt 固定 120 token：真实英文段落的前 96 个 token，加问题的前 24 个 token。
  这是固定 tokenized prose 负载，不使用 chat template；原始文本、完整 token 和解码文本入档。
- 每组八请求、各输出八 token，Greedy、EOS 关闭。到达时间固定为
  0/0/25/25/50/50/75/75 ms。全部输出由独立 Transformers CPU FP32 eager full-prefix
  oracle 检查，参考计算和 tokenizer 在采样前结束。

| 负载 | 结构 | 目的 |
|---|---|---|
| shared_96 | 八请求共享前 96 token，问题在下一页内分支 | 测量高共享收益及 cold 流中的自动填充 |
| mixed_50pct | 前四请求共享 96 token，后四请求不同根 | 测量较低共享比例，避免只选全命中 |
| no_shared | 八个根的首 token 均不同 | 零命中控制和缓存管理成本；因果 key 不得误命中后续相同内容 |

所有场景的 prompt/输出长度、到达过程和调度配置相同。不同场景输出可能不同，
但同场景 off/cold/warm 的每个输出必须完全一致。no_shared 的条目数会超过容量，
可观察 LRU 淘汰；本轮没有主动制造物理池耗尽或抢占。

## 关闭、冷、热的定义

每配置/负载/轮次使用一个新 worker；默认五轮，共 3×3×5=45 个独立进程。
按轮次轮换负载和模式顺序，全部串行，采样时不并行运行验证、编译或其他 benchmark。
模式顺序以稳定的负载身份轮换，避免负载位置的轮换将其抵消；单元测试要求同一负载
在连续三轮内由 off/cold/warm 各领先一次。修正顺序前的中断试采不进入最终五轮结果。
worker 内先执行三条代表性 warmup，再清空缓存。所有模式再运行同一条 97-token seed，
seed 为共同的 96-token 根加一个不同尾 token，只发布六个完整页面。

- off：缓存容量为零；seed 只是对齐计算预热，测量流内始终无缓存。
- cold：开启缓存，但 seed 后清空；测量流开始时零缓存，允许流内自然命中与填充。
- warm：保留 seed 的六个公共页面；没有预先计算八条实际问题的 suffix。
  在 no_shared 中，这六页与任何请求都不匹配，形成缓存污染控制。

seed 的执行时间、输入量与 before/after 缓存状态单列。常规 warm window 不包含 seed，
额外提供 warm 的 **prime-inclusive 吞吐**：仅将测量流的 64 个输出 token 作为分子，
分母为测量窗口+seed 执行时间。它展示本组八请求承担一次 seed 成本的摊销情况；
不把预填充当免费工作，也不将人工预热 seed 成本加给 off/cold 的应用侧成本。
启动、权重读取/上传和其余算子 warmup 不计入该摊销口径。

## 指标与检查

- TTFT 从计划到达到控制器收到首 token，包含提交延迟、排队、IPC、forward、采样和
  请求释放。单列 controller_submit_delay，不估算无法直接观察的 native 排队时间。
- TPOT 为每请求 `(last-first)/(outputs-1)` 后取均值，ITL P95 为相邻 token 间隔分位数。
- 吞吐为 64 个输出 token / 从首个计划到达到最后 token 交付的窗口；含到达空闲及排队。
- 每轮分位数采用 nearest rank；八请求的 TTFT P95 等于该轮最大值。汇总五轮
  median/min/max/样本标准差；off 对 cold/warm 的速度比按同轮配对后再汇总。
  不以中位数之比替代配对分布，不声称生产 SLA、置信区间或统计显著性。
- 记录请求命中比例、复用 token/总 prompt token、实际 forward token、缓存页/字节、
  淘汰次数、CUDA token/KV/权重/页号传输与分配。无共享场景不得出现任何复用。
- `actual_forward = sum(prompt+output-1) - reused`；CUDA token H2D 必须为实际输入×4。
  warm weights/KV H2D 必须为零增量，workspace/pool 不增长，仅允许每序列一次页表分配。
- 完成时没有请求或设备页表存活；所有物理 live 页必须仅由缓存持有。显式 clear 后
  live KV 为零。缓存关闭的测量流结束时也必须为零。
- 缓存 payload 是 retained pages×全层 K/V 每页字节数，不包括 vector/map 主机索引开销。
  pool 预留量单列；`peak_kv_bytes` 为进程启动以来峰值，包含 warmup/seed，不冒充流内峰值。
  所有 payload 均不等于 RSS、总 VRAM 或驱动/cuBLAS 内部内存。

计时窗口不查询 STEP_STATS，也不执行 STATS、clear、oracle、解码或 JSON 写盘。
窗口后检查输出与计数，失败立即中止，不生成通过报告。source/model/worker/build 指纹、
环境、GPU 前后快照、全部请求 token/timestamp、步骤和计数器均保留。

## 重现

沿用 [真实模型准备](real_model.md)，先确认当前二进制通过 [增量 prefill 验收](incremental_prefill.md)：

```powershell
$py = 'build-miniprefix-env/Scripts/python.exe'
$env:HF_HOME = "$PWD/build-miniprefix-models/hf-cache"
# CUDA bin 必须在 PATH；沿用已有 ROG 构建环境。
& $py miniprefix/benchmarks/prefix_benefit.py --model-dir build-miniprefix-models/smollm2-135m-instruct --worker build-phase54-cuda/miniprefix/miniprefix_cuda_worker.exe --backend cuda --verification miniprefix/docs/acceptance/incremental-prefill-verification.json --runs 5 --output benchmark-results/miniprefix/prefix-benefit-cuda-fp32.json
```

当前 worker/model 若与验收指纹不同，先重新执行 verify_incremental_prefill.py 并将
新验收路径传给 --verification。可显式改为 CPU worker/--backend cpu 重现另一后端；
当前主实验结果仅代表实际运行的 CUDA 配置，不把 CPU oracle 当作 CPU runner 性能。

结果见 [真实收益报告](acceptance/prefix-benefit-cuda-fp32.md)。原 dense/paged 八请求基线、
短 prompt 的 paired 实验和本轮固定长前缀负载分别保留，不跨负载拼接速度比。
