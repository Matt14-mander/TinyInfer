# 非整页前缀复用的成本观测与收益测量

CPU/CUDA runner 新增独立的 `CostStats`，原有 `RunnerStats` 25 字段和 worker v3 的 `STATS`
保持不变。Python 客户端通过 `cost_stats()` 读取累计值，通过 `set_cost_timing(bool)` 开关耗时采集。
新客户端调用旧 worker 时会得到 unknown command，不能将不支持的成本字段当成零。

## 统计契约

`COST_STATS` 返回以下 8 个无符号整数，顺序固定：

|字段|含义|
|---|---|
|lookup_calls|KVManager 前缀查找次数，缓存关闭的调用也计数|
|lookup_ns|查找、命中引用/元数据建立及容量检查的主机时间|
|publish_calls|KVManager 发布调用次数，重复键和缓存关闭的调用也计数|
|publish_ns|前缀发布、索引/LRU 和引用操作的主机时间|
|cow_pages|已准备替换页并进入后端复制路径的次数，包含后续失败的尝试|
|cow_bytes|CPU 完成或 CUDA 成对成功提交的 K+V 复制字节；仅有效 token、所有层|
|cow_copy_ns|CPU 复制回调区间时间，或 CUDA 同 stream 复制区间的 event 时间|
|cow_timed_copies|完成计时的 COW 复制区间数量|

计数始终启用，耗时默认关闭。`COST_TIMING 1` 开启，`COST_TIMING 0` 关闭，不重置累计值。
release、clear、清空前缀也不重置；实验通过区间前后快照作差。复制后模型计算失败不撤销
已发生的复制成本；若 CUDA 错误导致未完成计时，不能用 cow_pages 假定所有区间都有时间。
CPU/CUDA 都按两份 payload（K 和 V）计算字节，不计未使用尾部、权重、页表和 token 上传。
每个复制回调的 K/V 成对完成提交才累计字节（合并后的 CUDA 回调包含全部层）；
若 CUDA 的 K 提交成功而 V 提交失败，孤立 K 的字节不计入，
因此故障场景的 cow_bytes 是已成对提交工作量，不保证涵盖所有失败提交前的设备流量。

CUDA event 在首次启用时创建并在 runner 销毁时释放。事件记录在模型 stream 上，仅包围 D2D
复制；在现有模型完成同步后读取 elapsed time，没有新增逐次复制同步。区间可能包含主机
提交间隙，因此它是 stream 上复制阶段的实际跨度，不是纯 DMA 带宽或端到端耗时的可直接减项。
复制计时不含 KV 页分配、页表上传和事务提交；这些成本包含在请求端到端时间中，本版没有
单独拆分其耗时。cold/off 差值同时包含发布、索引、COW 等成本，不能把整个差值归因于复制。
错误后的 Engine 禁止再次更改计时状态；只读累计统计仍可获取。

## 实验方法

入口：`miniprefix/benchmarks/partial_prefix.py`。固定 SmolLM2-135M-Instruct FP32、页长 16、
64 页池、缓存容量 32、prefill chunk/token budget 16、单请求、每次输出 8 tokens。
四类负载分别为相同 31-token prompt、多轮历史续接、在第 17 token 后分叉、首 token 不同的零共享。

- off：缓存关闭，先 warmup 模型与 workspace。
- cold：缓存启用，warmup 后清空，测量开始时无缓存。
- hot：warmup 后清空，再生成指定历史建立缓存；prime 不计入测量区间。

每轮每个模式使用新 worker，模式和负载执行顺序随轮次轮换。CPU 与 CUDA 顺序运行。
每个组合至少 3 轮，默认 5 轮；报告 median/min/max 和同轮 off/hot 配对比值，保留原始记录。
另运行一轮相同负载的 profiling，**不将 profiling 的 TTFT/TPOT 混入性能汇总**。

时间边界：客户端 submit 之前到 token 事件交付。包括 submit/step IPC、调度和模型执行；
不包含进程/权重加载、分词、warmup、prime、统计查询或缓存清理。测量期间不调用 STATS/STEP_STATS。

- TTFT = 第一 token 交付时间 − submit 开始时间。
- TPOT =（最后 token − 第一 token 交付时间）/（输出 token 数 − 1）。
- 输出吞吐 = 输出 token 数 /（最后 token 交付时间 − submit 开始时间）。

这是单请求输出吞吐，不能当作多客户端饱和吞吐。每个组合对照缓存关闭 CPU 输出，逐 token
校验，并检查实际复用、forward 数、终态、资源释放和无 KV/权重 H2D。现有独立 Transformers
正确性证据仍保留；本实验的 CPU 参考用于隔离缓存/后端差异，不替代独立框架验证。

原始 JSON 包含输入/输出、逐 token 时间戳、统计快照、成本差值、模型/worker/script SHA256、
配置和环境；自动生成同名 Markdown 表格。未结束的输出标为 `complete: false`，不得用于验收。

```text
python miniprefix/benchmarks/partial_prefix.py --model-dir MODEL_DIR --cpu-worker CPU_WORKER --cuda-worker CUDA_WORKER --repeats 5 --output RESULT.json
```

hot 的 prime 成本需要后续请求摊销；cold/off 对照显示缓存本身的成本。当前没有旧版整页模式
对照，因此不能把 off/hot 的全部收益归因于“非整页”新增能力。页长、容量、共享比例、输出长度
变化可能改变结论，下一步优化应根据实测 COW 与发布成本决定。

本次结果及异常处理见[验收与结论](acceptance/partial-prefix-cost-review.md)，两组完整原始数据均保留。
