# 跨请求 CPU 张量 Batching

日期：2026-10-22。复用既有 Static/Continuous 调度、KV 最大预算准入、Chunked Prefill
和墙钟驱动，在真实 C++ worker 中增加跨请求计算合并。

后续已加入可选 [CPU PagedAttention](PAGED_ATTENTION.md)，分页模式替换本文的连续 KV
存储和 attention 步骤，共享投影不变；本文 49 项测试与基准记录保留为 batching 阶段历史。

## 计算路径

每次 STEP 先解析完整 batch，检查请求 ID 唯一性、状态进度、输入 token 和容量，
再将每个 work item 的输入 token 打包为 `[sum(input_tokens), 16]`，无需 padding。
每行保留请求所属 State 与绝对位置，位置项沿用该请求的 KV 长度，不使用 batch 行号。

1. 整个矩阵分别执行 Q、K、V 投影，共三次 MatMul。
2. 将 K/V 按请求与绝对位置写入各自的连续 KV Tensor。
3. 每个 query 独立执行 QK/Softmax/PV，只读取自己截至当前位置的因果前缀。
   即使同一 chunk 的未来 K/V 已写入，未来位置也不参加计算。
4. 合并所有 context 行，批量执行输出投影、残差、LayerNorm、两次 FFN 投影、
   激活、残差、LayerNorm 和词表投影。LayerNorm 始终沿最后一维逐行归一化。
5. 每个 work item 取最后一个输入行的 logits；未完成 prefill 返回 None，完成
   prefill 或 decode 才采样一个 token。输出顺序与输入 work item 顺序一致。

每个非空 STEP 的共享线性投影固定为 **7 次 MatMul**，attention 另有每输入 token
两次 MatMul。旧路径线性投影需要 `7 × 输入 token 数` 次调用。计数减少验证的是
计算合并，不等同于墙钟加速；微型模型中打包、Tensor 分配、attention 和 IPC 仍有成本。

输入解析/验证或计算异常会清空本 worker 的全部请求 KV 与上轮 logits 并返回 ERROR；
沿用调度器终止本轮并清理逻辑预算的规则，不返回部分成功结果。校验可能创建新 State，
但不会在完整 batch 校验通过之前执行投影。

## 接口与诊断

Python 的 Request、WorkItem、StepResult、serve 调用方式不变。启动握手增加
`BATCHED 1` 协议标记，旧 worker 会明确被拒绝，需要重新构建。

`backend_stats` 新增：

| 字段 | 定义 |
|---|---|
| batch_steps | 成功完成的 STEP 次数 |
| linear_matmul_calls | 上述七类共享投影的调用次数，不含 attention |
| linear_matmul_rows | 每次共享投影输入行数之和，应为 7 × forward_tokens |
| peak_batch_requests | 单次 STEP work item 数峰值 |
| peak_batch_tokens | 单次 STEP 打包输入 token 数峰值 |

forward_tokens 只计服务推理输入，独立参考推理不再污染该计数。
RESET 清空所有计数；CLEAR/RELEASE 回收 KV，保留累计计数供报告查询。

`last_step_logits()` 返回最近成功 STEP 各 work item 最后输入行的 32 维 logits，
包括未完成 prefill；`reference_logits(prefix)` 用旧的单 token forward 从零重算完整前缀。
这些诊断通过额外 IPC 获取，测试在计时窗口外使用；基准不逐轮查询 logits。
完整前缀 Greedy 参考也保持逐 token 路径，独立于打包实现。

real_benchmark 报告增加 revision=tensor_batching、batching 描述与硬件 batching 标记。
此处 hardware_batched=true 指 CPU 张量合并；没有 GPU kernel、线程并行或批量变长 attention。
模型仍是单层、单头、随机权重、FP32 的工程样例，KV 仍是每请求连续预分配。

## 构建与验收

沿用 STAGE4.md 构建命令，重建 tinyinfer_serving_worker。Windows 示例：

```powershell
$env:TINYINFER_SERVING_WORKER = (Resolve-Path build-phase54-contract/examples/tinyinfer_serving_worker.exe).Path
python -m unittest discover -s examples/serving_scheduler -v
python -m examples.serving_scheduler.real_benchmark --worker $env:TINYINFER_SERVING_WORKER --runs 3 --output benchmark-results/serving-tensor-batching.json
```

新增五项测试覆盖混合 decode/prefill、不同 chunk 长度、未完成 chunk 的 logits、
输出重排、完成释放后补位、8 请求 decode 合并、同请求不同同伴、错误 batch 清理与复用、
固定随机工作负载的 Static/Continuous 和多种 token budget。
Logits 固定容差为 `abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`，
Greedy token 要求一致。算子调用数、输入行数和最终资源占用也必须通过检查。

旧 results/stage4 数据保持原样，不能作为当前批量实现的性能结果。

2026-10-22 本地验收：Windows 11、MSVC 14.43、Release、native CPU 指令开启、
Python 3.12.3；完整 Python 套件 49/49，无 skip；CMake 注册的
tinyinfer_serving_scheduler_tests 通过。三轮独立 worker、每轮四配置、每配置 30 请求，
共 12 份真实报告全部通过完整前缀参考、投影计数、输入 token 和 KV 回收检查。
每份报告输入 token 为 300，linear_matmul_rows 为 2100，峰值同批请求数为 3。
8 请求 decode 专项测试验证每轮仍只调用 7 次共享投影 MatMul。

本地原始基准保存在仓库根目录的 `benchmark-results/serving-tensor-batching.json`
（该目录按项目规则被 Git 忽略）。本次只验收正确性与计算合并，未进行同环境旧 worker
的配对性能对照，不声明吞吐或延迟加速比。
