# 抢占重算、前缀共享与 CUDA PagedAttention

日期：2026-10-22。三项能力均已接入真实 worker，默认仍为连续 KV / 保守准入，
按参数开启分页、共享、动态容量与 CUDA。模型范围保持单层、单头、D=16、FP32。

## 运行

```python
from examples.serving_scheduler.backend import Request
from examples.serving_scheduler.tinyinfer_backend import TinyInferBackend
from examples.serving_scheduler.realtime import serve

# CPU: tinyinfer_serving_worker；CUDA: tinyinfer_serving_cuda_worker。
with TinyInferBackend(worker, block_size=4, kv_layout="paged", kv_capacity_blocks=8,
                      attention_mode="cuda", prefix_sharing=True) as backend:
    result = serve([Request("A", 0, 9, 5), Request("a", 0, 9, 5)], backend,
                   max_batch_size=3, kv_capacity_blocks=8, kv_block_size=4,
                   token_budget=8, prefill_chunk_size=4, kv_allocation="dynamic")
```

CPU worker 使用 `attention_mode="direct"` 或 `"gather"`。请求 ID A/a 产生相同的
样例整数 prompt，方便观察共享；没有真实文本 tokenizer。worker 握手已更新到 `PAGED 2`，
需要重新构建，不兼容旧二进制。

## 动态容量与抢占重算

`serve(..., kv_allocation="dynamic")` 使用独立的 `dynamic_scheduler.py`，只支持
Continuous + wait + 有限物理分页 pool；其他组合明确报错。默认 `"reserved"` 保留
原调度器和最大预算预留行为。

动态模式不在准入时预留整个最大长度，允许多个请求的最终 KV 总量超过 pool。
每个请求自身的 `ceil((prompt+output)/S)` 仍必须能独占装入 pool，否则拒绝。
每轮优先服务已有 decode，然后重建被抢占请求，最后处理新 prefill；三类工作共享
同一 token budget、同一打包矩阵和七次线性投影。prefill chunk 上限也用于重算 chunk。

执行前通过 native CAN_APPEND 试算整批增长和 COW，试算不修改 pool。不能装入时，
从活跃队尾选择较晚准入请求，释放其 native KV，将其放回等待队首。最早仍活跃请求
受保护，避免所有请求同时等待扩容；对于有限固定长度工作负载，它可独占完成并释放空间。
已经进入 worker 的请求计为 preemption；尚未计算的请求退回队列计为 kv_wait。

抢占保留 RequestResult 中已经交付的 token 和原始时间戳。恢复时分块重算之前缓存的
`prompt + generated[:-1]`，尚未完成 prefill 则只重算其已处理部分。重算最后一步
重新预测的 token 必须等于此前交付的最后一个 token；所有重算结果都返回 None，
不重复交付，不重新采样已交付序列。恢复完成后再继续正常 decode/prefill。
重算途中再次被抢占则重新排队，重建进度从零开始；有限工作负载下受保护请求继续前进。

等待恢复期间不计 native 占用，最初准入时刻保持不变，重算时间进入实际端到端延迟。
此策略不是在线负载的公平性或 SLA 承诺。动态模式目前 FIFO prefill，没有优先级、
EOS、取消、swap-to-host 或网络服务。

## 精确前缀共享与 COW

`prefix_sharing=True` 启用当前 worker/model 内的不可变 KV 快照缓存。key 是从位置零
开始的完整整数 token 前缀，使用精确 vector 比较，无哈希碰撞。只有 token 相同且
位置相同的因果前缀可复用；不同请求可以共享物理 block。

成功 STEP 发布完整 block 边界、最后一个 token 之前的前缀和当前完整前缀的快照。
新 prefill 或重算的首 chunk 查找最长匹配，但至少保留一个 query 行以生成 logits。
命中的 KV 行不重复执行 embedding、投影、attention、FFN 或 head。
后续 chunk 不再查找缓存；这是显式的首版复用边界。

引用计数包含请求 table 和缓存快照两种持有者。完整共享 blocks 只读；共享的部分尾块
继续追加时先复制 K/V block，然后写入私有副本。prepare_append 在临时 metadata 上
完整规划分配、COW 和缓存驱逐，成功后才复制 payload/提交；失败不留下部分变更。
直接写入仍共享的物理 block 会报错。

缓存快照在源请求 RELEASE 后仍可保留；空间不足时先按确定性的 key 顺序驱逐快照，
直到可分配或证明不足。正在被其他请求引用的 blocks 不会提前回收。CLEAR/RESET 清空
所有请求与快照，避免残留引用；每次 serve 仍在开始 RESET、结束 CLEAR，缓存只在
同一请求流内共享。手动使用持久 worker 的多次 STEP/RELEASE 可跨请求复用。

本例采用 map/vector metadata 副本和快照引用计数，便于验证事务；尚未实现 radix tree、
LRU、跨 worker/model 缓存、磁盘持久化或多层/GQA。

## CUDA kernel 与构建

打开项目既有 `TINYINFER_ENABLE_CUDA=ON` 后新增：

- `tinyinfer_serving_paged_cuda`：分页 attention CUDA 库。
- `tinyinfer_serving_cuda_worker`：CPU 模型投影 + CUDA attention 的混合 worker。
- `tinyinfer_cuda_paged_attention_tests`：在真实设备上执行原生数值/布局测试。

使用既有 ROG CUDA 12.5.1 + MSVC 14.29 构建目录：

```powershell
# 在 tools/build_cuda_rog.ps1 同样的开发环境与 CUDA bin PATH 中执行。
cmake --build build-phase54-cuda --target tinyinfer_serving_worker tinyinfer_serving_cuda_worker tinyinfer_paged_attention_tests tinyinfer_cuda_paged_attention_tests --parallel 4
ctest --test-dir build-phase54-cuda -R 'tinyinfer_(serving_scheduler|paged_attention|cuda_paged_attention)_tests' --output-on-failure
```

CUDA 可选目标不影响 SDK-free CPU 构建。CPU worker 接到 CUDA 配置会明确失败。
kernel 每个 query 使用一个 warp，16 个 lane 分别持有输出维度，32 个 lane 均参与
shuffle 点积归约，按 block table 逐 token 在线更新 softmax。一个 STEP 的所有 query
同一次 launch，支持混合长度、causal prefill 和 decode，无完整 KV gather。
实现使用的 warp shuffle 语义见 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html#warp-shuffle-functions)。

设备 K/V 与工作缓冲按需分配并复用，使用一个非默认 stream。首版每 STEP 把完整 CPU
K/V pool、query、table 和 length 上传，再下载 context 并同步；CPU 是 KV 权威副本，
所以共享、驱逐、COW、重算在上传后对 kernel 一致可见。Q/K/V 投影、FFN、head 仍在 CPU。
此路径是实际 CUDA attention kernel 基线，不是全 GPU 模型或零拷贝服务。kernel/传输失败
传播错误，等待终止后清理请求状态，不自动重跑 CPU；不声明吞吐或延迟加速比。

## 指标与基准

除既有指标外，backend_stats 新增：

| 字段 | 含义 |
|---|---|
| preemptions | 动态调度实际释放 native 请求的次数 |
| replay_input_tokens | 调度给重算的输入行数，含缓存命中 |
| recompute_tokens | 实际重新计算的 token 行数，不含命中 |
| prefix_cache_hits / prefix_reused_tokens | 成功复用次数与跳过的计算行数 |
| cow_block_copies | 实际复制的共享尾块数 |
| prefix_cache_entries / prefix_cache_evictions | 当前快照数与累计驱逐数 |
| cuda_attention_queries | 实际 GPU query 行数 |
| cuda_kernel_launches / cuda_uploaded_bytes | 实际 launch 次数与 H2D payload 字节 |

preemptions/replay_input_tokens 由动态调度器写入结果，其他字段由 native worker 计数。
RESET 清零全部统计（包括 CUDA）；CLEAR 清空 live KV/缓存并保留累计统计。
prefix 开启时，live_physical_blocks 和 live_kv_bytes 包括缓存持有的 blocks，不能
将 RELEASE 后的非零缓存占用解释为泄漏；serve 结束 CLEAR 后两者必须归零。
动态模式没有逻辑完整预留，trace 单列 kv_physical_blocks，保留字段 kv_reserved_blocks=0。

所有配置校验以下恒等式：

```text
prefill_input_tokens + decode_input_tokens + recompute_input_tokens
    = forward_tokens + prefix_reused_tokens
linear_matmul_calls = 7 * batch_steps
linear_matmul_rows  = 7 * forward_tokens
cuda_kernel_launches = batch_steps   # CUDA attention 模式
```

`recompute_input_tokens` 是报告中的调度行数，与启用缓存时的实际 recompute_tokens 不同。
动态模式的基准比较 continuous_full/continuous_chunked，保守模式仍比较四配置。

```powershell
python -m examples.serving_scheduler.real_benchmark --worker build-phase54-cuda/examples/tinyinfer_serving_cuda_worker.exe --runs 3 --kv-layout paged --attention-mode cuda --prefix-sharing --kv-allocation dynamic --kv-capacity-blocks 8 --output benchmark-results/serving-cuda-dynamic-shared.json
```

## 验收

本机 Windows 11、RTX 4070 Laptop GPU、驱动 555.97、CUDA 12.5.1、MSVC 14.29、Release/native
实际执行。完整 Python 套件 63/63（包括 2 项真实 CUDA 测试，无 skip）；CPU 与 CUDA
原生测试通过，CTest 三个目标通过。SDK-free CPU 构建也通过其 CPU 测试，CUDA 测试类按配置跳过。

验证包括：输出和 logits 对完整前缀 oracle 一致、重算不重复交付、重算计入 token budget、
部分前缀 COW 后分叉输出、源请求释放后复用、容量失败原子性、缓存驱逐、错误清理、
GPU 大幅 score、NaN 尾块屏蔽、非连续 blocks、1/3/7 block size、129-token 混合长度、
kernel/上传计数和 reset。随机工作负载覆盖多个 budget 和共享开关。

CPU dynamic（共享开/关）、CUDA dynamic+共享、CUDA reserved+共享四组各三个 worker，
共 30 份真实报告全部通过参考输出、资源与计数检查。原始文件在 Git 忽略的
`benchmark-results/serving-{cpu-dynamic,cpu-dynamic-shared,cuda-dynamic-shared,cuda-shared}.json`，
验收摘要见 [verification.json](results/advanced_serving/verification.json)。这些运行是
正确性与资源压力实验，并行采集的墙钟数据不用于性能比较。
