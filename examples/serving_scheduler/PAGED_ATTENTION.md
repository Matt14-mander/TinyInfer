# CPU PagedAttention

日期：2026-10-22。已在跨请求张量 batching 后端中加入物理分页 KV 与直接分页 attention。
适用模型仍为单层、单头、D=16、FP32 的随机权重 Transformer。
后续已实现可选抢占重算、精确前缀共享/COW 和真实 CUDA attention kernel，见
[ADVANCED_SERVING.md](ADVANCED_SERVING.md)。本文描述默认关闭共享的首版 CPU 路径；
当前协议版本为 `PAGED 2`，下方 56 项测试与验收记录保留为首版历史。

## 使用

保持默认连续 KV 路径用于回归；通过参数显式开启分页：

```python
from examples.serving_scheduler.backend import Request
from examples.serving_scheduler.tinyinfer_backend import TinyInferBackend
from examples.serving_scheduler.realtime import serve

with TinyInferBackend(worker, block_size=4, kv_layout="paged",
                      kv_capacity_blocks=32) as backend:
    result = serve([Request("a", 0, 13, 5), Request("b", 0, 7, 3)],
                   backend, max_batch_size=4, kv_capacity_blocks=32,
                   kv_block_size=4, token_budget=8, prefill_chunk_size=4)
```

物理 pool 容量必须与调度器容量一致，block size 也必须一致，否则在执行前报错。
分页模式需要有限非负容量，零容量合法；block size 为 1..4096。模型的取整容量仍须
不超过 4096。现有连续模式接口保持默认行为。需要重建 worker；握手增加 `PAGED 1`。

真实基准入口（以下 Windows 路径对应本地既有 Release 构建）：

```powershell
$worker = 'build-phase54-contract/examples/tinyinfer_serving_worker.exe'
python -m examples.serving_scheduler.real_benchmark --worker $worker --runs 3 --kv-layout paged --output benchmark-results/serving-paged-attention.json
# 诊断对照：分页存储，但先 gather 再使用已有 MatMul/Softmax
python -m examples.serving_scheduler.real_benchmark --worker $worker --runs 3 --kv-layout paged --attention-mode gather --output benchmark-results/serving-paged-gather.json
# 原连续 KV 张量 batching 路径
python -m examples.serving_scheduler.real_benchmark --worker $worker --runs 3 --output benchmark-results/serving-contiguous-kv.json
```

## 存储与生命周期

`native/paged_kv_cache.h` 定义统一 K/V pool，物理形状各为 `[N, S, 16]`。
每请求只保存状态元数据和逻辑 block 到物理 block 的映射，不再分配连续 K/V Tensor。

```text
logical_block = position / S
offset        = position % S
physical_block = block_table[request_id][logical_block]
```

Python 仍在准入时保守预留 `ceil((prompt+output)/S)` 个逻辑 blocks，防止已准入请求
增长时发生容量死锁。C++ 只为本轮实际写入后的 KV 长度领取物理 blocks。生成首 token
时 KV 长度等于 prompt；之后每次 decode 写入上轮生成的 token，最终长度为
`prompt+output-1`，不写入已经结束请求的最后一个输出 token。

完整 batch 验证通过后，`prepare_append` 在副本上构建所有 table/free-list 修改，
全部成功后交换提交；容量失败或主机分配异常不留下部分映射。成功领取后才开始投影。
现阶段复制 table/free-list 以明确事务边界，不作为优化实现。

RELEASE 将独占 blocks 归还 pool；CLEAR 和 worker 异常清空所有映射，RESET 额外清空
计数。物理 pool 保持分配，可继续复用。worker 的 STEP 失败沿用现有整轮失败语义：
丢弃所有请求状态，由 Python 调度器回收逻辑预留，不返回部分成功 token。

## 直接分页 attention

Q/K/V、输出投影、FFN、词表投影继续按打包 token 行批量计算。attention 对每个 query
遍历自己的 block table，逐块计算 QK score、局部最大值和加权 V。
维护累计最大值 m、分母 l、未归一化输出 u；新 block 最大值为 mb 时：

```text
m_new = max(m, mb)
scale = exp(m - m_new)
l_new = scale*l + sum(exp(score - m_new))
u_new = scale*u + sum(exp(score - m_new)*V)
output = u / l
```

scratch 为 `O(S+D)`，直接路径不 gather 完整 KV，不构造完整前缀的 scores/probabilities。
每个 query 的 length 严格为绝对位置+1；未来 token 和尾块未用位置不参与运算，
即使同一 chunk 的未来 K/V 已写入。非有限 score/output 明确报错并清理。

`attention_mode="gather"` 是诊断路径：通过相同 block table 收集有效 K/V，再执行
既有 MatMul/Softmax。完整前缀 reference 始终使用原单 token、连续 KV forward，
不共享分页存储或 online softmax。`last_step_logits()` 与 `block_table(request_id)`
仅在测试/诊断时额外查询，真实基准不在每轮查询它们。

设计参考：[PagedAttention](https://arxiv.org/abs/2309.06180) 的分页管理思想；
online softmax 的分块重缩放可参见 [FlashAttention](https://arxiv.org/abs/2205.14135)。
本例是 CPU 标量 attention 基线，不是上述论文的 GPU kernel 或性能复现。

## 指标

| backend_stats 字段 | 含义 |
|---|---|
| pool_capacity_blocks | 物理 pool block 上限 |
| pool_kv_bytes | 预分配 K/V payload 总字节，CLEAR/RESET 后保留 |
| live_physical_blocks / peak_physical_blocks | 当前/峰值归属请求的 blocks |
| free_physical_blocks | 可领取的 blocks |
| live_kv_bytes / peak_kv_bytes | 当前/峰值归属请求的 block payload 字节，含尾部空位 |
| peak_cached_tokens | 峰值有效 KV token 数 |
| paged_attention_queries | 直接分页 attention 处理的 query 行数 |
| gather_attention_queries | gather 参考路径处理的 query 行数 |

分页模式的 live_kv_bytes 为占用 block 字节，不等于进程实际已分配的 pool 字节；请求
完成后 live 为零，pool_kv_bytes 仍为 `N*S*16*2*4`。以上均不包含权重、临时张量、
table/free-list/allocator 或 IPC 内存，不代表 RSS。逻辑预留、物理占用、pool 分配分开报告。
原七次共享线性投影计数保持不变，直接分页 attention 的标量点积不计为 MatMul 调用。

## 验收

重建 `tinyinfer_serving_worker` 与 `tinyinfer_paged_attention_tests` 后执行：

```powershell
$env:TINYINFER_SERVING_WORKER = (Resolve-Path $worker).Path
python -m unittest discover -s examples/serving_scheduler -v
ctest --test-dir build-phase54-contract -R 'tinyinfer_(serving_scheduler|paged_attention)_tests' --output-on-failure
```

新增七项 Python 集成测试覆盖三路径 logits/Greedy 对照、混合阶段与请求重排、
非连续 blocks、跨块增长和释放复用、S-1/S/S+1、多个 block size、257-token 长上下文、
容量失败/重复请求清理、物理按需领取、pool 保留、Static/Continuous、多种 token budget、
wait/reject、零容量和配置不匹配。Logits 使用既定 `2e-5+2e-5*abs(reference)` 容差，
生成 token 必须一致。

C++ 原生测试验证 batch 分配事务、物理碎片、double precision softmax 参考、
大幅正负 score（约 ±4000）、NaN 尾块屏蔽、请求隔离、零容量和释放/重置生命周期。
本地 Release/native CPU 构建通过 56/56 Python 测试（无 skip）及原生测试。
连续、分页 gather、直接分页三条路径各三轮/四配置/30 请求，36 份真实报告全部通过
输出参考、共享投影计数、有效输入 token 与最终物理/逻辑回收检查。验收摘要见
[verification.json](results/paged_attention/verification.json)。原始 JSON 在 Git 忽略的
`benchmark-results/serving-{paged-attention,paged-gather,contiguous-kv}.json`。
这些运行用于正确性与资源验收，未作为隔离的配对性能实验，不声明加速比。

默认保留最大预算准入；动态准入、抢占重算、prefix sharing/COW 和 GPU kernel 可按
[高级能力说明](ADVANCED_SERVING.md) 开启。多层、多头/GQA 尚未实现。
