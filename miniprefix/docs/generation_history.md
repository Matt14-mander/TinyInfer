# 多轮生成历史的完整页复用

## 发布时机

Engine 保留 prefill 完成后的发布时机，不在中间 prefill chunk 或 submit 时发布。
在 decode 中新增以下顺序：

1. 把上一步采样的 token 交给 runner.append，完成模型执行、同步和 KV 事务提交。
2. 校验返回 logits 的维度和有限值。
3. 如果已提交 token 数刚好落在完整页边界，调用 cache_prefix 发布已提交历史的完整页。
4. 采样新 token，产生事件；EOS/长度结束时再释放请求引用。

已提交长度在 decode 发布点是 `prompt.size() + output.size()`（output 尚未追加本轮采样值）。
最后一个输出 token，包括 EOS，在该请求结束时尚未作为输入执行，因此没有对应 KV，不能缓存。
没有为了补齐它而偷偷运行额外 forward。下轮 prompt 中包含它时，作为未命中 suffix 正常计算。

例如页长 4、prompt `[a,b,c]`，生成 `[d,e]`：prefill 后已计算 3 个 token，采样 d；
下轮 append(d) 后第 1 页 `[a,b,c,d]` 才发布，随后采样 e 并结束。下一轮带上 e 和新的用户
内容，可以复用前 4 个 token，e 仍要计算。已提交但不满页的尾部同样不会发布。

## 接口、共享与容量

调用方为下一请求提交完整的历史 token + 新输入，使用新的 request ID。当前没有会话存储、
文本历史重分词或 chat template 的自动拼接；这些转换必须保证想复用的 token 前缀完全相同。
前缀树仍按所有因果历史精确匹配，模型/位置/RoPE/精度的作用域不变。

发布通过现有 KVManager/ExactPrefixCache，不改变 GPU payload，不复制完整 KV pool。
只共享已经填满且不再写入的页；源请求后续 decode 和新分支各自使用私有尾页/新页，
无需部分页 COW。源请求仍活跃时，跟随请求也能在同一步后续调度中命中刚发布的历史页。

只有跨完整页边界才新增 decode 发布调用，不在每个未满页 token 上扫描索引。
发布仍触碰其覆盖的根到完整页路径，可能改变 LRU、缓存占用和 evictions；这些统计与旧的
“仅 prefill 发布”策略不应强求相等。索引内部匹配/引用行为保持前一步的差分语义。

保持一次最多发布最前 `prefix_cache_blocks` 页的根优先策略。因此缓存容量不足以覆盖
历史时，较后面的生成页不会全部保存；需为多轮历史配置足够容量。本次不改变淘汰策略或
按共享折扣的准入，仍使用最坏情况预约。缓存关闭时无 KV 缓存引用；默认配置仍为关闭。

## 结束、取消、错误与清空

- Length/EOS：完整已提交页在释放请求前已发布，缓存 lease 使它们继续存活。
- Cancel：保留此前边界已发布的页，不为取消额外 forward，也不发布私有尾页。
- logits 校验或发布失败：沿用后端错误处理，Engine 失败并尝试清理；不采样/交付本轮新 token。
- 容量请求错误：沿用请求级隔离及预约释放，不改变其他请求已发布页的引用。
- clear_prefix_cache：移除当前缓存引用；活跃请求继续运行，在后续页边界可能再次发布。
- shutdown/后端清理：回收请求与缓存引用。清理失败时仍保留原资源所有权规则。

始终保留至少一个 prompt token 用于计算 logits；若下轮 prompt 本身完整页对齐，即使已有
全部 KV，仍会重新计算最后一页，未增加 cached-logits 路径。

## 回归

`miniprefix_engine_history_tests` 覆盖发布快照、最终未计算 token、页对齐、EOS/取消、
无效 logits/发布异常、同一步共享和缓存关闭。使用真实 KVManager/前缀索引管理引用。

`verify_generation_history.py` 对真实 CPU/CUDA 分别运行 cache off/on，覆盖三轮续接、
页对齐、单 token 结束、EOS、带私有尾页的取消及活跃源请求的分叉。逐 token 对照独立
Transformers CPU FP32 eager full-prefix，另对复用后的完整 logits 使用既有容差和相同 argmax。
检查实际 forward = prompt + 输出数 - 1 - reused，以及 step/runner 计数一致和最终资源归零。
输入/输出 token、命中和模型/worker SHA256 都写入 JSON，禁止以进程重启掩盖多轮状态问题。

```text
python miniprefix/tools/verify_generation_history.py --model-dir MODEL_DIR --cpu-worker CPU_WORKER --cuda-worker CUDA_WORKER --output generation-history.json
```

本次不是新吞吐基线，不将跳过 token 数直接宣称为端到端加速；部分页 COW 留待下一步。

本次结果与原始证据见 [2026-10-29 验收报告](acceptance/generation-history.md)。
