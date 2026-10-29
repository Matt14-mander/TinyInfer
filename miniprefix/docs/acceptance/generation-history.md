# 多轮生成历史完整页复用验收：2026-10-29

基于 main@542580b，加上此前未提交的精确页 trie/LRU 优化与本次 Engine 发布策略。
所有修改继续保留在工作区，没有提交。发布契约见 [多轮历史](../generation_history.md)。

## 验证结果

- CPU 完整重建与 CTest：55/55；CUDA 完整重建与 CTest：63/63（真实设备）。
- 真实 SmolLM2-135M-Instruct：CPU/CUDA × cache off/on，共 44 个场景，对照 Transformers CPU FP32 eager、use_cache=False。
- 三轮完整历史续接、页对齐重算末页、单 token 结束、EOS、取消、同轮活跃分叉均通过；输出 token 精确一致。
- 复用后的全词表 logits 共检查 196,608 个值，容差 `2e-3 + 2e-4*abs(reference)`，argmax 相同。
- forward 与 prefill/decode/reused 计数符合实际执行量；结束 clear 后序列、KV 和设备页表归零，CUDA 无权重/KV 重复上传。
- 小页池容量专项 CPU/CUDA 通过；正常容量压力仍只影响对应请求。
- C++ 单测验证发布发生在成功提交/校验之后、采样之前，异常时不发本轮新 token，完整页引用保留且尾页私有。

## 逐轮实际计算量

| 后端 | 场景 | prompt tokens | 输出 tokens | off forward | on forward | on reused |
|---|---|---:|---:|---:|---:|---:|
| cpu | turn1 | 3 | 7 | 9 | 9 | 0 |
| cpu | turn2 | 13 | 5 | 17 | 9 | 8 |
| cpu | turn3 | 20 | 3 | 22 | 6 | 16 |
| cuda | turn1 | 3 | 7 | 9 | 9 | 0 |
| cuda | turn2 | 13 | 5 | 17 | 9 | 8 |
| cuda | turn3 | 20 | 3 | 22 | 6 | 16 |

这些数据验证实际少执行的 token，不是端到端耗时/吞吐结论。prompt 是确定性 token 历史，
没有自动会话管理或 chat template 拼接；生产调用方需保证前缀 token 一致。

## 行为变化与限制

decode 每补满一页会发布并触碰完整前缀路径，因此 LRU、缓存占用和驱逐统计可与旧策略不同。
例如小页池专项的 evictions 从此前 1 变为 5；输出、容量保护和最终回收不变。不能用旧策略的
全 STATS 字段相等作为本次验收条件。已有索引结构差分仍由原单测独立覆盖。
发布仍最多保留根部 cache capacity 页；超过容量的历史不保证全命中。最终采样 token 没有 KV，
不增加额外 forward；部分页 COW、cached logits、生成尾页共享与抢占均未引入。
本次未重跑长期 soak 或 sanitizer，也未更新历史性能基线。

## 证据

本目录 generation-history.json 保存每场景 prompt/output、命中量、logits 误差和模型/worker SHA256；
generation-history-capacity.json 与两个 tests.log 保存容量和全量回归。历史验收文件保留不覆盖。
