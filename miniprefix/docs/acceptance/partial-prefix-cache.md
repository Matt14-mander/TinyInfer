# 非整页前缀缓存验收（2026-10-29）

基于 `6c479e8`，实现范围与容量规则见 [设计说明](../partial_prefix_cache.md)。

| 验证 | 结果 | 证据 |
|---|---|---|
| CPU 完整 CTest | 56/56 | [日志](partial-prefix-cpu-tests.log) |
| CUDA 完整 CTest，要求真实设备 | 65/65 | [日志](partial-prefix-cuda-tests.log) |
| 最终 CUDA memcheck/full leak | 应用完成标记，0 errors、0 bytes leaked | [日志](partial-prefix-memcheck.log) |
| SmolLM2 CPU/CUDA 容量回归 | 两后端通过，最终资源归零 | [JSON](partial-prefix-capacity.json) |
| CPU 混合请求压力 | 10 波、80 请求、333 输出 token，通过 | [原始记录](partial-prefix-stress-cpu.jsonl) |
| CUDA 混合请求压力 | 10 波、80 请求、333 输出 token，通过 | [原始记录](partial-prefix-stress-cuda.jsonl) |
| SmolLM2 CPU/CUDA 多轮独立 oracle | 44 场景、196,608 logits | [JSON](partial-prefix-generation-history.json) |

索引差分采用独立的平铺全历史参考实现，5 种容量各 3,000 次随机操作。验证逐 token 命中、
物理块引用、LRU 结果、计数和索引剪枝；附加长短快照共享一张物理页的统计/淘汰测试。
Engine 回归更新了页内命中的合法性，并验证增量输入从准确位置继续。

实际 CPU/CUDA runner 回归覆盖页长 4、历史长 1/3/4/7/8，页内分叉、整页与非整页相同 prompt、
命中后异常回滚和一页池压力。两层 GQA 小模型全部 logits 对照 CPU 独立重算，
绝对误差 < 2e-5 且 argmax 相同。CUDA memcheck 执行真实 D2D 与 kernel，并包含部分缓存测试完成标记。

真实模型使用固定 SmolLM2-135M-Instruct FP32。多轮测试对照 Transformers CPU eager 全前缀推理，
覆盖 cache off/on、未计算的最终 token、EOS、取消和活跃源分叉，检查 forward = 输入历史长度
加实际 decode 输入减复用 token。相同对齐 prompt 现在只重算最后一个 token；原有完整页
缓存结果中的命中数量不是新版本的验收预期。

混合压力复用既有原生 stress 的无缓存参考，覆盖页面淘汰、容量等待、取消和清空；每波检查
输出及 step/runner 计数。此次是短程压力回归，测试进程可能同时运行，不据其时长声称性能收益，
也不替代此前长期压力验收。非整页快照的发布频率、COW 流量与 TTFT 收益留待专门 paired 测量。

MSVC 14.29、CUDA 12.5.1、RTX 4070 Laptop。CPU/CUDA 均先完整清理构建，后续修正增量构建，
最终完整 CTest 再次通过。最终源码及测试/stress 二进制哈希见 [SHA256](partial-prefix-source-sha256.json)，
模型与 worker 哈希在真实模型 JSON 中。

```text
ctest --test-dir build-kv-manager-cpu --output-on-failure
ctest --test-dir build-kv-manager-cuda --output-on-failure
python miniprefix/tools/verify_generation_history.py --model-dir MODEL_DIR --cpu-worker CPU_WORKER --cuda-worker CUDA_WORKER --output RESULT.json
python miniprefix/tools/verify_capacity_admission.py --model MODEL --cpu-worker CPU_WORKER --cuda-worker CUDA_WORKER --output RESULT.json
miniprefix_stress MODEL cpu 5 10
miniprefix_stress MODEL cuda 5 10
```
