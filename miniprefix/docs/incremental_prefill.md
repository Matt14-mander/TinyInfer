# 命中后的增量 prefill

2026-10-26：在已有“恢复命中页面并跳过 prefix”的路径上，补齐长 suffix 分块执行的
范围校验、每轮 token 计数和专门验收。未新增 KV 复制、部分页共享或 COW。

## 执行路径

1. 请求首次获得计算预算时创建序列，调用一次 `reuse_prefix`，恢复完整页面及已计算位置。
2. Engine 验证返回范围小于 prompt 长度，且非零命中在完整页面边界结束。
   首版没有 cached logits，因此至少保留一个 prompt token 计算。无效范围在 append 前失败清理。
3. `prefilled` 从命中位置开始。每次只输入
   `prompt[prefilled : prefilled + min(remaining_suffix, prefill_chunk, remaining_budget)]`。
4. 后续轮次只推进 suffix，不重新查询缓存或重置位置。runner 使用其恢复/推进的 `used`
   作为 RoPE 位置与 causal attention 范围，append 直接访问共享页及私有尾页。
5. suffix 全部完成、首 logits 形状及有限值检查通过后，发布完整 prompt 页面并采样首 token。
   中间 prefill chunk 不交付生成事件。随后每轮输入上一个生成 token，进入 decode。

复用 token 不占 forward budget；已有 decode 请求先获得预算，prefill 使用剩余预算。
一轮内 runner 仍逐请求调用，没有跨请求张量 batching。取消部分 prefill 的请求会释放
请求引用；未完成的 suffix 不发布到缓存，缓存已有引用继续有效。

## 可检查计数

C++ `Engine::last_step_stats()` 和 Python `NativeEngine.last_step_stats()` 返回最近
一轮的成功计算量；每次 STEP 开始重置，idle STEP 为零，不清空累计 runner 计数。

| 字段 | 口径 |
|---|---|
| prefix_reused_tokens | 本轮新请求一次性恢复的 prompt token；不属于计算预算 |
| prefill_tokens | 本轮实际 append 的剩余 prompt token |
| decode_tokens | 本轮实际 append 的已生成 token |

应满足 `prefill_tokens + decode_tokens <= token_budget`；请求完成后的累计实际输入量为
`prompt_length - reused_tokens + output_length - 1`。最后一个输出 token 不再输入模型。
CUDA 输入 H2D 应严格为实际输入量×4 字节，KV/权重 H2D 不因命中增加。

worker 增加只读 `STEP_STATS` 命令，返回上述三个整数；原 STEP 事件和 STATS 格式不变，
协议仍为版本 3。检查该计数会增加一次 IPC，因此验证过程不用于报告性能。

```python
engine.submit("request", prompt, max_new_tokens=3)
while engine.pending:
    events = engine.step()
    counts = engine.last_step_stats()
```

## 重现与验收

`miniprefix_engine_prefill_tests` 使用记录型 runner 验证传入的每个 token 及绝对位置、
一次性恢复、decode 优先、chunk/budget 约束、首事件边界、取消，以及非法命中和 NaN logits
不能发布缓存。真实模型使用独立 Transformers CPU FP32 eager full-prefix 对照：

```powershell
$py = 'build-miniprefix-env/Scripts/python.exe'
& $py miniprefix/tools/verify_incremental_prefill.py --model-dir build-miniprefix-models/smollm2-135m-instruct --cpu-worker build-phase54-cuda/miniprefix/miniprefix_worker.exe --cuda-worker build-phase54-cuda/miniprefix/miniprefix_cuda_worker.exe --output benchmark-results/miniprefix/incremental-prefill-verification.json
```

默认页长 16，pool 64/cache 8，budget 5、chunk 7。113-token prompt 命中 48 token 后，
只 prefill 剩余 65 token，单请求需 13 轮才交付首 token；三个输出共计算 67 token，
关闭缓存的同请求为 115 token。页对齐的 48-token prompt 复用 32、重算最后 16 token；
49-token 完整命中用例复用 48、只重算一个 prompt token。

进一步检查长 suffix 与 decode 混合、增量途中取消、不同 chunk 切分下的逐步 logits。
[冻结的真实模型验收](acceptance/incremental-prefill-smollm2-fp32.md) 保留原始逐轮事件、
Engine/runner 计数和源码/模型/worker 指纹。本次不重测端到端性能、不改写历史前缀基线。
