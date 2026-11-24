# CUDA 跨请求 batched decode

首版合并已经被 Scheduler 选中的 decode 请求，不改变准入、decode 优先级、稳定请求
顺序、token budget、prefill chunk 或同轮不补位规则。CPU runner 默认返回无批量结果，
继续逐请求执行；单请求与 prefill 仍沿用原路径。

## 执行与 KV 生命周期

Engine 用计划副本取得本轮预算内的 decode 前缀，调用可选 append_decode_batch。
CUDA 将各请求的一个输入 token 组成矩阵行：各层投影、MLP、最终词表投影共用 GEMM；
RoPE、分页 KV 写入与 attention 根据每行独立的位置和设备页表处理不同上下文长度。
这是真实跨请求矩阵计算，不是把多个逐请求 append 包进一个循环。

KVManager 批量准备先验证整批预约，再只读模拟 LRU 回收能否覆盖所有新页和同时存活的 COW 工作页。
完整计划可行时才按当前 LRU 顺序执行驱逐，并在 GPU 启动前取得整批页面；不可行、重复请求或
非法输入返回空组且不驱逐。应用后的驱逐效果与普通 append 一样不随失败回滚恢复。
详细边界见 [批量物理页准备](batch_workpages.md)。

每行持有独立 Append 事务；所有设备读写完成前不能提交、回滚、发布、清理或另开普通
追加。活动事务计数保证一行结束不能提前解锁其余行。整批 logits 有限后提交各行，
Engine 再按原顺序发布历史、采样、完成和释放请求，然后才处理 prefill；后者仍可
在同一步命中刚发布的 decode 前缀。尚未写入 KV 的新采样 token 不发布。

非有限 logits 会在同步后回滚整组、标记设备页表 dirty，并回退逐请求执行，让原路径
处理成功的前序请求及失败请求。CUDA 执行/同步错误仍进入 Engine 的致命错误清理。
成功合批的 KV 提交先于逐行发布；若后续发布或清理本身失败，Engine 仍整体失败并清理，
已完成的批内计算会体现在 forward 计数中。

合批期间多行同时占用新页，可能比串行拥有更高的瞬时页数；在串行释放或驱逐原本可以
解除共享的场景，也可能多做 COW。这里保证 token/KV 正确性、发布顺序和容量回退，
不承诺物理页编号、峰值内存或失败时已执行工作量与串行逐位相同。FP32 GEMM 形状变化
也可能产生舍入差异，回归检查数值容限与 greedy 输出，不声称普遍位级相同。

## 资源与观测

Workspace 按最高行数复用激活、整批 logits、位置数组和页表指针数组。每请求设备
页表仍只上传增量/dirty 部分，KV 池和模型权重常驻设备。host logits 暂存仍按次分配；
prefill batching、混合 prefill/decode GEMM、CUDA Graph 与异步重叠不在本次范围。

协议新增只读 DECODE_STATS，返回成功批次数、成功合批行数、回退次数。原 STATS、
COST_STATS、STEP_STATS 的字段和 v3 握手保持不变；新 Python 方法为 decode_batch_stats。
合批计数只记录成功组，回退中的实际 COW/上传工作仍按原成本口径统计。CPU 默认计数为零。

## 验证和基线

- KV 组事务：重复/非法输入、无驱逐容量回退、部分事务结束后的全局锁、回滚与重试。
- CPU/CUDA：不同位置和长度、整页边界、共享尾页、源释放后的双分支、非有限结果的整批
  回滚、dirty 页表修复、两页小池回退、稳定 workspace 及资源归零。
- Engine：逐请求与批处理逐步对照事件、预算统计和发布历史，覆盖预算 1/2/6、
  decode 后同轮 prefill 命中、取消、EOS 和长度结束。
- 保留既有 CPU/CUDA 全量回归、设备 memcheck 和独立 Transformers 真实模型对照。

batched_decode.py 原样复用 baseline.py 的四组真实 prompt、独立 Transformers FP32
输出参考和 measure 指标计算；八请求改为同时到达，固定调度负载，避免 wall-clock
到达导致两版选择不同请求。分别设置并发上限 1/4 和 off/cold/hot，每组合五轮旧/新
进程交替，轮换模式及版本顺序。旧版为 e21b1da 的串行 CUDA worker。

每次先用相同模式 warmup，清缓存并在 hot 重新 prime，计时区间不包含加载、分词、
warmup、prime、统计查询。TTFT 从计划到达算到客户端事件（包含排队），TPOT 为
各请求首末输出间的平均间隔，吞吐为八请求全部输出 / 总窗口。每次校验输出、事件
顺序、forward/复用/淘汰计数、GPU 常驻及释放，并用 DECODE_STATS 证明实际发生合批。
并发 1 是控制组，不将 batch=4 相对 batch=1 的差异混同为同并发旧/新收益。

```text
python miniprefix/benchmarks/batched_decode.py --model-dir MODEL_DIR --before OLD_CUDA_WORKER --after NEW_CUDA_WORKER --repeats 5 --output RESULT.json
```

本机结果与完整原始样本见[验收报告](acceptance/batched-decode-review.md)。固定八请求结果
不代表持续到达下的饱和吞吐或 p99 SLA；hot prime 成本仍需后续请求摊销。
