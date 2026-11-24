# 批量工作页与前缀缓存保留

在共享准入、批量物理页准备之上增加可选执行策略。默认 `BatchPolicy::Full` 保持整批准备；
`Adaptive` 只在已有调度序列内缩短一次执行的连续前缀，不改变 Scheduler 顺序、token budget 或准入承诺。

## 决策与边界

`KVManager::plan_batch` 只读验证输入并模拟按 LRU 释放引用，报告扩页、初始 COW、驱逐后的 COW 工作页、
需要驱逐的引用和实际可释放页。不可行计划的回收字段不代表部分执行结果。它不分配载荷、不 touch 缓存、
不发布、不改变命中或驱逐统计。有效计划仍需由 `prepare_batch` 在当前状态重新验证和执行；不能跨状态变化复用。

Adaptive 对至少三行的候选组预估整批，再从大到小寻找不少于两行的连续前缀：

- 整批空间不可行时，选择最大的可行前缀。
- 整批可行时，只有减少至少两个驱逐引用，且每行驱逐引用下降至少 25%，才选择该前缀。
- 整批输入无效、不满足阈值或只有两行时仍走原路径；不隐藏后续行错误，也不等待未来释放空间。

这些阈值是保守的实验启发式，不是对未来命中或延迟的预测。行长、计算成本和缓存引用价值并不相同。
剩余行在同一 `Engine::step` 中继续按原顺序执行，重新按当前状态决定下一组；不会扣减探测阶段的 token budget。
若前缀选择器返回零或超出输入长度，Engine 按后端契约错误处理并清理资源。

未 resident 请求的恢复边界保持不变。每行提交后仍按原顺序发布、采样事件和处理终态；不为凑批提前发布。
拆批改变了驱逐与发布之间的交错时点，因此命中次数、驱逐数量和物理页 ID 可以变化，不承诺缓存轨迹逐项一致。
驱逐策略仍为原 LRU，已应用的驱逐在后续失败回滚后保留；工作页和请求映射按原事务规则回滚。

## 配置与观测

通过 CPU/CUDA `ModelRunner::set_batch_policy(BatchPolicy::Adaptive)` 在空闲事务边界开启，
默认 Full；不修改 worker 文本协议、Python 默认行为或现有配置文件。CPU 保留串行计算参考路径。

`batch_space_stats()` 是独立的累计计数器，clear 后不清零，测量窗口用前后差：

| 字段 | 含义 |
|---|---|
| prepare_calls / prepared_batches / rejected_batches | 准备调用、成功准备事务组、无效/容量拒绝；异常抛出不一定计入拒绝 |
| growth_pages / cow_pages / cow_work_pages | 已应用计划的扩页、初始 COW、驱逐后 COW 工作页；不是当前存活页，也不是 GPU 已完成复制次数 |
| cancelled_cow_pages | 驱逐缓存引用使尾页独占，取消的 COW 需求 |
| evicted_references / freed_pages | 批量准备实际驱逐的缓存引用、由这些驱逐释放的物理页；两者不能混用 |
| selection_calls / split_calls / rows_deferred | Adaptive 选择调用、拆批次数、留给后续执行的行数（不是延后到下一个 step） |
| immediate_victims_avoided | 相比当时可行整批计划少驱逐的引用数；后续行仍可能驱逐，**不能当作最终缓存净收益** |
| prepare_ns / selection_ns | `set_cost_timing(true)` 时的 host 准备/选择时间，准备不包含 GPU 执行；默认关闭 |

净效果结合全路径 `prefix_evictions`、`prefix_reused_tokens`、`forward_tokens`、实际批量回退及端到端表现判断。
计数器不包含单行 append 的准备成本；诊断租约或失败等路径可能使“少驱逐”不产生后续复用。

## 配对实验

capacity 原生基线新增可选参数（旧命令保持兼容）：

```text
capacity MODEL cpu|cuda conservative|shared off|cold|hot COUNT RATE SEED POOL_PAGES [CACHE_PAGES] [full|adaptive] [uniform|mixed] [SHARED_PREFIX_TOKENS] [timing-off|profile]
python miniprefix/benchmarks/capacity.py --binary BINARY --model MODEL --backend cuda --policies shared --batch-policies full adaptive --workloads mixed --shared-prefixes 0 16 --pools 32 64 --rates 40 --requests 48 --repeats 3 --output balance.jsonl.gz
```

uniform 保留 65/68/71/74 输入、各 8 输出；mixed 使用 17/33/65/89 输入和 4/8/16/24 输出。
`shared-prefixes` 控制不同模板之间的公共 token 长度；零公共前缀仍有同模板重复请求，不能称为完全无复用。
off/cold/hot、池大小、策略、负载、前缀长度和 profiling 分开汇总。逐请求输出数量参与 TPOT 和吞吐计算，
每个 token 与同后端无缓存 oracle 对照。端到端测量默认关闭成本计时；`--profile` 单独运行，不能与主结果混合。
`--before-binary` 仅支持旧默认负载与 Full；本轮 Full/Adaptive 使用同一个可执行文件。

实测与适用范围见 [验收报告](acceptance/batch-cache-balance.md)。
