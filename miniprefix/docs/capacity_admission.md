# KV 容量感知准入（首版，无抢占）

Engine 在每轮 FIFO 补位时调用 runner.reserve_request；CPU/CUDA runner 均委托统一 KVManager。
Scheduler 仍负责活跃槽位、decode 优先和 token 预算，Engine 协调容量结果与请求生命周期。

## 预约与进展保证

每个请求按 `ceil((prompt_tokens + max_new_tokens - 1) / block_tokens)` 预约最坏情况页数。
最后一个输出 token 不再输入模型，因此不占 KV；现有 context 校验规则保持不变。
预约是 host metadata，不提前申请物理页、不驱逐缓存，也不改变 STATS 的物理页统计。

容量承诺 = 所有预约页数 + 未预约序列当前持有的页数。共享页按每个请求分别计入，
不以可能命中的前缀抵扣预约；仅缓存持有的页可驱逐，不占预约额度。
这种保守算法可能降低并发，也可能拒绝本来能依靠共享或提前 EOS 完成的请求，但不依赖
未来命中或提前结束来保证活跃请求的完整 decode 空间。

- Ready：记录预约并进入 active，序列创建与前缀恢复仍延迟到获得计算预算时。
- Wait 且存在 active：队首等待，停止本轮补位，让已有请求继续计算；不跳过队首。
- Impossible：单请求最坏情况超过池总容量，只发该请求的 Error 终态，继续检查后续请求。
- Wait 且没有 active：没有引擎内请求能释放空间（例如诊断序列占满池），立即发该请求的
  Error 终态；不依赖外部释放、不无限返回空步骤。外部释放后新请求仍可提交。

预约不会因 prefill 结束而释放；长度结束、EOS、取消、请求失败及 shutdown 都释放。
尚未执行 create 的 active 请求也持有预约，取消它同样归还额度。完成后的补位仍发生在下一轮。
只要调用方继续 step，正 token budget 和有限 max_new_tokens 保证至少一个活跃请求前进；
不存在多个活跃请求互相等待扩页的情况。没有抢占、重算或新的公平性策略。

## 所有权与错误

KVManager 在未预约的诊断 append/reuse 上同样保护已有预约，防止诊断命令抢走未来页。
正常物理页申请仍使用现有缓存驱逐和追加事务。拒绝的 prefix reuse 会撤销新序列引用；
缓存查找的命中/LRU 统计可保留，但不会留下序列页。清理后实际页、页表和逻辑预约均可复用。

新增 KvCapacityError，仅表示可在请求范围内处理的容量不足。Engine 捕获该类型，安全
release 当前序列并发一次 Error，其他请求继续执行。CUDA 分配失败、同步失败、非法 logits
等仍属于后端错误，沿用 Engine 失败隔离和清理异常保护；如果容量错误后的 release 失败，
同样进入后端失败处理，不掩盖资源安全问题。

自定义有界 ModelRunner 必须实现 reserve_request/release_reservation；默认实现只适合
无容量约束的参考 runner。API 为单线程同步约定。release_reservation 为 noexcept；当序列仍
存在时，取消预约会将它归入未预约序列的实际页计数，不能凭空释放物理页。

worker 协议仍是 v3，终态 Error 使用既有 reason=3；协议不传错误文本，C++ GenerationEvent
保留具体原因。25 项 STATS 和 3 项 STEP_STATS 不变；预约计数暂不暴露为新协议字段。

## 回归

- engine_prefill_test：超大队首、按容量串行、跨页 decode、预约未执行时取消、容量错误隔离、
  外部占页导致的无进展拒绝，以及随后继续提交。
- kv_manager_test：逻辑预约无物理分配、Wait/Impossible、阻止诊断偷占、缓存驱逐、释放与 clear。
- verify_capacity_admission.py：真实 SmolLM2 CPU/CUDA、3 页池，对比 32 页参考的输出；
  覆盖前缀共享、取消、提前 EOS、外部占页、诊断保护、最后资源归零。循环有 100 步上界。
- verify_real_model.py：保留独立 Transformers 数值检查；更新容量场景，验证拒绝后仍可生成。

运行真实模型专项测试：

```text
python miniprefix/tools/verify_capacity_admission.py --model MODEL.mpf --cpu-worker CPU_WORKER --cuda-worker CUDA_WORKER --output capacity-admission.json
```

这次修改不构成性能收益结论；容量紧张时允许并发与事件交错改变，但成功请求输出必须一致。

## 本次验收（2026-10-28）

基于 main@92281cb 的未提交工作区；CPU/CUDA 完整重建后，最终增量构建与回归均通过。
CPU CTest 54/54，CUDA CTest 62/62（要求实际设备）。真实模型容量专项在两个后端均通过，
3 页池与大池参考生成相同 token `[282,260,905,30]`，最终 live sequences/KV blocks 均为零。
独立 Transformers FP32/eager 回归通过：54 次 logits 对照、2,654,208 个值、9 次 CUDA 常驻检查。
`git diff --check` 通过。本次未重新测量性能，也未运行新的 Compute Sanitizer 验收。

证据：[CPU](acceptance/capacity-cpu-tests.log)、[CUDA](acceptance/capacity-cuda-tests.log)、
[真实容量专项](acceptance/capacity-admission.json)、[独立数值对照](acceptance/capacity-real-model.json)。
JSON 保留模型和 worker SHA256，历史验收文件未覆盖。

## 共享容量基础（第一步，尚未启用共享抵扣）

`KVManager::capacity_ledger()` 提供空闲事务边界上的只读账本。字段含义：

- `total_blocks / physical_blocks / free_blocks`：池总页数、实际存活页、空闲页。
- `pinned_blocks`：序列及独立前缀租约持有的物理页并集，按页 ID 去重。
- `cache_only_blocks`：仅缓存持有、可驱逐的物理页。
- `shared_reference_savings`：序列和独立租约的页引用数减去其物理页并集大小；不包括缓存引用。
- `logical_reserved_blocks`：现有请求预约页数之和，继续使用保守算法。
- `future_private_blocks`：各预约减去对应序列已拥有的逻辑页数，取非负值后相加。
  这是尚未分配的扩页额度，不包含部分尾页的 COW 替换，也不是新的准入承诺公式。
- `prefix_lease_blocks`：独立前缀租约持有的物理页并集；它是 pinned_blocks 的子集。
- `cow_tail_candidates`：当前尾页非整页且引用数大于一的序列/独立租约数量。
  它是潜在复制需求，不是已经预约的 COW 峰值空间。

账本查询在 Append 事务进行中会拒绝；此时新旧页并存由现有事务及物理池统计负责。
下一步接入准入时，必须明确建立 COW 峰值额度规则，不能直接把以上字段相加作为可用容量。

`probe_prefix(prompt)` 仅返回可复用 token 数，不保留页 ID，不改变 LRU、命中统计或引用。
结果只代表查询时刻，不能用于未经锁定的容量抵扣。仍保留至少一个 prompt token 计算 logits。
`acquire_prefix_lease(id, prompt)` 重新匹配并锁定页引用，由 KVManager 持有，缓存清空或驱逐
不会撤销租约。它不改变 LRU/命中统计，不创建计算序列，不授予共享准入额度。
租约 ID 是独立命名空间；重复获取报错，重复释放为空操作；空命中也登记 ID，需正常释放。
获取失败撤销登记；当前按完整租约页数保守计费，保护已有请求预约和未预约诊断序列。
`release_prefix_lease(id)` 显式释放，`clear()` 和管理器析构兜底释放全部租约。
事务期间禁止获取/释放租约，避免 GPU 使用中的页引用发生变化。

ExactPrefixCache 的 `probe/acquire` 共享既有 trie 匹配算法，后者把引用交给调用方提供的
空 PagedSequence。底层调用者必须在同一池上 release，且池须比租约存活更久；业务层应使用
KVManager 的托管接口。现有 reuse 的 LRU、命中统计及生成路径保持原样，worker 协议不变。

新增回归覆盖只读查询/LRU、租约保活、重复释放、原共享者退出、预约保护、部分页 COW
复制失败回滚，以及 clear 资源归零。实际共享感知 Ready/Wait 决策将在第二步接入。
