# KV 容量感知准入（共享感知，无抢占）

当前 Engine 使用下文“共享感知准入”策略。首版记录及第一步基础接口说明保留用于解释演进；
两参数 reserve_request 继续提供原保守策略，三参数接口用于共享感知准入。

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

## 共享感知准入（第二步）

Engine 传递 `reserve_request(id, max_kv_tokens, prompt)`。CPU/CUDA 委托同一 KVManager；
自定义 runner 的默认三参数实现转发旧两参数接口，既有有界 fake runner 保持兼容。
`max_kv_tokens = prompt.size() + max_new_tokens - 1`，单请求最大页数 P 不变。

每个共享感知预约维护待恢复前缀，Ready 前已取得完整页引用。仅完整页可抵扣：

```
容量承诺 = 所有共享感知请求完整页的物理 ID 并集大小
         + sum(每请求最大页数 P - 该请求持有的完整页数)
         + 旧式预约页数 + 未预约序列页数 + 独立诊断前缀租约页数
```

未执行请求使用待恢复前缀；已执行请求使用其已提交序列。原共享者完成后，其余请求的引用
继续保住物理页，账本不依赖某个请求承担“共享页所有者”。缓存独占页可驱逐，不计入承诺。
如两个请求各最多 3 页，锁定同一个完整前缀页，则总承诺为 1+2+2=5 页，而不是 6 页。
冷启动请求执行并发布完整页后，也可为随后到达的请求提供共享信用。

候选查找不影响 LRU/命中统计；获取页引用、登记预约和容量校验作为同步事务完成。
Wait/异常撤销新登记及引用；Impossible 仍表示单请求最大页数超过池容量。
FIFO、无抢占、无活跃请求时的无进展拒绝策略不变；并发数及事件交错可以因准入改善而改变。

### 恢复与生命周期

实际获得计算预算时重新查询缓存。更长或等长的新候选只有在替换后容量承诺仍不超限时才采用；
否则恢复 Ready 时锁定的完整页。等 token 不代表等物理页，替换必须重新核算物理 ID 并集。
旧租约在候选校验结束前保持有效；成功后引用转交序列，并按实际复用 token 数记录命中。
LRU 仅触碰缓存中仍存在且物理页匹配的路径。即便缓存已被清空，也能恢复已锁定的前缀。

取消未执行请求直接释放待恢复租约和预约；正常完成、EOS、请求范围容量错误、shutdown、
后端失败的 clear 路径释放全部相关引用。release_reservation 可重复调用。
reset_sequence 在共享感知预约存在时，将已提交完整页转回待恢复租约，保护其余请求依赖的
容量承诺；它不立即释放这些完整页，最终由 release/release_reservation/clear 释放。
未经恢复不能直接 append 持有待恢复前缀的请求；不能 fork 到共享感知预约目标。
旧接口的 reset/fork 行为保持不变。CUDA 恢复页表上传失败时仍使用 reset，随后沿用 Engine
后端失败清理流程；不增加异步设备访问。

### 部分页与 COW 预算

准入阶段只锁定完整页，不锁定或抵扣部分尾页。恢复时仍支持最新的非整页前缀。
每请求的 `P - full_pages` 覆盖其私有尾页和未来扩页。部分尾页若还被另一个活跃请求持有，
那个请求也保留自己的尾页额度；若源是别的请求的完整页，则源已在完整页并集中计费。
仅缓存持有的旧页可被驱逐，最后一个引用变成独占时可原地写，不必 COW。
因此顺序 append 保留旧页用于回滚的同时，仍能在已有保守额度内申请目标页。
批量准备现在先模拟实际空闲页与 LRU 可回收引用，确认整批 COW 峰值可满足后才执行驱逐；
完整计划仍不能满足时返回串行路径，不让请求互相等待。参见 [批量物理页准备](batch_workpages.md)。
本版本不把瞬时节省的部分尾页转借给新请求，也不依赖提前 EOS 才能完成生成。

`capacity_ledger().committed_blocks` 暴露上述实际准入承诺；原 logical_reserved_blocks 继续显示
未抵扣的 P 之和。pending 前缀纳入 pinned_blocks/prefix_lease_blocks，future_private_blocks
扣除已锁定的页。账本仍仅支持 Append 事务边界查询，不改变 worker 的 25 字段 STATS 协议。

### 回归范围

KVManager 覆盖 5 页池并发两个最大 3 页请求、部分尾页 COW/回滚、Wait 无副作用、原共享者
退出、缓存清空、恢复失败重置、较新前缀、等 token 异物理页替换回退、取消前缀租约和无共享拒绝。
Engine 覆盖已准入未执行时取消、后续补位、请求范围失败后继续生成与清理。
verify_capacity_admission.py 新增真实 CPU/CUDA 的冷/热缓存比较：5/7 页池，整页/非整页复用，
热缓存首轮两个请求同时 resident，冷缓存首轮仅一个；全部输出对照关闭缓存的大池参考。
这些是容量与正确性验收，不是 TTFT/吞吐收益结论。

本步验收（2026-11-23）：基于基础提交 `8745aa6` 的工作区实现，CPU CTest 56/56、CUDA
CTest 65/65 通过；真实 SmolLM2-135M 的原容量专项及新增 8 个 CPU/CUDA 冷热案例全部通过。
5/7 页池热缓存首轮 resident 数为 2，冷缓存为 1，输出与关闭缓存的大池参考一致，清理后
live_sequences/kv_live_blocks 均归零。模型与最终 worker SHA256 及案例结果见
[共享准入验收](acceptance/shared-capacity-admission.json)。尚未测量持续到达的性能收益。

## 第三步：压力与收益验收

已完成保守/共享准入在 off/cold/hot 下的持续到达对照，以及固定缓存预算的物理池诊断。
CPU/CUDA 各至少五分钟生命周期压力通过。容量收益明确，但小池热缓存中的批量回退限制了
吞吐收益；下一项执行优化应关注批量 COW 的临时物理页空间，不能仅靠增加活跃请求数。
完整边界与复现见 [测量协议](shared_capacity_measurement.md)，结果见
[共享容量验收报告](acceptance/shared-capacity-measurement.md)。
