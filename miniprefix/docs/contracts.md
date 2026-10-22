# 接口契约与不变量

本文为后续实现约束。除 include/miniprefix/request.h 外，以下名称是目标接口描述，
尚未提供可调用 Engine/ModelRunner/KVManager API。

## 公共边界

Engine 目标操作：submit(tokenized_request)、step()、poll_events()、cancel(id)、
drain()/shutdown()、stats()。文本接口经 tokenizer adapter 转换后调用相同 token API。
每个 engine 生命周期内 request ID 唯一；非法输入在进入执行前失败。首版明确支持
Greedy、max_new_tokens、EOS；不暴露暂未支持的采样参数。

GenerationEvent 可包含 token、终态或二者；取消/错误可无 token。完成和失败必须区分。
Tokenizer、模型加载、warmup 不混入 warm serving 的请求延迟，但单独保留冷启动测量。

## Scheduler → Engine / ModelRunner

SchedulePlan 描述 request/sequence handle、phase（prefill/decode/recompute）、
绝对位置、输入 token 范围和预计 KV 增量，不携带裸设备地址。packed batch 保留稳定
的 request 到输出映射；token budget 覆盖正常 prefill、decode 与重算，缓存跳过行
另外计数。各 query 的可见长度严格等于对应绝对位置+1。

ModelRunner 的目标是按计划产出每个需要采样位置的 logits，而不是自行改写输出队列。
未完成 prefill、recompute、padding 都不能被当作新输出 token。
prefill 可计算全部必需 attention 行，但只为需要的生成位置保留/计算 head 输出。

## 资源事务与前缀

KVManager 目标操作：dry-run/prepare、commit、rollback、release_sequence、
retain/release_cached_prefix、stats。申请结果包含 sequence handle 和资源 lease；
异步场景必须使用代次/epoch 防止旧 completion 引用已复用 slot。

- prepare 失败不能留下部分映射、引用或可见 token。
- 失败/取消先等待在途设备访问终止，再释放引用；禁止对同一 batch 静默 CPU 重跑。
- 共享块不可直接写；尾块追加通过 COW，完整块可共享。
- prefix key 包含完整因果前缀身份和 cache namespace：模型/权重 revision、位置编码
  配置、KV 布局/dtype，以及未来 adapter 等影响计算的上下文。
- 索引命中只代表可复用长度/lease，不能虚构完成 prefill 或重复发 token。
- prefix cache 生命周期独立于单次 generate；flush_cache 与 reset_stats 明确分离。
- 整轮 CLEAR、单请求释放、清空前缀、清零统计必须是不同操作。

## 错误与统计

输入/配置错误、容量等待、正常长度/EOS 完成、取消、后端执行错误分别记录。
首版遇到设备执行错误可终止 engine 当前流，禁止输出伪成功；复杂故障隔离后置。

统一计数：scheduled_input = executed_input + reused_input，分别保留 prefill、decode、
recompute；交付吞吐只统计提交给用户的 token。TTFT 从提交到首 token，TPOT/ITL
从交付时间戳计算，排队、重算、传输与同步不能从端到端窗口中偷偷剔除。
slot 占用与 GPU utilization 不等同，cache hit rate 与实际跳过计算量分开报告。
