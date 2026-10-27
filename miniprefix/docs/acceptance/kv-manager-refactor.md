# KVManager 提取与行为回归：2026-10-27

CPU/CUDA runner 已统一使用 KVManager，include 保留接口，公共实现在 src/kv 编译一次。
本次提取所有权与协调逻辑，保持模型权重格式、物理布局、精度、缓存策略、配置参数、
worker 协议版本 3 和全部 25 个 STATS 字段。没有新增 COW、抢占或跨请求矩阵 batching。

## 模块边界

| 接口 | 实现 | 责任 |
|---|---|---|
| include/miniprefix/kv_manager.h | src/kv/kv_manager.cpp | 序列 metadata、输入校验、前缀恢复/发布、追加事务、回收与 KvStats |
| include/miniprefix/paged_kv.h | src/kv/paged_kv.cpp | free list、页面引用、布局地址、申请/回滚和峰值 |
| include/miniprefix/prefix_cache.h | src/kv/prefix_cache.cpp | 完整因果 token key、页面 LRU、缓存引用与压力淘汰 |
| ModelRunner / RunnerStats | 原 CPU/CUDA runner 与 src/model/model.cpp | 模型计算、设备资源及统一统计适配 |

每个 runner 拥有独立 KVManager，模型/配置作用域保持隔离。manager 不包含 Tensor、CUDA、
设备指针或同步操作。CPU runner 保留 K/V 数组；CUDA runner 保留 device payload、设备页表、
cuBLAS/stream/workspace 和传输计数。CPU 按层提前解析 token offset，读取同一物理地址，
浮点归约顺序不变；CUDA append/attention kernel 数学和寻址代码不变。

新增 CMake 目标 `MiniPrefix::kv`，只依赖 C++17 `MiniPrefix::contracts`。
`MiniPrefix::engine` 传递链接它；单独使用页池/缓存方法的调用方须链接 kv 库，
contracts 继续只提供头文件数据契约，不携带实现库。既有页池/缓存单测已切换链接目标。
KvCacheConfig 的三个字段及默认值保持不变，配置拆分留待单独接口演进。

## 追加、失败与完成边界

`prepare_append(id,tokens)` 校验并保存输入快照、预留历史容量和页面，返回不可复制/移动的
Append 事务。只读 sequence 视图包含原 used 位置和新预留的 block table。

- backend 完成计算后 commit，才追加 token history、推进 used 和累计实际 forward。
- CPU 异常退出自动回滚；CUDA 异常路径先等待 stream，再显式 rollback，随后析构为空操作。
- rollback 只归还本次新增页；原已提交 token/页面保持有效，缓存淘汰仍只释放缓存引用。
- pending append 期间禁止重置/释放相应序列或 clear 全部状态，防止借用视图失效；
  事务必须在 manager 生命周期内结束。接口仍为单线程同步模型，不提供异步 completion/epoch。
- 设备前缀表上传失败时 reset_sequence 归还请求引用并保留空序列；设备表归 backend 管理。
- release、clear_prefix_cache、clear 语义区分，累计 forward/hit/eviction 统计不因此重置。

RunnerStats 保持原字段顺序、大小和标准布局，通过唯一的 update_kv_stats 适配器合并
公共字段，设备计数不被覆盖。单元测试固定 25 字段大小和关键 offset。

## 结果

| 验证 | 结果 | 证据 |
|---|---|---|
| 新目录 Release CUDA 全套 | 58/58，无跳过 | [日志](kv-manager-cuda-tests.log) |
| 新目录 Release CPU-only 全套 | 51/51，无跳过 | [日志](kv-manager-cpu-tests.log) |
| 重构前后 worker 精确差分 | 94 检查点；10 向量/491,520 个 FP32 值的协议还原字节一致 | [原始差分](kv-manager-differential.json) |
| 原真实模型完整/分块/decode | 54 logits 对照、9 次 CUDA 常驻检查通过 | [验收](kv-manager-real-model-verification.json) |
| 长 suffix 增量 prefill | 14 场景、24 logits 对照通过 | [验收](kv-manager-incremental-verification.json) |
| 精确前缀/共享/取消/LRU/失败 | 18 场景、12 logits 对照通过 | [验收](kv-manager-prefix-verification.json) |
| 实际 CUDA 模型 memcheck/full leak | 0 errors，0 bytes leaked / 0 allocations | [过滤日志](kv-manager-memcheck.log) |

独立 Transformers CPU FP32 eager full-prefix 对照合计 90 向量、4,423,680 个值，
最大绝对误差 `1.1372566223144531e-4`，argmax 全部一致；容差保持
`abs(actual-reference) <= 2e-3 + 2e-4*abs(reference)`。
精确差分还逐检查点比较全部 STATS 字段、生成/取消/错误事件、STEP_STATS、错误文字，
涵盖非连续页面、复用孔洞、共享前缀、清空时保留活跃引用、失败拒绝与等待请求补位。

新 KVManager 单测覆盖输入快照、commit、后端异常自动 rollback、pending 操作拒绝、
共享引用、reset、容量压力与累计统计。原新增接口只使用只读视图，runner 不直接改 used、
页表、历史或缓存。原日常 build-phase54-cuda worker 也已重新构建，CPU/CUDA smoke 的
logits 和统计与干净构建一致。

## 构建与复核

验证使用独立目录 `build-kv-manager-cuda` 与 `build-kv-manager-cpu`，MSVC 14.29、CUDA
12.5.1、Release、CPU native，GPU SM 89。公共头变更后采用干净构建，避免旧对象混用。
当前 Windows 工具链的默认 Debug compiler probe 遇到 PDB manager 不匹配，配置显式
使用 `-DCMAKE_TRY_COMPILE_CONFIGURATION=Release` 后通过；没有跳过 compiler 检查。

首次增量差分发现统计布局改变与旧对象混用的问题，最终保留原 RunnerStats 布局并补上
布局/映射回归，再完成以上干净构建。沙箱内 sanitizer 无法附加超时未计入通过；
随后主机侧同一 CUDA worker 完成 memcheck。日志省略三行完整 logits，保留控制响应、
最终 live KV=0 和 sanitizer 诊断，完整原始日志 SHA256 记录在过滤日志与汇总 JSON。

同版本的模型与参考依赖在本地准备后，重新执行数值验收：

```powershell
$py = 'build-miniprefix-env/Scripts/python.exe'
# CUDA bin 必须在 PATH；模型/依赖准备沿用 real_model.md。
& $py miniprefix/tools/verify_real_model.py --model-dir build-miniprefix-models/smollm2-135m-instruct --cpu-worker build-kv-manager-cuda/miniprefix/miniprefix_worker.exe --cuda-worker build-kv-manager-cuda/miniprefix/miniprefix_cuda_worker.exe --output benchmark-results/miniprefix/kv-manager-real-model-verification.json
& $py miniprefix/tools/verify_incremental_prefill.py --model-dir build-miniprefix-models/smollm2-135m-instruct --cpu-worker build-kv-manager-cuda/miniprefix/miniprefix_worker.exe --cuda-worker build-kv-manager-cuda/miniprefix/miniprefix_cuda_worker.exe --output benchmark-results/miniprefix/kv-manager-incremental-verification.json
& $py miniprefix/tools/verify_prefix_cache.py --model-dir build-miniprefix-models/smollm2-135m-instruct --cpu-worker build-kv-manager-cuda/miniprefix/miniprefix_worker.exe --cuda-worker build-kv-manager-cuda/miniprefix/miniprefix_cuda_worker.exe --runs 3 --output benchmark-results/miniprefix/kv-manager-prefix-verification.json
```

前后差分需要保留重构前的独立 worker，路径示例：

```powershell
& $py miniprefix/tools/verify_kv_manager_refactor.py --model-dir build-miniprefix-models/smollm2-135m-instruct --before-cpu-worker build-kv-manager-before/miniprefix_worker.exe --cpu-worker build-kv-manager-cuda/miniprefix/miniprefix_worker.exe --before-cuda-worker build-kv-manager-before/miniprefix_cuda_worker.exe --cuda-worker build-kv-manager-cuda/miniprefix/miniprefix_cuda_worker.exe --output benchmark-results/miniprefix/kv-manager-differential.json
```

[汇总与证据 SHA256](kv-manager-refactor-verification.json) 保留重构前代码参照、
模型/前后 worker 指纹、构建参数和全部验收文件哈希。原 binaries 保存在 Git 忽略的
工作区构建目录，不将二进制或模型权重加入仓库。

本次没有运行独立性能验收。prefix verifier 内含三组附带计时，仅用于该脚本的功能
回归，不能替代历史五轮基线或推导重构后的速度比；所有历史性能原始数据保持不变。
