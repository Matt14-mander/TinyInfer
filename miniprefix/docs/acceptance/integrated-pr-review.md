# PR #2–#8 整体评审与合并验证（2026-10-28）

## 评审范围

集成分支：`codex/integrated-review`，起点 `main@4e36dd7c73def6548aec261507e153c27b25a09d`。
按 #2 → #8 顺序合并原始 PR，保留各 PR 提交和独立 merge commit，未修改远程 main。
本报告和后续修复提交一起构成验证后的集成版本；原 PR 分支不自动包含这些冲突解决及修复。

| PR | 审查的 head | 内容 |
|---|---|---|
| #2 | fd51bcc0cbc255db3ddca3859aa8f6750c6ab94c | 清理异常与 Engine 失败处理 |
| #3 | 0db565a69b2699d38b456353823d3e4a0943b006 | runner 实际 context 上限 |
| #4 | 2bb9ae9537a8bf0f3eaf5fb60eee8149e1f1ff26 | Scheduler 提取 |
| #5 | 8b4135bc5bc3c8e9e53fe17f5e3f79f46ca6623b | 终态处理统一 |
| #6 | eec0d265bdb7c181298defbef972419e0e2190bc | CUDA 资源组件 |
| #7 | cd99970fd5f9ac4eb441dc3296d3e23fbba854e7 | worker 协议拆分 |
| #8 | 4ab68a8c30372f818c3976466af18f74089f7eea | KVManager 容量协调 |

## 发现与修复

1. **P1：#2 与 #7 的失败隔离不完整。** Engine 已进入失败状态后，FORWARD、REUSE、CACHE、RELEASE、CLEAR_PREFIX 仍直接调用同一个 runner；clear 失败时可能操作未安全回收的后端。新增只读 `Engine::failed()`，在协议分发处拒绝这些调用。统计查询可用，shutdown 可重试。新增清理成功/失败两条路径的协议回归，检查精确响应及 runner 操作次数。
2. **P1：#2 与 #5 的错误处理合并不能假设 clear 成功。** 终态路径必须先毒化 Engine、捕获二次清理异常，并由 runner 保留资源所有权。合并时使用 `RunnerManaged` 模式，避免强行逐请求 release；保留原始错误与清理错误，并对已完成请求去重。保留两项 PR 的全部失败处理测试。
3. **P1：#3 与 #5/#7 的测试 runner 接口不兼容。** 新增的纯虚 `context_limit()` 未在后两项 PR 的 mock 中实现，组合后无法编译。补齐 override，完整重建验证。
4. **P2：#4 与 #5 的部分 shutdown 后续调度缺少组合覆盖。** 新增回归：一个请求已取消、下一个 release 抛错，重试调度时只保留未完成请求，等待请求按既有预算进入，已完成请求不重复 append 或发终态。
5. **P2：CUDA sanitizer 验证存在假阳性空间。** 资源测试故意申请 SIZE_MAX 触发分配失败，在本机 sanitizer 下曾只出现工具汇总而没有应用完成标记。新增 `--sanitizer`，仅跳过该故意非法分配场景；普通 CTest 仍执行它。此次要求应用完成标记和工具零错误/零泄漏同时出现。

## 冲突解决

#2、#8 可直接合并。#3/#4 在 engine_prefill_test 发生 mock、测试函数及 main 调用的增量冲突；#4 另有 Engine 构造函数冲突。全部保留，并同时初始化 Scheduler、校验模型和 runner context 上限。

#5 在 engine.h/engine.cpp 发生语义冲突，保留 Scheduler 和 State.finished，移除旧局部 done/remaining 状态，使用统一终态处理且保留 #2 的清理异常保护。#5/#6/#7 在 tests/CMakeLists.txt 或 tests/README.md 有增量冲突，全部测试目标和说明均保留。

另审查 KVManager.reserve_pages 与 ExactPrefixCache.evict_one 的边界、被序列引用的缓存页、空间不足及回滚；CUDA runner 与资源组件的同步/释放、workspace 复用和统计；协议解析、响应编码及诊断访问；未发现需要额外修改的组合问题。此结论仅限本次审查与覆盖场景。

## 验证结果

- Windows / MSVC 14.29 / Release / Ninja；CPU 与 CUDA 均在合并后完整清理重建，再运行最终增量构建和 CTest。
- CPU：54/54；CUDA：62/62。CUDA 配置要求真实设备，RTX 4070 Laptop，CUDA 12.5.1，SM89。
- SmolLM2-135M-Instruct FP32（Llama 架构）：`verify_kv_manager_refactor.py` 对照 main 基线 worker 通过；覆盖分页跨页、前缀恢复、容量耗尽、交错所有权、缓存清理与 Engine 请求。逐后端 logits、token/event、资源统计按脚本断言一致。
- `verify_worker_protocol.py`：CPU、CUDA 各两组 transcript，returncode/stdout/stderr 与基线逐字节一致。新失败状态保护由故障注入单测覆盖，不属于旧正常协议对照。
- Compute Sanitizer memcheck：资源组件测试（`--require-device --sanitizer`）有应用完成标记；真实模型 worker 完成 FORWARD/CACHE/RELEASE/REUSE/跨页 append/CLEAR_PREFIX/STATS/QUIT，复用 6 tokens，最终 live sequences/KV blocks 为 0。两者均零错误、零泄漏。
- `git diff --check` 通过。

## 证据与复现

本目录 `integrated-cpu-tests.log`、`integrated-cuda-tests.log` 保存 CTest 输出；`integrated-differential.json` 保存模型/worker/source 哈希和逐项差分；`integrated-protocol-{cpu,cuda}.json` 保存原始响应哈希与长度；两个 integrated memcheck 日志保存工具结果。worker memcheck summary 省略长度超过 500 字符的 logits 行，完整原始日志位于本地 `build-kv-manager-cuda/integrated-worker-memcheck.log`。

CTest：`ctest --test-dir build-kv-manager-{cpu,cuda} --output-on-failure`。
差分脚本参数及 before/after 哈希可参照 JSON，before worker 来自 `build-cuda-resources-before`，after worker 来自 `build-kv-manager-cuda/miniprefix`。
memcheck 参数：`--tool memcheck --leak-check full --target-processes all --launch-timeout 30 --error-exitcode 99`；worker 参数为 `MODEL cuda 64 2 8 3 3 12 6`。

这次验证行为与资源正确性，不是新的吞吐或延迟性能基线。远程 CI 结果需另行查看。集成分支包含七项 PR 的全部改动；若继续单独合并原 PR，需要应用这里的冲突解决和最终修复后重新验证，不能仅依据单个 PR 显示可合并来判断整体兼容。
