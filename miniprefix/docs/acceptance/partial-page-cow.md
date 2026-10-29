# 部分页 COW 验收（2026-10-29）

基于 `a13fd22`，实现边界见 [设计说明](../partial_page_cow.md)。本次交付 runner/KVManager
fork 与部分尾页 COW，不包含非整页前缀缓存、Engine fork 命令或吞吐提升结论。

| 验证 | 结果 | 证据 |
|---|---|---|
| CPU 干净构建及完整 CTest | 56/56 通过 | [日志](cow-cpu-tests.log) |
| CUDA 干净构建及完整 CTest | 65/65 通过，要求真实设备 | [日志](cow-cuda-tests.log) |
| 最终新增边界回归 | CPU 构建 1/1、CUDA 构建 2/2 COW 测试通过 | `miniprefix_cow_tests` |
| 最终 CUDA COW memcheck/full leak | 应用完成标记、0 errors、0 bytes leaked | [日志](cow-memcheck.log) |
| SmolLM2 CPU/CUDA 多轮回归 | 44 场景，196,608 logits；独立 Transformers oracle | [JSON](cow-generation-history.json) |
| SmolLM2 CPU/CUDA 容量准入回归 | 两后端通过，最终请求/KV/页表资源归零 | [JSON](cow-capacity.json) |

CPU/CUDA COW 的数值验证使用确定性两层 GQA 小模型，对照 CPU 独立历史重算，覆盖整页、
部分页和跨页追加；全部 logits 绝对误差 < 2e-5 且 argmax 一致。失败注入分别发生在第一层
真实 K/V 复制之后，以及模型计算产生非有限 logits 时。后者覆盖源释放后的重试与 CUDA 页表修复。
真实 SmolLM2 测试验证现有完整页缓存/Engine 路径保持行为；其 worker 尚未暴露 fork，
因此不能将这组结果描述成真实模型非整页缓存收益验证。

工具链：MSVC 14.29、CUDA 12.5.1、RTX 4070 Laptop；CPU/CUDA Release 均显式启用测试断言。
完整构建后只增加了测试阶段标记及目标预约不足/跨页失败用例，最终针对 COW 重新构建与运行，
并重新执行 memcheck。源文件、最终测试二进制哈希见 [SHA256](cow-source-sha256.json)，
真实模型与 worker 哈希位于对应 JSON。

```text
ctest --test-dir build-kv-manager-cpu --output-on-failure
ctest --test-dir build-kv-manager-cuda --output-on-failure
ctest --test-dir build-kv-manager-cuda -R miniprefix_cow --output-on-failure
compute-sanitizer --tool memcheck --leak-check full --target-processes all --launch-timeout 30 --error-exitcode 99 build-kv-manager-cuda/tests/miniprefix_cow_tests.exe --cuda --require-device
```
