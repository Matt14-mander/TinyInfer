# CUDA Backend

状态：llama_runner.cu 已实现 FP32 全模型 GPU forward、分页 KV 与权重常驻，append 和
attention kernel 直接使用 block table 寻址。只上传新增页号，不上传 KV payload。
完整页面的精确前缀复用已接入，只复制页号，不复制 KV。当前使用私有 RAII/cuBLAS/stream，
尚未统一到 core 设备 API，尚无部分页 COW。
公共序列/页池/前缀管理交由 KVManager；本后端保留 GPU payload、页表分配/上传和完成同步。
追加异常必须先同步，再 rollback，防止在途 kernel 使用已回收页面。
资源组件现在独立于模型执行，声明在私有 `resources.h`，实现在 `resources.cpp`：

- `Allocation`：唯一所有权、按需扩容、不缩容；扩容不保留旧内容。成功扩容才计入
  device_allocations，申请失败保留旧指针和容量；调用方负责在释放/扩容前结束设备访问。
- `StreamBlas`：管理 nonblocking stream 与绑定的 cuBLAS handle，保持 PEDANTIC math
  和禁止 atomics 的设置。显式 initialize 在 runner 参数校验后执行；初始化失败回收
  已有资源。checked synchronize 与析构/reset 的 best-effort 清理分开。
- `Workspace`：管理 11 个模型临时 buffer，保留原有分配顺序和高水位容量；bytes()
  只统计 workspace，不包括权重、KV 和 block table。

runner 保留 kernel、权重上传、KV/页表、完成同步与事务回滚的编排。析构先同步并释放
页表和参数，再关闭 execution resources；workspace/KV allocation 随成员析构释放。
这些组件仅在 CUDA target 内编译，不成为 TinyInfer 通用设备 API 或公共 SDK 接口。
`miniprefix_cuda_resources_tests` 在真实设备上验证复用/增长、分配失败、handle 配置、
SGEMM、重复初始化/清理和异常退出；无 GPU 时仅允许非 require-device 配置 skip。

2026-10-28 重构验证：CPU CTest 51/51、CUDA CTest 59/59 通过；复用
`tools/verify_kv_manager_refactor.py` 对重构前后 worker 做严格差分，94 个检查点、
491,520 个 FP32 logits、事件/STEP_STATS 和全部 25 个 STATS 字段一致，原始证据见
[差分结果](../../../docs/acceptance/cuda-resources-differential.json)。真实 worker 的
prefill、前缀共享、decode、release/退出经 Compute Sanitizer memcheck 验证为 0 errors、
0 leaked bytes，见 [过滤日志](../../../docs/acceptance/cuda-resources-worker-memcheck.log)。
这是行为/资源验证，不重新声称端到端性能收益。
真实模型与基线边界见 docs/real_model.md、docs/baseline.md。

目标是完整 GPU 模型执行、GPU 常驻权重/KV、packed/ragged attention、追加、COW 与
table 的增量更新。设备资源与 completion 优先复用 TinyInfer::cuda；模型专用 kernel
放在此处，通用设备能力扩展进入 TinyInfer backend。

首批交付：与 ModelRunner 一致的设备执行边界和真实设备数值对照。M3 验收必须证明
warm step 不再上传完整 KV pool。异步重叠/CUDA Graph 等在稳定基线后增加。
构建服从 TINYINFER_ENABLE_CUDA；CPU-only 配置不读取 CUDA header 或要求 SDK。
