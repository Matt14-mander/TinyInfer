# 与 serving_scheduler 的增量迁移

现有 examples/serving_scheduler 是可运行参考，MiniPrefix 已有真实模型 CPU/CUDA 闭环与
独立性能基线；真实模型已实现 CPU/CUDA 分页池与直接寻址，prefix cache 和抢占重算
仍待按下表迁移。新页池 metadata 见 include/miniprefix/paged_kv.h。
迁移必须产生单一的通用实现，不在两个目录长期维护同一算法的副本。

| 现有来源 | 目标位置 | 迁移方式 |
|---|---|---|
| scheduler.py / dynamic_scheduler.py | src/scheduler + engine | 保留 Python 轨迹/策略参考；C++ 计划接口就绪后逐项迁入 |
| tinyinfer_worker.cpp | src/model / engine / apps | 分离模型、packed forward、采样、状态与 IPC；旧协议留在适配层 |
| native/paged_kv_cache.h | src/kv + src/prefix + backend/cpu | 先参数化层/head/dtype，再分离 block 所有权、索引和 attention |
| native/cuda_paged_attention.h/.cu | src/backend/cuda | 保留旧 host 镜像路径作对照；新常驻路径通过 TinyInfer 设备抽象实现 |
| tinyinfer_backend.py | bindings/python 或迁移适配器 | 先兼容旧测试，等 Engine 稳定后消除热循环文本 IPC |
| realtime.py / metrics.py / real_benchmark.py | benchmarks / apps | 复用指标定义和数据格式，区分旧模拟/混合路径与新真实模型 |
| test_* 与原生 paged_attention_test.cpp | tests + 原示例回归 | 复用场景和 oracle，增加独立框架数值/真实设备检查 |

每次提取先补齐行为边界与回归，再修改旧入口引用新模块；只有同时保留两条路径用于
明确的差分验证时才暂留旧实现，并在文档注明退出条件。当前真实模型使用独占分页 KV，
没有共享/COW；旧示例保留其教学算法与回归，历史 dense 基线作为冻结数据保留。

原 examples 的 Phase/Stage 验收保持历史含义，新结果写入 MiniPrefix 专属输出目录。
原 speculative_decoding 也是独立模拟参考，未经真实 runner/KV 接入不能作为 MiniPrefix
已支持推测解码的证据。核心静态 Graph/Prepared 路径不因建立本目录而更改完成状态。
