# CUDA Backend

状态：规划目录；原示例已有可运行的混合 CPU/CUDA 分页 attention 基线。

目标是完整 GPU 模型执行、GPU 常驻权重/KV、packed/ragged attention、追加、COW 与
table 的增量更新。设备资源与 completion 优先复用 TinyInfer::cuda；模型专用 kernel
放在此处，通用设备能力扩展进入 TinyInfer backend。

首批交付：与 ModelRunner 一致的设备执行边界和真实设备数值对照。M3 验收必须证明
warm step 不再上传完整 KV pool。异步重叠/CUDA Graph 等在稳定基线后增加。
构建服从 TINYINFER_ENABLE_CUDA；CPU-only 配置不读取 CUDA header 或要求 SDK。
