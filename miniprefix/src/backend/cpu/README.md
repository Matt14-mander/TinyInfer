# CPU Backend

状态：llama_runner.cpp 已实现配置驱动的多层/GQA/RoPE/RMSNorm/SwiGLU CPU forward，
复用 TinyInfer Tensor、持久 packed RHS 和算子；通过真实权重的独立框架 logits/Greedy 对照。
KV 已使用共享物理池和序列页表，attention 直接访问非连续页面，与 CUDA 共用所有权管理。
序列/页池/缓存/追加事务和公共统计已提取为 KVManager；本文件保留 CPU payload 与模型计算。

负责参考 forward、attention 和必要 KV payload 操作。保持 FP32 参考与清楚的因果边界，
验证完整前缀、增量 KV、分块、共享/COW、重算和打包计算的一致性。

不要重新实现已有通用 Tensor/MatMul；需要的通用算子扩展进入 TinyInfer core。
CPU 参考不作为 CUDA 执行失败后的隐式重跑路径。
