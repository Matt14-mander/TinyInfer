# CPU Backend

状态：规划目录，复用 TinyInfer eager CPU 算子与原微型模型 oracle。

负责参考 forward、attention 和必要 KV payload 操作。保持 FP32 参考与清楚的因果边界，
验证完整前缀、增量 KV、分块、共享/COW、重算和打包计算的一致性。

不要重新实现已有通用 Tensor/MatMul；需要的通用算子扩展进入 TinyInfer core。
CPU 参考不作为 CUDA 执行失败后的隐式重跑路径。
