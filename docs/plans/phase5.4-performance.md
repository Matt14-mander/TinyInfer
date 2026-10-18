# Phase 5.4 — ROG CUDA 性能验证协议

冻结日期：2026-10-18，开始测量前制定。只测现有实现，本轮不改算子、等待或复用策略。

## 固定范围与门槛

ROG RTX 4070 Laptop GPU，同机 Release native Prepared CPU Auto 对照。
沿用跨后端既有模型定义：两个 ONNX fixture、control M=64/K=N=512、
target M=128/K=N=1024 的 FP32 Gemm/vector bias/ReLU（alpha=beta=1，无转置），
fallback 为 64x512 GELU，显式 WholeModelCpu。数据生成与 Metal 基准相同。

两轮、每轮三个顺序独立进程，每种 warm mode 5 次预热/31 次采样，
交替 backend 和进程/模型起始顺序，不剔除样本。nearest-rank ceil(n*p)。
各轮分别取 process 比值的中位数：target E2E p50 CPU/GPU >=1.2，
target E2E p95 GPU/CPU <=1.05；fallback E2E p50 wrapper/CPU <=1.05。
若失败如实报告，不事后换 target、改阈值或筛进程。这是独立 ROG 门槛，不改变 Mac 5.3 状态。

## 计时与正确性

记录构建/import、prepare、context create、first inference、warm host-to-host E2E、
独立 bind / completed run / host output、resident-only。公开接口同步完成，
不以提交耗时代替 GPU 完成；completed run 是 host wall time，不是纯 kernel timestamp。
E2E 包含 lvalue 输入快照、H2D、推理、D2H、独立输出及其析构。
诊断阶段不能相加代替 E2E。准备/首跑是 process/case-first，不声称清空驱动缓存。
每个 E2E 输出在计时外与同机 CPU 比较，阈值 1e-5*(1+abs(reference))。
resident 模式检查 TinyInfer allocations/uploads/downloads/copies 不增加；warm staging 分配不增加。

## 内存与环境证据

记录 CPU packed weights/arena、CUDA constants/arena/累计逻辑设备请求、pinned cache。
Windows GetProcessMemoryInfo 记录整个进程的 peak working set（两个后端合计，不归属单后端）。
独立 memory-only 进程在 CUDA context 初始化之后取基线，分别记录模型准备、一个/两个 context、
运行和销毁后的进程 working set/private usage，以及 cudaMemGetInfo 的设备级 free/total 快照。
这些设备快照受其他进程/WDDM/驱动缓存影响，不是本进程精确 VRAM 峰值或泄漏证明。
不向用户承诺当前工具无法获得的 WDDM 每进程显存峰值。

保留全部原始样本、stderr、OS/CPU/GPU/电源/温度、开始结束时间、build cache、
Git 状态、源码与二进制 SHA-256。测量时不同时启动其他基准或 sanitizer，不修改系统电源策略。
