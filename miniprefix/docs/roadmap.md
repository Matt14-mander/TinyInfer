# 开发路线与验收

2026-10-26：M0 已完成；M1 已有请求生命周期/配置/原生 runner，调度与高级 KV 提取未完成；
M2 已完成首个 SmolLM2-135M-Instruct CPU/CUDA 正确性闭环；M3 已有分页 KV/权重常驻
GPU、完整 forward 与设备增量页表更新，COW 尚未实现；M4 已建立无缓存 dense/paged FP32 基线，
已接入最小精确前缀缓存、增量 prefill 与冷/热 paired 实验；共享/混合/零共享八请求流的
45 进程 CUDA [收益测量](prefix_benefit.md) 已完成。M5 的完整页面共享、页面 LRU 已开始，
索引优化和长时间负载未完成。验收边界见 [精确前缀缓存](prefix_cache.md)、[真实模型](real_model.md)
和 [基线协议](baseline.md)，不把局部实现当整个阶段全部完成。

| 阶段 | 交付 | 验收门槛 |
|---|---|---|
| M0 架构骨架 | 目录、模块职责、初始请求类型、可选 CMake 入口 | SDK-free 契约可编译；原示例入口保留 |
| M1 引擎与参考路径 | Engine 状态机、ModelConfig、可配置微型 CPU runner；从示例提取 KV/调度接口 | EOS/长度/取消/失败/重算不重复输出；现有轨迹与数值回归 |
| M2 真实模型正确性 | 选定一种小型预训练模型架构、配置/权重/tokenizer；多层/多头/GQA 与必要 RoPE/RMSNorm/MLP | 真实权重 logits/Greedy 对独立框架参考；全前缀/增量/分块一致 |
| M3 GPU 完整执行 | GPU forward、权重与 KV 常驻、设备追加/COW/table 更新 | 实际设备对照通过；warm step 无完整 KV pool H2D；GPU 资源完成后安全复用 |
| M4 端到端基线 | 本地 CLI、生成事件流、冷/热测量、缓存关闭/开启对照 | 固定模型/精度/输入，重复测量 TTFT/TPOT/吞吐/显存；保留原始数据与环境 |
| M5 前缀复用优化 | 替换线性扫描索引、合理驱逐、跨 generate 缓存、多轮/分叉工作负载 | 无跨上下文误命中；输出一致；命中、节省计算、内存与端到端收益均可解释 |

M1 可分批迁入现有 continuous batching、chunked prefill、共享/COW 和抢占机制；
M2/M3 保留这些机制的开关，以便隔离模型与设备问题。M5 优化已有前缀能力，不意味着
此前阶段要重复实现一套索引或禁用所有缓存验证。

首个真实模型已选定 SmolLM2-135M-Instruct 和 MPFX0001 FP32 权重格式，模型 revision
与数值结果固定于验收文档。首版只实现一种架构，模型准备工具需显式调用。
低精度执行按硬件和数值验证选择，FP32 CPU oracle 保留；量化不作为 M0/M1 的依赖。

CUDA Graph、CPU/GPU 异步重叠、量化、真实推测解码、HTTP、multi-GPU、PD 分离、
多模态与 MoE 均在 M4 基线之后按实际瓶颈安排，不纳入第一版承诺。

下一项具体工作：扩展共享 system prefix/多轮负载，测量索引成本，随后引入部分页 COW、
容量准入与抢占恢复。保留历史 dense/paged 基线与独立框架 oracle，复用基线
对照正确性、实际跳过计算量、显存和端到端收益。
