# Phase 5 — Hardware Backends

- 规划日期：2026-10-11（Asia/Shanghai）。
- 状态：开发目标已规划，尚未开始实现。
- 基线：`924a7e2`，Phase 4.5 静态 FP32 CPU 优化收口。

## 核心目标

让同一个静态 FP32 Model 在 CPU 和 GPU 上执行，形成可复用的设备执行计划，
明确内存驻留、数据传输、执行完成和回退行为，并用真实设备上的端到端测量验收。
沿用现有 ONNX 导入、Graph 优化和 CPU 参考路径。

Phase 4.5 已完成其测量支持边界内的验收；其 Auto 白名单不是 GPU 选择规则。
当前 Backend 只按 OpType 判断支持，ExecutionContext 使用可被 CPU 直接访问的
Tensor/Buffer，Executor 按主机步骤释放中间值。CUDA/Metal 目前只有占位说明。
因此首个工作项是补齐设备运行时契约，随后完成一个后端的模型执行闭环。

## 开发顺序

| 子阶段 | 开发目标 | 交付与退出条件 |
| --- | --- | --- |
| 5.0 设备契约 | 设备身份、节点能力判断、设备分配/复制、执行完成与资源生命周期 | 设计 ADR、CPU 适配测试、目标 GPU/驱动/SDK 清单；禁用 GPU 时现有 CPU 构建和 API 可用 |
| 5.1 首个 GPU 后端 | 分配与传输、单执行队列、矩阵与逐元素算子 | 真机单算子数值测试、错误路径和资源释放检查通过 |
| 5.2 Prepared GPU 模型 | 常量权重上传一次、设备中间值驻留、复用执行计划与激活缓冲 | 两个 ONNX fixture 全图在 GPU 执行，重复绑定/运行/取出结果正确，回退可解释 |
| 5.3 性能验收 | GPU 执行与真实应用端到端测量，CPU 对照、交叉点分析 | 冷/热启动、传输、同步、内存和 p50/p95 报告；达到预先固定的目标工作负载门槛 |
| 5.4 第二 GPU 后端 | 用同一契约接入 Metal 或 CUDA 的另一方 | 同一组模型/语义/生命周期测试跨后端通过，独立实机性能报告 |

建议首先验证 ROG 是否具备可用 NVIDIA GPU/CUDA 工具链；具备条件时先做 CUDA，
随后 Metal。现有 CPU 报告没有提供 GPU 型号，不能据此宣称 CUDA 已可用。
若 CUDA 条件不满足而 Mac Metal 设备可用，先交付 Metal，保持契约和验收目标相同。
两个 GPU 后端顺序实现。5.0–5.3 是首个交付里程碑；Phase 5 完整收口包含 5.4。

## 5.0 必须明确的运行时契约

1. **能力判断**：根据节点、dtype、shape、layout、属性及设备判断支持，返回原因。
   保留现有 CPU 入口，避免仅凭 `supports(OpType)` 承诺所有输入都支持。
2. **内存边界**：主机 Tensor 保持既有深拷贝/视图语义。设备分配使用明确的设备
   资源表示和传输接口，设备指针不能伪装成可供 `Tensor::at()` 解引用的主机内存。
3. **完成语义**：首版公开同步推理接口，内部使用一条队列/stream；返回主机结果前
   等待执行和复制完成。异步任务引用的资源必须保持到完成，不能仅按主机循环步数释放。
4. **准备与运行**：不可变设备权重和计划可复用，各 context 拥有独立可写激活。
   设备初始化、权重上传和算子准备与 warm run 分离；记录上传/分配/回退计数。
5. **回退策略**：GPU 严格模式在准备阶段报告不支持节点；允许回退时首版整模型
   使用现有 CPU Prepared 路径，并记录原因。逐节点混合执行和跨设备图分区留待后续。
6. **失败与构建**：GPU 支持可选构建；无设备、SDK 缺失、复制/执行失败有明确诊断。
   执行失败不静默重跑 CPU；无 GPU 构建继续可运行 CPU 全套测试。

## 首个 GPU 算子范围

- 静态 FP32、rank-2 MatMul、Gemm、FusedGemmActivation，以及 Add、ReLU、Tanh、Softmax。
- 以现有 `phase3_mlp_gemm.onnx` 和 `rl_actor_mlp_tanh.onnx` 为模型验收样本；
  增加计算量更大的矩阵/MLP 工作负载，避免仅以微型模型评价 GPU 加速。
- 覆盖 transA/transB、alpha/beta、既有 bias 广播、尾部、零尺寸和运行时输入变化。
  非连续输入可在边界显式整理；相应复制必须计入诊断与端到端时间。
- CUDA 首版矩阵计算优先接入 cuBLAS，自有 kernel 实现必要的 bias/激活/逐元素操作。
  优先完成正确模型路径，后续再按测量结果开发自有矩阵 kernel。
- 能力不支持的属性/布局必须拒绝或按显式整模型策略回退，不静默改变语义。
- 首轮采用明确的 FP32 计算策略，避免默认降精度导致参考比较失真。

## 验收边界

### 正确性与资源

- CPU 原有完整测试通过；GPU 算子、两个真实 fixture 及准备/运行集成测试真机通过。
- Fixture 保留现有 `1e-5` 独立参考门槛。矩阵对比先沿用
  `1e-5 * (1 + abs(reference))`；若不同归约次序需要其他门槛，必须依据误差分析
  单独制定，不能遇到失败后整体放宽。
- 明确 NaN、Inf、signed zero 和 ReLU 语义；改变输入、动态权重/bias、零 K、
  重复运行、context 隔离、计划销毁后的资源存活和失败清理有覆盖。
- 稳态运行不重复上传常量权重；GPU 支持图的中间结果保持设备驻留。
- 设备测试不能被“编译成功”或 CPU mock 替代；未执行的硬件明确标为待验收。

### 性能

- 分别报告导入/准备、首次执行、设备完成时间、H2D/D2H 复制和
  **输入在主机 → 推理 → 独立主机输出**的 warm 端到端时间。
- GPU 计时包含等待工作完成；不以 kernel 提交时间作为执行时间。
  分阶段诊断不能直接相加替代实测端到端延迟。
- 对照当前 CPU Prepared Auto；同模型、输入、精度、结果所有权约定。
  增加设备驻留计时作为诊断，不能据此宣称主机调用端到端加速。
- 固定目标硬件和矩阵/MLP 用例后至少三次独立进程复测，记录 warmup、采样、
  顺序、p50/p95、内存、GPU/驱动/SDK及源码版本，保留全部结果。
- 首个 GPU 交付的建议性能门槛：至少一个预先选定的中大工作负载，两轮独立
  测量的 warm 端到端 p50 加速比中位数均达到 **1.2×**；5.0 确认并冻结门槛。
  同时审查 p95，不要求微型模型一定加速，报告 CPU/GPU 适用交叉点。
- 对未走 GPU 的 CPU 回退控制，调查可重复超过 5% 的新增延迟回退。

## 本阶段不扩大的范围

动态 shape、Transformer/KV cache、Conv、Int8/FP16、CPU 多线程、自有高性能 GPU
GEMM、自动调优、多 GPU、并发多 stream 和通用 MLIR lowering 均属于后续独立工作。
Phase 4 的这些延期能力保持在路线图中，不阻塞本次静态 FP32 GPU 里程碑。
设备后端先消费现有 Graph/Prepared 表示；出现实际 lowering/codegen 需求时再引入低层 IR。

## 技术依据

- [CUDA 异步执行](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)：执行完成、传输与同步契约。
- [cuBLAS](https://developer.nvidia.com/cublas)：首版 CUDA 矩阵库路径。
- [Metal 资源](https://developer.apple.com/documentation/metal/resource-fundamentals)：设备资源与存储边界。
- [Metal 命令结构](https://developer.apple.com/documentation/metal/setting-up-a-command-structure)：提交与完成语义。
