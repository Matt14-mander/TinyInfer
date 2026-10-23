# SmolLM2 FP32 基线验收：2026-10-23

已在本机真实 CPU/CUDA 上完成。以下是每配置五个独立进程样本的中位数，方括号为 min/max；
共 15 次测量、120 请求、960 个输出 token，另有每进程三次 warmup，不计入吞吐。

| 配置 | 输出 token/s | TTFT P50 ms | TTFT P95 ms | 平均 TPOT ms |
|---|---:|---:|---:|---:|
| cpu_b1 | 7.55 [6.48, 10.41] | 3011.66 [2757.74, 5221.74] | 7860.40 [5803.91, 9530.10] | 60.67 [44.49, 68.81] |
| cuda_b1 | 118.71 [109.91, 125.20] | 216.77 [202.84, 236.75] | 463.97 [437.49, 505.14] | 6.49 [6.10, 7.01] |
| cuda_b2 | 118.47 [106.15, 122.67] | 181.91 [173.88, 220.90] | 436.58 [413.97, 490.54] | 12.41 [12.01, 14.20] |

每轮只有 8 请求，nearest-rank P95 等于该轮最大值；表中的 P95 是五轮此值的中位数，
不是生产尾延迟或 SLA。全部输出与独立 Transformers CPU full-prefix Greedy 一致。

## 环境与固定工作量

- CPU：13th Gen Intel Core i9-13980HX，运行后读取本机注册表确认。未锁核、锁频或隔离桌面负载。
- GPU：RTX 4070 Laptop，8188 MiB，驱动 555.97；CUDA 12.5.1，SM 89。
- Windows 11 build 26200；MSVC 14.29、Release /O2、AVX2；CUDA FP32 pedantic cuBLAS、禁用 FMA contraction。
- Python 3.12.3、Torch 2.5.1+cpu、Transformers 4.46.3；完整依赖与构建参数在 JSON。
- 模型：HuggingFaceTB/SmolLM2-135M-Instruct，原始 BF16 权重转 FP32，原生文件 538,076,734 字节。
- revision：`12fd25f77366fa6b3b4b768ec3050bf629380bac`。
- context 256、token budget 32、prefill chunk 16；4 种 chat prompt 重复两遍，prompt token 长度 [37, 39, 43, 54, 37, 39, 43, 54]。
- 到达时间 0/0/10/10/20/20/30/30 ms，每请求 8 输出 token，固定负载关闭 EOS；自然 EOS 在独立验收中测试。

计时从计划 tokenized 到达到 Python 控制器收到 token event，包含排队、提交延迟、IPC、forward、采样与 KV 释放；
不含 tokenizer、模型启动、warmup、oracle、解码和写盘。STEP 整轮完成后交付事件。
启动另报：新进程到首 token 的中位数为 CPU 1409.25 ms、CUDA b1 2377.87 ms、CUDA b2 2224.31 ms。
未清空 OS 文件缓存和驱动缓存，因此不称物理冷启动。

## 正确性与常驻性证据

- 数值门槛：42 次 logits 比较，2,064,384 个 FP32 值；最大绝对误差 9.86903906e-05，全部 argmax 一致。
- 预设容差为 `abs(a-b) <= 2e-3 + 2e-4*abs(b)`，参考使用完整 prefix 重算，不复用待测 KV。
- 覆盖 full/chunk prefill、decode、EOS、长度终止、并发请求、取消、非法输入和 KV 回收；9 次 CUDA warm decode 额外检查零新分配、零权重/KV H2D。
- 15 轮性能输出全部匹配参考，结束后 live sequence/KV 均为零。每轮实际 forward 402 token，CUDA token H2D 1,608 字节，权重/KV H2D 增量为零。
- CUDA 每轮只新增 16 次设备分配，即 8 序列各一对 dense K/V；不是零分配调度。持久 workspace 恒为 602,176 字节。
- CUDA device weights payload 651,306,240 字节（tied head 另有矩阵副本）；KV peak 为 b1 11,796,480、b2 23,592,960 字节。
- 上述内存仅记录自有 payload，不包括 cuBLAS/驱动内部内存，不代表总 VRAM 峰值或 RSS。每轮 logits D2H 16,121,856 字节，仍有优化空间。

## 结果解释与后续比较边界

cuda_b2 表示两个活跃请求，由 runner 逐 sequence 计算；目前没有跨请求张量批处理。
本次 b2 的 TTFT 有所降低，但 TPOT 接近 b1 两倍、吞吐基本相同，不能据此声称 GPU batching 提速。
CPU 吞吐范围 6.48–10.41 token/s，波动明显；没有剔除慢样本，也没有足够证据归因于某一种后台负载或降频。
GPU 前后空闲快照温度 58→61°C，未采集运行中频率轨迹；本结果适合本机后续同口径回归，跨机器不能直接作因果速度比较。
这是短 prompt、短输出、dense KV、无前缀缓存的首个真实模型基线；尚未覆盖长上下文、饱和负载、共享前缀命中率、分页、抢占重算或 HTTP 服务。
后续实现这些能力时应保留此组无缓存对照，并增加对应负载；本报告不把示例模块已实现的算法计入 MiniPrefix。

## 可复核材料

- [完整性能原始数据](baseline-smollm2-fp32.json)：全部 15 轮、逐请求 token/timestamp、逐 STEP 事件、计数器、warmup、源码/模型/worker 指纹和构建参数。
- [数值与生命周期验收](real-model-verification.json)：同一模型和 worker，benchmark 在指纹不符时拒绝运行。
- [协议与重现命令](../baseline.md)、[真实模型准备与执行](../real_model.md)。

冻结文件 SHA256：

```text
dbd1392cf7d90037c83a7fd04ae3e2a61d027d993bda4386ddda09421c5a47e5  baseline-smollm2-fp32.json
456ab164da1e161962e89a39e7c0211f6769ef7788729cbf3082c11e58586b09  real-model-verification.json
```

模型 SHA256：`fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef`。
CPU worker SHA256：`2802167c4900da82d40cd5a5e2247c02a913b9040511c45b3eb54b7131384078`。
CUDA worker SHA256：`787e83277ea0bf41db2653584f9a42700d30b2759517ffe2f2eb176fc7d66ad7`。

源码指纹覆盖 MiniPrefix 的 .py/.cpp/.cu/.h；原始构建 cache 指纹保留采样时版本。
采样完成后仅补充文档和 CTest 注册，未重建或修改被测运行代码。

采样后回归：CTest 的 baseline metrics、serving scheduler、CPU PagedAttention、CUDA
PagedAttention 四项全部通过；metrics 内含三项手算时间戳/分位数与错误输出拒绝测试。
冻结的 15 个 worker PID、120 条请求、源码和二进制指纹另行复核通过，文档本地链接检查通过。
