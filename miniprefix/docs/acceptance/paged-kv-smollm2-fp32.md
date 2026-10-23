# 分页 KV 真实模型验收：2026-10-23

CPU/CUDA 已从逐序列 dense KV 改为物理 block 池与序列 block table。所有页面当前独占，
本次未加入前缀缓存、COW、抢占或跨请求矩阵 batching。实现边界见 [分页 KV](../paged_kv.md)。

## 正确性与回收

- 独立 Transformers CPU FP32 full-prefix oracle：54 次 logits 对照、2,654,208 个值，最大绝对误差 0.000113725662；全部 argmax 一致。
- 预设容差 `abs(a-b) <= 2e-3 + 2e-4*abs(b)`，未因分页放宽。
- CPU/CUDA 均覆盖：16 token 常规页、3 token 非整齐页、完整/分块 prefill、跨页 decode、非连续物理页、孔洞复用和失败后的原序列重试。
- 容量不足、context 溢出和非法 token 不改变已提交 token 数与页面占用。Engine 池耗尽后为全部三个未完成请求各发一次 error，live KV/page/table 均归零。
- 9 次 CUDA warm decode 检查零新分配、零 KV/权重 H2D，仅输入 token 与新增页号上传。
- CTest 五项通过：MiniPrefix 页池、基线指标、原 serving scheduler、原 CPU 和 CUDA PagedAttention。页池测试包含地址、容量失败原子性、回滚、复用和溢出配置。

## 同口径无缓存性能

仍为 SmolLM2-135M-Instruct、FP32、context 256、budget 32、chunk 16，默认页长 16、128 页。
同一组 8 请求、每条输出 8 token、相同计划到达和独立参考，EOS 关闭。每配置五个新进程、
各三次 warmup，合计 15 轮、120 请求、960 输出 token。数字为五轮中位数 [min,max]。

| 配置 | 输出 token/s | TTFT P95 ms | 平均 TPOT ms |
|---|---:|---:|---:|
| cpu_b1 | 6.10 [5.54, 10.79] | 9917.91 [5591.76, 10902.13] | 75.55 [44.12, 87.60] |
| cuda_b1 | 120.67 [119.01, 123.60] | 456.30 [444.88, 460.95] | 6.30 [6.15, 6.43] |
| cuda_b2 | 124.32 [110.88, 126.25] | 406.36 [400.64, 465.49] | 12.29 [11.97, 13.51] |

每轮仅 8 请求，nearest-rank TTFT P95 等于该轮最大值；不能当作生产 SLA。计时包含提交
延迟、排队、IPC、forward、Greedy、native release，tokenization/startup/warmup/reference/IO 在窗口外。
完整口径与重现见 [基线协议](../baseline.md)。启动时间、标准差和逐 token 事件均在原始 JSON。

| 配置 | 历史 dense 中位吞吐 | 当前 paged 中位吞吐 |
|---|---:|---:|
| cpu_b1 | 7.55 | 6.10 |
| cuda_b1 | 118.71 | 120.67 |
| cuda_b2 | 118.47 | 124.32 |

两组是同机先后采样，未锁核/锁频、不是随机交替的成对 A/B。CUDA 区间重叠，暂未观察到
明显吞吐退化，不能把中位数上涨归因于分页。CPU 当前范围 5.54–10.79 token/s，
历史范围 6.48–10.41；当前中位数更低，波动较大，不能据此断言分页开销比例或性能改善。
全部慢样本保留。本轮未同时运行编译、验证或其他 benchmark。

## 物理池、占用与传输

- 每页全层 K/V payload 737,280 字节；默认池预留 94,371,840 字节（90 MiB），运行期间不扩容，结束后继续保留。
- 当前已分配页面的 peak payload：
  - cpu_b1：4 页，2,949,120 字节；历史 dense 为 11,796,480 字节。
  - cuda_b1：4 页，2,949,120 字节；历史 dense 为 11,796,480 字节。
  - cuda_b2：8 页，5,898,240 字节；历史 dense 为 23,592,960 字节。

- 占用下降不是总预留显存下降：当前池比历史短负载的 dense allocation 更大，空闲页可供后续序列复用。
- 每轮输入 forward 402 token；CUDA token H2D 1,608 字节、页号 H2D 112 字节；权重/KV H2D 零增量。
- 每轮 CUDA 新分配 8 次，仅为序列设备页表；全局 K/V pool 已在启动时分配。warm decode 跨页申请不触发 cudaMalloc。
- 结束时 live sequence/page/KV 和设备页表均归零；workspace 恒为 602,176 字节。自有 payload 不含驱动/cuBLAS 内部开销，不是总 VRAM/RSS 峰值。

## 冻结证据

- [完整分页性能原始数据](baseline-smollm2-paged-fp32.json)
- [分页数值与生命周期验收](paged-kv-verification.json)
- [不可变的历史 dense 对照](baseline-smollm2-fp32.md)

模型 revision：`12fd25f77366fa6b3b4b768ec3050bf629380bac`；模型 SHA256：`fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef`。
CPU worker SHA256：`1bd7d22a3546d92cf2ebd1b045ebcfa52bbe5bce596d564ea288b706f02bf332`。
CUDA worker SHA256：`8999bbab7fd86c38988daf96fa2f3d568d75d9895cb6b1764f12d4711b3756b5`。

```text
e63114148dc04f0afed27177455aefa6a0da2763606e72d19851b04b17574e03  baseline-smollm2-paged-fp32.json
7ba055115edd1f677ca3d76ecb3286b5e827fab47b3d4b17eed45666d8b65477  paged-kv-verification.json
```

原始 JSON 保留模型/worker/源码指纹、构建参数与 cache 指纹、依赖、GPU 温度/功率/时钟前后快照、全部进程与请求。
后续前缀优化应以本分页无缓存配置为对照，并新增共享前缀的冷/热负载。
