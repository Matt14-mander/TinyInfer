# 最小精确前缀缓存验收：2026-10-25

真实 SmolLM2-135M-Instruct CPU/CUDA 路径已经接入完整页面精确复用。模型 FP32、
revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`，GPU 为 RTX 4070 Laptop。
实现与使用见 [精确前缀缓存](../prefix_cache.md)。本报告不修改历史 dense/paged 数据。

## 正确性与资源

- 开启缓存：18 项行为验收、12 次 logits 对照，共 589,824 个值，最大绝对误差 9.6321106e-05，argmax 一致。
- 关闭缓存回归：54 次 logits 对照、2,654,208 个值，9 次 CUDA 常驻/分配检查全部通过。
- 所有 oracle 都来自独立 Transformers 4.46.3 / Torch 2.5.1 CPU FP32 eager full-prefix 重算；容差保持 `2e-3 + 2e-4*abs(reference)`。
- 覆盖重复请求、分支只命中相同部分、相同后续块但不同此前 token 不误命中、页对齐首 logits、单 token prompt。
- 两个活跃请求共享物理页；取消一个请求、清空缓存后，另一个请求仍产生与 oracle 一致的输出，最终所有页面归零。
- 页面 LRU 与物理池压力淘汰通过；活跃引用无法淘汰时，Engine 给全部未完成请求各一个 error 终态并回收请求/缓存/设备页表。
- CUDA 输入 H2D 严格等于实际 forward token 数×4，复用时 KV/权重 H2D 不增加。缓存命中后上传恢复的页号，不上传 KV。
- 六项 CTest 全部通过：页池、前缀缓存、基线指标、原 serving scheduler、原 CPU 和 CUDA PagedAttention。

## 五组小负载对照

每个 backend 各一个 off worker 与一个 on worker，二者预热后重复五组流；off 位于
cold→warm 配对的前/后交替。37 token 的同一 chat prompt，固定四个输出 token、EOS 关闭，
block 4、pool 64、cache 16、context 128、budget 16、chunk 8。oracle 在所有计时前完成。
这些是同一暖进程内的 paired flows，不是五个独立进程，也不是原八请求到达负载。

| Backend | 模式 | TTFT ms，中位数 [min,max] | 末 token ms，中位数 | 实际 forward token | 复用 token |
|---|---|---:|---:|---:|---:|
| cpu | off | 865.94 [775.21, 899.96] | 1098.24 | 40 | 0 |
| cpu | cold | 856.38 [831.17, 907.98] | 1117.30 | 40 | 0 |
| cpu | warm | 83.50 [71.92, 95.38] | 347.19 | 4 | 36 |
| cuda | off | 34.52 [34.06, 36.68] | 56.31 | 40 | 0 |
| cuda | cold | 37.45 [35.63, 39.72] | 59.66 | 40 | 0 |
| cuda | warm | 8.19 [7.49, 12.37] | 29.16 | 4 | 36 |

每次四个生成 token 中，只有前三个再输入模型，故关闭/冷缓存计算 37+3=40 token。
热缓存复用 36 个 prompt token，只计算剩余一个 prompt token 与三个 decode token，
合计四个 forward token。页长 4 是该实验配置；默认页长 16 下，同一个 prompt 最多复用 32 token，不能套用这里的 36-token 数字。
冷缓存仍要执行相同 forward，并增加索引发布成本；热缓存收益应结合真实工作负载命中率。
计时从 tokenized submit 到控制器收到对应 token event；包含 IPC/forward/Greedy/请求释放，
不含 tokenizer、参考计算、stats、清缓存、启动和 warmup。不声称生产 P95、SLA 或独立进程置信区间。

## 内存与范围

- block 4 的 SmolLM2 每页全层 K/V 184,320 字节；64 页 pool 固定预留 11,796,480 字节（11.25 MiB）。
- 重复 base prompt 后缓存持有 9 页、1,658,880 字节。idle 时保留它们是预期行为；clear 后 live KV 归零，pool 预留保持不变。
- 并发热请求的中间状态为 11 个唯一物理页：9 个共享完整页、每个请求一个私有尾页。不是每请求各复制 9 个 KV 页面。
- 容量上限针对缓存条目，活跃请求引用独立存在；索引为完整 vector key + map，尚无 radix/块哈希或查找时间拆分。
- 缓存只跨同一固定模型 runner 的请求；无部分页共享、COW、持久化、多 adapter、抢占或跨请求矩阵合并。

## 冻结证据

- [完整缓存验收与原始 paired 测量](exact-prefix-verification.json)：所有样本、token、时间戳、before/after 计数、源码/模型/二进制 hash、依赖与 GPU 快照。
- [同一二进制的缓存关闭回归](prefix-cache-off-verification.json)。

CPU worker SHA256：`f35e4a34dc5b647cd89c2eeff920c377f9c2fc64af04a01edc316bb88f2e851e`。
CUDA worker SHA256：`67cde44849ca649a0758c4f219bc776873b3b6b6524c1d8f10c82cbd0aaef169`。

```text
fb235442947b8a75ce1695d538be838f8c59d24295033cc786b9989701c8a00a  exact-prefix-verification.json
93ddbdede1bd36afad30b6f839766760a3b3615634452e1515d5977f9ef4482b  prefix-cache-off-verification.json
```

模型 SHA256：`fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef`。后续优化保留 off/cold/warm 对照，
扩展更长共享 system prefix、多轮分支与持续到达负载，再判断整体吞吐和容量收益。
