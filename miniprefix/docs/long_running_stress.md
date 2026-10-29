# 长期压力测试与瓶颈测量

`miniprefix_stress` 是直接调用 C++ Engine 的独立基准程序，`benchmarks/stress.py` 负责
外部 watchdog、保留 JSONL 原始样本、检查完成标记并汇总。它不修改 worker 协议，也不在生产
热路径插入计时。CUDA 模式直接创建 CUDA runner，设备不可用时失败，不回退 CPU。

## 固定负载与正确性门槛

- 同一进程、同一 Engine，持续提交新的请求 ID，不在每轮销毁 Engine。
- context 128、活跃上限 4、token budget 16、chunk 8；页长 4、物理池 24 页、缓存 12 页。
- 12 种确定性 token prompt，长度 8/17/33/65，前 6 种共享最多 8-token 根；其余根不同。
  token 由源码公式生成，属于机制测试，不评价自然语言回答质量。
- 使用闭环 burst 负载：每轮一次提交 8 个请求，排空后再提交下一轮，模板随轮次轮换。4 个正常长度完成、1 个提前 EOS、
  1 个计算一步后取消、1 个排队取消、1 个最坏情况超过池容量的请求级 Error。
- 每个正常请求最多输出 8 token。65-token prompt 的预约使多个大请求不能同时进入池，
  因而持续覆盖 FIFO 容量等待；小请求、命中与不同长度组合覆盖同轮混合执行。
- 每 5 轮清空前缀缓存，穿插冷启动、流内复用和缓存污染/驱逐。缓存容量远小于模板集合。
- 启动时用同一后端、关闭缓存且大页池的 runner 生成全部模板的 8-token reference。
  压测逐 token 比对，包括取消前已输出的 token；检查 EOS、Error、取消和 Length 各自的数量，
  任何重复终态、终态后事件、错误输出、Engine 失败或超过每轮 256 步均使进程非零退出。
- 对比每轮 Engine prefill/decode/reused 计数与 runner 增量；检查无权重/KV 重复 H2D。
- 每轮结束要求无序列/设备页表残留、只剩缓存引用；clear 后要求 KV 为零；退出时再次检查。

该 reference 验证缓存、调度和生命周期行为，不替代独立框架数值 oracle。真实模型与后端数学
正确性沿用独立 Transformers 回归；本次没有修改模型运算。无随机种子依赖。

## 计时与内存口径

Profile runner 用 steady_clock 包围 admission、create、reuse、publish、prefill append、decode
append、release 和 clear。prefill/decode 按请求已消费位置区分，不以 chunk 长度猜测。
CUDA append 时间含 CPU 准备、H2D、kernel/cuBLAS、logits D2H 和同步，不是 GPU kernel 时间；
reuse 同样包括设备页表更新与同步，不是纯索引耗时。

另报告 Engine step 时间、提交到准入的队列时间、提交到首 token 的 TTFT，以及成功请求的
`(last-first)/(output_tokens-1)` 平均 TPOT。取消和错误请求不进入 TTFT/TPOT 分位数。
吞吐分子包含全部实际发出的 token（含取消前的输出），分母为各轮原生测量窗口 wall time。
不包含 tokenizer、worker IPC、模型启动、参考计算或 JSONL 写盘，不能和历史 Python worker
端到端基线直接比较。剩余 wall 时间包括 Engine、校验、事件处理及测量自身开销。

保留所有轮次，稳态汇总仅排除前 10 轮；不足 11 轮的 smoke 不排除。分位数为 nearest rank。
单进程时间序列不是独立重复样本，不据此宣称置信区间、缓存加速比或生产 P95 SLA。

Windows 每轮采集进程 PrivateUsage（私有提交量）及 WorkingSetSize（驻留集），另报告
原生 workspace 和实际页计数。进程指标包含模型、运行库、allocator 和 Engine 历史 ID；
不能把 RSS/PrivateUsage 的增长直接判成 KV 泄漏，也不能把页归零当作整个进程内存有界。
其他平台的内存查询当前返回 0，表示未实现采集，不能用于内存结论。

## 构建与复现

开启既有 `TINYINFER_BUILD_MINIPREFIX` 构建 `miniprefix_stress`；CUDA build 同时支持 cpu/cuda，
CPU-only build 只支持 cpu。没有下载模型或新增运行时 Python 包。

```text
cmake --build BUILD --target miniprefix_stress
python miniprefix/benchmarks/stress.py --executable BUILD/miniprefix/miniprefix_stress.exe --model MODEL.mpf --backend cuda --seconds 600 --min-waves 128 --output stress-cuda.json
```

程序满足最短时长及最少轮数后才结束。Python watchdog 为 `max(seconds*3,180)` 秒，包含
启动与 reference；超时或异常保留 `.jsonl` 和 `.stderr.log`，不生成 passed 报告。成功报告
核对 executable/model 前后 SHA256，并保存原始 JSONL 哈希。最终原始报告中的命令保留本机路径。

## 阶段边界

这项工作补齐 MiniPrefix 的持续压力回归和原生调用级瓶颈测量。它不是整个仓库 Phase 6
（Transformer/compiler direction）的完成声明；也不覆盖多小时 soak、多租户、HTTP、
跨请求矩阵 batching、COW、抢占、多 GPU 或大缓存索引扩展性。后续优化需在相同协议下复测，
不能用本测试替代缓存 off/cold/warm 的独立进程收益实验。

## 索引规模诊断

`miniprefix_prefix_index_bench` 只链接 KV metadata，不执行模型、不分配 CUDA payload。
分别测试 16/64/256/1024 个缓存页（4 token/页）：一次预热后保留 10 次完整 publish、
完整前缀命中、首 token 不匹配及逐页 LRU 淘汰的原始微秒计时。每轮核对命中长度、
淘汰数量及引用归零。source 在淘汰期间仍引用页面，因此淘汰测的是缓存引用/LRU 操作，
不冒充 GPU 物理释放成本。

`stored_key_token_bytes` 当前由缓存实际保留的边 token 数量计算（不含节点、
vector 容量余量和 allocator）。历史完整 key 实现使用
`pages*(pages+1)/2 * block_tokens * sizeof(int32)`；历史报告不改写。
这是索引扩展性诊断，不能把其时间直接加到真实模型端到端结果或作为新索引的加速证据。

```text
cmake --build BUILD --target miniprefix_prefix_index_bench
BUILD/miniprefix/miniprefix_prefix_index_bench.exe
```

本机实测见 [2026-10-29 验收报告](acceptance/long-running-stress.md)：CUDA 10 分钟、
CPU 至少 128 轮，附完整原始记录、内存变化、调用耗时与索引规模实验。
