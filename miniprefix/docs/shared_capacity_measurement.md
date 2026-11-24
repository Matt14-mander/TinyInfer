# 共享容量准入：压力与收益测量

本基线比较同一个 `miniprefix_capacity_bench` 二进制中的两种准入接口：conservative 调用原两参数
reserve_request，shared 调用传入 prompt 的三参数接口。模型、KV 配置、生成路径、调度参数和到达
时间表保持相同。包装层转发 packed prefill、batched generation 和 GPU greedy，不通过逐请求
logits 路径替代正常 CUDA 生成。

## 工作负载和计时

使用本地 SmolLM2-135M-Instruct FP32，context=128、batch=4、token budget=32、chunk=16、
block=8、pool=20 页。四种确定性 token prompt 长度为 65/68/71/74，共享前 64 token，输出各 8 token，
不使用提前 EOS。token 是合法合成 ID，不代表自然语言负载或真实用户的命中分布。

off 关闭缓存；cold 预热后清空缓存并允许在线学习；hot 预先生成四种历史。每个案例使用独立进程，
模型加载、同后端无缓存输出参考、8 请求预热及 prime 均在测量区间之外。每次输出逐 token 对照参考，
并要求每请求恰好一个 Length 终态。结束时检查预约、KV 和设备页表归零。

按固定到达槽加确定性整数抖动构造开放到达时间表，控制器不等待上一请求完成。同步 Engine 只能在
step 边界接收新请求，因此 TTFT/E2E 从计划到达时间计起，包含提交延迟及排队。每轮对全部速率、缓存
和准入策略的执行次序做确定性打乱；配对案例使用相同 seed。测试不与编译、CPU 压测或其他 GPU 压测并行。

- TTFT：首 token 减计划到达时间；TPOT：单请求七个输出间隔的平均；E2E：末 token 减计划到达时间。
- queue_ms：Ready 减实际 submit；capacity_wait_ms：首次 Wait 到 Ready 的墙钟时间，包含这段时间内
  等待下一调度轮的时间，不是纯准入函数 CPU 耗时。未收到 Wait 的请求取 0。另记录 Wait 尝试次数。
- arriving_tok_s：第 20% 至 80% 到达点之间交付的 token/s；service_tok_s：第 20% 至 80% 完成点
  之间交付的 token/s。积压比例使用“已到达未完成请求数 > 4”，结合两个窗口判断是否饱和。
- 活跃数在 Ready 与 release_reservation 之间积分；resident 峰值在 create/release 时记录。
  logical_reserved 是未抵扣页数；committed 是实际承诺；saved 是二者之差。三者的峰值可能发生在不同时间。
- physical_peak 是整个 worker 生命周期的峰值，包含预热/prime；它不等于准入额度。每步不拷贝 KV 载荷。

这是带主机观测的原生基线，计时包含事件轮询、正确性检查和容量账本查询，不包含 tokenizer、worker IPC、
HTTP 或 JSON 编码；不能直接与已有 Python worker 端到端数字做绝对值比较。每个案例保留原始请求时间线、
二进制/模型 SHA256 和 GPU 快照。汇总中的分位数是每轮分位数的中位数，吞吐也按轮取中位数，并保留范围。

## 生命周期压力

既有 `miniprefix_stress` 的 Profile 现在转发三参数准入接口，避免误测保守策略。每波检查容量承诺不超过
24 页、波次结束预约为零，并记录共享准入调用次数和共享节省页数。覆盖不同长度、前缀复用、取消、EOS、
超容量拒绝和周期性缓存清空，所有成功 token 对照无缓存同后端参考。

该工具沿用逐请求同步 logits 执行，用于生命周期压力及资源回收验证；其吞吐不用于估计 CUDA 批量执行收益。
已有失败注入和 COW 回滚用例继续由 CPU/CUDA CTest 覆盖。有限时长压力测试不能证明无限期无泄漏或所有
随机工作负载均可进展；报告必须保留实际时长、轮次、请求数和主机内存变化。

## 复现

```text
python miniprefix/benchmarks/capacity.py --binary CUDA_CAPACITY_BENCH --model MODEL.mpf --backend cuda --rates 10 40 100 --requests 64 --repeats 3 --output sweep.jsonl.gz
python miniprefix/benchmarks/stress.py --executable CPU_STRESS --model MODEL.mpf --backend cpu --seconds 300 --min-waves 32 --output soak-cpu.json
python miniprefix/benchmarks/stress.py --executable CUDA_STRESS --model MODEL.mpf --backend cuda --seconds 300 --min-waves 32 --output soak-cuda.json
python -m unittest miniprefix.tests.test_capacity -v
```

## 批量空间诊断

追加 hot 模式、40 请求/s、256 请求、三轮重复的实验，将物理池 20/64 页与两种准入策略交错运行，
缓存预算固定为 20 页。每种配置到达窗口约 6.4 秒，比最短扫描窗口更长。补录 decode/prefill 的成功批次数
及回退次数；放大物理池用于定位瞬时空闲页限制，不作为修改生产默认池大小的建议。

```text
python miniprefix/benchmarks/capacity.py --binary CUDA_CAPACITY_BENCH --model MODEL.mpf --backend cuda --rates 40 --requests 256 --repeats 3 --modes hot --pools 20 64 --cache-pages 20 --output headroom.jsonl.gz
```

扫描和诊断分别保留二进制指纹：诊断版仅新增缓存预算参数及现有回退计数输出，生产准入/执行代码相同。
若大池到达窗口积压比例较低，吞吐接近输入负载只能视为已达到该输入速率，不能称为饱和上限。
结果与解释见 [验收报告](acceptance/shared-capacity-measurement.md)。
