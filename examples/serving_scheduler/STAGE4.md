# 第四阶段：真实 TinyInfer CPU 后端工程验收

日期：2026-10-15。按用户确认，采用本地微型随机权重 Transformer 完成工程验证。
**本阶段已完成本地真实数值推理、实际 K/V 存储、墙钟调度与指标测量；不是预训练大模型服务。**

2026-10-22 更新：当前 worker 已增加跨请求 CPU 张量 batching，见 [BATCHING.md](BATCHING.md)。
本文的串行实现描述、44 项测试和 results/stage4 性能数据均保留为历史第四阶段记录。

## 后端具体是什么

`tinyinfer_serving_worker` 是链接 TinyInfer C++ core 的持久 CPU 子进程。模型有单层、
单头因果 attention，隐藏维度 16，FFN 维度 32，词表 32，FP32，最大预留上下文 4096。
权重由固定 uint32 LCG seed=20261015 初始化，没有训练或语言理解能力，也没有 tokenizer。
输入是固定规则生成的整数 token，token 0 不当作 EOS，按指定输出预算结束。

每个输入 token 执行 embedding + 位置项、Q/K/V MatMul、因果 attention Softmax、
attention 输出投影、残差、LayerNorm、ReLU FFN、残差/LayerNorm、词表 logits 和 Greedy。
计算使用 TinyInfer `ops::matmul/softmax/add/layer_norm/relu`，不是模拟 token 公式。
K/V 存入每请求独立的真实 Tensor，后续 decode 复用；chunked prefill 延续相同位置和 K/V。

同一调度 step 的请求在 C++ 内**逐个计算**，没有跨请求 tensor batching 或 GPU kernel。
这里的 Static/Continuous 是真实模型调用上的准入/共调度策略；不能把这些数字当成
cuBLAS/GPU batching、预训练 LLM 或生产 serving 的吞吐。微型计算中 IPC/调度开销占比很高。

## 构建与运行

```bash
cmake -S . -B build-serving -DCMAKE_BUILD_TYPE=Release \
  -DTINYINFER_ENABLE_NATIVE_ARCH=ON -DTINYINFER_BUILD_EXAMPLES=ON \
  -DTINYINFER_BUILD_TESTS=ON
cmake --build build-serving --target tinyinfer_serving_worker -j 4

# 完整 Python 测试，必须提供真实 worker；未提供时 integration tests 会 skip
TINYINFER_SERVING_WORKER=build-serving/examples/tinyinfer_serving_worker \
  python3 -m unittest discover -s examples/serving_scheduler -v

# 三个独立 worker，每种配置 30 请求；含完整请求时间戳、原始 trace 与真实 KV 计数
python3 -m examples.serving_scheduler.real_benchmark \
  --worker build-serving/examples/tinyinfer_serving_worker --runs 3 \
  --output serving-real-results.json
```

Windows 使用实际生成的 Release/tinyinfer_serving_worker.exe 路径，环境变量按终端语法设置。
Mac SDK 的 libc++ include workaround 沿用项目既有 Phase 5.1 构建说明。不要拿 Debug/ASan
数据当 Release 性能数据。启用 examples/tests 且找到 Python 3.7+ 时，CMake 自动注册
`tinyinfer_serving_scheduler_tests`；先构建 worker 后运行该 CTest。

`benchmark.py` 仍是前三阶段模拟入口；**只有 `real_benchmark.py` 是本阶段墙钟入口**。
没有下载模型、安装第三方 Python 库，也不需要 CUDA/Metal。

## 时间、正确性与内存边界

- `serve()` 使用 perf_counter_ns，按真实 arrival 等待，不推进虚拟时间，也不信任后端
  的 duration 字段。额外测试将该字段改为 1e12，仍得到实际时间。
- 起点是每轮请求流的墙钟 origin；首次到达前空闲不计吞吐窗口。模型启动、权重准备和
  reset 在窗口外，warmup 不采样；release/CLEAR 成本计入总完成窗口，统计查询在窗口外。
- token 交付时间是 C++ 完成整个 step、Python 收到/校验结果之后的时刻；包含 IPC、
  等待、调度开销，不包含网络、tokenizer 或报告 JSON 写盘。不是逐请求计算中间点计时。
- TTFT、TPOT、请求延迟/P95 定义沿用主 README，但 clock=wall_ms。TPOT 是有效请求平均，
  单 token 不填零。P95 为 nearest rank；每配置 30 请求，汇总是 3 轮各自分位数的中位数，
  不宣称统计置信区间或生产 SLA。原始轮次、拒绝、每请求时间全部保留，不删除离群点。
- 全量 prefill、chunked prefill、Static、Continuous 的所有完成输出均与**每次生成都从
  新状态重算完整前缀**的 Greedy 参考一致。参考不跨生成步复用 KV，且在计时窗口外执行。
  它共享模型算子实现，不是第二套数学/框架实现；算子数值由既有 core 测试承担。
- 逻辑 KV 仍按 ceil((prompt+output)/block_size) 预留；后端 block size 必须与调度器一致。
  worker 首次处理时实际分配同样取整后的 K/V Tensor，每预留 token 为 2×16×4=128 bytes。
  未执行的 prefill 可以已占逻辑预算但尚未分配 native Tensor；物理 KV 不得超过逻辑上限。
- `peak_kv_bytes/live_kv_bytes` 只含 K/V Tensor payload，不含权重、临时 attention 张量、
  allocator/process/IPC 开销；不是 RSS 或 GPU 显存。`forward_tokens` 应等于实际 prefill+
  decode 输入 token 数。正常完成及异常后 live KV 均归零。
- worker 是同步 CPU 后端，回复即计算完成；完成请求先释放 native KV 再释放逻辑预算。
  非法输入/计算错误清理 native 状态并明确失败；IPC 中断/失去 framing 时终止并等待 worker，
  再回收预算，不自动重跑。中途失败终止本轮，不输出伪造成功样本。

## 实验协议与验收证据

固定 30 请求，每 6 请求重复长度模式五次，arrival 0..10 ms；prompt/output 长度见 JSON。
比较 static_full、continuous_full、static_chunked、continuous_chunked；batch=3，
KV=32 blocks×4 token slots，分块 T=8/C=4。每轮新建 worker，配置顺序轮转，每配置先
warmup 一个请求，再测完整流；所有模式的目标输出 token 总量都是 120。

- Release 完整 Python 套件：44/44，包括之前 39 项与新增 5 项真实后端测试；无 skip。
- ASan/UBSan C++ worker 真实集成：5/5；只验证 CPU host 内存/边界。
- CTest 可发现并运行真实集成，不以未执行/skip 充当验收。
- 三个独立真实 worker、四种配置、共 12 份原始报告；所有输出参考与真实 KV 计数检查通过。

性能表与构建环境见 [results/stage4/report.md](results/stage4/report.md)，原始数据在
[cpu-benchmark.json](results/stage4/cpu-benchmark.json)，验收状态在 verification.json。

本例四个学习阶段在上述边界内收尾。预训练模型、真实文本 tokenizer、GPU 后端、跨请求
张量 batching、HTTP/网络流式交付、EOS、取消、请求超时和生产故障隔离均未实现；也不改变
TinyInfer Phase 5.3 尚未通过的 Metal 性能门槛或 Phase 5.4 CUDA 开发状态。
