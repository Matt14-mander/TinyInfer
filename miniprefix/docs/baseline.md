# FP32 端到端性能基线协议

固定 SmolLM2-135M-Instruct revision 与 native model/worker SHA256。性能采集前，必须
提供同一模型、同一 worker 二进制通过的真实数值验收 JSON；模型或二进制变化即拒绝运行。
构建、Python/参考库版本、源码 SHA、GPU/驱动/状态、原始时间戳和输出全部进入报告。

## 固定配置

| 配置 | Backend | 活跃请求上限 | token budget | prefill chunk | context |
|---|---|---:|---:|---:|---:|
| cpu_b1 | 原生 CPU | 1 | 32 | 16 | 256 |
| cuda_b1 | 全模型 CUDA | 1 | 32 | 16 | 256 |
| cuda_b2 | 全模型 CUDA | 2 | 32 | 16 | 256 |

全部 FP32、Greedy、无 prefix cache。历史首轮为 dense KV；当前脚本使用分页 KV，
16 token/页、128 页，并将结果写入独立的 paged 文件。batch 上限表示调度并存请求数，runner
仍逐 sequence 调用；cuda_b2 不代表已经跨请求合并 cuBLAS 或 attention。

四种 chat prompt 重复两遍，共 8 请求，到达时间为 0/0/10/10/20/20/30/30 ms，
每请求固定输出 8 token；EOS 为保持相同输出工作量而明确关闭。完整文本、token ID、
目标输出保留在 JSON。所有完成序列与独立 Transformers CPU full-prefix Greedy 对照。
自然 EOS 行为已在数值验收中另测，固定长度基准不替代正常生成展示。

每配置五轮独立 worker；每个 worker 只测一次固定流，先运行三条代表性 warmup。
配置顺序按轮次轮转，全部串行采集，禁止并行 benchmark/验证/编译污染测量。
源码修改后重新采集，不删除离群值。GPU 频率、功率和温度记录为上下文，不强行锁频。

## 计时口径

- warm window：首个计划到达到最后 token event 被 Python 控制器接收。
  包含控制器延迟、排队、IPC、native forward、Greedy 和 native KV release。
- tokenizer/model preparation、warmup、stats、参考校验、解码与 JSON 写盘在窗口外。
  这是 tokenized engine 的端到端基线，不是 HTTP/text gateway 的延迟。
- worker_startup：新进程创建到 READY，包含读取/校验权重与 GPU 权重上传。
- fresh_process_first_token：从创建进程到第一条 warmup 的首 token。CPU 的首次 packing
  等懒初始化在其中；OS 文件缓存和 GPU 驱动缓存未清空，不称为物理冷缓存实验。
- TTFT = 首 token 交付 − 计划到达；controller_submit_delay 单列，native 排队没有凭空估算。
- TPOT 是每请求 `(last-first)/(outputs-1)` 的有效请求均值；单 token 为 null。
  ITL P95 使用所有相邻 token 间隔样本，二者口径不同。
- 吞吐 = 窗口内交付 token / 窗口秒数，包含到达空闲和排队，不包含 prompt token。
- 每轮分位数用 nearest rank；8 请求时 TTFT P95 实际等于该轮最大值。报告五轮中位数、
  min/max、样本标准差，不声称生产 P95、SLA 或统计置信区间。
- STEP 内请求逐个计算，event 在整轮完成后交付；不把计算中途时刻当用户收 token 时刻。

## 内存与传输

分别报告 FP32 device weights、每层 KV 当前/峰值 payload、持久 workspace payload、
allocation 次数、权重/token/KV H2D 和 logits D2H。它们不包括 cuBLAS/驱动内部内存，
不等于进程 RSS 或设备总 VRAM 峰值。tied head 当前有单独的 device 矩阵副本，权重
payload 因此大于 model.mpf 中唯一权重的 payload。
分页版额外报告物理池预留量、live/peak block 数、设备页表与页号 H2D；不能将
live KV 下降解读为整个预留池显存下降。字段口径见 [分页 KV](paged_kv.md)。

warm window 必须满足：实际输入 token 数 = prompt + 每请求输出数−1；最终 KV 归零；
CUDA 权重/KV H2D 为零增量，token H2D 为输入数×4；分页版仅允许每序列一次设备页表
分配，物理池与 workspace 不增长，结束后 live 页数和设备页表归零。
历史 dense 版为每序列两次 K/V buffer 分配。违反计数或输出错误会中止。

## 重现

先按 [真实模型文档](real_model.md) 准备依赖、模型、编译并运行数值验收，然后：

```powershell
$py = 'build-miniprefix-env/Scripts/python.exe'
$env:HF_HOME = "$PWD/build-miniprefix-models/hf-cache"
# 沿用 ROG 工具链的 CUDA bin PATH。
& $py miniprefix/tools/verify_real_model.py --model-dir build-miniprefix-models/smollm2-135m-instruct --cpu-worker build-phase54-cuda/miniprefix/miniprefix_worker.exe --cuda-worker build-phase54-cuda/miniprefix/miniprefix_cuda_worker.exe --output benchmark-results/miniprefix/paged-kv-verification.json
& $py miniprefix/benchmarks/baseline.py --model-dir build-miniprefix-models/smollm2-135m-instruct --cpu-worker build-phase54-cuda/miniprefix/miniprefix_worker.exe --cuda-worker build-phase54-cuda/miniprefix/miniprefix_cuda_worker.exe --verification benchmark-results/miniprefix/paged-kv-verification.json --runs 5 --output benchmark-results/miniprefix/baseline-smollm2-paged-fp32.json
```

原始运行输出保留在 Git 忽略的 benchmark-results/miniprefix；冻结基线和验收副本存于
docs/acceptance/，不包含模型权重。手算 timestamp/TPOT、nearest rank 和错误输出拒绝
由 tests/test_baseline.py 验证，运行 `python -m unittest discover -s miniprefix/tests -v`。
结果解释见 [历史 dense 报告](acceptance/baseline-smollm2-fp32.md) 和
[paged 报告](acceptance/paged-kv-smollm2-fp32.md)。历史 JSON 不覆盖；当前脚本拒绝 dense 验收。
前缀收益使用单独的共享/混合/零共享八请求流，见 [测量协议](prefix_benefit.md)，
不与此处四种短 chat prompt 的历史数字拼接速度比。
