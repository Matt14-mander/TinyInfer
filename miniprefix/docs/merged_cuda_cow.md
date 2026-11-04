# CUDA COW 跨层合并复制

KV 的物理布局是 `[block][layer][block_tokens][kv_width]`，K、V 各一个池。
旧实现逐层提交 K/V，一次 COW 需要 `2 * layers` 次 cudaMemcpyAsync。
当前将层看作二维矩形的行，使用两次 cudaMemcpy2DAsync：

- source/destination：对应物理页第 0 层的起始地址。
- pitch：`block_tokens * kv_width * sizeof(float)`。
- width：`valid_tokens * kv_width * sizeof(float)`。
- height：模型层数。

每层只搬运有效 token，行尾未使用部分不复制。SmolLM2 的 30 层从 60 次提交减为 2 次，
数据量不变；不将全部页复制后再把额外字节算作有效流量。

KVManager 新增 copy_shared_tail_region，提供以 float 元素为单位的二维区域。
CPU 继续通过原逐层接口复制，该接口委托同一个事务完成检查。新回调抛异常时不允许 commit；
CUDA runner 在已有异常路径先等待设备访问结束，再恢复旧页表并归还新页。事件计时、
共享引用、容量预约、页表 dirty 修复和发布策略沿用已有行为。

COW 页数与正常复制字节数保持原统计口径；CUDA 成本回调现在覆盖全部层，只有两份 K/V
均成功提交后才累计字节。单独 K 成功、V 提交失败的孤立流量仍不计入 cow_bytes。

## 验证

- CPU/CUDA 原有 COW、非整页缓存、容量及模型数值回归。
- 增加真实二维 K/V 复制的有效区域与行尾哨兵检查。
- 在 K 复制提交后、V 提交前注入异常；同步后验证源未变、旧映射恢复、新页释放、重试可用。
- 原有模型计算失败后恢复、源释放后重试、精确字节计数与计时开关测试继续执行。

## 对比方法

compare_cuda_cow.py 复用现有 partial_prefix.py 的四组输入、CPU 无缓存输出参考、generate
及 TTFT/TPOT/吞吐计算。保存 329b240 的旧 worker，核对历史记录中的 SHA256 后，与新 worker
在同机交替运行；每轮交换旧/新顺序，并轮换 off/cold/hot 与负载顺序。

正式测量默认 6 轮，计时插桩关闭；独立 profiling 默认 3 轮，仅比较复制区间。每个模式和
版本启动新 worker，各负载先 warmup、清空缓存，hot 再 prime。测量区间不查询统计。
所有旧/新样本必须输出一致，复用量、forward 量、COW 页数与字节数一致，最终资源归零。
原始样本保留异常点，配对加速比按同轮计算。

吞吐仍是单请求输出 token/s，prime 不计入热请求延迟。CUDA event 时间包含复制提交间隙，
不等于纯 DMA 时间；不能直接用复制时间减少量推算端到端提升。旧基线的历史数据保留，
本次结论优先使用同机交替运行的数据，避免跨日期性能差异。

```text
python miniprefix/benchmarks/compare_cuda_cow.py --model-dir MODEL_DIR --cpu-worker CPU_WORKER --before OLD_CUDA_WORKER --after NEW_CUDA_WORKER --repeats 6 --profiles 3 --output RESULT.json
```

本次[验收报告](acceptance/merged-cow-review.md)保留全部旧/新结果、波动与原始数据。
