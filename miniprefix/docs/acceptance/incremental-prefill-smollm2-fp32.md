# 增量 prefill 验收：2026-10-26

已有前缀恢复路径现在补齐 Engine 命中范围校验、长 suffix 分块调度计数与专门回归。
模型为真实 SmolLM2-135M-Instruct，FP32，revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`。
本次是正确性/资源与计算量验收，没有性能采样，不更新历史吞吐或 TTFT 结论。

## 真实 CPU/CUDA 结果

- 14 项场景、24 次 logits 对照、1,179,648 个值全部通过。
- 最大绝对误差 7.86781311e-05，argmax 完全一致；固定容差 `2e-3 + 2e-4*abs(reference)`。
- 参考为 Transformers 4.46.3 / Torch 2.5.1 CPU FP32 eager，每个检查从完整前缀重算，不复用 native KV。
- 默认页长 16，pool 64、cache 8、context 256；token budget 5、prefill chunk 7。
- 逐步 logits 覆盖命中 48 token 后按 1/7/3/11 token 循环追加，跨多个物理页；所有比较都在相同绝对位置。

| Backend | 场景 | prompt | 复用 | 实算 prefill | decode 输入 | 首输出 STEP |
|---|---|---:|---:|---:|---:|---:|
| cpu | long | 113 | 48 | 65 | 2 | 13 |
| cpu | aligned | 48 | 32 | 16 | 2 | 4 |
| cpu | full-hit | 49 | 48 | 1 | 2 | 1 |
| cpu | branch | 113 | 16 | 97 | 2 | 20 |
| cuda | long | 113 | 48 | 65 | 2 | 13 |
| cuda | aligned | 48 | 32 | 16 | 2 | 4 |
| cuda | full-hit | 49 | 48 | 1 | 2 | 1 |
| cuda | branch | 113 | 16 | 97 | 2 | 20 |

113-token long 请求只计算 65 个 suffix token，13 轮 prefill 后生成首 token；
三个输出只需两次 decode 输入，合计 67 个实际 forward token。关闭缓存为 113+2=115 token，
输出完全相同。对齐 prompt 保留最后整页重算，完整命中且有私有尾 token 时仅重算尾 token。

## 调度、事件和回收

- 长 suffix 与已有 decode 请求混合时，第一轮先用 1 token decode，再用剩余 4 token prefill；新请求本轮恢复 48 token，不占 forward budget。
- 每轮 prefill+decode 实际输入不超过 5；命中只恢复一次。整个 suffix 完成前没有该请求的生成事件。
- 三个生成 token 和 length 终态与独立参考一致，无首 token 丢失或重复；中途取消只交付一个 cancelled 终态。
- 取消未完成 prefill 不发布 suffix 页面，释放请求及设备页表；显式 clear 后 live KV 归零。
- CUDA 输入 H2D 与实际 forward×4 完全一致，KV/权重 H2D 不增加。
- 记录型 runner 的 C++ 测试逐一核对 append token、绝对位置、chunk、budget、decode 优先；错误命中范围在 append 前拒绝，NaN 首 logits 不发布缓存。

## 证据与重现

- [原始逐轮事件、计数、logits 误差与指纹](incremental-prefill-verification.json)。
- [执行路径、接口与重现命令](../incremental_prefill.md)。
- CTest 新增 `miniprefix_engine_prefill_tests`，模型资源不作为普通单元测试依赖。
- 当前 ROG 构建全套 CTest 57/57 通过，无跳过；冻结源码/worker 指纹、逐轮 budget 与文档链接另行复核通过。

CPU worker SHA256：`a565f4fa409b4156bb24b45fc4f2c0c9441635a5cef05a9bb7bbc1a53779624a`。
CUDA worker SHA256：`2f605077541b19e2c8cc138260ee6ee7f640c7fd80224e1f1b6f737f053ca92f`。
原始验收 SHA256：`7c45a57dc381b2ea8081a043f26af616b21ef775689a403115c4f5d1647ce481`。

数据包含模型 SHA256、当前源码指纹、全部实际 token、每轮预算与 Engine/runner 计数。
STEP_STATS 是新增只读命令，原 STEP/STATS 格式和协议版本 3 保持兼容。
