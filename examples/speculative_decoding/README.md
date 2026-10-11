# 推测解码学习实验：Greedy Draft → Verify

第一阶段目标是学习候选生成、验证、接受/拒绝，并保证与 **Target Greedy 的 token
序列完全一致**。使用 Python 标准库与确定性模拟模型，无需下载模型或安装依赖。
本目录是独立的 Python 算法实验，尚未调用 TinyInfer C++ Runtime。

## 运行

需要 Python 3.7+，在 TinyInfer 仓库根目录执行：

```bash
# 今日验收：正确性测试（失败时返回非零退出码）
python3 examples/speculative_decoding/benchmark.py --test

# 三种 draft 情况，每种比较 K=1、2、4；逐行检查与 Target Greedy 相同
python3 examples/speculative_decoding/benchmark.py

# 观察首次错误：错误候选与后续候选被丢弃，Target 给出纠正 token
python3 examples/speculative_decoding/benchmark.py --scenario first_wrong --tokens 5 --trace

# 看混合接受/拒绝过程
python3 examples/speculative_decoding/benchmark.py --scenario mixed --trace

# JSON 包含完整 baseline、统计和可选逐轮 trace
python3 examples/speculative_decoding/benchmark.py --scenario mixed --trace --format json

# 也支持模块方式运行
python3 -m examples.speculative_decoding.benchmark --test
```

`--tokens` 是最大新增 token 数，包含 EOS、不包含 prompt；`--prompt` 接收 token ID
列表，默认 `[1]`，单独指定 `--prompt` 表示空 prompt。`--seed` 控制模拟 Target。
EOS=0，其他 token ID 为 1..16。输出包含原 prompt 和新增 tokens；EOS 保留在结果中，
生成后立即终止。如果 prompt 最后一个 token 已为 EOS，则不再调用模型。

## 文件分工

| 文件 | 职责 |
| --- | --- |
| `greedy_baseline.py` | 独立的逐 token Target Greedy 参考路径；前缀相关的 ToyTarget |
| `draft_proposer.py` | DraftProposer 在临时候选前缀上自回归产生最多 K 个 token |
| `greedy_verifier.py` | 验证、首错纠正、全匹配 bonus、长度/EOS 控制、统计与 trace |
| `benchmark.py` | K=1/2/4 实验、表格/JSON 输出、12 项 unittest 正确性测试 |
| `README.md` | 算法说明、验收结果与后续扩展路线 |

## 算法：只提交经过 Target 确认的前缀

假设已经提交的前缀为 `P`：

1. Draft 在 `P` 上预测 `d1`，再在 `P + d1` 上预测 `d2`，依次产生最多 K 个候选。
   这些是临时候选，不立即写入最终输出；遇到 draft EOS 停止提案。
2. Target 从 `P` 开始预测下一个 token `t1`。
3. 如果 `d1 == t1`，接受 `d1`，在确认后的 `P + d1` 上验证下一位置。
4. 首次不匹配时，输出 Target 的纠正 token，丢弃该错误候选与全部后续候选。
   从新的已提交前缀重新开始一轮 draft；不能继续使用错误前缀下的后续预测。
5. 整段候选全匹配时，若长度预算仍允许且未遇到 EOS，再输出一个 Target 的
   **bonus token**。下一轮 draft 从包含 bonus 的新前缀开始。
6. 任何已提交 EOS 都立即终止；最后一轮候选数量与 bonus 必须遵守剩余长度预算。

### 两个例子

假设 Target Greedy 的新增序列开头是 `[7, 6, 9, ...]`：

| Draft 候选 | 验证过程 | 本轮输出 |
| --- | --- | --- |
| `[8, 9, 5, 2]` | 第 0 位：8 != 7；不再验证后面三位 | `[7]`，全部候选不接受 |
| `[7, 8, 5, 2]` | 接受 7；第 1 位：8 != 6；丢弃余下后缀 | `[7, 6]`，其中 6 是纠正 token |
| `[7, 6]` | 全部匹配且预算充足 | `[7, 6, 9]`，其中 9 是 bonus |

为什么与 Greedy 相同？用前缀归纳即可：最初前缀相同；每个接受 token 都等于
Target 在同一前缀上的预测，每个纠正/bonus token 直接来自 Target。因此每一步提交
之后，前缀仍与 Target Greedy 相同。EOS 与新增长度规则也相同，所以完整结果相同。
该证明要求 `next_token(prefix)` 是确定性的，且不隐式改变模型状态。

## 模拟模型与三个实验场景

`ToyTarget` 根据**整个 token 前缀**计算下一个 token，不是读取提前写好的目标答案。
长度达到设置的终止位置后返回 EOS。实验中的 draft 使用独立 ToyTarget 对象与
可控错误规则；它们是为了观察算法行为而设计的模拟器，不代表真实模型速度或质量。

- `exact`：Draft 与 Target 规则一致，候选全部匹配。
- `mixed`：根据当前候选前缀，在部分位置主动产生错误，观察接受前缀和丢弃后缀。
- `first_wrong`：Draft 在所有前缀上都预测错误，所以每轮第 0 个候选被拒绝。

这三个场景的 draft 计算不会调用实验中的 Target 对象。真实 Target/Draft 后续均
可接到 `next_token(prefix) -> token_id` 接口；不要求它们共享规则。

## 指标定义

| 指标 | 定义 |
| --- | --- |
| `acceptance_rate` | 接受的 draft token / 所有提出的 draft token；无候选时为 0 |
| `proposed_tokens` / `draft_calls` | Draft 实际提出的 token 数，包含被丢弃的后缀 |
| `accepted_tokens` | 连续通过 Target 验证的 draft token 数；纠正/bonus 不计入 |
| `checked_tokens` | Target 比较过的候选数，等于 accepted + mismatches |
| `mismatches` | 首次不匹配事件数，也是 Target 纠正 token 数 |
| `discarded_tokens` | 未继续检查的候选后缀数；不是额外拒绝事件数 |
| `bonus_tokens` | 整段匹配后追加的 Target token 数 |
| `verification_rounds` | draft → verify 循环次数 |
| `target_calls` | 实际逐 token Target 函数调用次数，包含纠正和 bonus |

可检查的统计恒等式：

```text
proposed = accepted + mismatches + discarded
new_output_tokens = accepted + mismatches + bonus
```

接受率的分母包含浪费的提案，因此 K 增大不保证接受率更高。纠正 token 可能等于
后缀里的某个 token，但那不算接受；必须在正确位置和前缀上通过验证。

### 默认参数实测（2026-10-11）

prompt=`[1]`，seed=7，最多新增 24 tokens（最后一个为 EOS）：

| Draft | K | 与 Greedy 相同 | 接受/提出 | 接受率 | 验证轮数 | Target 调用 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| exact | 1 | 是 | 12/12 | 100.0% | 12 | 24 |
| exact | 2 | 是 | 16/16 | 100.0% | 8 | 24 |
| exact | 4 | 是 | 20/20 | 100.0% | 5 | 24 |
| mixed | 1 | 是 | 10/15 | 66.7% | 15 | 24 |
| mixed | 2 | 是 | 15/19 | 78.9% | 10 | 24 |
| mixed | 4 | 是 | 16/33 | 48.5% | 9 | 24 |
| first_wrong | 1 | 是 | 0/24 | 0.0% | 24 | 24 |
| first_wrong | 2 | 是 | 0/46 | 0.0% | 24 | 24 |
| first_wrong | 4 | 是 | 0/85 | 0.0% | 24 | 24 |

Target Greedy 本身调用 Target 24 次。本阶段所有 verification 都是**串行调用**；
即使验证轮数减少，Target 调用仍为 24，且多了 draft 的工作。
**验证轮数不是 batched Target forward 次数，以上数据不证明实际加速。**
`benchmark.py` 当前统计算法行为，不输出真实模型延迟或吞吐量。

## 今日验收

| 检查项 | 结果与证据 |
| --- | --- |
| 正确性 | 所有默认实验与独立 Target Greedy token 序列完全相同 |
| 首次预测错误 | 每轮只检查第 0 位，提交 Target 纠正 token，不查询被拒绝后缀 |
| K=1/2/4 | exact/mixed/first_wrong 各完成三种 K 对照 |
| 指标统计 | 表格/JSON 输出接受率、验证轮数、Target/Draft 调用及拒绝/丢弃情况 |
| 正确性测试 | 12 项 unittest 通过，包含 200 组随机确定性模型 × 3 种 K 的 600 组序列对照 |
| 工程文档 | 本文解释流程、精确性依据、计数口径、运行方式和限制 |

额外测试覆盖：每个错误位置、候选前缀依赖、全匹配 bonus、拒绝 draft EOS、
接受/纠正/bonus EOS、零生成预算、空 prompt、已结束 prompt、最后不足 K 个位置、
参数错误、调用方 prompt 不被修改。测试使用显式断言检查，不依赖 Python `assert`。

## 后续两阶段

### 第二阶段：实际模型的 Logits 与 KV Cache

- 将真实 target/draft logits 转为 greedy argmax，固定相同 tokenizer、token ID、EOS
  和并列最大值规则。继续逐序列检查与独立 Target Greedy 完全一致。
- 分别管理两个模型的 KV cache、位置与已提交长度。候选拒绝后回滚未提交的 cache，
  纠正/bonus token 按正确前缀更新；覆盖跨轮一致性和 EOS/预算边界。
- 本实验尚未提供 logits、模型加载、tokenizer、GPU 或 KV cache 适配器。

### 第三阶段：批量 Target Verification 与真实延迟

- Target 一次处理候选块，提供各候选位置的预测和全匹配时的额外位置预测。
  找出首个不匹配后只提交连续匹配前缀与纠正 token，丢弃无效后缀。
- 保留今天的串行 verifier 和 Greedy baseline 作为正确性参考。
- 测量真实端到端：draft、target 批量验证、cache 提交/回滚、同步与输出处理；
  对比 K=1/2/4，报告重复测量、p50/p95、吞吐量与接受率。
- 区分串行 token 检查数、真实 Target forward 次数、prefill 和 decode 成本。
  批量验证才可能将多位置验证合并为一次 Target forward；是否更快需实测。

当前仅实现 **greedy 精确匹配**。随机采样的推测解码还需要概率接受和拒绝后的修正
分布，本实现不包含该算法，不能据此声称保持采样分布。

## 学习参考

- [Fast Inference from Transformers via Speculative Decoding（ICML 2023）](https://arxiv.org/abs/2211.17192)：原始推测解码与精确采样方法；本文原型只练习 greedy 分支。
- [Draft & Verify（ACL 2024）](https://aclanthology.org/2024.acl-long.607/)：draft/verification 机制；论文使用自推测方法，本实验用独立模拟 draft。
