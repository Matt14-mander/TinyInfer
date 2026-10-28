# 应用入口

状态：worker.cpp 提供原生 CPU/可选 CUDA 本地协议，generate.py 提供 tokenizer 到
token event 再解码的文本 CLI。运行见 docs/real_model.md；尚无 HTTP server。

worker.cpp 保留进程参数、权重加载和 backend/Engine 初始化。私有 `protocol/` 组件通过
miniprefix_worker_protocol 静态库供 CPU/CUDA worker 复用，不作为公共 SDK：

- command：解析 stdin 行并保持 v3 的校验顺序，无 Engine/runner 副作用。
- response：编码 READY、OK/ERROR、事件、9 位有效数字 logits、STEP_STATS 和 25 字段 STATS。
- diagnostics：管理 `reference/` ID 与 FORWARD/REUSE/CACHE/RELEASE/CLEAR_PREFIX，
  保留创建成功但 append/reuse 失败后的序列状态，与 Engine 请求 ID 隔离。
- session：分发命令，逐行捕获异常、编码错误并 flush；QUIT 成功时立即返回，EOF
  不额外提交 shutdown，资源清理由 Engine 析构负责。

保持现有协议宽松规则：忽略尾随字段，沿用流式整数解析；CACHE/RELEASE/CANCEL 的 ID
校验仍由下游处理。事件不额外编码 error_message，错误文本、字段顺序、握手版本不变。
Engine 进入 failed 后，FORWARD/REUSE/CACHE/RELEASE/CLEAR_PREFIX 也拒绝执行，避免
绕过 Engine 使用同一故障后端；STATS/STEP_STATS 仍可读取，QUIT 可重试 shutdown 清理。
解析/编码/状态轨迹由 miniprefix_worker_protocol_tests 验证。真实 worker 的完整输出
差分证据见 [CPU](../docs/acceptance/worker-protocol-cpu.json) 与
[CUDA](../docs/acceptance/worker-protocol-cuda.json)，使用 tools/verify_worker_protocol.py 复现。

首批入口为本地单模型 CLI：显式配置/权重/tokenizer 路径，Greedy 文本或 token 输入，
token 事件流与独立统计输出。CLI 调用同一个 Engine，不自行做 batching/KV 分配。
网络协议、鉴权、多用户服务和部署在 M4 基线之后另行规划。
