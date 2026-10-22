# 应用入口

状态：规划目录，尚无可运行 CLI 或 HTTP server。

首批入口为本地单模型 CLI：显式配置/权重/tokenizer 路径，Greedy 文本或 token 输入，
token 事件流与独立统计输出。CLI 调用同一个 Engine，不自行做 batching/KV 分配。
网络协议、鉴权、多用户服务和部署在 M4 基线之后另行规划。
