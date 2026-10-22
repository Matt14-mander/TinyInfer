# 配置

状态：配置字段约定，尚无配置解析器或可执行示例配置。

分为 ModelConfig 与 EngineConfig：

- 模型：架构、权重/tokenizer revision 与路径、层/head/维度、位置编码、上下文和 dtype。
- 引擎：backend/device、batch 上限、token budget、prefill chunk、KV block size/容量、
  prefix cache 开关/容量策略、抢占策略、观测选项。

配置加载先校验语义/形状/设备能力，未知或不支持的选项明确失败，不静默回退。
模型路径由本地运行者提供；不提交权重、大型 tokenizer 资产或机器专属绝对路径。
解析格式与首个模型选型在 M1/M2 实现时锁定，当前不伪造一个看似能运行的 JSON。
