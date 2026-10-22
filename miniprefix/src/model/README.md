# Model 与 ModelRunner

状态：规划目录，尚无真实模型加载器。

负责 ModelConfig、权重加载/映射、tokenizer adapter 和模型 forward。配置包含层数、
hidden/FFN 维度、query/KV head 数、head dimension、词表、上下文和位置编码参数；
加载时验证形状、GQA 分组关系、精度与 backend 能力。

首批交付：配置驱动的微型 CPU runner，随后一种真实小型模型架构与 tokenizer。
runner 接受 token ID/绝对位置/sequence handle 与 KV view，返回 logits/完成句柄。
权重和 tokenizer 文件通过显式路径提供，不在代码中隐式下载，也不把生成采样混入 forward。
