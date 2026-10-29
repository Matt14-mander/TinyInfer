# 性能与前缀复用实验

状态：baseline.py 已建立 SmolLM2-135M-Instruct FP32 原生 CPU/CUDA 首个端到端基线，
见 docs/baseline.md 的测量协议与 docs/acceptance/baseline-smollm2-fp32.md 的冻结结果。
prefix_benefit.py 提供真实模型八请求负载的 off/cold/warm 独立进程 ablation，覆盖
共享 96-token 根、半数共享和零共享控制；协议见 ../docs/prefix_benefit.md。

保留 cold start 与 warm serving 两种窗口。记录模型/权重/tokenizer revision、精度、
硬件/驱动、构建、warmup、到达过程、输入/输出长度、原始时间戳与重复进程次数。

首批工作负载：无共享、共享系统提示、不同共享比例、多轮对话、分叉前缀、混合长度、
容量压力/抢占；同负载比较缓存关闭/开启，逐请求输出校验后再讨论性能。
报告 TTFT、TPOT/ITL、吞吐、排队、重算、实际跳过 token、缓存索引开销、显存/传输。

原始输出放仓库根目录 benchmark-results/miniprefix/（已被根 .gitignore 忽略），
经审阅的摘要与验证元数据再放文档目录。模拟轨迹、旧混合后端和新 GPU 常驻模型的数据分开。

`stress.cpp` / `stress.py` 提供同一 Engine 的长期混合压力测试及原生调用级耗时归因，
包含 watchdog、终态/输出/容量/回收检查及 Windows 进程内存采样。
协议和口径见 [长期压力测试](../docs/long_running_stress.md)。
