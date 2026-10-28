# 模型与参考数据准备

状态：prepare_model.py 显式下载并固定 revision、导出 FP32 MPF 和 manifest；
verify_real_model.py 提供独立 Transformers full-prefix 数值/生命周期验收。
verify_prefix_cache.py 验证真实 CPU/CUDA 完整页面前缀共享、清理和淘汰，并测量关闭/冷/热
缓存的五组暖进程 paired flows；原始证据与口径见 ../docs/prefix_cache.md。
verify_incremental_prefill.py 验证默认页长下长 suffix、budget/chunk、逐步 logits、
decode 混合与取消；只验收正确性及计算量，不报告性能。
verify_kv_manager_refactor.py 对保存的重构前 worker 与新 worker 做精确差分：FP32 logits、
事件、逐轮计数、错误及全部 STATS 字段，不把相近输出当作行为相同。
verify_worker_protocol.py 对保存的新旧 worker 输入相同合法/非法命令混合流和 EOF 场景，
比较完整 stdout/stderr 字节、退出码，保存 worker/model 与输出 SHA256；无需 Python ML 包。
示例：`python miniprefix/tools/verify_worker_protocol.py --before-worker OLD --worker NEW
--model MODEL.mpf --backend cpu --output result.json`（CUDA 将 backend 改为 cuda）。
requirements-reference.txt 固定参考依赖，CPU PyTorch 单独安装于工作区虚拟环境。

后续提供显式触发的权重检查/转换、tokenizer 验证、独立框架参考导出和 fixture 准备。
记录来源、revision、校验和、许可证与工具版本；参考数据与性能计时分离。
依赖按工具单独声明，不让配置/构建 TinyInfer 默认安装 Python 包或下载模型。
