# 真实模型推理闭环

2026-10-23：原生 CPU/CUDA runner 和本地生成入口已运行 SmolLM2-135M-Instruct。
模型为 30 层、hidden=576、9 query heads / 3 KV heads、head dimension=64、词表 49152，
使用 RoPE、RMSNorm、SwiGLU 和 tied embeddings。权重从 BF16 checkpoint 转为 FP32，
CPU、CUDA 和独立 Transformers 参考都执行 FP32。固定 revision：
`12fd25f77366fa6b3b4b768ec3050bf629380bac`。
来源：[模型卡](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)，Apache-2.0。

## 已实现与当前边界

- ModelConfig 与严格的 MPFX0001 权重加载，校验名称、形状、字节数和有限数值。
- CPU 使用 TinyInfer Tensor、持久 RHS packing/MatMul 和模型专用参考计算。
- CUDA 的 embedding、投影、RMSNorm、RoPE、GQA attention、SwiGLU、FFN、head 都在 GPU。
  权重上传一次，分页 K/V 常驻设备，warm append 上传输入 token 和新增页号、下载 logits。
- C++ Engine 管理等待/活跃队列，轮间补位、decode 优先、chunked prefill、token budget、
  Greedy、EOS/长度结束、排队/活跃取消与失败终态。Python 只做 tokenizer、IPC 和展示。
- 原生 runner 每次处理一个 sequence 的 chunk，同轮请求依次执行；没有跨请求矩阵合并。
- CPU/CUDA KV 使用共享物理 block 池与序列页表，直接分页寻址，见 [分页 KV](paged_kv.md)。
  KVManager 统一持有序列 metadata、页面引用、追加事务与公共统计；backend 保留 payload、
  设备页表和同步。接口为 include/miniprefix/kv_manager.h，实现位于 src/kv。
  2026-10-25 已接入 [完整页面精确前缀共享](prefix_cache.md)，默认关闭；部分页 COW、
  抢占重算尚未迁入真实模型路径，不能用旧示例的 feature flag 宣称支持。

CUDA 使用 MiniPrefix 私有 RAII allocation、cuBLAS handle 和一个 stream，复用项目既有
CUDA SDK 配置。core 当前没有模型所需 RMSNorm/RoPE/GQA 执行接口，因此暂未统一到
CudaDeviceRuntime；这是与目标架构的待收敛差异。首版每次 append 同步完成，失败不回退 CPU。
绑定仍为本地文本 IPC，token event 在 STEP 整体完成后交付；不提供 HTTP 或异步流水。

## 准备与运行

所有下载都是显式工具操作；CMake/生成入口不会联网或安装依赖。工作区虚拟环境和模型
放 `build-miniprefix-env/` 与 `build-miniprefix-models/`，两者已被根 Git ignore 排除。

```powershell
python -m venv build-miniprefix-env
$py = 'build-miniprefix-env/Scripts/python.exe'
& $py -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r miniprefix/tools/requirements-reference.txt
$env:HF_HOME = "$PWD/build-miniprefix-models/hf-cache"
& $py miniprefix/tools/prepare_model.py --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --output build-miniprefix-models/smollm2-135m-instruct
```

原始 safetensors、tokenizer、配置、模型卡和校验和 manifest 保留在模型目录。导出的
model.mpf 为 538076734 bytes；此首版格式为 little-endian FP32、单个 safetensors 源，
支持 bias-free、非 interleaved、无 rope scaling 的 Llama 结构，不声称兼容全部 HF 模型。

在已配置的 ROG CUDA 开发环境（沿用 tools/build_cuda_rog.ps1）中：

```powershell
cmake -S . -B build-phase54-cuda -DTINYINFER_BUILD_MINIPREFIX=ON
cmake --build build-phase54-cuda --target miniprefix_worker miniprefix_cuda_worker --parallel 4
# CUDA bin 必须在 PATH 中；也可用 miniprefix_worker + --backend cpu。
& $py -m miniprefix.apps.generate --worker build-phase54-cuda/miniprefix/miniprefix_cuda_worker.exe --model-dir build-miniprefix-models/smollm2-135m-instruct --backend cuda --prompt 'What is the capital of France?' --max-new-tokens 16 --context 128
```

实际 CPU/CUDA 均生成 `The capital of France is Paris.`，token 为
`[504,3575,282,4649,314,7042,30,2]`，自然 EOS 结束。
JSON 输出加 `--json`；`--raw` 跳过 chat template。CLI 统计模型准备与引擎生成时间，
不要把单次展示运行当作稳定性能结果；性能基线见 [协议](baseline.md)。

## 独立数值与生命周期验收

```powershell
& $py miniprefix/tools/verify_real_model.py --model-dir build-miniprefix-models/smollm2-135m-instruct --cpu-worker build-phase54-cuda/miniprefix/miniprefix_worker.exe --cuda-worker build-phase54-cuda/miniprefix/miniprefix_cuda_worker.exe --output benchmark-results/miniprefix/real-model-verification.json
```

参考为 Transformers 4.46.3 / PyTorch 2.5.1 CPU FP32 eager，每次从完整前缀重算，
不共享 native 模型代码或 KV。当前 54 次 logits 检查共覆盖 2654208 个值，固定容差
`2e-3 + 2e-4*abs(reference)`，要求 argmax 完全相同；当前最大绝对误差约 1.14e-4。
覆盖完整/分块 prefill、增量 decode、Unicode tokenization、两个 backend、自然/强制 EOS、
固定长度、轮间调度、排队/活跃取消、非法输入与最终 KV 归零；另有跨页、碎片页复用、
失败重试和池耗尽清理检查。历史 dense 验收 JSON 保留 42 次检查，属于旧二进制。

9 次实际 CUDA warm decode 检查：每次输入 H2D 增加 4 bytes，KV/权重 H2D 不增加，
device allocation 不增加。检查的是 native 拥有的资源，不能据此声称 cuBLAS/驱动没有
内部内存开销。验收输出、模型/worker hash 与基线一起保留在文档 acceptance 目录。
