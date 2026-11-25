# SmolLM2-135M-Instruct FP32 CPU decode

CPU decode packs one token per scheduled request for Q/K/V, output projection,
gate/up/down and LM head. Attention reads each sequence's paged history and
its staged new K/V. The engine appends staged rows in order through ordinary
KV transactions. A single decode row and prefill use the serial path.

Linux x86-64 AMD EPYC 9V74, GCC 14, Release, native architecture flag off.
Use the pinned exporter from the repository root and run:

```sh
python3 miniprefix/tools/prepare_model.py --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --output weights
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_MINIPREFIX=ON -DTINYINFER_ENABLE_CUDA=OFF
cmake --build build --target miniprefix_worker -j4
python3 miniprefix/benchmarks/decode_batch.py build/miniprefix/miniprefix_worker weights/model.mpf --pairs 2
```

The script checks the FP32 export digest. Each sample starts a fresh worker,
warms weights, then submits all requests before the first step. Both modes use
identical prompts of four tokens, six output tokens, chunk 8, budget 32,
context 64, 128 four-token KV pages, disabled cache and disabled EOS. Only
the decode mode changes. Pairs alternate execution order and assert identical
per-request token streams and termination. Throughput counts all output tokens
from first submission through completion, including IPC. TTFT uses the common
burst arrival time. TPOT averages each request's first-to-last interval divided
by five. Peak RSS is the worker's Linux VmHWM since process start.

| Concurrency | Output tok/s | P95 TTFT ms | Mean TPOT ms | Peak RSS MiB |
|---:|---:|---:|---:|---:|
| 1 | 10.00 / 10.11 | 168 / 172 | 86 / 85 | 1053 / 1053 |
| 2 | 10.10 / 11.26 | 335 / 356 | 171 / 144 | 1053 / 1053 |
| 4 | 9.83 / 13.29 | 723 / 676 | 349 / 226 | 1053 / 1054 |
| 8 | 9.78 / 14.43 | 1396 / 1375 | 709 / 391 | 1053 / 1055 |

Values are serial / batched. Throughput is the median of two runs per mode.
The nearest-rank P95 over 2–16 requests is the maximum for these small samples.
At concurrency 1 both modes use the same forward path. Measured throughput
increased 1%, while TTFT rose about 4 ms. At 8, throughput increased 48%
with about 2 MiB additional peak RSS. These results cover short homogeneous
prompts on this host. [Raw samples](cpu-decode-batch.json).
