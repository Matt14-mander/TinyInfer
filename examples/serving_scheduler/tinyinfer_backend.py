"""Actual local CPU attention/logits/KV via a persistent TinyInfer C++ worker."""
from pathlib import Path
import subprocess
import time
try:
    from .backend import StepResult
except ImportError:
    from backend import StepResult


class TinyInferBackend:
    real_backend = True
    hardware_batched = True  # CPU tensor batching; ragged attention is per query.

    def __init__(self, executable, block_size=1, kv_layout="contiguous",
                 kv_capacity_blocks=None, attention_mode="direct", prefix_sharing=False):
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block size must be positive")
        if kv_layout not in ("contiguous", "paged"):
            raise ValueError("kv_layout must be contiguous or paged")
        if attention_mode not in ("direct", "gather", "cuda") or (attention_mode != "direct" and kv_layout != "paged"):
            raise ValueError("gather/CUDA attention requires paged KV")
        if type(prefix_sharing) is not bool or (prefix_sharing and kv_layout != "paged"):
            raise ValueError("prefix sharing requires paged KV")
        if kv_layout == "paged":
            if type(kv_capacity_blocks) is not int or kv_capacity_blocks < 0:
                raise ValueError("paged KV requires a nonnegative integer capacity")
            if block_size > 4096:
                raise ValueError("paged block size must be at most 4096")
        elif kv_capacity_blocks is not None:
            raise ValueError("physical capacity is only configured for paged KV")
        self.block_size = block_size
        self.kv_layout, self.kv_capacity_blocks = kv_layout, kv_capacity_blocks
        self.attention_mode = attention_mode
        self.prefix_sharing = prefix_sharing
        self.handles, self.outputs = {}, {}
        self.process = subprocess.Popen([str(Path(executable).resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        if self.process.stdout.readline().strip() != "READY 16 32 4096 BATCHED 1 PAGED 2":
            self.close()
            raise RuntimeError("invalid worker handshake")
        if kv_layout == "paged":
            try:
                self._command(["CONFIG_PAGED", kv_capacity_blocks, block_size, {"direct": 0, "gather": 1, "cuda": 2}[attention_mode]])
                if prefix_sharing:
                    self._command(["PREFIX_SHARING", 1])
            except BaseException:
                self.close()
                raise

    def _command(self, values, convert=int):
        if self.process.poll() is not None:
            raise RuntimeError("TinyInfer worker exited")
        try:
            self.process.stdin.write(" ".join(map(str, values)) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline().strip()
        except BaseException:
            # A cancelled/failed read loses framing. Stop and join the CPU worker
            # before any caller can release its logical KV reservation.
            self.close()
            raise
        if line.startswith("ERROR "):
            raise RuntimeError("TinyInfer worker error: " + line)
        if not line or line.split()[0] != "OK":
            self.close()
            raise RuntimeError("invalid TinyInfer worker response: " + line)
        try:
            return [convert(x) for x in line.split()[1:]]
        except ValueError:
            self.close()
            raise

    @staticmethod
    def prompt(request):
        seed = sum((i + 1) * ord(c) for i, c in enumerate(request.request_id))
        return [(seed + i * 7) % 32 for i in range(request.prompt_tokens)]

    def step(self, batch):
        start = time.perf_counter_ns()
        mixed = any(getattr(item, "is_replay", False) for item in batch)
        command = ["MIXED" if mixed else "STEP", len(batch)]
        for item in batch:
            r = item.request
            key = r.request_id
            if key not in self.handles:
                self.handles[key] = len(self.handles) + 1
                self.outputs[key] = []
            capacity = ((r.prompt_tokens + r.output_tokens + self.block_size - 1) // self.block_size) * self.block_size
            replay = getattr(item, "is_replay", False)
            if mixed:
                command.append(int(replay))
                if replay:
                    command.extend([r.prompt_tokens, r.output_tokens, len(self.outputs[key]),
                                    self.outputs[key][-1] if self.outputs[key] else -1])
            if replay:
                prefix = self.prompt(r) + self.outputs[key][:-1]
                tokens = prefix[item.prefill_start:item.prefill_start + item.input_tokens]
            else:
                tokens = (self.prompt(r)[item.prefill_start:item.prefill_start + item.input_tokens]
                          if item.is_prefill else [self.outputs[key][-1]])
            command.extend([self.handles[key], 0 if replay else item.generated, item.prefill_start, item.input_tokens,
                            item.prefix_length if replay else r.prompt_tokens, 1 if replay else r.output_tokens, capacity] + tokens)
        result = self._command(command)
        if len(result) != len(batch):
            raise RuntimeError("worker result arity")
        for item, token in zip(batch, result):
            if token >= 0:
                self.outputs[item.request.request_id].append(token)
        return StepResult((time.perf_counter_ns() - start) / 1000000.0,
                          [None if t == -1 else t for t in result])

    def can_step(self, batch):
        if not batch:
            return True
        command = ["CAN_APPEND", len(batch)]
        for item in batch:
            key = item.request.request_id
            if key not in self.handles:
                self.handles[key] = len(self.handles) + 1
                self.outputs[key] = []
            length = (item.prefill_start + item.input_tokens
                      if item.is_prefill or getattr(item, "is_replay", False)
                      else item.request.prompt_tokens + item.generated)
            command.extend([self.handles[key], length])
        values = self._command(command)
        if values not in ([0], [1]):
            raise RuntimeError("invalid capacity plan response")
        return values == [1]

    def reference(self, request):
        return self._command(["REFERENCE", request.prompt_tokens, request.output_tokens] + self.prompt(request))

    def reference_logits(self, prefix):
        """Diagnostic full-prefix serial oracle, outside benchmark timing."""
        return self._command(["REFERENCE_LOGITS", len(prefix)] + list(prefix), float)

    def last_step_logits(self):
        """Final input row per work item, including unfinished prefill chunks."""
        values = self._command(["LOGITS"], float)
        if len(values) % 32:
            raise RuntimeError("worker logits arity")
        return [values[i:i + 32] for i in range(0, len(values), 32)]

    def block_table(self, request_id):
        """Diagnostic logical-to-physical block mapping; query outside timing."""
        if self.kv_layout != "paged":
            raise ValueError("block table requires paged KV")
        return self._command(["BLOCK_TABLE", self.handles[request_id]])

    def synchronize(self):
        # CPU worker replies only after computation and token publication finish.
        pass

    def release(self, request_id):
        self._command(["RELEASE", self.handles[request_id]])

    def reset(self):
        self._command(["RESET"])
        self.handles.clear()
        self.outputs.clear()

    def clear(self):
        try:
            self._command(["CLEAR"])
        except BaseException:
            self.close()
            raise

    def stats(self):
        values = self._command(["STATS"])
        names = ("live_requests", "live_kv_bytes", "peak_kv_bytes", "peak_cached_tokens", "forward_tokens",
                 "batch_steps", "linear_matmul_calls", "linear_matmul_rows", "peak_batch_requests", "peak_batch_tokens",
                 "pool_capacity_blocks", "live_physical_blocks", "peak_physical_blocks", "pool_kv_bytes",
                 "free_physical_blocks", "paged_attention_queries", "gather_attention_queries",
                 "prefix_cache_hits", "prefix_reused_tokens", "cow_block_copies", "prefix_cache_entries",
                 "prefix_cache_evictions", "recompute_tokens", "cuda_attention_queries", "cuda_kernel_launches", "cuda_uploaded_bytes")
        if len(values) != len(names):
            raise RuntimeError("worker stats arity")
        return dict(zip(names, values))

    def close(self):
        if getattr(self, "process", None) is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()

    def __enter__(self): return self
    def __exit__(self, exc_type, *args):
        try:
            if exc_type is None and self.process.poll() is None:
                self._command(["QUIT"])
                code = self.process.wait(timeout=5)
                if code:
                    raise RuntimeError("worker exited unsuccessfully: %d" % code)
        finally:
            self.close()
