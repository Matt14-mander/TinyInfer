"""Local native engine client; Python does no model forward or scheduling."""
from pathlib import Path
import subprocess
import time


class NativeEngine:
    def __init__(self, worker, model, backend="cpu", context=512, batch_size=2, token_budget=32, chunk_size=16,
                 block_tokens=16, kv_blocks=128, prefix_cache_blocks=0):
        self.process = subprocess.Popen([str(Path(worker).resolve()), str(Path(model).resolve()), backend,
                                         str(context), str(batch_size), str(token_budget), str(chunk_size), str(block_tokens), str(kv_blocks), str(prefix_cache_blocks)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.ids, self.diagnostics, self.pending = {}, {}, set()
        ready = self.process.stdout.readline().strip().split()
        if len(ready) != 7 or ready[:3] != ["READY", "MINIPREFIX", "3"]:
            self.close(force=True)
            raise RuntimeError("native worker failed to initialize: " + getattr(self, "last_stderr", ""))
        self.vocabulary, self.eos, self.context = map(int, ready[3:6])
        self.backend = ready[6]

    def _command(self, fields, convert=int):
        if self.process.poll() is not None:
            raise RuntimeError("native worker exited")
        try:
            self.process.stdin.write(" ".join(map(str, fields)) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline().strip()
        except BaseException:
            self.close(force=True)
            raise
        if line.startswith("ERROR "):
            raise RuntimeError(line[6:])
        if not line or line.split()[0] != "OK":
            self.close(force=True)
            raise RuntimeError("invalid native worker response")
        try:
            return [convert(value) for value in line.split()[1:]]
        except ValueError:
            self.close(force=True)
            raise RuntimeError("invalid native payload")

    def submit(self, request_id, prompt_tokens, max_new_tokens=32, eos_token_id="model"):
        if request_id in self.ids:
            raise ValueError("duplicate request ID")
        prompt_tokens = list(prompt_tokens)
        if (not isinstance(request_id, str) or not request_id or type(max_new_tokens) is not int
                or any(type(t) is not int for t in prompt_tokens)):
            raise ValueError("invalid request")
        handle = len(self.ids) + 1
        eos = self.eos if eos_token_id == "model" else -1 if eos_token_id is None else eos_token_id
        self._command(["SUBMIT", handle, max_new_tokens, eos, len(prompt_tokens)] + prompt_tokens)
        self.ids[request_id] = handle
        self.pending.add(request_id)

    def _events(self, values):
        if not values or len(values) != 1 + values[0] * 3:
            raise RuntimeError("invalid native event count")
        reverse = {v: k for k, v in self.ids.items()}
        result = []
        for i in range(values[0]):
            handle, token, reason = values[1 + i * 3:4 + i * 3]
            if handle not in reverse or reason not in (-1, 0, 1, 2, 3):
                raise RuntimeError("invalid native event identity/reason")
            key = reverse[handle]
            event = {"request_id": key, "token_id": None if token == -1 else token,
                     "finish_reason": None if reason == -1 else ("eos", "length", "cancelled", "error")[reason],
                     "received_ns": time.perf_counter_ns()}
            if reason != -1:
                self.pending.discard(key)
            result.append(event)
        return result

    def step(self):
        return self._events(self._command(["STEP"]))

    def last_step_stats(self):
        values = self._command(["STEP_STATS"])
        if len(values) != 3:
            raise RuntimeError("invalid engine step stats")
        return dict(zip(("prefix_reused_tokens", "prefill_tokens", "decode_tokens"), values))

    def cancel(self, request_id):
        return self._events(self._command(["CANCEL", self.ids[request_id]]))

    def events(self):
        while self.pending:
            yield from self.step()

    def forward(self, sequence_id, token_ids):
        if sequence_id not in self.diagnostics:
            self.diagnostics[sequence_id] = len(self.diagnostics) + 1
        tokens = list(token_ids)
        values = self._command(["FORWARD", self.diagnostics[sequence_id], len(tokens)] + tokens, float)
        if len(values) != self.vocabulary:
            raise RuntimeError("invalid logits shape")
        return values

    def release(self, sequence_id):
        self._command(["RELEASE", self.diagnostics[sequence_id]])

    def reuse_prefix(self, sequence_id, prompt_tokens):
        if sequence_id not in self.diagnostics:
            self.diagnostics[sequence_id] = len(self.diagnostics) + 1
        tokens = list(prompt_tokens)
        return self._command(["REUSE", self.diagnostics[sequence_id], len(tokens)] + tokens)[0]

    def cache_prefix(self, sequence_id):
        self._command(["CACHE", self.diagnostics[sequence_id]])

    def clear_prefix_cache(self):
        self._command(["CLEAR_PREFIX"])

    def stats(self):
        names = ("live_sequences", "live_kv_bytes", "peak_kv_bytes", "forward_tokens", "device_weight_bytes",
            "weight_uploaded_bytes", "token_uploaded_bytes", "kv_uploaded_bytes", "logits_downloaded_bytes", "device_allocations", "device_workspace_bytes",
            "kv_block_tokens", "kv_total_blocks", "kv_live_blocks", "kv_peak_blocks", "kv_reserved_bytes",
            "block_table_uploaded_bytes", "block_table_device_bytes", "prefix_cache_capacity", "kv_cached_blocks",
            "kv_cache_only_blocks", "prefix_hits", "prefix_misses", "prefix_reused_tokens", "prefix_evictions")
        values = self._command(["STATS"])
        if len(values) != len(names):
            raise RuntimeError("invalid native stats")
        return dict(zip(names, values))

    def close(self, force=False):
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream and not stream.closed:
                    if stream is process.stderr:
                        self.last_stderr = stream.read()
                    stream.close()
            return
        try:
            if not force:
                self._command(["QUIT"])
            else:
                process.terminate()
            process.wait(timeout=10)
        except BaseException:
            process.kill()
            process.wait()
            if not force:
                raise
        finally:
            if not process.stderr.closed:
                self.last_stderr = process.stderr.read()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream and not stream.closed:
                    stream.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(force=kind is not None)
