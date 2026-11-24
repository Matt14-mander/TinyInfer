"""Local native engine client; Python does no model forward or scheduling."""
from pathlib import Path
import subprocess
import time
from collections import deque


class NativeEngine:
    def __init__(self, worker, model, backend="cpu", context=512, batch_size=2, token_budget=32, chunk_size=16,
                 block_tokens=16, kv_blocks=128, prefix_cache_blocks=0, max_waiting_requests=1024, max_event_slots=65536,
                 history_capacity=4096, history_ttl_ms=300000):
        limits=(max_waiting_requests,max_event_slots,history_capacity,history_ttl_ms)
        if any(type(n) is not int or n<0 or n>2147483647 for n in limits) or not all(limits[:2]):
            raise ValueError("invalid lifecycle limits")
        self.history_capacity,self.history_ttl_ms=history_capacity,history_ttl_ms
        self.history=deque();self.next_handle=1
        self.process = subprocess.Popen([str(Path(worker).resolve()), str(Path(model).resolve()), backend,
                                         str(context), str(batch_size), str(token_budget), str(chunk_size), str(block_tokens), str(kv_blocks), str(prefix_cache_blocks), *(map(str,limits) if limits!=(1024,65536,4096,300000) else ())],
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

    def _prune_history(self):
        now=time.monotonic()
        while self.history and (len(self.history)>self.history_capacity or now>=self.history[0][1]):
            key,_=self.history.popleft();self.ids.pop(key,None)

    def submit(self, request_id, prompt_tokens, max_new_tokens=32, eos_token_id="model", timeout_ms=0):
        self._prune_history()
        if request_id in self.ids:
            raise ValueError("duplicate request ID")
        prompt_tokens = list(prompt_tokens)
        if (not isinstance(request_id, str) or not request_id or type(max_new_tokens) is not int
                or any(type(t) is not int for t in prompt_tokens)):
            raise ValueError("invalid request")
        if type(timeout_ms) is not int or timeout_ms<0 or timeout_ms>2147483647:
            raise ValueError("invalid timeout")
        handle = self.next_handle
        if handle>2147483647:raise RuntimeError("native handle space exhausted; open a new session")
        eos = self.eos if eos_token_id == "model" else -1 if eos_token_id is None else eos_token_id
        fields=(["SUBMIT_TIMEOUT",handle,max_new_tokens,eos,timeout_ms] if timeout_ms else ["SUBMIT",handle,max_new_tokens,eos])
        self._command(fields+[len(prompt_tokens)]+prompt_tokens)
        self.next_handle+=1
        self.ids[request_id] = handle
        self.pending.add(request_id)

    def _events(self, values):
        if not values or len(values) != 1 + values[0] * 3:
            raise RuntimeError("invalid native event count")
        reverse = {v: k for k, v in self.ids.items()}
        result = []
        for i in range(values[0]):
            handle, token, reason = values[1 + i * 3:4 + i * 3]
            if handle not in reverse or reason not in (-1, 0, 1, 2, 3, 4):
                raise RuntimeError("invalid native event identity/reason")
            key = reverse[handle]
            event = {"request_id": key, "token_id": None if token == -1 else token,
                     "finish_reason": None if reason == -1 else ("eos", "length", "cancelled", "error", "timeout")[reason],
                     "received_ns": time.perf_counter_ns()}
            if reason != -1:
                self.pending.discard(key)
                self.history.append((key,time.monotonic()+self.history_ttl_ms/1000))
            result.append(event)
        self._prune_history()
        return result

    def step(self):
        return self._events(self._command(["STEP"]))

    def lifecycle_stats(self):
        values=self._command(["LIFECYCLE_STATS"])
        if len(values)!=6:raise RuntimeError("invalid lifecycle stats")
        return dict(zip(("waiting","active","buffered_events","event_slots","tracked_ids","history_ids"),values))

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

    def set_cost_timing(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        self._command(["COST_TIMING", int(enabled)])

    def set_execution_timing(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        self._command(["EXEC_TIMING", int(enabled)])

    def execution_stats(self):
        fields = ("forwards", "forward_download_ns", "host_forward_ns", "stream_forward_ns", "admission_plan_ns")
        values = self._command(["EXEC_STATS"])
        if len(fields) != len(values):
            raise RuntimeError("invalid execution stats")
        return dict(zip(fields, values))

    def cost_stats(self):
        names = ("lookup_calls", "lookup_ns", "publish_calls", "publish_ns",
                 "cow_pages", "cow_bytes", "cow_copy_ns", "cow_timed_copies")
        values = self._command(["COST_STATS"])
        if len(values) != len(names):
            raise RuntimeError("invalid cost stats")
        return dict(zip(names, values))

    def decode_batch_stats(self):
        values = self._command(["DECODE_STATS"])
        if len(values) != 3:
            raise RuntimeError("invalid decode batch stats")
        return dict(zip(("batches", "rows", "fallbacks"), values))

    def prefill_batch_stats(self):
        names = ("prefill_batches", "mixed_batches", "prefill_tokens", "decode_tokens", "fallbacks")
        values = self._command(["PREFILL_STATS"])
        if len(values) != len(names):
            raise RuntimeError("invalid prefill batch stats")
        return dict(zip(names, values))

    def sampled_id_downloaded_bytes(self):
        values = self._command(["SAMPLE_BYTES"])
        if len(values) != 1:
            raise RuntimeError("invalid sampled ID byte count")
        return values[0]

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
