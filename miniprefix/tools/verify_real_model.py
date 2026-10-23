"""Independent Transformers FP32/eager checks for real native model runners.

All checks are outside performance windows. No CPU fallback counts as CUDA
acceptance. Model/tokenizer must already exist locally.
"""
import argparse
from datetime import date
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(model_dir, workers, output=None):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    root = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, torch_dtype=torch.float32,
                                               attn_implementation="eager").eval()
    records, transcripts, residency, paged_checks = [], [], [], []
    def reference(prefix):
        with torch.inference_mode():
            ids = torch.tensor([prefix], dtype=torch.long)
            return model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].numpy()
    def close(actual, expected, label):
        a = np.asarray(actual, dtype=np.float32)
        difference = np.abs(a - expected)
        if (a.shape != expected.shape or not np.isfinite(a).all()
                or not np.all(difference <= 2e-3 + 2e-4 * np.abs(expected))):
            raise AssertionError("logits differ: " + label + " max_abs=" + str(float(difference.max())))
        if int(a.argmax()) != int(expected.argmax()):
            raise AssertionError("Greedy mismatch: " + label)
        records.append({"case": label, "checked_logits": len(a), "max_abs_error": float(difference.max()),
                        "mean_abs_error": float(difference.mean()), "greedy_token": int(a.argmax())})
    for backend, worker in workers.items():
        with NativeEngine(worker, root / "model.mpf", backend, 256, 2, 8, 3) as engine:
            for case, prefix in enumerate(([1, 504, 3575, 30],
                                           tokenizer.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}], add_generation_prompt=True),
                                           tokenizer.encode("Hello 世界! Count: 1, 2, 3,", add_special_tokens=False))):
                sequence = backend + "-chunks-" + str(case)
                cuts = sorted(set((1, max(2, len(prefix) // 2), len(prefix))))
                previous = 0
                for end in cuts:
                    close(engine.forward(sequence, prefix[previous:end]), reference(prefix[:end]), backend + "/chunk/" + str(case) + "/" + str(end))
                    previous = end
                full_id = sequence + "-full"
                close(engine.forward(full_id, prefix), reference(prefix), backend + "/full/" + str(case))
                engine.release(full_id)
                growing = list(prefix)
                for step in range(3):
                    token = int(reference(growing).argmax())
                    before = engine.stats()
                    growing.append(token)
                    actual = engine.forward(sequence, [token])
                    after = engine.stats()
                    close(actual, reference(growing), backend + "/decode/" + str(case) + "/" + str(step))
                    if backend == "cuda":
                        if (after["kv_uploaded_bytes"] != 0 or before["weight_uploaded_bytes"] != after["weight_uploaded_bytes"]
                                or after["token_uploaded_bytes"] - before["token_uploaded_bytes"] != 4
                                or before["device_allocations"] != after["device_allocations"]):
                            raise AssertionError("warm CUDA decode violates residency/allocation contract")
                        expected_table_bytes=4 if (len(growing)-1)%before["kv_block_tokens"]==0 else 0
                        if after["block_table_uploaded_bytes"]-before["block_table_uploaded_bytes"]!=expected_table_bytes:
                            raise AssertionError("decode block table uploaded more than new entries")
                        residency.append({"case": case, "step": step, "token_h2d_delta": 4, "kv_h2d_delta": 0,
                                          "weight_h2d_delta": 0, "allocation_delta": 0,"block_table_h2d_delta":expected_table_bytes})
                engine.release(sequence)
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}], add_generation_prompt=True)
            expected, growing = [], list(prompt)
            for _ in range(16):
                token = int(reference(growing).argmax());expected.append(token);growing.append(token)
                if token == tokenizer.eos_token_id:
                    break
            engine.submit("natural-eos", prompt, 16)
            events = list(engine.events())
            actual = [e["token_id"] for e in events if e["token_id"] is not None]
            if actual != expected or events[-1]["finish_reason"] != "eos":
                raise AssertionError("natural EOS generation mismatch")
            transcripts.append({"backend": backend, "prompt_tokens": prompt, "output_tokens": actual,
                                "text": tokenizer.decode(actual, skip_special_tokens=True), "finish_reason": "eos"})
            engine.submit("force-eos", prompt, 5, expected[0])
            forced = list(engine.events())
            if len(forced) != 1 or forced[0]["token_id"] != expected[0] or forced[0]["finish_reason"] != "eos":
                raise AssertionError("forced EOS boundary")
            engine.submit("length", prompt, 3, None)
            engine.submit("cancel-active", prompt, 8, None)
            engine.submit("cancel-waiting", prompt, 8, None)
            if engine.cancel("cancel-waiting")[0]["finish_reason"] != "cancelled":
                raise AssertionError("queued cancellation")
            received = []
            while engine.pending:
                step_events = engine.step();received.extend(step_events)
                if any(e["request_id"] == "cancel-active" and e["token_id"] is not None for e in step_events):
                    received.extend(engine.cancel("cancel-active"))
            length = [e for e in received if e["request_id"] == "length"]
            if [e["token_id"] for e in length] != expected[:3] or length[-1]["finish_reason"] != "length":
                raise AssertionError("length/continuous scheduling")
            if sum(e["finish_reason"] == "cancelled" for e in received) != 1:
                raise AssertionError("active cancellation terminal count")
            for key, tokens, limit in (("bad-token", [engine.vocabulary], 1), ("bad-length", [1], 0), ("too-long", [1] * 255, 2)):
                try:
                    engine.submit(key, tokens, limit)
                except (RuntimeError, ValueError):
                    pass
                else:
                    raise AssertionError("invalid input accepted")
            stats = engine.stats()
            if stats["live_sequences"] or stats["live_kv_bytes"]:
                raise AssertionError("native KV leaked")
        # Interleave owners so A spans non-contiguous physical pages. Release B,
        # reuse its pages for C, then cross more boundaries without repacking KV.
        with NativeEngine(worker, root / "model.mpf", backend, 64, 2, 8, 4,
                          block_tokens=3, kv_blocks=12) as engine:
            a,b,c=[1,504,3575],[28,21781,198,504],[1,28,504,3575,282,30]
            close(engine.forward("a",a),reference(a),backend+"/paged/a-first")
            close(engine.forward("b",b),reference(b),backend+"/paged/b")
            tail=[4649,314,7042,30,2,28,198]
            close(engine.forward("a",tail),reference(a+tail),backend+"/paged/a-fragmented")
            a+=tail
            if engine.stats()["kv_live_blocks"]!=6:
                raise AssertionError("page rounding/interleaved ownership")
            engine.release("b")
            close(engine.forward("c",c),reference(c),backend+"/paged/reused-pages")
            close(engine.forward("a",tail),reference(a+tail),backend+"/paged/a-cross-again")
            a+=tail
            before=engine.stats()
            for key,tokens in (("exhausted",[1]*40),("a",[1]*48),("a",[engine.vocabulary])):
                try:
                    engine.forward(key,tokens)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("invalid/capacity append accepted")
                after=engine.stats()
                if (after["kv_live_blocks"]!=before["kv_live_blocks"] or
                        after["forward_tokens"]!=before["forward_tokens"]):
                    raise AssertionError("failed append changed committed pages/tokens")
            engine.release("exhausted")
            close(engine.forward("a",[504]),reference(a+[504]),backend+"/paged/retry-after-failure")
            for key in ("a","c"):
                engine.release(key)
            stats=engine.stats()
            if stats["kv_live_blocks"] or stats["live_sequences"] or stats["block_table_device_bytes"]:
                raise AssertionError("pages/block tables leaked")
            if stats["kv_total_blocks"]!=12 or stats["kv_block_tokens"]!=3:
                raise AssertionError("paged configuration not honored")
            paged_checks.append({"backend":backend,"case":"fragmentation/reuse/cross-page/atomic-rejection/retry",
                                 "stats":stats})
        # Resource exhaustion is explicit: current engine fails all unfinished
        # requests and releases every page. Preemption/admission is a later step.
        with NativeEngine(worker, root / "model.mpf", backend, 64, 2, 8, 4,
                          block_tokens=4, kv_blocks=1) as engine:
            for key in ("first","second","waiting"):
                engine.submit(key,[1]*9,2,None)
            events=list(engine.events())
            if len(events)!=3 or any(e["finish_reason"]!="error" or e["token_id"] is not None for e in events):
                raise AssertionError("exhaustion must terminalize all unfinished requests once")
            stats=engine.stats()
            if stats["kv_live_blocks"] or stats["live_sequences"] or stats["block_table_device_bytes"]:
                raise AssertionError("exhaustion cleanup leaked pages")
            paged_checks.append({"backend":backend,"case":"engine-pool-exhaustion-cleanup","stats":stats})
    report = {"date": date.today().isoformat(), "model": json.loads((root / "manifest.json").read_text()),
              "native_model_sha256": sha256(root / "model.mpf"), "dtype": "float32", "reference": "Transformers eager full-prefix CPU",
              "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform(),
              "tolerance": "abs(a-b) <= 2e-3 + 2e-4*abs(b); identical argmax", "logit_checks": records,
              "checked_logits": sum(r["checked_logits"] for r in records), "cuda_residency_checks": residency,
              "transcripts": transcripts, "paged_checks":paged_checks,
              "kv_layout":"paged, exclusive block ownership; no prefix cache",
              "worker_sha256": {k: sha256(v) for k, v in workers.items()}, "passed": True}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cpu-worker", required=True)
    parser.add_argument("--cuda-worker")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workers = {"cpu": args.cpu_worker}
    if args.cuda_worker:
        workers["cuda"] = args.cuda_worker
    result = verify(args.model_dir, workers, args.output)
    print(json.dumps({"passed": result["passed"], "logit_checks": len(result["logit_checks"]),
                      "checked_logits": result["checked_logits"], "cuda_residency_checks": len(result["cuda_residency_checks"])}))


if __name__ == "__main__":
    main()
