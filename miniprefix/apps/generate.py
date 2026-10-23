"""Real tokenizer -> native MiniPrefix -> greedy token events -> decoded text."""
import argparse
import json
from pathlib import Path
import sys
import time
from miniprefix.bindings.python.client import NativeEngine


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--raw", action="store_true", help="Do not apply model chat template")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    prepare = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompt = (tokenizer.encode(args.prompt, add_special_tokens=False) if args.raw else
              tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True))
    with NativeEngine(args.worker, args.model_dir / "model.mpf", args.backend, args.context,
                      chunk_size=args.chunk_size) as engine:
        preparation_ms = (time.perf_counter_ns() - prepare) / 1e6
        start = time.perf_counter_ns()
        engine.submit("request", prompt, args.max_new_tokens)
        output, times, printed = [], [], ""
        finish = None
        for event in engine.events():
            if event["token_id"] is not None:
                output.append(event["token_id"])
                times.append((event["received_ns"] - start) / 1e6)
                text = tokenizer.decode(output, skip_special_tokens=True)
                if not args.json:
                    # Emit complete decoded prefixes; retain an incomplete UTF-8
                    # byte token until the tokenizer can produce valid text.
                    if text.startswith(printed) and not text.endswith("\ufffd"):
                        print(text[len(printed):], end="", flush=True)
                        printed = text
            if event["finish_reason"]:
                finish = event["finish_reason"]
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if finish == "error":
            raise RuntimeError("native generation failed")
        result = {"backend": args.backend, "prompt_tokens": len(prompt), "output_tokens": output,
                  "text": tokenizer.decode(output, skip_special_tokens=True), "finish_reason": finish,
                  "preparation_ms": preparation_ms, "elapsed_ms": elapsed, "token_times_ms": times,
                  "ttft_ms": times[0] if times else None, "stats": engine.stats()}
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            if result["text"].startswith(printed):
                print(result["text"][len(printed):], end="")
            print()


if __name__ == "__main__":
    main()
