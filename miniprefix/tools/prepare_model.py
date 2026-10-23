"""Explicitly download a pinned HF Llama checkpoint and export native FP32 MPF.

Never invoked by CMake or inference. Tokenizer/config/license stay alongside
the native file; metadata records exact model revision and all source hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


def export_model(source, output):
    import numpy as np
    from safetensors import safe_open
    source, output = Path(source), Path(output)
    c = json.loads((source / "config.json").read_text())
    if (c.get("model_type") != "llama" or c.get("hidden_act") != "silu"
            or c.get("attention_bias", False) or c.get("mlp_bias", False)
            or c.get("rope_scaling") is not None or c.get("rope_interleaved", False)
            or c.get("pretraining_tp", 1) != 1):
        raise ValueError("only unscaled, bias-free, split-half RoPE Llama is supported")
    paths = sorted(source.glob("*.safetensors"))
    if len(paths) != 1:
        raise ValueError("first exporter requires one safetensors file")
    with safe_open(paths[0], framework="pt", device="cpu") as reader:
        names = sorted(reader.keys())
        if c.get("tie_word_embeddings", False):
            names = [n for n in names if n != "lm_head.weight"]
        temp = output.with_suffix(output.suffix + ".tmp")
        try:
            with temp.open("wb") as stream:
                stream.write(b"MPFX0001")
                stream.write(struct.pack("<8i2fII", c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"],
                                         c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"]),
                                         c["vocab_size"], c["max_position_embeddings"], c["eos_token_id"],
                                         c["rms_norm_eps"], c["rope_theta"], int(c.get("tie_word_embeddings", False)), len(names)))
                for name in names:
                    value = reader.get_tensor(name).float().numpy()
                    if value.ndim == 2 and name != "model.embed_tokens.weight":
                        value = value.T
                    value = np.asarray(value, dtype="<f4", order="C")
                    if not np.isfinite(value).all():
                        raise ValueError("nonfinite source weights")
                    encoded = name.encode("utf-8")
                    stream.write(struct.pack("<I", len(encoded)) + encoded)
                    stream.write(struct.pack("<I", value.ndim))
                    stream.write(struct.pack("<" + "i" * value.ndim, *value.shape))
                    stream.write(struct.pack("<Q", value.nbytes))
                    stream.write(value.tobytes())
            os.replace(temp, output)
        finally:
            if temp.exists():
                temp.unlink()


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--revision", default="main", help="Resolved to immutable commit before download")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info(args.model_id, revision=args.revision).sha
    snapshot_download(args.model_id, revision=revision, local_dir=args.output,
                      allow_patterns=["config.json", "generation_config.json", "*.safetensors", "tokenizer*", "special_tokens_map.json", "README.md", "LICENSE*"])
    target = args.output / "model.mpf"
    export_model(args.output, target)
    metadata = {"model_id": args.model_id, "revision": revision, "format": "MPFX0001", "dtype": "float32",
                "source": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                           if p.is_file() and p.name != "manifest.json"}, "native_bytes": target.stat().st_size}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_id": args.model_id, "revision": revision, "native_bytes": metadata["native_bytes"]}))


if __name__ == "__main__":
    main()
