#!/usr/bin/env python3
"""Resumable, fixed-batch greedy inference for the v1.1 main experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def model_fingerprint(model_dir: Path) -> dict[str, Any]:
    config = model_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    weight_files = sorted(model_dir.glob("*.safetensors"))
    index = model_dir / "model.safetensors.index.json"
    files = weight_files if weight_files else ([index] if index.is_file() else [])
    if not files:
        raise FileNotFoundError(f"No local model weights found in {model_dir}")
    return {
        "path": str(model_dir),
        "config_sha256": digest(config),
        "weight_files": [{"name": path.name, "size_bytes": path.stat().st_size} for path in files],
    }


def adapter_fingerprint(adapter_dir: Path | None) -> dict[str, Any] | None:
    if adapter_dir is None:
        return None
    candidates = [adapter_dir / "adapter_model.safetensors", adapter_dir / "policy" / "adapter_model.safetensors",
                  adapter_dir / "default" / "adapter_model.safetensors"]
    weight = next((path for path in candidates if path.is_file()), None)
    if weight is None:
        raise FileNotFoundError(f"No PEFT adapter weights found under {adapter_dir}")
    config = weight.parent / "adapter_config.json"
    return {"path": str(weight.parent), "weights_sha256": digest(weight),
            "config_sha256": digest(config) if config.is_file() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("The frozen main-experiment generation batch size is 8")
    if not args.model.is_dir() or args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("Invalid local model or generation settings")
    rows = read_jsonl(args.input)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if not rows or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Input must contain nonempty unique request IDs")
    if any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in rows):
        raise ValueError("Every inference input must have a nonempty prompt")
    if any(any(k in row for k in ("target", "target_constraints", "new_state", "delta", "new_rule")) for row in rows):
        raise ValueError("Model-visible inference input contains hidden labels")
    if args.method == "Pure-Qwen+ICE":
        if not any("当前更新后的上层规则：" in row["prompt"] for row in rows):
            raise ValueError("ICE inputs do not contain the updated upper rule")
    elif any("当前更新后的上层规则：" in row["prompt"] for row in rows):
        raise ValueError("Closed-book condition unexpectedly exposes the updated upper rule")
    if args.method in {"Pure-Qwen", "Pure-Qwen+ICE"} and args.adapter is not None:
        raise ValueError("Base/ICE methods must not load an adapter")
    if args.method not in {"Pure-Qwen", "Pure-Qwen+ICE"} and args.adapter is None:
        raise ValueError("An adapter path is required for this trained method")

    run_path = args.output.with_suffix(args.output.suffix + ".run.json")
    input_hash = digest(args.input)
    config = {
        "protocol": "main_experiment_formal_v1_1_batch8_greedy_v1",
        "method": args.method, "input_path": str(args.input), "input_sha256": input_hash,
        "output_path": str(args.output), "expected_rows": len(rows),
        "model": model_fingerprint(args.model), "adapter": adapter_fingerprint(args.adapter),
        "seed": args.seed, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "decode": "greedy", "dtype": "bfloat16", "attention": "sdpa", "padding_side": "left",
        "cuda_deterministic_algorithms": "warn_only", "status": "running",
    }
    if args.output.exists() or run_path.exists():
        if not args.resume or not run_path.is_file() or not args.output.is_file():
            raise FileExistsError("Output exists; pass --resume only for the exact same interrupted run")
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        compare_keys = set(config) - {"status"}
        if any(previous.get(key) != config.get(key) for key in compare_keys):
            raise ValueError("Resume configuration differs from the existing run manifest")
        if previous.get("status") == "complete":
            raise FileExistsError("This run is already complete; refusing to overwrite")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(run_path, config)

    existing = read_jsonl(args.output) if args.output.exists() else []
    done = {row.get("id"): row for row in existing}
    if len(done) != len(existing) or not set(done).issubset(set(ids)):
        raise ValueError("Existing partial output has duplicate or unexpected IDs")
    pending = [row for row in rows if row["id"] not in done]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda")
    if args.adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False, local_files_only=True)
        if any(param.requires_grad for param in model.parameters()):
            raise RuntimeError("Inference adapter unexpectedly has trainable parameters")
    model.eval()

    started = time.perf_counter()
    input_tokens = output_tokens = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            ) for row in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to("cuda")
            width = encoded.input_ids.shape[1]
            limit = getattr(model.config, "max_position_embeddings", None)
            if limit and width + args.max_new_tokens > limit:
                raise ValueError(f"Prompt plus generation limit exceeds model context: {width}+{args.max_new_tokens}>{limit}")
            input_tokens += int(encoded.attention_mask.sum().item())
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            suffixes = generated[:, width:]
            answers = tokenizer.batch_decode(suffixes, skip_special_tokens=True)
            for row, answer, tokens in zip(batch, answers, suffixes):
                token_ids = tokens.tolist()
                if tokenizer.eos_token_id in token_ids:
                    token_ids = token_ids[:token_ids.index(tokenizer.eos_token_id) + 1]
                output_tokens += len(token_ids)
                record = {key: row[key] for key in (
                    "id", "request_id", "case_family_id", "family_id", "event_id", "domain_id", "difficulty", "split"
                ) if key in row}
                record.update({"method": args.method, "answer": answer.strip(), "output_tokens": len(token_ids)})
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            done_count = len(done) + min(offset + len(batch), len(pending))
            elapsed = time.perf_counter() - started
            print(json.dumps({"method": args.method, "done": done_count, "total": len(rows),
                              "batch_size": args.batch_size, "elapsed_sec": round(elapsed, 2),
                              "output_tokens": output_tokens}, ensure_ascii=False), flush=True)
            del encoded, generated, suffixes
    final_rows = read_jsonl(args.output)
    if len(final_rows) != len(rows) or {row.get("id") for row in final_rows} != set(ids):
        raise RuntimeError(f"Incomplete inference output: {len(final_rows)}/{len(rows)}")
    elapsed = time.perf_counter() - started
    config.update({
        "status": "complete", "rows": len(final_rows), "output_sha256": digest(args.output),
        "elapsed_sec": round(elapsed, 3), "input_tokens_generated_this_resume": input_tokens,
        "output_tokens_generated_this_resume": output_tokens,
        "requests_per_sec_this_resume": round(len(pending) / max(elapsed, 1e-9), 4),
        "completed_at_unix": time.time(),
    })
    write_json(run_path, config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
