#!/usr/bin/env python3
"""Batch-8 greedy test inference for the v1.1 RTE prefix adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def t5_batch(tokenizer, texts: list[str], device, max_length: int):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    if encoded.input_ids.shape[1] > max_length:
        raise ValueError(f"T5 text length {encoded.input_ids.shape[1]} exceeds {max_length}")
    return {key: value.to(device) for key, value in encoded.items()}


def generate_batch(qwen, qtok, rte, t5tok, bank, rows, device, max_new_tokens, max_t5_query_tokens):
    import torch
    from rte_model import prefix_cache
    prompts = [qtok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                        tokenize=False, add_generation_prompt=True) for row in rows]
    qinputs = qtok(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
    if qinputs.input_ids.shape[1] + max_new_tokens + rte.prefix_tokens > qwen.config.max_position_embeddings:
        raise ValueError("RTE query plus generation budget exceeds Qwen context")
    tinputs = t5_batch(t5tok, [row["prompt"] for row in rows], device, max_t5_query_tokens)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prefix = rte(tinputs["input_ids"], tinputs["attention_mask"], bank)
        cache = prefix_cache(prefix, dtype=qwen.dtype)
        batch = len(rows)
        p_len = rte.prefix_tokens
        attention = torch.cat([
            torch.ones((batch, p_len), dtype=qinputs.attention_mask.dtype, device=device),
            qinputs.attention_mask,
        ], dim=1)
        result = qwen(input_ids=qinputs.input_ids, attention_mask=attention,
                      past_key_values=cache, use_cache=True, return_dict=True)
        logits = result.logits[:, -1, :]
        past = result.past_key_values
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in rows]
        pad_id = qtok.pad_token_id if qtok.pad_token_id is not None else qtok.eos_token_id
        eos_id = qtok.eos_token_id
        for _ in range(max_new_tokens):
            candidate = logits.argmax(dim=-1)
            was_finished = finished.clone()
            for i, token in enumerate(candidate.tolist()):
                if not was_finished[i] and token != eos_id:
                    generated[i].append(token)
            newly_finished = candidate.eq(eos_id) & ~was_finished
            finished |= newly_finished
            if bool(finished.all()):
                break
            step_ids = candidate.clone()
            step_ids[was_finished] = pad_id
            mask_add = (~was_finished).to(attention.dtype).unsqueeze(1)
            attention = torch.cat([attention, mask_add], dim=1)
            result = qwen(input_ids=step_ids.unsqueeze(1), attention_mask=attention,
                          past_key_values=past, use_cache=True, return_dict=True)
            past = result.past_key_values
            logits = result.logits[:, -1, :]
        answers = [qtok.decode(tokens, skip_special_tokens=True).strip() for tokens in generated]
        lengths = [len(tokens) for tokens in generated]
    del qinputs, tinputs, prefix, cache, result
    return answers, lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt from train_main_rte.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--t5", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    # The longest frozen v1.1 test prompt is 563 mT5 tokens; avoid truncation.
    parser.add_argument("--max-t5-query-tokens", type=int, default=640)
    parser.add_argument("--expected-rows", type=int, default=5100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rte_model import RTEPrefix
    if args.batch_size != 8:
        raise ValueError("Frozen v1.1 generation batch size is 8")
    rows = read_jsonl(args.input)
    rules = read_jsonl(args.rules)
    if len(rows) != args.expected_rows or len(rules) != 9:
        raise ValueError(f"Unexpected test/rule counts: {len(rows)}/{len(rules)}")
    ids = [row.get("id") for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError("Test request IDs must be nonempty and unique")
    for row in rows:
        if any(k in row for k in ("target", "target_constraints", "new_state", "delta", "new_rule")):
            raise ValueError("Model-visible test input contains hidden gold")
        if "当前更新后的上层规则：" in row["prompt"]:
            raise ValueError("RTE closed-book input must not expose the updated rule")
    if any(not row.get("event_id") for row in rows):
        raise ValueError("RTE test request is missing event metadata")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    run_path = args.output.with_suffix(args.output.suffix + ".run.json")
    config = {
        "protocol": "main_experiment_formal_v1_1_rte_batch8_greedy_v1",
        "method": "RTE", "input_sha256": digest(args.input), "rules_sha256": digest(args.rules),
        "checkpoint_sha256": digest(args.checkpoint), "model_config_sha256": digest(args.model / "config.json"),
        "expected_rows": len(rows), "batch_size": 8, "max_new_tokens": args.max_new_tokens,
        "seed": 42, "decode": "greedy", "dtype": "bfloat16", "status": "running",
    }
    if args.output.exists() or run_path.exists():
        if not args.resume or not args.output.is_file() or not run_path.is_file():
            raise FileExistsError("Output exists; use --resume only for the same interrupted run")
        old = json.loads(run_path.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k, v in config.items() if k != "status"):
            raise ValueError("RTE resume manifest does not match this run")
        if old.get("status") == "complete":
            raise FileExistsError("RTE run is already complete")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(run_path, config)

    existing = read_jsonl(args.output) if args.output.exists() else []
    done = {row.get("id") for row in existing}
    if len(done) != len(existing) or not done.issubset(set(ids)):
        raise ValueError("Existing RTE output has duplicate/unexpected IDs")
    pending = [row for row in rows if row["id"] not in done]

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("RTE inference requires CUDA BF16")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    qtok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="left")
    if qtok.pad_token_id is None:
        qtok.pad_token = qtok.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    for p in qwen.parameters():
        p.requires_grad_(False)
    t5tok = AutoTokenizer.from_pretrained(str(args.t5), local_files_only=True)
    rte = RTEPrefix(qwen.config, str(args.t5), prefix_tokens=24, cross_attention_blocks=4,
                    local_files_only=True).to(device).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("base_model") != str(args.model) or payload.get("t5_model") != str(args.t5):
        raise ValueError("RTE checkpoint base-model paths differ from inference configuration")
    rte.load_state_dict(payload["rte"], strict=True)
    rule_texts = [f"问题：{row['question']}\n答案：{row['new_rule']}" for row in rules]
    rule_inputs = t5_batch(t5tok, rule_texts, device, 256)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bank = rte.encode_rulebank(rule_inputs["input_ids"], rule_inputs["attention_mask"])

    started = time.time()
    input_tokens = output_tokens = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            answers, lengths = generate_batch(qwen, qtok, rte, t5tok, bank, batch, device,
                                              args.max_new_tokens, args.max_t5_query_tokens)
            for row, answer, length in zip(batch, answers, lengths):
                record = {key: row[key] for key in (
                    "id", "request_id", "case_family_id", "family_id", "event_id", "domain_id", "difficulty", "split"
                ) if key in row}
                record.update({"method": "RTE", "answer": answer, "output_tokens": length})
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                output_tokens += length
            input_tokens += len(batch)
            stream.flush()
            print(json.dumps({"method": "RTE", "done": len(done) + min(offset + len(batch), len(pending)),
                              "total": len(rows), "batch_size": 8, "elapsed_sec": round(time.time() - started, 1),
                              "output_tokens": output_tokens}, ensure_ascii=False), flush=True)

    final_rows = read_jsonl(args.output)
    if len(final_rows) != len(rows) or {row.get("id") for row in final_rows} != set(ids):
        raise RuntimeError(f"Incomplete RTE output: {len(final_rows)}/{len(rows)}")
    config.update({"status": "complete", "output_sha256": digest(args.output),
                   "elapsed_sec": round(time.time() - started, 2), "rows": len(final_rows),
                   "generated_this_resume": input_tokens, "output_tokens_generated_this_resume": output_tokens,
                   "completed_at_unix": time.time()})
    write_json(run_path, config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
