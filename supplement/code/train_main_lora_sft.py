#!/usr/bin/env python3
"""Train the v1.1 Rule+Case LoRA-SFT main-experiment baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def decision_handling_target(row: dict[str, Any]) -> str:
    state = row["new_state"]
    return f"最终处理判断：{state['final_decision']}。当前处置：{state['current_handling']}。"


def encode_batch(tokenizer, rows: list[dict[str, Any]], torch, max_total_tokens: int):
    sequences, label_rows, weights = [], [], []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=True,
            add_generation_prompt=True,
        )
        target_ids = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        if not prompt_ids or len(target_ids) < 2:
            raise ValueError(f"Empty prompt/target: {row.get('id')}")
        if len(prompt_ids) + len(target_ids) > max_total_tokens:
            raise ValueError(f"Would truncate sample {row.get('id')}: {len(prompt_ids)}+{len(target_ids)}")
        sequences.append(prompt_ids + target_ids)
        label_rows.append([-100] * len(prompt_ids) + target_ids)
        weights.append(float(row.get("weight", row.get("sample_weight", 1.0))))
    max_len = max(map(len, sequences))
    pad_id = tokenizer.pad_token_id
    input_ids = [seq + [pad_id] * (max_len - len(seq)) for seq in sequences]
    labels = [lab + [-100] * (max_len - len(lab)) for lab in label_rows]
    attention = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in sequences]
    device = "cuda"
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "weights": torch.tensor(weights, dtype=torch.float32, device=device),
    }


def weighted_batch_loss(model, tokenizer, rows, torch, max_total_tokens):
    batch = encode_batch(tokenizer, rows, torch, max_total_tokens)
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    logits = output.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    valid = labels.ne(-100)
    token_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.clamp_min(0).reshape(-1), reduction="none"
    ).reshape_as(labels)
    counts = valid.sum(dim=1).clamp_min(1)
    per_row = (token_loss * valid).sum(dim=1) / counts
    weights = batch["weights"]
    loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, {"examples": len(rows), "mean_target_tokens": float(counts.float().mean().detach())}


def eval_nll(model, tokenizer, rows, torch, max_total_tokens, batch_size: int) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            eval_rows = [{"id": x["id"], "prompt": x["prompt"],
                          "target": decision_handling_target(x), "weight": 1.0} for x in group]
            loss, _ = weighted_batch_loss(model, tokenizer, eval_rows, torch, max_total_tokens)
            total += float(loss) * len(group)
            count += len(group)
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    train = read_jsonl(args.data / "train_sft.jsonl")
    dev = read_jsonl(args.data / "dev_closed_book.jsonl")
    rule_count = sum(row.get("kind") == "rule" for row in train)
    case_count = len(train) - rule_count
    if (case_count, rule_count, len(dev)) != (1620, 18, 540):
        raise ValueError(f"Unexpected main-run train/dev counts: {case_count}/{rule_count}/{len(dev)}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {args.output}")
    if args.epochs < 1 or args.micro_batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("Training schedule values must be positive")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires CUDA BF16 on the H100")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # The H100 cuDNN SDPA backward produced nonfinite LoRA gradients at update 6
    # in the frozen seed-42 run. Disable that backend while keeping SDPA enabled;
    # this was verified to make the same failing update finite.
    torch.backends.cuda.enable_cudnn_sdp(False)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda")
    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora)
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    args.output.mkdir(parents=True, exist_ok=False)
    best_dir = args.output / "best_adapter"
    history: list[dict[str, Any]] = []
    started = time.time()
    manifest = {
        "method": "LoRA-SFT-RuleCase", "protocol": "main_experiment_formal_v1_1",
        "seed": args.seed, "model": str(args.model), "source_data": str(args.data),
        "source_sha256": {name: digest(args.data / name) for name in
                          ("train_sft.jsonl", "dev_closed_book.jsonl", "rule_pairs.jsonl", "prepared_manifest.json")},
        "train_case_variants": case_count, "train_rule_views": rule_count,
        "train_case_family_count": 540, "family_weighting": "each of three wording variants has weight 1/3",
        "rule_qa_loss_share_target": 0.20,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "epochs": args.epochs, "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_approx": args.micro_batch_size * args.gradient_accumulation,
        "checkpoint_selection": "minimum closed-book development final-decision/current-handling NLL",
        "attention_backend": {"implementation": "sdpa", "cudnn_sdp_enabled": False},
        "test_split_read": False, "test_gold_read": False,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__, "status": "running",
    }
    write_json(args.output / "run_manifest.json", manifest)
    optimizer = torch.optim.AdamW((param for _, param in trainable), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    micro = args.micro_batch_size
    accumulation = args.gradient_accumulation
    updates_total = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_loss = 0.0
        seen = 0
        group_size = micro * accumulation
        for group_start in range(0, len(order), group_size):
            chosen_indices = order[group_start:group_start + group_size]
            chunks = [chosen_indices[i:i + micro] for i in range(0, len(chosen_indices), micro)]
            optimizer.zero_grad(set_to_none=True)
            for chunk in chunks:
                minibatch = [train[i] for i in chunk]
                loss, _ = weighted_batch_loss(model, tokenizer, minibatch, torch, args.max_total_tokens)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite SFT loss in epoch {epoch}")
                (loss / len(chunks)).backward()
                epoch_loss += float(loss.detach()) * len(minibatch)
                seen += len(minibatch)
            nonfinite_grads = [name for name, param in trainable
                               if param.grad is not None and not torch.isfinite(param.grad).all()]
            if nonfinite_grads:
                raise RuntimeError(f"Nonfinite LoRA gradients in epoch {epoch}: {nonfinite_grads[:5]}")
            grad_norm = torch.nn.utils.clip_grad_norm_((param for _, param in trainable), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite LoRA gradient norm in epoch {epoch}")
            optimizer.step()
            nonfinite_params = [name for name, param in trainable if not torch.isfinite(param).all()]
            if nonfinite_params:
                raise RuntimeError(f"Nonfinite LoRA parameters after update in epoch {epoch}: {nonfinite_params[:5]}")
            updates_total += 1
            if updates_total % 20 == 0:
                print(json.dumps({"epoch": epoch, "updates": updates_total,
                                  "epoch_progress": min(group_start + group_size, len(order)),
                                  "train_loss_so_far": epoch_loss / max(seen, 1),
                                  "elapsed_sec": round(time.time() - started, 1),
                                  "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
                                 ensure_ascii=False), flush=True)

        dev_score = eval_nll(model, tokenizer, dev, torch, args.max_total_tokens, args.eval_batch_size)
        record = {"epoch": epoch, "updates_total": updates_total,
                  "train_loss": epoch_loss / max(seen, 1), "dev_decision_handling_nll": dev_score,
                  "elapsed_sec": round(time.time() - started, 1),
                  "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        history.append(record)
        write_json(args.output / "history.json", history)
        if not best_dir.exists() or dev_score < min(x["dev_decision_handling_nll"] for x in history[:-1]):
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
            write_json(args.output / "selection.json", {"epoch": epoch, "dev_nll": dev_score})
        print(json.dumps(record, ensure_ascii=False), flush=True)

    adapter_path = best_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise RuntimeError("Best LoRA adapter checkpoint was not written")
    manifest.update({"status": "complete", "elapsed_sec": round(time.time() - started, 1),
                     "best_selection": json.loads((args.output / "selection.json").read_text()),
                     "adapter_sha256": digest(adapter_path), "optimizer_updates": updates_total,
                     "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    write_json(args.output / "run_manifest.json", manifest)
    (args.output / "complete.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
