#!/usr/bin/env python3
"""Train the v1.1 RuleBank/T5-prefix RTE method on train domains only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def chat_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
    )


def lm_loss(qwen, tokenizer, rte, rulebank, texts: list[str], targets: list[str],
            weights: list[float], max_total_tokens: int) -> torch.Tensor:
    from rte_model import prefix_cache
    device = next(qwen.parameters()).device
    prompt_rows = [chat_ids(tokenizer, text) for text in texts]
    target_rows = [tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
                   for target in targets]
    if any(not ids for ids in prompt_rows) or any(len(ids) < 2 for ids in target_rows):
        raise ValueError("RTE batch contains an empty prompt/target")
    prompt_width = max(map(len, prompt_rows))
    target_width = max(map(len, target_rows))
    if prompt_width + target_width > max_total_tokens:
        raise ValueError(f"RTE prompt/target would exceed context: {prompt_width}+{target_width}")
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(texts), prompt_width + target_width), pad_id,
                           dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    labels = torch.full((len(texts), target_width), -100, dtype=torch.long, device=device)
    for i, (prompt, target) in enumerate(zip(prompt_rows, target_rows)):
        start = prompt_width - len(prompt)
        input_ids[i, start:prompt_width] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention[i, start:prompt_width] = 1
        input_ids[i, prompt_width:prompt_width + len(target)] = torch.tensor(target, dtype=torch.long, device=device)
        attention[i, prompt_width:prompt_width + len(target)] = 1
        labels[i, :len(target)] = torch.tensor(target, dtype=torch.long, device=device)

    t5_tokens = rte._active_t5_tokens
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prefix = rte(t5_tokens[0], t5_tokens[1], rulebank)
        cache = prefix_cache(prefix, dtype=qwen.dtype)
        full_attention = torch.cat(
            [torch.ones((len(texts), rte.prefix_tokens), dtype=attention.dtype, device=device), attention], dim=1
        )
        # Only project the suffix needed to predict target tokens. Qwen2's
        # logits_to_keep selects the last sequence positions before lm_head;
        # keeping target_width + 1 positions preserves the prompt-final token
        # that predicts target[0], while avoiding full-vocabulary logits for
        # the (often much longer) prompt. This is loss-equivalent to the full
        # logits slice below, but substantially reduces peak training memory.
        logits = qwen(input_ids=input_ids, attention_mask=full_attention,
                      past_key_values=cache, use_cache=True, return_dict=True,
                      logits_to_keep=target_width + 1).logits
    predicted = logits[:, :target_width].float()
    token_loss = F.cross_entropy(predicted.reshape(-1, predicted.shape[-1]), labels.reshape(-1),
                                 ignore_index=-100, reduction="none").reshape(len(texts), target_width)
    per_row = token_loss.sum(1) / labels.ne(-100).sum(1).clamp_min(1)
    row_weights = torch.tensor(weights, dtype=per_row.dtype, device=device)
    return (per_row * row_weights).sum() / row_weights.sum().clamp_min(1e-8)


def encode_t5_batch(tokenizer, texts: list[str], device: torch.device, max_length: int):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    if encoded.input_ids.shape[1] > max_length:
        raise ValueError(f"T5 input length {encoded.input_ids.shape[1]} exceeds {max_length}; refusing truncation")
    return {key: value.to(device) for key, value in encoded.items()}


def load_components(args, need_train: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rte_model import RTEPrefix, trainable_parameter_report
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("RTE main experiment requires CUDA BF16")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")
    qtok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="left")
    if qtok.pad_token_id is None:
        qtok.pad_token = qtok.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device)
    qwen.eval()
    qwen.config.use_cache = True
    for parameter in qwen.parameters():
        parameter.requires_grad_(False)
    t5tok = AutoTokenizer.from_pretrained(str(args.t5), local_files_only=True)
    rte = RTEPrefix(qwen.config, str(args.t5), prefix_tokens=24, cross_attention_blocks=4,
                    local_files_only=True).to(device)
    report = trainable_parameter_report(qwen, rte)
    if report["qwen_trainable"] != 0 or report["rte_trainable"] == 0:
        raise RuntimeError(f"Invalid frozen/trainable boundary: {report}")
    if need_train:
        for component in (rte.amortizer, rte.query_encoder):
            if hasattr(component, "gradient_checkpointing_enable"):
                component.gradient_checkpointing_enable()
    print(json.dumps({"qwen_trainable": report["qwen_trainable"],
                      "rte_trainable": report["rte_trainable"],
                      "t5_model_type": rte.amortizer.config.model_type,
                      "device": str(device)}, ensure_ascii=False), flush=True)
    return qwen, qtok, rte, t5tok, device


def get_rulebank(rte, t5tok, rules, device, max_length: int):
    texts = [f"问题：{row['question']}\n答案：{row['new_rule']}" for row in rules]
    batch = encode_t5_batch(t5tok, texts, device, max_length)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return rte.encode_rulebank(batch["input_ids"], batch["attention_mask"])


def replay_rulebank(rte, t5tok, rules, device, max_length: int,
                    cpu_rng_state: torch.Tensor, cuda_rng_state: torch.Tensor):
    """Recompute the shared RuleBank with the same dropout mask, without
    advancing the RNG stream used by the subsequent query encoder calls.

    This permits per-loss backward passes to release Qwen activations while
    preserving the original one-shared-RuleBank stochastic objective.
    """
    current_cpu_rng = torch.get_rng_state()
    current_cuda_rng = torch.cuda.get_rng_state(device)
    try:
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)
        return get_rulebank(rte, t5tok, rules, device, max_length)
    finally:
        torch.set_rng_state(current_cpu_rng)
        torch.cuda.set_rng_state(current_cuda_rng, device)


def validate_data(data: Path):
    manifest = json.loads((data / "prepared_manifest.json").read_text(encoding="utf-8"))
    train = read_jsonl(data / "train_rte.jsonl")
    dev = read_jsonl(data / "dev_closed_book.jsonl")
    rules = read_jsonl(data / "rule_pairs.jsonl")
    if (len(train), len(dev), len(rules)) != (1620, 540, 9):
        raise ValueError(f"Unexpected RTE train/dev/rule counts: {len(train)}/{len(dev)}/{len(rules)}")
    if any(row.get("split") != "train" for row in train):
        raise ValueError("RTE train data contains a non-train row")
    if any(row.get("split") != "development" for row in dev):
        raise ValueError("RTE dev data contains a non-development row")
    if any("当前更新后的上层规则：" in row["prompt"] for row in train + dev):
        raise ValueError("Case prompts must not expose the updated rule")
    if len({row["event_id"] for row in rules}) != 9:
        raise ValueError("RuleBank must contain exactly one update per event")
    return train, dev, rules, manifest


def check(args):
    train, dev, rules, _ = validate_data(args.data)
    qwen, qtok, rte, t5tok, device = load_components(args)
    for text in [row["prompt"] for row in train[:8] + dev[:8]]:
        batch = encode_t5_batch(t5tok, [text], device, args.max_t5_query_tokens)
        if batch["input_ids"].shape[1] > args.max_t5_query_tokens:
            raise RuntimeError("Unexpected T5 query truncation")
    bank = get_rulebank(rte, t5tok, rules, device, args.max_t5_rule_tokens)
    rte.eval()
    sample = train[:args.batch_size]
    batch = encode_t5_batch(t5tok, [row["prompt"] for row in sample], device, args.max_t5_query_tokens)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prefix = rte(batch["input_ids"], batch["attention_mask"], bank)
    expected = (qwen.config.num_key_value_heads, rte.prefix_tokens, rte.head_dim)
    if len(prefix) != qwen.config.num_hidden_layers or tuple(prefix[0][0].shape[1:]) != expected:
        raise RuntimeError(f"RTE KV shape mismatch: {prefix[0][0].shape}, expected [B,{expected}]")
    # The no-prefix path must remain exactly the unmodified frozen Qwen.
    prompt = qtok.apply_chat_template([{"role": "user", "content": train[0]["prompt"]}],
                                     tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits_a = qwen(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=False).logits
        logits_b = qwen(input_ids=prompt, attention_mask=torch.ones_like(prompt), past_key_values=None,
                        use_cache=False).logits
    if not torch.equal(logits_a, logits_b):
        raise RuntimeError("No-prefix Qwen logits differ from base model")
    suffix_width = min(32, prompt.shape[1])
    with torch.no_grad():
        suffix_logits = qwen(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=False,
                             logits_to_keep=suffix_width).logits
    if not torch.equal(suffix_logits, logits_a[:, -suffix_width:, :]):
        raise RuntimeError("Target-only logits differ from the matching full-sequence suffix")
    print(json.dumps({"passed": True, "train_rows": len(train), "dev_rows": len(dev),
                      "rule_events": len(rules), "prefix_shape": list(prefix[0][0].shape),
                      "no_prefix_exact_match": True,
                      "target_logits_suffix_exact_match": True}, ensure_ascii=False), flush=True)


def smoke(args):
    train_rows, _, rules, _ = validate_data(args.data)
    qwen, qtok, rte, t5tok, device = load_components(args, need_train=True)
    batch = train_rows[:args.batch_size]
    optimizer = torch.optim.AdamW(rte.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=False)
    optimizer.zero_grad(set_to_none=True)
    bank_cpu_rng = torch.get_rng_state()
    bank_cuda_rng = torch.cuda.get_rng_state(device)
    bank = get_rulebank(rte, t5tok, rules, device, args.max_t5_rule_tokens)
    qbatch = encode_t5_batch(t5tok, [row["prompt"] for row in batch], device,
                             args.max_t5_query_tokens)
    rte._active_t5_tokens = (qbatch["input_ids"], qbatch["attention_mask"])
    case_loss = lm_loss(qwen, qtok, rte, bank,
                        [row["prompt"] for row in batch], [row["target"] for row in batch],
                        [float(row.get("sample_weight", 1.0)) for row in batch],
                        args.max_total_tokens)
    del rte._active_t5_tokens
    if not torch.isfinite(case_loss):
        raise RuntimeError("RTE smoke case loss is not finite")
    case_nll = float(case_loss.detach())
    (args.instance_loss_weight * case_loss).backward()
    del bank, qbatch, case_loss

    rule = rules[0]
    rule_values = []
    for index, rule_query in enumerate([rule["question"], rule["rephrase"]]):
        bank = replay_rulebank(rte, t5tok, rules, device, args.max_t5_rule_tokens,
                               bank_cpu_rng, bank_cuda_rng)
        rbatch = encode_t5_batch(t5tok, [rule_query], device, args.max_t5_query_tokens)
        rte._active_t5_tokens = (rbatch["input_ids"], rbatch["attention_mask"])
        rule_loss = lm_loss(qwen, qtok, rte, bank, [rule_query], [rule["new_rule"]], [1.0],
                            args.max_total_tokens)
        del rte._active_t5_tokens
        if not torch.isfinite(rule_loss):
            raise RuntimeError("RTE smoke loss is not finite")
        rule_values.append(float(rule_loss.detach()))
        loss_weight = 1.0 if index == 0 else args.rephrase_loss_weight
        (loss_weight * rule_loss).backward()
        del bank, rbatch, rule_loss
    qwen_has_grad = any(parameter.grad is not None for parameter in qwen.parameters())
    rte_grad_norm = torch.nn.utils.clip_grad_norm_(rte.parameters(), float("inf"))
    rte_has_grad = any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       and parameter.grad.abs().sum() > 0 for parameter in rte.parameters())
    if qwen_has_grad or not rte_has_grad or not torch.isfinite(rte_grad_norm):
        raise RuntimeError(f"RTE smoke gradient boundary failed: qwen_grad={qwen_has_grad}, rte_grad={rte_has_grad}")
    optimizer.step()
    torch.cuda.synchronize()
    rule_nll = rule_values[0]
    print(json.dumps({"passed": True, "smoke_batch": len(batch),
                      "loss": args.instance_loss_weight * case_nll + rule_values[0]
                              + args.rephrase_loss_weight * rule_values[1],
                      "case_nll": case_nll, "rule_nll": rule_nll,
                      "rephrase_rule_nll": rule_values[1],
                      "qwen_trainable": 0, "rte_nonzero_gradient": True,
                      "optimizer_step": True,
                      "rte_grad_norm": float(rte_grad_norm),
                      "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
                     ensure_ascii=False), flush=True)


def train(args):
    train_rows, dev_rows, rules, source_manifest = validate_data(args.data)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {args.output}")
    qwen, qtok, rte, t5tok, device = load_components(args, need_train=True)
    # foreach=False avoids the large temporary tensor lists that can otherwise
    # spike peak memory for this ~1.77B-parameter trainable RTE module.
    optimizer = torch.optim.AdamW(rte.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=False)
    args.output.mkdir(parents=True, exist_ok=False)
    input_files = ("train_rte.jsonl", "dev_closed_book.jsonl", "rule_pairs.jsonl", "prepared_manifest.json")
    manifest = {
        "method": "RTE", "protocol": "main_experiment_formal_v1_1",
        "seed": args.seed, "base_model": str(args.model), "rule_encoder": str(args.t5),
        "train_rows": len(train_rows), "train_families": 540, "dev_rows": len(dev_rows),
        "rule_events": len(rules), "rulebank_content": "updated upper rules only; no downstream cases or answers",
        "architecture": {"prefix_tokens": 24, "cross_attention_blocks": 4,
                         "frozen_qwen": True, "qwen_kv_heads": qwen.config.num_key_value_heads},
        "optimizer": "AdamW", "optimizer_foreach": False,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "loss": "0.1*case_NLL + direct_rule_NLL + 0.1*rephrased_rule_NLL",
        "loss_logits": "target-only Qwen logits_to_keep suffix; equivalent target positions to full-sequence projection",
        "backward_schedule": "loss-equivalent sequential backward per case/direct/rephrase term; recompute RuleBank with identical CPU/CUDA RNG states to preserve dropout mask and release Qwen activations",
        "case_family_weighting": "each wording variant has sample_weight=1/3",
        "checkpoint_selection": "minimum closed-book dev decision/current-handling token NLL",
        "data_sha256": {name: digest(args.data / name) for name in input_files},
        "dataset_manifest_sha256": hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest(),
        "test_data_read": False, "test_gold_read": False, "status": "running",
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
    }
    write_json(args.output / "run_manifest.json", manifest)
    best_dir = args.output / "best.pt"
    best_dev = float("inf")
    history = []
    started = time.time()
    update = 0
    for epoch in range(1, args.epochs + 1):
        rte.train()
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_loss = 0.0
        groups = 0
        for start in range(0, len(order), args.batch_size):
            batch = [train_rows[i] for i in order[start:start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            bank_cpu_rng = torch.get_rng_state()
            bank_cuda_rng = torch.cuda.get_rng_state(device)
            bank = get_rulebank(rte, t5tok, rules, device, args.max_t5_rule_tokens)
            qbatch = encode_t5_batch(t5tok, [row["prompt"] for row in batch], device,
                                     args.max_t5_query_tokens)
            rte._active_t5_tokens = (qbatch["input_ids"], qbatch["attention_mask"])
            case_loss = lm_loss(qwen, qtok, rte, bank,
                                [row["prompt"] for row in batch], [row["target"] for row in batch],
                                [float(row.get("sample_weight", 1.0)) for row in batch],
                                args.max_total_tokens)
            del rte._active_t5_tokens
            if not torch.isfinite(case_loss):
                raise RuntimeError(f"Nonfinite RTE case loss at update {update + 1}")
            case_value = float(case_loss.detach())
            (args.instance_loss_weight * case_loss).backward()
            del bank, qbatch, case_loss

            rule = rules[(update + epoch - 1) % len(rules)]
            rule_texts = [rule["question"], rule["rephrase"]]
            rule_values = []
            for index, query in enumerate(rule_texts):
                # Re-encode the small RuleBank graph so each Qwen activation
                # graph can be freed immediately after its backward pass.
                bank = replay_rulebank(rte, t5tok, rules, device, args.max_t5_rule_tokens,
                                       bank_cpu_rng, bank_cuda_rng)
                rbatch = encode_t5_batch(t5tok, [query], device, args.max_t5_query_tokens)
                rte._active_t5_tokens = (rbatch["input_ids"], rbatch["attention_mask"])
                rule_loss = lm_loss(qwen, qtok, rte, bank, [query], [rule["new_rule"]], [1.0],
                                    args.max_total_tokens)
                del rte._active_t5_tokens
                if not torch.isfinite(rule_loss):
                    raise RuntimeError(f"Nonfinite RTE rule loss at update {update + 1}")
                rule_values.append(float(rule_loss.detach()))
                loss_weight = 1.0 if index == 0 else args.rephrase_loss_weight
                (loss_weight * rule_loss).backward()
                del bank, rbatch, rule_loss

            torch.nn.utils.clip_grad_norm_(rte.parameters(), args.max_grad_norm)
            optimizer.step()
            update += 1
            total_loss = (args.instance_loss_weight * case_value + rule_values[0]
                          + args.rephrase_loss_weight * rule_values[1])
            epoch_loss += total_loss
            groups += 1
            if update % args.log_interval == 0:
                print(json.dumps({"epoch": epoch, "update": update, "updates_per_epoch": (len(train_rows) + args.batch_size - 1) // args.batch_size,
                                  "loss": total_loss, "case_nll": case_value,
                                  "rule_nll": rule_values[0],
                                  "elapsed_sec": round(time.time() - started, 1),
                                  "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
                                 ensure_ascii=False), flush=True)

        dev_score = evaluate_dev(args, qwen, qtok, rte, t5tok, rules, dev_rows, device)
        record = {"epoch": epoch, "updates": update, "mean_train_loss": epoch_loss / max(groups, 1),
                  "dev_decision_handling_nll": dev_score, "elapsed_sec": round(time.time() - started, 1),
                  "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        history.append(record)
        write_json(args.output / "history.json", history)
        if dev_score < best_dev:
            best_dev = dev_score
            tmp = best_dir.with_suffix(".pt.tmp")
            torch.save({"rte": rte.state_dict(), "epoch": epoch, "dev_nll": dev_score,
                        "base_model": str(args.model), "t5_model": str(args.t5), "seed": args.seed}, tmp)
            tmp.replace(best_dir)
            record["selected"] = True
            write_json(args.output / "selection.json", {"epoch": epoch, "dev_nll": dev_score,
                                                          "checkpoint_sha256": digest(best_dir)})
        write_json(args.output / "history.json", history)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    manifest.update({"status": "complete", "elapsed_sec": round(time.time() - started, 1),
                     "best_dev_nll": best_dev, "checkpoint_sha256": digest(best_dir),
                     "optimizer_updates": update,
                     "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    write_json(args.output / "run_manifest.json", manifest)
    (args.output / "complete.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"), encoding="utf-8")


def evaluate_dev(args, qwen, qtok, rte, t5tok, rules, rows, device) -> float:
    rte.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        bank = get_rulebank(rte, t5tok, rules, device, args.max_t5_rule_tokens)
        for start in range(0, len(rows), args.eval_batch_size):
            batch = rows[start:start + args.eval_batch_size]
            qbatch = encode_t5_batch(t5tok, [row["prompt"] for row in batch], device,
                                     args.max_t5_query_tokens)
            rte._active_t5_tokens = (qbatch["input_ids"], qbatch["attention_mask"])
            losses = lm_loss(qwen, qtok, rte, bank,
                             [row["prompt"] for row in batch], [row["target"] for row in batch],
                             [1.0] * len(batch), args.max_total_tokens)
            del rte._active_t5_tokens
            total += float(losses) * len(batch)
            count += len(batch)
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "train"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--t5", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--instance-loss-weight", type=float, default=0.1)
    parser.add_argument("--rephrase-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--max-t5-rule-tokens", type=int, default=256)
    # Full v1.1 prompts reach 549 mT5 tokens in train/dev; keep headroom and
    # reject any future overflow rather than silently truncating the query.
    parser.add_argument("--max-t5-query-tokens", type=int, default=640)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.epochs < 1:
        parser.error("batch sizes and epochs must be positive")
    if args.action == "check":
        check(args)
    elif args.action == "smoke":
        smoke(args)
    else:
        if args.output is None:
            parser.error("--output is required for train")
        train(args)


if __name__ == "__main__":
    main()
