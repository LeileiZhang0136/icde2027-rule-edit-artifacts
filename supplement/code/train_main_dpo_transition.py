#!/usr/bin/env python3
"""Train DPOTransition on the frozen v1.1 train split only."""
from __future__ import annotations

import argparse
import difflib
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


def remove_updated_rule_line(prompt: str) -> str:
    """Create the closed-book case prompt by removing only the update line."""
    lines = prompt.splitlines()
    marker = "当前更新后的上层规则："
    matches = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one updated-rule line, found {len(matches)}")
    return "\n".join(line for i, line in enumerate(lines) if i != matches[0])


def maybe_remove_updated_rule_line(prompt: str, probability: float, rng: random.Random) -> tuple[str, bool]:
    dropped = rng.random() < probability
    return (remove_updated_rule_line(prompt), True) if dropped else (prompt, False)


def finalize_existing(args: argparse.Namespace) -> None:
    """Finalize an already completed run whose PEFT adapter is nested by name."""
    output = args.output
    manifest_path = output / "run_manifest.json"
    selection_path = output / "selection.json"
    history_path = output / "history.json"
    adapter_dir = output / "best_adapter" / "policy"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    required = (manifest_path, selection_path, history_path, adapter_config, adapter_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot finalize DPO run; missing: " + ", ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if manifest.get("method") != "DPOTransition" or manifest.get("status") != "running":
        raise ValueError("Expected an unfinished DPOTransition manifest")
    if not isinstance(selection.get("step"), int) or not isinstance(selection.get("dev_nll"), (int, float)):
        raise ValueError("Invalid DPO checkpoint selection record")
    if not history or max(int(row.get("step", 0)) for row in history) != args.steps:
        raise ValueError("DPO training history does not contain the configured final step")
    if not any(int(row.get("step", 0)) == selection["step"] and
               row.get("dev_decision_handling_nll") == selection["dev_nll"] for row in history):
        raise ValueError("Selected DPO checkpoint is not supported by the saved development history")
    if not any(int(row.get("step", 0)) == args.steps and
               "dev_decision_handling_nll" in row for row in history):
        raise ValueError("Final DPO development evaluation is missing")
    if selection["step"] > args.steps or adapter_path.stat().st_size == 0:
        raise ValueError("Invalid selected DPO adapter checkpoint")

    final_record = next(row for row in reversed(history) if int(row.get("step", 0)) == args.steps)
    manifest.update({
        "status": "complete",
        "elapsed_sec": final_record.get("elapsed_sec"),
        "best_selection": selection,
        "adapter_sha256": digest(adapter_path),
        "adapter_relative_path": "best_adapter/policy",
        "cuda_peak_gib": max(float(row.get("cuda_peak_gib", 0.0)) for row in history),
        "finalized_from_saved_training_history": True,
    })
    write_json(manifest_path, manifest)
    (output / "complete.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"), encoding="utf-8")


def encode_pair(tokenizer, prompt: str, completion: str, max_total_tokens: int) -> tuple[list[int], int]:
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True,
    )
    answer_ids = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prompt_ids or len(answer_ids) < 2:
        raise ValueError("Empty prompt/completion")
    if len(prompt_ids) + len(answer_ids) > max_total_tokens:
        raise ValueError(f"Would truncate prompt/completion: {len(prompt_ids)}+{len(answer_ids)}")
    return prompt_ids + answer_ids, len(prompt_ids)


def sequence_logp(model, tokenizer, prompt: str, completion: str, torch, max_total_tokens: int):
    ids, prompt_len = encode_pair(tokenizer, prompt, completion, max_total_tokens)
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    attention = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    logits = output.logits[:, prompt_len - 1:-1].float()
    labels = input_ids[:, prompt_len:]
    logps = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return logps.sum(dim=-1)


def changed_token_positions(chosen_ids: list[int], rejected_ids: list[int]) -> tuple[list[int], list[int]]:
    """Return changed-token positions plus one shared boundary token on each side.

    Keeping a small boundary makes insertions (for example, an appended false
    claim) score both the inserted text and the chosen completion's stopping
    point. Equal spans elsewhere in the two answers are excluded from DPO.
    """
    matcher = difflib.SequenceMatcher(a=chosen_ids, b=rejected_ids, autojunk=False)
    chosen_mask: set[int] = set()
    rejected_mask: set[int] = set()
    has_difference = False
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            continue
        has_difference = True
        chosen_lo = max(0, a0 - 1)
        chosen_hi = min(len(chosen_ids), max(a1, a0 + 1) + 1)
        rejected_lo = max(0, b0 - 1)
        rejected_hi = min(len(rejected_ids), max(b1, b0 + 1) + 1)
        chosen_mask.update(range(chosen_lo, chosen_hi))
        rejected_mask.update(range(rejected_lo, rejected_hi))
    if not has_difference:
        raise ValueError("DPO chosen/rejected token sequences are identical")
    if not chosen_mask or not rejected_mask:
        raise ValueError("Changed-token DPO mask unexpectedly empty")
    return sorted(chosen_mask), sorted(rejected_mask)


def completion_token_ids(tokenizer, completion: str) -> list[int]:
    ids = tokenizer.encode(completion, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    if not ids:
        raise ValueError("Empty DPO completion tokenization")
    return ids


def sequence_logp_at_positions(model, tokenizer, prompt: str, completion: str,
                               positions: list[int], torch, max_total_tokens: int):
    """Sum completion log probabilities only at the selected token positions."""
    ids, prompt_len = encode_pair(tokenizer, prompt, completion, max_total_tokens)
    answer_len = len(ids) - prompt_len
    if not positions or min(positions) < 0 or max(positions) >= answer_len:
        raise ValueError("Changed-token mask is outside the completion token range")
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    attention = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    logits = output.logits[:, prompt_len - 1:-1].float()
    labels = input_ids[:, prompt_len:]
    logps = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    index = torch.tensor(positions, dtype=torch.long, device=logps.device)
    return logps.index_select(1, index).sum(dim=-1)


def decision_handling_target(row: dict[str, Any]) -> str:
    state = row["new_state"]
    return f"最终处理判断：{state['final_decision']}。当前处置：{state['current_handling']}。"


def eval_nll(model, tokenizer, rows, torch, max_total_tokens: int) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for row in rows:
            target = decision_handling_target(row)
            logp = sequence_logp(model, tokenizer, row["prompt"], target, torch, max_total_tokens)
            total += float((-logp / max(1, len(tokenizer.encode(target, add_special_tokens=False)) + 1)).item())
            count += 1
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rule-aux-weight", type=float, default=0.05)
    parser.add_argument("--chosen-sft-weight", type=float, default=0.0)
    parser.add_argument("--preference-scope", choices=("full_answer", "changed_token_span"),
                        default="full_answer",
                        help="Apply DPO to the whole completion or only to differing token spans plus boundary tokens")
    parser.add_argument("--rule-dropout-prob", type=float, default=0.0,
                        help="Probability of removing the explicit updated-rule line from case prompts")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after this many consecutive dev NLL evaluations fail to improve; 0 disables")
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="Validate and finalize a completed run with a named PEFT adapter folder")
    args = parser.parse_args()
    if not 0.0 <= args.rule_dropout_prob <= 1.0:
        parser.error("--rule-dropout-prob must be between 0 and 1")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be nonnegative")
    if args.finalize_existing:
        finalize_existing(args)
        return
    train = read_jsonl(args.data / "train_dpo.jsonl")
    dev = read_jsonl(args.data / "dev_closed_book.jsonl")
    rule_pairs = read_jsonl(args.data / "rule_pairs.jsonl")
    if len(train) != 1620 or len(dev) != 540 or len(rule_pairs) != 9:
        raise ValueError("Unexpected v1.1 DPO train/dev/rule counts")
    if any(row["chosen"] == row["rejected"] for row in train):
        raise ValueError("A DPO pair has identical chosen/rejected completions")
    if args.rule_dropout_prob and any(
            sum(line.startswith("当前更新后的上层规则：") for line in row["prompt"].splitlines()) != 1
            for row in train):
        raise ValueError("Rule dropout requires exactly one explicit updated-rule line in every train case prompt")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {args.output}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("DPOTransition requires CUDA BF16 on the H100")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(args.initial_adapter), is_trainable=True,
                                      adapter_name="policy", local_files_only=True)
    model.load_adapter(str(args.initial_adapter), adapter_name="reference", is_trainable=False,
                       local_files_only=True)
    model.set_adapter("policy")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name and "policy" in name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable DPO policy adapter parameters")
    model.config.use_cache = False

    args.output.mkdir(parents=True, exist_ok=False)
    best_dir = args.output / "best_adapter"
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    manifest = {
        "method": "DPOTransition", "protocol": "main_experiment_formal_v1_1",
        "seed": args.seed, "model": str(args.model), "initial_adapter": str(args.initial_adapter),
        "initial_adapter_sha256": digest(args.initial_adapter / "adapter_model.safetensors"),
        "data_sha256": {name: digest(args.data / name) for name in
                        ("train_dpo.jsonl", "dev_closed_book.jsonl", "rule_pairs.jsonl", "prepared_manifest.json")},
        "train_rows": len(train), "train_families": 540, "events": len(rule_pairs),
        "objective": ("DPO beta=0.1 against frozen copy of LoRA-SFT-RuleCase, plus 0.05 rule-QA auxiliary CE; "
                      f"preference_scope={args.preference_scope}"),
        "preference_scope": args.preference_scope,
        "negative_protocol": "stale old-rule answer for changed cases; explicit false update-effect claim for unaffected cases",
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "beta": args.beta, "rule_aux_weight": args.rule_aux_weight,
        "chosen_sft_weight": args.chosen_sft_weight, "weight_decay": 0.01,
        "early_stop_patience": args.early_stop_patience,
        "rule_dropout_prob": args.rule_dropout_prob,
        "case_prompt_policy": ("randomly remove only the explicit updated-rule line; keep rule-QA auxiliary prompts unchanged"
                               if args.rule_dropout_prob else "keep the explicit updated-rule line"),
        "checkpoint_selection": "minimum closed-book development final-decision/current-handling NLL",
        "test_split_read": False, "test_gold_read": False,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__, "status": "running",
    }
    write_json(args.output / "run_manifest.json", manifest)
    rng = random.Random(args.seed)
    history = []
    best_dev = float("inf")
    stale_evaluations = 0
    early_stopped = False
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        losses, margins = [], []
        dpo_mask_token_counts = []
        dropped_rule_rows = 0
        for row in batch:
            case_prompt, dropped_rule = maybe_remove_updated_rule_line(
                row["prompt"], args.rule_dropout_prob, rng)
            dropped_rule_rows += int(dropped_rule)
            model.set_adapter("policy")
            if args.preference_scope == "changed_token_span":
                chosen_ids = completion_token_ids(tokenizer, row["chosen"])
                rejected_ids = completion_token_ids(tokenizer, row["rejected"])
                chosen_positions, rejected_positions = changed_token_positions(chosen_ids, rejected_ids)
                policy_chosen = sequence_logp_at_positions(
                    model, tokenizer, case_prompt, row["chosen"], chosen_positions,
                    torch, args.max_total_tokens)
                policy_chosen_full = (sequence_logp(model, tokenizer, case_prompt, row["chosen"], torch,
                                                    args.max_total_tokens)
                                      if args.chosen_sft_weight else None)
                dpo_mask_token_counts.append(len(chosen_positions) + len(rejected_positions))
            else:
                policy_chosen = sequence_logp(model, tokenizer, case_prompt, row["chosen"], torch,
                                              args.max_total_tokens)
                policy_chosen_full = policy_chosen
                rejected_positions = None
                dpo_mask_token_counts.append(0)
            if args.preference_scope == "changed_token_span":
                policy_rejected = sequence_logp_at_positions(
                    model, tokenizer, case_prompt, row["rejected"], rejected_positions,
                    torch, args.max_total_tokens)
            else:
                policy_rejected = sequence_logp(model, tokenizer, case_prompt, row["rejected"], torch,
                                                args.max_total_tokens)
            with torch.no_grad():
                model.set_adapter("reference")
                if args.preference_scope == "changed_token_span":
                    ref_chosen = sequence_logp_at_positions(
                        model, tokenizer, case_prompt, row["chosen"], chosen_positions,
                        torch, args.max_total_tokens)
                    ref_rejected = sequence_logp_at_positions(
                        model, tokenizer, case_prompt, row["rejected"], rejected_positions,
                        torch, args.max_total_tokens)
                else:
                    ref_chosen = sequence_logp(model, tokenizer, case_prompt, row["chosen"], torch,
                                               args.max_total_tokens)
                    ref_rejected = sequence_logp(model, tokenizer, case_prompt, row["rejected"], torch,
                                                 args.max_total_tokens)
            model.set_adapter("policy")
            margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            loss = -torch.nn.functional.logsigmoid(args.beta * margin).mean()
            if args.chosen_sft_weight:
                if policy_chosen_full is None:
                    raise RuntimeError("Full chosen-answer log probability missing for SFT anchor")
                chosen_nll = -policy_chosen_full.mean() / max(1, len(tokenizer.encode(row["chosen"], add_special_tokens=False)) + 1)
                loss = loss + args.chosen_sft_weight * chosen_nll
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite DPO loss at step {step}")
            (loss / args.batch_size).backward()
            losses.append(float(loss.detach()))
            margins.append(float(margin.detach().mean()))

        rule = rule_pairs[(step - 1) % len(rule_pairs)]
        question = rule["question"] if step % 2 else rule["rephrase"]
        target = rule["new_rule"]
        rule_logp = sequence_logp(model, tokenizer, question, target, torch, args.max_total_tokens)
        rule_loss = -rule_logp.mean() / max(1, len(tokenizer.encode(target, add_special_tokens=False)) + 1)
        (args.rule_aux_weight * rule_loss).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if step <= 5:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * step / 5.0
        optimizer.step()
        record = {"step": step, "dpo_loss": sum(losses) / len(losses),
                  "policy_minus_reference_margin": sum(margins) / len(margins),
                  "mean_dpo_masked_tokens": (sum(dpo_mask_token_counts) / max(1, len(dpo_mask_token_counts))
                                              if args.preference_scope == "changed_token_span" else None),
                  "rule_dropped_rows": dropped_rule_rows,
                  "rule_aux_nll": float(rule_loss.detach()),
                  "elapsed_sec": round(time.time() - started, 1),
                  "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        history.append(record)
        if step == 1 or step % 10 == 0:
            print(json.dumps(record, ensure_ascii=False), flush=True)

        if step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            model.set_adapter("policy")
            dev_nll = eval_nll(model, tokenizer, dev, torch, args.max_total_tokens)
            record["dev_decision_handling_nll"] = dev_nll
            write_json(args.output / "history.json", history)
            if dev_nll < best_dev:
                best_dev = dev_nll
                stale_evaluations = 0
                model.save_pretrained(best_dir, safe_serialization=True, selected_adapters=["policy"])
                tokenizer.save_pretrained(best_dir)
                write_json(args.output / "selection.json", {"step": step, "dev_nll": dev_nll})
            else:
                stale_evaluations += 1
            write_json(args.output / "history.json", history)
            print(json.dumps({"step": step, "dev_decision_handling_nll": dev_nll,
                              "best_dev_nll": best_dev,
                              "stale_evaluations": stale_evaluations}, ensure_ascii=False), flush=True)
            if args.early_stop_patience > 0 and stale_evaluations >= args.early_stop_patience:
                early_stopped = True
                print(json.dumps({"early_stop": "development NLL patience reached",
                                  "step": step, "best_dev_nll": best_dev}, ensure_ascii=False), flush=True)
                break

    adapter_path = best_dir / "policy" / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise RuntimeError("No best DPO adapter checkpoint was written")
    manifest.update({"status": "early_stopped_dev_nll" if early_stopped else "complete",
                     "steps_completed": history[-1]["step"], "elapsed_sec": round(time.time() - started, 1),
                     "best_selection": json.loads((args.output / "selection.json").read_text()),
                     "adapter_sha256": digest(adapter_path), "adapter_relative_path": "best_adapter/policy",
                     "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    write_json(args.output / "run_manifest.json", manifest)
    (args.output / "complete.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
