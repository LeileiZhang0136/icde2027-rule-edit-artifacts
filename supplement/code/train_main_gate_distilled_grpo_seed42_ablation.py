#!/usr/bin/env python3
"""Train v1.1 GateDistilledGRPO from fresh SFT/DPO adapters, train split only.

The frozen decision-disagreement gate routes SFT/DPO teacher outputs on train
rows. A deterministic field verifier retains useful train-only replay cases;
GRPO then optimizes a decision-gated eight-field reward, with DPO-reference KL,
verifier-gated target replay, and an upper-rule QA auxiliary objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


FIELD_LABELS = {
    "decision": "最终处理判断",
    "handling": "当前处置",
    "obligations_added": "新增义务",
    "obligations_removed": "取消义务",
    "obligations_unchanged": "保持义务",
    "human_review": "人工复核",
    "reasoning": "理由",
}
ALL_LABELS = tuple(FIELD_LABELS.values()) + (
    "义务状态变化", "新增程序", "取消程序", "当前适用义务及状态", "当前许可", "当前禁止",
)
BASIS_TEXT = {
    "action_date": "行为发生日期", "age_threshold": "适用年龄门槛",
    "birth_date": "出生日期及行为时年龄", "domain_rule": "下游领域规则",
    "secondary_gate_met": "领域前置条件是否满足", "upper_rule_version": "当前上层规则版本",
    "feature_type": "特征类型", "identity_inference": "是否推断身份",
    "identity_feature_generated": "是否生成身份特征", "separate_consent": "单独授权状态",
    "encrypted": "加密状态", "purpose_matches_domain": "用途是否符合领域规则",
    "retention_days": "保存期限", "impact_level": "影响程度",
    "adverse_proposed": "是否提出不利处理", "formal_confirmation": "正式确认状态",
    "materials_checked": "材料审阅情况", "system_reason_checked": "理由审阅情况",
    "reviewer_can_override": "审查人能否修改建议", "review_reason_recorded": "是否记录审查理由",
    "days_since_last_check": "距上次核验的时间", "independent_source_count": "独立证据来源数量",
    "core_claim_true": "核心事实是否成立", "affirmative_authorization": "是否存在明确授权",
    "explicit_objection": "是否存在明确反对", "restricted_condition": "限制条件是否适用",
    "urgent": "是否紧急", "immediate_safety_risk": "是否存在即时安全风险",
    "safe_alternative_available": "是否有安全替代方案",
    "upper_safety_restriction_applies": "上层安全限制是否适用",
    "local_exception_allows_action": "下游例外是否允许行动", "direct_conflict": "上下层规则是否冲突",
    "independent_safeguard_complete": "独立保障条件是否完成", "filed_date": "提交日期",
    "request_status": "申请当前状态", "legacy_requirement_met": "旧标准条件是否满足",
}
FIELDS = ("decision", "handling", "obligations_added", "obligations_removed",
          "obligations_unchanged", "human_review", "reasoning", "internal_consistency")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_all(text: str, label: str) -> list[str]:
    starts = list(re.finditer(rf"{re.escape(label)}\s*[：:]", text))
    out = []
    next_pattern = re.compile(r"(?:" + "|".join(re.escape(x) for x in ALL_LABELS if x != label) + r")\s*[：:]")
    for match in starts:
        end = len(text)
        next_match = next_pattern.search(text, match.end())
        if next_match:
            end = next_match.start()
        value = text[match.end():end].strip().strip(" \t\r\n。.!?；;")
        out.append(value)
    return out


def extract(text: str, label: str) -> str | None:
    values = extract_all(text, label)
    return values[0] if values else None


def parse_set(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    value = raw.strip().strip("。.!?；;")
    if value.lower() in {"无", "none", "null", "不适用"}:
        return ()
    items = [part.strip().strip("。.!?；;") for part in re.split(r"[；;]", value)]
    return tuple(sorted(set(part for part in items if part and part not in {"无", "none", "null"})))


def parse_review(raw: str | None) -> tuple[Any, ...] | None:
    if raw is None:
        return None
    compact = re.sub(r"\s+", "", raw)
    if any(term in compact for term in ("不需要", "无需", "不要求", "不触发", "不适用")):
        return (False, None, None)
    if "需要" not in compact and "应当复核" not in compact:
        return None
    standard = re.search(r"标准为\s*(.+?)(?=[，,;；。.!?]|$)", raw)
    status = re.search(r"状态为\s*(.+?)(?=[，,;；。.!?]|$)", raw)
    if not standard or not status:
        return None
    return (True, re.sub(r"\s+", "", standard.group(1)), re.sub(r"\s+", "", status.group(1)))


def score_answer(answer: str, row: dict[str, Any], reward_mode: str = "decision_gated") -> dict[str, Any]:
    expected = row["target_constraints"]
    delta = row["delta"]
    state = row["new_state"]
    actual = {
        "decision": extract(answer, FIELD_LABELS["decision"]),
        "handling": extract(answer, FIELD_LABELS["handling"]),
        "obligations_added": parse_set(extract(answer, FIELD_LABELS["obligations_added"])),
        "obligations_removed": parse_set(extract(answer, FIELD_LABELS["obligations_removed"])),
        "obligations_unchanged": parse_set(extract(answer, FIELD_LABELS["obligations_unchanged"])),
        "human_review": parse_review(extract(answer, FIELD_LABELS["human_review"])),
    }
    want_review = expected["human_review"]
    expected_review = ((True, re.sub(r"\s+", "", str(want_review["reason"])),
                        re.sub(r"\s+", "", str(want_review["status"])))
                       if want_review["required"] else (False, None, None))
    wanted = {
        "decision": expected["final_decision"],
        "handling": expected["current_handling"],
        "obligations_added": tuple(sorted(delta["obligations_added"])),
        "obligations_removed": tuple(sorted(delta["obligations_removed"])),
        "obligations_unchanged": tuple(sorted(delta["obligations_unchanged"])),
        "human_review": expected_review,
    }
    reasoning = extract(answer, FIELD_LABELS["reasoning"])
    reasoning_ok = bool(reasoning)
    basis = state.get("basis", [])
    missing_basis = [BASIS_TEXT.get(key, key) for key in basis if BASIS_TEXT.get(key, key) not in (reasoning or "")]
    reasoning_ok = reasoning_ok and not missing_basis
    outcome = delta.get("result_type")
    if outcome == "unaffected":
        compact = re.sub(r"\s+", "", reasoning or "")
        no_effect = any(phrase in compact for phrase in ("不改变本案处理结果", "不改变本案", "不影响本案"))
        false_change = (("改变本案" in compact and "不改变本案" not in compact)
                        or ("影响本案" in compact and "不影响本案" not in compact))
        reasoning_ok = reasoning_ok and no_effect and not false_change
    if outcome == "exception_preserves":
        reasoning_ok = reasoning_ok and any(term in (reasoning or "") for term in ("例外", "豁免", "阻断", "不适用"))

    fields = {key: actual[key] is not None and actual[key] == wanted[key]
              for key in ("decision", "handling", "obligations_added", "obligations_removed",
                          "obligations_unchanged", "human_review")}
    fields["reasoning"] = reasoning_ok

    duplicates_ok = True
    for field, label in FIELD_LABELS.items():
        values = extract_all(answer, label)
        if len(values) > 1:
            parsed_values = [parse_set(v) if field.startswith("obligations_") else
                             parse_review(v) if field == "human_review" else v for v in values]
            if len({json.dumps(v, ensure_ascii=False, sort_keys=True) for v in parsed_values}) > 1:
                duplicates_ok = False
    sets = [actual[name] for name in ("obligations_added", "obligations_removed", "obligations_unchanged")]
    sets_complete = all(value is not None for value in sets)
    pairwise_disjoint = sets_complete and not (set(sets[0]) & set(sets[1]) or
                                               set(sets[0]) & set(sets[2]) or
                                               set(sets[1]) & set(sets[2]))
    contradiction_phrase = any(term in answer for term in (
        "与上文结论相反", "与前述处置相反", "自相矛盾", "以上判断互相冲突"))
    fields["internal_consistency"] = bool(
        all(actual[key] is not None for key in wanted) and duplicates_ok and pairwise_disjoint
        and not contradiction_phrase
    )
    if reward_mode == "decision_gated":
        # A wrong decision yields zero; otherwise score the remaining seven fields.
        if not fields["decision"]:
            reward = 0.0
        else:
            reward = sum(int(fields[key]) for key in FIELDS if key != "decision") / 7.0
    elif reward_mode == "all_fields":
        reward = sum(int(fields[key]) for key in FIELDS) / len(FIELDS)
    else:
        raise ValueError(f"Unknown online reward mode: {reward_mode}")
    return {"score": reward, "fields": fields, "missing_basis": missing_basis,
            "duplicates_consistent": duplicates_ok, "obligation_sets_disjoint": bool(pairwise_disjoint)}


def calibrate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = 0
    false_accepts = Counter()
    mutations = 0
    for row in rows:
        base = score_answer(row["target"], row)
        accepted += int(all(base["fields"].values()))
        wrong = {
            "decision": "__wrong_decision__", "handling": "__wrong_handling__",
            "obligations_added": "__wrong_added__", "obligations_removed": "__wrong_removed__",
            "obligations_unchanged": "__wrong_unchanged__", "human_review": "不需要",
            "reasoning": "理由：不相关。",
        }
        for field, label in FIELD_LABELS.items():
            values = extract_all(row["target"], label)
            if not values:
                continue
            replacement = wrong[field]
            if field.startswith("obligations_"):
                replacement = "__wrong_item__"
            elif field == "human_review" and not row["target_constraints"]["human_review"]["required"]:
                replacement = "需要，标准为__wrong_standard__，状态为pending"
            old = values[0]
            mutated = row["target"].replace(f"{label}：{old}", f"{label}：{replacement}", 1)
            result = score_answer(mutated, row)
            mutations += 1
            false_accepts[field] += int(result["fields"].get(field, False))
        # Add a conflicting duplicate decision section; consistency must fail.
        conflict = row["target"] + " 最终处理判断：__conflicting_decision__。"
        conflict_result = score_answer(conflict, row)
        mutations += 1
        false_accepts["internal_consistency"] += int(conflict_result["fields"]["internal_consistency"])
    total = len(rows)
    passed = accepted == total and all(value == 0 for value in false_accepts.values())
    report = {
        "protocol": "main_v1_1_gate_grpo_verifier_calibration_v1",
        "rows": total, "gold_all_field_acceptance": accepted,
        "mutations": mutations, "mutation_false_accepts_by_field": dict(false_accepts),
        "decision_gate": "wrong decision yields zero; otherwise mean of remaining seven binary fields",
        "passed": passed,
        "limitation": "Deterministic lexical/constraint verifier for canonical train targets; not a universal semantic judge for arbitrary prose.",
    }
    if not passed:
        raise RuntimeError(f"GRPO verifier calibration failed: {report}")
    return report


def decision_handling_target(row):
    s = row["new_state"]
    return f"最终处理判断：{s['final_decision']}。当前处置：{s['current_handling']}。"


def chat_ids(tok, prompt: str, answer: str | None = None):
    prompt_ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                         add_generation_prompt=True)
    if answer is None:
        return prompt_ids, None
    answer_ids = tok.encode(answer, add_special_tokens=False) + [tok.eos_token_id]
    return prompt_ids, answer_ids


def mean_sequence_logp(model, tok, prompt: str, answer: str, torch, max_total_tokens: int):
    prompt_ids, answer_ids = chat_ids(tok, prompt, answer)
    if len(prompt_ids) + len(answer_ids) > max_total_tokens:
        raise ValueError(f"GRPO sequence would truncate: {len(prompt_ids)}+{len(answer_ids)}")
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device="cuda")
    mask = torch.ones_like(ids)
    output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    start = len(prompt_ids) - 1
    logits = output.logits[:, start:start + len(answer_ids)].float()
    labels = ids[:, len(prompt_ids):]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return logp.mean()


def generate_group(model, tok, prompt: str, count: int, max_new_tokens: int,
                   temperature: float, top_p: float, torch, seed: int) -> list[str]:
    model.eval()
    prompt_text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tok(prompt_text, return_tensors="pt", truncation=False).to("cuda")
    if encoded.input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("GRPO prompt plus output budget exceeds Qwen context")
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=True, temperature=temperature,
                                top_p=top_p, num_return_sequences=count,
                                max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                                eos_token_id=tok.eos_token_id, use_cache=True)
    start = encoded.input_ids.shape[1]
    return [tok.decode(seq[start:], skip_special_tokens=True).strip() for seq in output]


def build_gate_memory(model, tok, rows, output: Path, args, torch):
    memory = []
    audit = []
    route_counts = Counter()
    for start in range(0, len(rows), args.probe_batch_size):
        batch = rows[start:start + args.probe_batch_size]
        for adapter_name in ("sft_teacher", "reference"):
            model.set_adapter(adapter_name)
            model.eval()
            prompts = [tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                               tokenize=False, add_generation_prompt=True) for row in batch]
            encoded = tok(prompts, return_tensors="pt", padding=True, truncation=False).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=args.teacher_max_new_tokens,
                                            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                                            use_cache=True)
            answers = tok.batch_decode(generated[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
            if adapter_name == "sft_teacher":
                sft_answers = [x.strip() for x in answers]
            else:
                dpo_answers = [x.strip() for x in answers]
            del encoded, generated
        for row, sft_answer, dpo_answer in zip(batch, sft_answers, dpo_answers):
            sft_score = score_answer(sft_answer, row)
            dpo_score = score_answer(dpo_answer, row)
            sft_decision = extract(sft_answer, FIELD_LABELS["decision"])
            dpo_decision = extract(dpo_answer, FIELD_LABELS["decision"])
            agrees = bool(sft_decision and dpo_decision and sft_decision == dpo_decision)
            if args.gate_selection == "random_teacher":
                source = args.gate_rng.choice(("LoRA-SFT", "DPOTransition"))
            else:
                source = "LoRA-SFT" if agrees else "DPOTransition"
            chosen = sft_answer if agrees else dpo_answer
            judged = sft_score if agrees else dpo_score
            verified = [field for field, ok in judged["fields"].items() if ok]
            route_counts[source] += 1
            route_counts["decision_agreement"] += int(agrees)
            route_counts["decision_parse_failure"] += int(not sft_decision or not dpo_decision)
            if verified:
                replay_weight = max(0.1, sum(int(judged["fields"][f]) for f in FIELDS) / len(FIELDS))
                memory.append({"id": row["id"], "case_family_id": row["case_family_id"],
                               "prompt": row["prompt"], "target": row["target"],
                               "sample_weight": float(row.get("sample_weight", 1 / 3)) * replay_weight,
                               "gate_teacher": source, "verified_fields": verified,
                               "verifier_reward": judged["score"], "initial_answer": chosen})
            audit.append({"id": row["id"], "case_family_id": row["case_family_id"],
                          "sft_decision": sft_decision, "dpo_decision": dpo_decision,
                          "decision_agreement": agrees, "selected_teacher": source,
                          "sft_reward": sft_score["score"], "dpo_reward": dpo_score["score"],
                          "selected_reward": judged["score"], "verified_fields": verified,
                          "sft_sha256": hashlib.sha256(sft_answer.encode()).hexdigest(),
                          "dpo_sha256": hashlib.sha256(dpo_answer.encode()).hexdigest()})
        print(json.dumps({"gate_probe_done": min(start + len(batch), len(rows)), "total": len(rows),
                          "retained": len(memory)}, ensure_ascii=False), flush=True)
    if not memory:
        raise RuntimeError("Frozen gate/verifier produced no train-only replay examples")
    write_jsonl(output / "gate_memory.jsonl", memory)
    write_json(output / "gate_audit.json", {
        "protocol": ("v1_1_train_only_random_teacher_verifier_replay"
                     if args.gate_selection == "random_teacher"
                     else "v1_1_train_only_decision_disagreement_gate_verifier_replay"),
        "gate_selection": args.gate_selection,
        "rows_probed": len(rows), "rows_retained": len(memory), "route_counts": dict(route_counts),
        "rows": audit, "test_split_read": False, "test_gold_read": False,
        "teacher_adapter_sha256": file_digest(args.sft_adapter / "adapter_model.safetensors"),
        "dpo_adapter_sha256": adapter_digest(args.dpo_adapter),
    })
    return memory


def build_random_gate_memory_from_cache(model, tok, rows, output: Path, args, torch,
                                       cached_memory: list[dict[str, Any]],
                                       cached_audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Randomly route teachers while regenerating only uncached teacher answers.

    The frozen full run stores the selected teacher answer for every train row.
    For rows whose random teacher matches that stored teacher, the exact answer
    is reused. Only rows routed to the other frozen teacher require generation.
    """
    memory_by_id = {str(item["id"]): item for item in cached_memory}
    audit_by_id = {str(item["id"]): item for item in cached_audit["rows"]}
    if set(memory_by_id) != {str(row["id"]) for row in rows}:
        raise ValueError("Random-gate cache must cover every frozen train row")

    output_memory: list[dict[str, Any]] = []
    output_audit: list[dict[str, Any]] = []
    route_counts = Counter()
    reused_answers = 0
    regenerated_answers = 0
    for start in range(0, len(rows), args.probe_batch_size):
        batch = rows[start:start + args.probe_batch_size]
        selected_by_id: dict[str, str] = {}
        answer_by_id: dict[str, str] = {}
        regenerate: dict[str, list[dict[str, Any]]] = {"LoRA-SFT": [], "DPOTransition": []}
        for row in batch:
            row_id = str(row["id"])
            selected = args.gate_rng.choice(("LoRA-SFT", "DPOTransition"))
            selected_by_id[row_id] = selected
            cached = memory_by_id[row_id]
            if selected == cached["gate_teacher"]:
                answer_by_id[row_id] = cached["initial_answer"]
                reused_answers += 1
            else:
                regenerate[selected].append(row)

        for teacher, teacher_rows in regenerate.items():
            for offset in range(0, len(teacher_rows), args.probe_batch_size):
                chunk = teacher_rows[offset:offset + args.probe_batch_size]
                adapter_name = "sft_teacher" if teacher == "LoRA-SFT" else "reference"
                model.set_adapter(adapter_name)
                model.eval()
                prompts = [tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                   tokenize=False, add_generation_prompt=True)
                           for row in chunk]
                encoded = tok(prompts, return_tensors="pt", padding=True, truncation=False).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded, do_sample=False, max_new_tokens=args.teacher_max_new_tokens,
                        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
                answers = tok.batch_decode(generated[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
                for row, answer in zip(chunk, answers):
                    answer_by_id[str(row["id"])] = answer.strip()
                regenerated_answers += len(chunk)
                del encoded, generated

        for row in batch:
            row_id = str(row["id"])
            selected = selected_by_id[row_id]
            answer = answer_by_id[row_id]
            judged = score_answer(answer, row)
            verified = [field for field, ok in judged["fields"].items() if ok]
            route_counts[selected] += 1
            if verified:
                replay_weight = max(0.1, len(verified) / len(FIELDS))
                output_memory.append({
                    "id": row_id, "case_family_id": row["case_family_id"],
                    "prompt": row["prompt"], "target": row["target"],
                    "sample_weight": float(row.get("sample_weight", 1 / 3)) * replay_weight,
                    "gate_teacher": selected, "verified_fields": verified,
                    "verifier_reward": judged["score"], "initial_answer": answer,
                })
            prior = audit_by_id[row_id]
            output_audit.append({
                "id": row_id, "case_family_id": row["case_family_id"],
                "event_id": row.get("event_id"), "domain_id": row.get("domain_id"),
                "difficulty": row.get("difficulty"),
                "selected_teacher": selected,
                "cached_gate_teacher": memory_by_id[row_id]["gate_teacher"],
                "decision_agreement": prior.get("decision_agreement"),
                "selected_reward": judged["score"], "verified_fields": verified,
                "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "test_split_read": False, "test_gold_read": False,
            })
        print(json.dumps({"random_gate_probe_done": min(start + len(batch), len(rows)),
                          "total": len(rows), "retained": len(output_memory),
                          "reused_answers": reused_answers,
                          "regenerated_answers": regenerated_answers}, ensure_ascii=False), flush=True)

    if not output_memory:
        raise RuntimeError("Random teacher routing produced no verifier-backed replay cases")
    write_jsonl(output / "gate_memory.jsonl", output_memory)
    write_json(output / "gate_audit.json", {
        "protocol": "v1_1_random_teacher_train_only_verifier_replay_from_frozen_teacher_cache",
        "gate_selection": "uniform random between frozen LoRA-SFT and DPOTransition teachers",
        "rows_probed": len(rows), "rows_retained": len(output_memory),
        "route_counts": dict(route_counts), "reused_answers": reused_answers,
        "regenerated_answers": regenerated_answers,
        "source_gate_audit_sha256": file_digest(args.gate_memory_cache / "gate_audit.json"),
        "source_gate_memory_sha256": file_digest(args.gate_memory_cache / "gate_memory.jsonl"),
        "rows": output_audit, "test_split_read": False, "test_gold_read": False,
        "teacher_adapter_sha256": file_digest(args.sft_adapter / "adapter_model.safetensors"),
        "dpo_adapter_sha256": adapter_digest(args.dpo_adapter),
    })
    return output_memory


def file_digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def adapter_digest(path: Path) -> str:
    for name in ("adapter_model.safetensors", "policy/adapter_model.safetensors", "default/adapter_model.safetensors"):
        candidate = path / name
        if candidate.is_file():
            return file_digest(candidate)
    raise FileNotFoundError(f"No LoRA weights below {path}")


def load_cached_gate_memory(cache_dir: Path, train_rows: list[dict[str, Any]], args
                           ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load the completed main-run gate memory after verifying its provenance."""
    required = ("run_manifest.json", "gate_audit.json", "gate_memory.jsonl")
    missing = [name for name in required if not (cache_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete gate-memory cache: {missing}")
    source_manifest = json.loads((cache_dir / "run_manifest.json").read_text(encoding="utf-8"))
    source_audit = json.loads((cache_dir / "gate_audit.json").read_text(encoding="utf-8"))
    memory = read_jsonl(cache_dir / "gate_memory.jsonl")
    if source_manifest.get("status") != "complete":
        raise ValueError("Gate-memory source run is incomplete")
    if source_manifest.get("gate") != "choose SFT when SFT/DPO decision fields agree; otherwise choose DPO; gate probes train split only":
        raise ValueError("Gate-memory source does not use the frozen decision-disagreement gate")
    if source_manifest.get("test_split_read") is not False or source_manifest.get("test_gold_read") is not False:
        raise ValueError("Gate-memory source manifest is not explicitly train-only")
    if source_audit.get("test_split_read") is not False or source_audit.get("test_gold_read") is not False:
        raise ValueError("Gate-memory source audit is not explicitly train-only")
    for name in ("train_grpo.jsonl", "dev_closed_book.jsonl", "rule_pairs.jsonl", "prepared_manifest.json"):
        if source_manifest.get("data_sha256", {}).get(name) != file_digest(args.data / name):
            raise ValueError(f"Gate-memory cache used different prepared data: {name}")
    sft_sha = file_digest(args.sft_adapter / "adapter_model.safetensors")
    dpo_sha = adapter_digest(args.dpo_adapter)
    if source_manifest.get("sft_adapter_sha256") != sft_sha or source_audit.get("teacher_adapter_sha256") != sft_sha:
        raise ValueError("Gate-memory cache uses a different SFT teacher")
    if source_manifest.get("dpo_adapter_sha256") != dpo_sha or source_audit.get("dpo_adapter_sha256") != dpo_sha:
        raise ValueError("Gate-memory cache uses a different DPO teacher")

    train_by_id = {str(row["id"]): row for row in train_rows}
    memory_by_id = {str(item["id"]): item for item in memory}
    audit_rows = source_audit.get("rows", [])
    audit_by_id = {str(item["id"]): item for item in audit_rows}
    expected_ids = set(train_by_id)
    if (len(train_by_id) != len(train_rows) or len(memory_by_id) != len(memory)
            or len(audit_by_id) != len(audit_rows) or set(memory_by_id) != expected_ids
            or set(audit_by_id) != expected_ids):
        raise ValueError("Gate-memory cache does not cover exactly the frozen train rows")
    for row_id, row in train_by_id.items():
        item = memory_by_id[row_id]
        audit = audit_by_id[row_id]
        if item.get("case_family_id") != row.get("case_family_id") or item.get("target") != row.get("target"):
            raise ValueError(f"Gate-memory cached row differs from current train row: {row_id}")
        if audit.get("case_family_id") != row.get("case_family_id"):
            raise ValueError(f"Gate audit family differs from current train row: {row_id}")
    cache_info = {
        "reused": True, "source_dir": str(cache_dir), "rows": len(memory),
        "source_run_manifest_sha256": file_digest(cache_dir / "run_manifest.json"),
        "source_gate_audit_sha256": file_digest(cache_dir / "gate_audit.json"),
        "source_gate_memory_sha256": file_digest(cache_dir / "gate_memory.jsonl"),
        "test_split_read": False, "test_gold_read": False,
    }
    return memory, source_audit, cache_info


def evaluate_dev(model, tok, rows, torch, args) -> float:
    model.set_adapter("policy")
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for row in rows:
            answer = decision_handling_target(row)
            logp = mean_sequence_logp(model, tok, row["prompt"], answer, torch, args.max_total_tokens)
            total += float(-logp)
            count += 1
    return total / max(1, count)


def train(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train_rows = read_jsonl(args.data / "train_grpo.jsonl")
    dev_rows = read_jsonl(args.data / "dev_closed_book.jsonl")
    rules = read_jsonl(args.data / "rule_pairs.jsonl")
    if (len(train_rows), len(dev_rows), len(rules)) != (1620, 540, 9):
        raise ValueError("Unexpected frozen v1.1 train/dev/rule counts")
    if any(row.get("split") != "train" for row in train_rows) or any(row.get("split") != "development" for row in dev_rows):
        raise ValueError("GRPO training/dev split contamination")
    if len({row["case_family_id"] for row in train_rows}) != 540:
        raise ValueError("GRPO train family count must be 540")
    calibration = calibrate(train_rows)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {args.output}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GateDistilledGRPO requires CUDA BF16")
    random.seed(args.seed)
    rng = random.Random(args.seed)
    # Keep random teacher routing deterministic without shifting the training
    # sampler's seed-42 sequence.
    args.gate_rng = random.Random(args.seed + 104729)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(args.dpo_adapter), is_trainable=True,
                                      adapter_name="policy", local_files_only=True)
    model.load_adapter(str(args.dpo_adapter), adapter_name="reference", is_trainable=False, local_files_only=True)
    model.load_adapter(str(args.sft_adapter), adapter_name="sft_teacher", is_trainable=False, local_files_only=True)
    model.set_adapter("policy")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name and "policy" in name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable policy LoRA parameters")
    model.config.use_cache = True

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "verifier_calibration.json", calibration)
    if args.gate_selection == "random_teacher" and args.gate_memory_cache is not None:
        cached_memory, source_audit, gate_cache_info = load_cached_gate_memory(
            args.gate_memory_cache, train_rows, args)
        gate_memory = build_random_gate_memory_from_cache(
            model, tok, train_rows, args.output, args, torch, cached_memory, source_audit)
        gate_cache_info["mode"] = "reuse_matching_frozen_teacher_answers_regenerate_randomly_selected_alternates"
    elif args.gate_memory_cache is not None:
        gate_memory, source_audit, gate_cache_info = load_cached_gate_memory(
            args.gate_memory_cache, train_rows, args)
        copied_audit = dict(source_audit)
        copied_audit["reused_from"] = str(args.gate_memory_cache)
        write_jsonl(args.output / "gate_memory.jsonl", gate_memory)
        write_json(args.output / "gate_audit.json", copied_audit)
    else:
        gate_memory = build_gate_memory(model, tok, train_rows, args.output, args, torch)
        source_audit = None
        gate_cache_info = {"reused": False, "rows": len(gate_memory),
                           "test_split_read": False, "test_gold_read": False}
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    manifest = {
        "method": "GateDistilledGRPO", "protocol": "main_experiment_formal_v1_1",
        "seed": args.seed, "model": str(args.model), "sft_teacher": str(args.sft_adapter),
        "dpo_initial_policy_and_reference": str(args.dpo_adapter),
        "sft_adapter_sha256": adapter_digest(args.sft_adapter), "dpo_adapter_sha256": adapter_digest(args.dpo_adapter),
        "data_sha256": {n: file_digest(args.data / n) for n in ("train_grpo.jsonl", "dev_closed_book.jsonl", "rule_pairs.jsonl", "prepared_manifest.json")},
        "train_rows": len(train_rows), "train_families": 540, "dev_rows": len(dev_rows), "rule_events": len(rules),
        "gate": ("randomly choose between the same frozen SFT/DPO teachers; train rows only"
                 if args.gate_selection == "random_teacher" else
                 "choose SFT when SFT/DPO decision fields agree; otherwise choose DPO; gate probes train split only"),
        "gate_selection": args.gate_selection,
        "replay": "canonical train target replay for rows where selected teacher answer has at least one verifier-passed field",
        "replay_sampling": ("uniform over train variants/families" if args.replay_sampling == "uniform"
                            else "proportional to train-row base weight times selected-teacher verifier reward"),
        "reward": ("mean accuracy over all eight fields" if args.online_reward_mode == "all_fields"
                   else "wrong decision=0; otherwise mean accuracy of seven remaining binary verifier fields"),
        "online_policy_gradient": not args.disable_online_policy_gradient,
        "ablation": args.method_name,
        "steps": args.steps, "group_size": args.group_size, "temperature": args.temperature,
        "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate, "kl_weight": args.kl_weight,
        "gate_replay_weight": args.gate_replay_weight, "rule_aux_weight": args.rule_aux_weight,
        "gate_memory_cache": gate_cache_info,
        "checkpoint_selection": "minimum closed-book development final-decision/current-handling NLL",
        "test_split_read": False, "test_gold_read": False,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__, "status": "running",
    }
    write_json(args.output / "run_manifest.json", manifest)
    best_dir = args.output / "best_adapter"
    best_dev = float("inf")
    history = []
    train_by_id = {str(row["id"]): row for row in train_rows}
    if args.replay_sampling == "uniform":
        replay_weights = [max(float(train_by_id[str(item["id"])].get("sample_weight", 1 / 3)), 1e-6)
                          for item in gate_memory]
    else:
        replay_weights = [max(float(item["sample_weight"]), 1e-6) for item in gate_memory]
    train_weights = [max(float(row.get("sample_weight", 1 / 3)), 1e-6) for row in train_rows]
    started = time.time()
    skipped = 0
    for step in range(1, args.steps + 1):
        row = rng.choices(train_rows, weights=train_weights, k=1)[0]
        model.set_adapter("policy")
        candidates = generate_group(model, tok, row["prompt"], args.group_size, args.max_new_tokens,
                                    args.temperature, args.top_p, torch, args.seed + step)
        judged = [score_answer(answer, row, args.online_reward_mode) for answer in candidates]
        rewards = [item["score"] for item in judged]
        score_tensor = torch.tensor(rewards, dtype=torch.float32, device="cuda")
        spread = float(score_tensor.std(unbiased=False))
        optimizer.zero_grad(set_to_none=True)
        policy_logps, reference_logps = [], []
        if spread > 1e-6 or args.disable_online_policy_gradient:
            if spread > 1e-6:
                advantages = (score_tensor - score_tensor.mean()) / (score_tensor.std(unbiased=False) + 1e-4)
            else:
                advantages = torch.zeros_like(score_tensor)
                skipped += 1
            for index, answer in enumerate(candidates):
                model.set_adapter("policy")
                model.train()
                policy_logp = mean_sequence_logp(model, tok, row["prompt"], answer, torch, args.max_total_tokens)
                with torch.no_grad():
                    model.set_adapter("reference")
                    model.eval()
                    ref_logp = mean_sequence_logp(model, tok, row["prompt"], answer, torch, args.max_total_tokens)
                model.set_adapter("policy")
                model.train()
                policy_gradient_loss = (policy_logp * 0.0 if args.disable_online_policy_gradient
                                        else -advantages[index] * policy_logp)
                loss = policy_gradient_loss + args.kl_weight * (policy_logp - ref_logp).pow(2)
                (loss / args.group_size).backward()
                policy_logps.append(float(policy_logp.detach()))
                reference_logps.append(float(ref_logp.detach()))
        else:
            skipped += 1

        replay = rng.choices(gate_memory, weights=replay_weights, k=1)[0]
        model.set_adapter("policy")
        model.train()
        replay_logp = mean_sequence_logp(model, tok, replay["prompt"], replay["target"], torch, args.max_total_tokens)
        replay_loss = -replay_logp * args.gate_replay_weight
        replay_loss.backward()

        rule = rules[(step - 1) % len(rules)]
        rule_prompt = rule["question"] if step % 2 else rule["rephrase"]
        rule_logp = mean_sequence_logp(model, tok, rule_prompt, rule["new_rule"], torch, args.max_total_tokens)
        rule_loss = -rule_logp * args.rule_aux_weight
        rule_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if step <= args.warmup_steps:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * step / max(1, args.warmup_steps)
        optimizer.step()
        record = {"step": step, "train_id": row["id"], "case_family_id": row["case_family_id"],
                  "mean_reward": sum(rewards) / len(rewards), "max_reward": max(rewards),
                  "reward_spread": spread, "group_rewards": rewards,
                  "skipped_policy_update": spread <= 1e-6, "skipped_updates": skipped,
                  "mean_policy_logp": sum(policy_logps) / max(1, len(policy_logps)),
                  "mean_reference_logp": sum(reference_logps) / max(1, len(reference_logps)),
                  "gate_replay_loss": float(replay_loss.detach()), "rule_aux_loss": float(rule_loss.detach()),
                  "elapsed_sec": round(time.time() - started, 2),
                  "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        history.append(record)
        if step == 1 or step % args.log_interval == 0:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if step % args.eval_interval == 0 or step == args.steps:
            dev_nll = evaluate_dev(model, tok, dev_rows, torch, args)
            record["dev_decision_handling_nll"] = dev_nll
            write_json(args.output / "history.json", history)
            if dev_nll < best_dev:
                best_dev = dev_nll
                if best_dir.exists():
                    import shutil
                    shutil.rmtree(best_dir)
                model.set_adapter("policy")
                model.save_pretrained(best_dir, safe_serialization=True, selected_adapters=["policy"])
                tok.save_pretrained(best_dir)
                write_json(args.output / "selection.json", {"step": step, "dev_nll": dev_nll,
                                                              "adapter_sha256": adapter_digest(best_dir)})
            print(json.dumps({"step": step, "dev_decision_handling_nll": dev_nll,
                              "best_dev_nll": best_dev}, ensure_ascii=False), flush=True)
    write_json(args.output / "history.json", history)
    manifest.update({"status": "complete", "elapsed_sec": round(time.time() - started, 2),
                     "best_dev_nll": best_dev, "selection": json.loads((args.output / "selection.json").read_text()),
                     "adapter_sha256": adapter_digest(best_dir), "skipped_policy_updates": skipped,
                     "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    write_json(args.output / "run_manifest.json", manifest)
    (args.output / "complete.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--dpo-adapter", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--gate-replay-weight", type=float, default=0.1)
    parser.add_argument("--rule-aux-weight", type=float, default=0.05)
    parser.add_argument("--method-name", default="GateDistilledGRPO")
    parser.add_argument("--gate-selection", choices=("decision_disagreement", "random_teacher"),
                        default="decision_disagreement")
    parser.add_argument("--replay-sampling", choices=("verifier_weighted", "uniform"),
                        default="verifier_weighted")
    parser.add_argument("--online-reward-mode", choices=("decision_gated", "all_fields"),
                        default="decision_gated")
    parser.add_argument("--disable-online-policy-gradient", action="store_true")
    parser.add_argument("--gate-memory-cache", type=Path)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    if args.group_size < 2 or args.probe_batch_size < 1 or args.steps < 1:
        parser.error("group size must be >=2 and batch/steps positive")
    train(args)


if __name__ == "__main__":
    main()
