#!/usr/bin/env python3
"""Train v1.1 GateDistilledGRPO, train split only.

The frozen decision-disagreement gate routes SFT/DPO teacher outputs on train
rows. A deterministic field verifier retains useful train-only replay cases;
GRPO then optimizes a decision-gated eight-field reward, with DPO-reference KL,
verifier-gated target replay, and an upper-rule QA auxiliary objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def build_train_sampling_weights(train_rows: list[dict[str, Any]],
                                 gate_memory: list[dict[str, Any]],
                                 hard_sampling_alpha: float) -> tuple[list[float], dict[str, Any]]:
    """Optionally upweight case families that train-only teachers do not solve.

    The family-level mean verifier reward is shared by that family's three
    difficulty paraphrases, preserving equal sampling across easy/medium/hard.
    """
    if not math.isfinite(hard_sampling_alpha) or hard_sampling_alpha < 0:
        raise ValueError("hard_sampling_alpha must be a finite non-negative number")
    if hard_sampling_alpha == 0:
        weights = [max(float(row.get("sample_weight", 1 / 3)), 1e-6) for row in train_rows]
        return weights, {
            "mode": "base_sample_weight_only", "hard_sampling_alpha": 0.0,
            "train_rows": len(train_rows), "train_families": len({r["case_family_id"] for r in train_rows}),
        }

    train_family_by_id = {str(row["id"]): str(row["case_family_id"]) for row in train_rows}
    if len(train_family_by_id) != len(train_rows):
        raise ValueError("GRPO train row IDs must be unique for hard sampling")
    audit_ids: set[str] = set()
    by_family: dict[str, list[float]] = {}
    for item in gate_memory:
        item_id = str(item["id"])
        family = str(item["case_family_id"])
        if item_id in audit_ids or train_family_by_id.get(item_id) != family:
            raise ValueError(f"Train-only teacher audit ID/family mismatch: {item_id}")
        audit_ids.add(item_id)
        reward = float(item["verifier_reward"])
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError(f"Invalid train-only verifier reward for {family}: {reward}")
        by_family.setdefault(family, []).append(reward)

    train_families = {str(row["case_family_id"]) for row in train_rows}
    if audit_ids != set(train_family_by_id) or set(by_family) != train_families:
        raise ValueError("Train-only teacher audit must cover exactly the GRPO train rows/families")
    family_reward = {family: sum(values) / len(values) for family, values in by_family.items()}
    family_counts = Counter(str(row["case_family_id"]) for row in train_rows)
    if any(count != 3 for count in family_counts.values()):
        raise ValueError("Family-aware hard sampling expects exactly three difficulty rows per family")
    if any(len(by_family[family]) != family_counts[family] for family in train_families):
        raise ValueError("Teacher audit must contain all three difficulty rows for every family")

    weights = []
    for row in train_rows:
        family = str(row["case_family_id"])
        base_weight = max(float(row.get("sample_weight", 1 / 3)), 1e-6)
        weights.append(base_weight * (1.0 + hard_sampling_alpha * (1.0 - family_reward[family])))

    total = sum(weights)
    imperfect_mass = sum(weight for row, weight in zip(train_rows, weights)
                          if family_reward[str(row["case_family_id"])] < 1.0)
    uniform_imperfect_mass = sum(1 for family, reward in family_reward.items() if reward < 1.0) / len(family_reward)
    return weights, {
        "mode": "train_only_family_teacher_hardness",
        "formula": "base_sample_weight * (1 + alpha * (1 - mean_family_verifier_reward))",
        "hard_sampling_alpha": hard_sampling_alpha,
        "train_rows": len(train_rows), "train_families": len(train_families),
        "imperfect_teacher_families": sum(reward < 1.0 for reward in family_reward.values()),
        "uniform_probability_imperfect_families": uniform_imperfect_mass,
        "weighted_probability_imperfect_families": imperfect_mass / total,
        "mean_family_teacher_reward": sum(family_reward.values()) / len(family_reward),
        "family_mean_teacher_rewards": family_reward,
    }


def build_sft_error_sampling_weights(train_rows: list[dict[str, Any]],
                                     sft_audit_rows: list[dict[str, Any]],
                                     alpha: float) -> tuple[list[float], dict[str, Any]]:
    """Upweight train families where the frozen SFT baseline makes errors.

    The SFT audit must exactly cover the GRPO training rows. Family-level mean
    SFT reward is shared across the three language-difficulty variants so the
    update targets SFT weaknesses without changing within-family proportions.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("sft_error_sampling_alpha must be finite and non-negative")

    train_family_by_id = {str(row["id"]): str(row["case_family_id"]) for row in train_rows}
    if len(train_family_by_id) != len(train_rows):
        raise ValueError("GRPO train row IDs must be unique for SFT-error sampling")

    audit_ids: set[str] = set()
    by_family: dict[str, list[float]] = {}
    for item in sft_audit_rows:
        item_id = str(item["id"])
        family = str(item["case_family_id"])
        if item_id in audit_ids or train_family_by_id.get(item_id) != family:
            raise ValueError(f"Train-only SFT audit ID/family mismatch: {item_id}")
        audit_ids.add(item_id)
        reward = float(item["sft_reward"])
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError(f"Invalid train-only SFT verifier reward for {family}: {reward}")
        by_family.setdefault(family, []).append(reward)

    train_families = {str(row["case_family_id"]) for row in train_rows}
    if audit_ids != set(train_family_by_id) or set(by_family) != train_families:
        raise ValueError("Train-only SFT audit must cover exactly the GRPO train rows/families")
    family_counts = Counter(str(row["case_family_id"]) for row in train_rows)
    if any(count != 3 for count in family_counts.values()):
        raise ValueError("SFT-error family sampling expects exactly three difficulty rows per family")
    if any(len(by_family[family]) != family_counts[family] for family in train_families):
        raise ValueError("SFT audit must contain all three difficulty rows for every family")

    family_reward = {family: sum(values) / len(values) for family, values in by_family.items()}
    weights = []
    for row in train_rows:
        family = str(row["case_family_id"])
        base_weight = max(float(row.get("sample_weight", 1 / 3)), 1e-6)
        weights.append(base_weight * (1.0 + alpha * (1.0 - family_reward[family])))

    total = sum(weights)
    imperfect_mass = sum(weight for row, weight in zip(train_rows, weights)
                          if family_reward[str(row["case_family_id"])] < 1.0)
    uniform_imperfect_mass = sum(reward < 1.0 for reward in family_reward.values()) / len(family_reward)
    return weights, {
        "mode": "train_only_family_sft_error_hardness",
        "formula": "base_sample_weight * (1 + alpha * (1 - mean_family_sft_reward))",
        "sft_error_sampling_alpha": alpha,
        "train_rows": len(train_rows), "train_families": len(train_families),
        "imperfect_sft_families": sum(reward < 1.0 for reward in family_reward.values()),
        "uniform_probability_imperfect_sft_families": uniform_imperfect_mass,
        "weighted_probability_imperfect_sft_families": imperfect_mass / total,
        "mean_family_sft_reward": sum(family_reward.values()) / len(family_reward),
        "family_mean_sft_rewards": family_reward,
        # Preserve the existing per-step logging key; SFT is one of the audited teachers.
        "family_mean_teacher_rewards": family_reward,
    }


def remove_updated_rule_line(prompt: str) -> str:
    """Drop only the explicit rule-update sentence from a case prompt."""
    lines = prompt.splitlines()
    marker = "当前更新后的上层规则："
    matches = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one updated-rule line, found {len(matches)}")
    return "\n".join(line for i, line in enumerate(lines) if i != matches[0])


def maybe_remove_updated_rule_line(prompt: str, probability: float,
                                   rng: random.Random) -> tuple[str, bool]:
    dropped = rng.random() < probability
    return (remove_updated_rule_line(prompt), True) if dropped else (prompt, False)


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


def normalize_state_label(raw: str | None, field: str) -> str | None:
    """Canonicalize a few conservative state-name aliases for train rewards.

    Free-form explanations and obligation strings remain exact; this only
    handles explicit enum-like decision/handling labels whose aliases are
    unambiguous in the controlled task vocabulary.
    """
    if raw is None:
        return None
    value = re.sub(r"[\s-]+", "_", raw.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    if field == "decision":
        for suffix in ("_not_authorized", "_unauthorized", "_not_allowed", "_disallowed", "_denied"):
            if value.endswith(suffix):
                return value[:-len(suffix)] + "_permission_denied"
        for suffix in ("_authorized", "_allowed", "_permitted"):
            if value.endswith(suffix):
                return value[:-len(suffix)] + "_permission_granted"
    if field == "handling" and value == "direct_dispatch":
        return "execute_dispatch"
    return value


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


def score_answer(answer: str, row: dict[str, Any], strict_reward_weight: float = 0.0,
                 decision_dense_weight: float = 0.0,
                 wrong_decision_field_weight: float = 0.0) -> dict[str, Any]:
    if (not math.isfinite(decision_dense_weight) or not math.isfinite(wrong_decision_field_weight)
            or decision_dense_weight < 0.0 or wrong_decision_field_weight < 0.0
            or (decision_dense_weight == 0.0 and wrong_decision_field_weight != 0.0)
            or (decision_dense_weight > 0.0 and decision_dense_weight <= wrong_decision_field_weight)):
        raise ValueError("dense reward requires decision_dense_weight > wrong_decision_field_weight >= 0")
    expected = row["target_constraints"]
    delta = row["delta"]
    state = row["new_state"]
    actual = {
        "decision": normalize_state_label(extract(answer, FIELD_LABELS["decision"]), "decision"),
        "handling": normalize_state_label(extract(answer, FIELD_LABELS["handling"]), "handling"),
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

    fields = {key: actual[key] is not None and (
                  actual[key] == normalize_state_label(wanted[key], key)
                  if key in {"decision", "handling"} else actual[key] == wanted[key])
              for key in ("decision", "handling", "obligations_added", "obligations_removed",
                          "obligations_unchanged", "human_review")}
    fields["reasoning"] = reasoning_ok

    duplicates_ok = True
    for field, label in FIELD_LABELS.items():
        values = extract_all(answer, label)
        if len(values) > 1:
            parsed_values = [parse_set(v) if field.startswith("obligations_") else
                             parse_review(v) if field == "human_review" else
                             normalize_state_label(v, field) if field in {"decision", "handling"} else v
                             for v in values]
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
    # Decision-gated reward mirrors the frozen main metric: a wrong decision is 0.
    if not fields["decision"]:
        partial_reward = 0.0
    else:
        partial_reward = sum(int(fields[key]) for key in FIELDS if key != "decision") / 7.0
    strict_pass = bool(all(fields.values()))
    base_reward = ((1.0 - strict_reward_weight) * partial_reward
                   + strict_reward_weight * float(strict_pass))
    nondecision_field_score = sum(int(fields[key]) for key in FIELDS if key != "decision") / 7.0
    decision_aux_reward = decision_dense_weight * float(fields["decision"])
    wrong_decision_field_aux_reward = (wrong_decision_field_weight * nondecision_field_score
                                       if not fields["decision"] else 0.0)
    reward = base_reward + decision_aux_reward + wrong_decision_field_aux_reward
    return {"score": reward, "base_score": base_reward, "partial_score": partial_reward,
            "decision_aux_reward": decision_aux_reward,
            "wrong_decision_field_aux_reward": wrong_decision_field_aux_reward,
            "nondecision_field_score": nondecision_field_score,
            "strict_pass": strict_pass,
            "fields": fields, "missing_basis": missing_basis,
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
        "state_aliases": {"decision": "authorized/allowed/permitted suffixes; explicit negative permission suffixes",
                          "handling": {"direct_dispatch": "execute_dispatch"}},
        "limitation": "Deterministic lexical/constraint verifier with narrowly declared state aliases; not a universal semantic judge for arbitrary prose.",
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
        probe_rows = []
        for row in batch:
            prompt, dropped = maybe_remove_updated_rule_line(
                row["prompt"], args.rule_dropout_prob, args.rng)
            probe_rows.append((row, prompt, dropped))
        for adapter_name in ("sft_teacher", "dpo_teacher"):
            model.set_adapter(adapter_name)
            model.eval()
            prompts = [tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
                       for _, prompt, _ in probe_rows]
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
        for (row, prompt, dropped), sft_answer, dpo_answer in zip(probe_rows, sft_answers, dpo_answers):
            sft_score = score_answer(sft_answer, row, args.strict_reward_weight)
            dpo_score = score_answer(dpo_answer, row, args.strict_reward_weight)
            sft_decision = extract(sft_answer, FIELD_LABELS["decision"])
            dpo_decision = extract(dpo_answer, FIELD_LABELS["decision"])
            agrees = bool(sft_decision and dpo_decision and sft_decision == dpo_decision)
            if args.gate_selection == "verifier_best":
                # Gold is available only for these train rows; no dev/test labels
                # are used to construct the replay teacher.
                source = "LoRA-SFT" if sft_score["score"] >= dpo_score["score"] else "DPOTransition"
            else:
                source = "LoRA-SFT" if agrees else "DPOTransition"
            # The selected teacher must also determine which answer supplies
            # the verifier-backed replay weight. Keeping this tied to decision
            # agreement would silently make verifier_best behave like the old
            # agreement gate for memory filtering and weighting.
            if source == "LoRA-SFT":
                chosen, judged = sft_answer, sft_score
            else:
                chosen, judged = dpo_answer, dpo_score
            verified = [field for field, ok in judged["fields"].items() if ok]
            route_counts[source] += 1
            route_counts["decision_agreement"] += int(agrees)
            route_counts["decision_parse_failure"] += int(not sft_decision or not dpo_decision)
            if verified:
                replay_weight = max(0.1, sum(int(judged["fields"][f]) for f in FIELDS) / len(FIELDS))
                memory.append({"id": row["id"], "case_family_id": row["case_family_id"],
                               "prompt": prompt, "target": row["target"],
                               "sample_weight": float(row.get("sample_weight", 1 / 3)) * replay_weight,
                               "gate_teacher": source, "verified_fields": verified,
                               "verifier_reward": judged["score"], "initial_answer": chosen,
                               "updated_rule_line_dropped": dropped})
            audit.append({"id": row["id"], "case_family_id": row["case_family_id"],
                          "event_id": row["event_id"], "domain_id": row["domain_id"],
                          "difficulty": row["difficulty"],
                          "sft_decision": sft_decision, "dpo_decision": dpo_decision,
                          "decision_agreement": agrees, "selected_teacher": source,
                          "sft_reward": sft_score["score"], "dpo_reward": dpo_score["score"],
                          "selected_reward": judged["score"], "verified_fields": verified,
                          "updated_rule_line_dropped": dropped,
                          "sft_sha256": hashlib.sha256(sft_answer.encode()).hexdigest(),
                          "dpo_sha256": hashlib.sha256(dpo_answer.encode()).hexdigest()})
        print(json.dumps({"gate_probe_done": min(start + len(batch), len(rows)), "total": len(rows),
                          "retained": len(memory)}, ensure_ascii=False), flush=True)
    if not memory:
        raise RuntimeError("Frozen gate/verifier produced no train-only replay examples")
    write_jsonl(output / "gate_memory.jsonl", memory)
    write_json(output / "gate_audit.json", {
        "protocol": "v1_1_train_only_gate_verifier_replay",
        "gate_selection": args.gate_selection,
        "rows_probed": len(rows), "rows_retained": len(memory), "route_counts": dict(route_counts),
        "rows": audit, "test_split_read": False, "test_gold_read": False,
        "teacher_adapter_sha256": file_digest(args.sft_adapter / "adapter_model.safetensors"),
        "dpo_adapter_sha256": adapter_digest(args.dpo_adapter),
    })
    return memory


def load_cached_gate_memory(cache_dir: Path, train_rows: list[dict[str, Any]], args
                           ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse an exact, completed train-only SFT/DPO audit without regenerating it."""
    required = ("run_manifest.json", "gate_audit.json", "gate_memory.jsonl")
    missing = [name for name in required if not (cache_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cached gate audit is incomplete: {missing}")
    source_manifest = json.loads((cache_dir / "run_manifest.json").read_text(encoding="utf-8"))
    source_audit = json.loads((cache_dir / "gate_audit.json").read_text(encoding="utf-8"))
    memory = read_jsonl(cache_dir / "gate_memory.jsonl")
    if source_manifest.get("status") != "complete":
        raise ValueError("Cached gate memory must come from a completed run")
    if source_manifest.get("test_split_read") is not False or source_manifest.get("test_gold_read") is not False:
        raise ValueError("Cached gate memory source is not explicitly test-sealed")
    if source_audit.get("test_split_read") is not False or source_audit.get("test_gold_read") is not False:
        raise ValueError("Cached gate audit is not explicitly train-only")
    if source_manifest.get("gate_selection") != args.gate_selection or source_audit.get("gate_selection") != args.gate_selection:
        raise ValueError("Cached gate selection differs from the requested run")
    if float(source_manifest.get("strict_reward_weight", -1)) != args.strict_reward_weight:
        raise ValueError("Cached teacher rewards used a different strict-reward weight")
    if float(source_manifest.get("rule_dropout_prob", -1)) != args.rule_dropout_prob:
        raise ValueError("Cached teacher prompts used a different rule-dropout probability")
    expected_hashes = {name: file_digest(args.data / name) for name in
                       ("train_grpo.jsonl", "dev_closed_book.jsonl", "rule_pairs.jsonl", "prepared_manifest.json")}
    for name, expected_hash in expected_hashes.items():
        if source_manifest.get("data_sha256", {}).get(name) != expected_hash:
            raise ValueError(f"Cached gate memory used different prepared data: {name}")
    current_sft = file_digest(args.sft_adapter / "adapter_model.safetensors")
    current_dpo = adapter_digest(args.dpo_adapter)
    if source_manifest.get("sft_adapter_sha256") != current_sft or source_audit.get("teacher_adapter_sha256") != current_sft:
        raise ValueError("Cached teacher adapter does not match the current LoRA-SFT adapter")
    if source_manifest.get("dpo_adapter_sha256") != current_dpo or source_audit.get("dpo_adapter_sha256") != current_dpo:
        raise ValueError("Cached teacher adapter does not match the current DPO adapter")

    train_by_id = {str(row["id"]): row for row in train_rows}
    if len(train_by_id) != len(train_rows):
        raise ValueError("Current train row IDs are not unique")
    audit_rows = source_audit.get("rows", [])
    audit_by_id = {str(row["id"]): row for row in audit_rows}
    memory_by_id = {str(row["id"]): row for row in memory}
    expected_ids = set(train_by_id)
    if len(audit_by_id) != len(audit_rows) or set(audit_by_id) != expected_ids:
        raise ValueError("Cached teacher audit does not cover exactly the current train rows")
    if len(memory_by_id) != len(memory) or set(memory_by_id) != expected_ids:
        raise ValueError("Cached replay memory does not cover exactly the current train rows")
    if source_audit.get("rows_probed") != len(train_rows) or source_audit.get("rows_retained") != len(memory):
        raise ValueError("Cached audit row counts do not match its contents")

    for row_id, source_row in train_by_id.items():
        audited = audit_by_id[row_id]
        cached = memory_by_id[row_id]
        family = str(source_row["case_family_id"])
        if str(audited.get("case_family_id")) != family or str(cached.get("case_family_id")) != family:
            raise ValueError(f"Cached teacher family mismatch for {row_id}")
        if cached.get("target") != source_row.get("target"):
            raise ValueError(f"Cached replay target mismatch for {row_id}")
        dropped = cached.get("updated_rule_line_dropped")
        expected_prompt = (remove_updated_rule_line(source_row["prompt"]) if dropped else source_row["prompt"])
        if not isinstance(dropped, bool) or cached.get("prompt") != expected_prompt:
            raise ValueError(f"Cached replay prompt mismatch for {row_id}")
        if cached.get("verifier_reward") != audited.get("selected_reward"):
            raise ValueError(f"Cached verifier reward mismatch for {row_id}")
    return memory, {
        "reused": True, "source_dir": str(cache_dir),
        "source_run_manifest_sha256": file_digest(cache_dir / "run_manifest.json"),
        "source_gate_audit_sha256": file_digest(cache_dir / "gate_audit.json"),
        "source_gate_memory_sha256": file_digest(cache_dir / "gate_memory.jsonl"),
        "rows": len(memory), "train_split_read": True, "test_split_read": False,
        "test_gold_read": False,
    }


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
    if args.rule_dropout_prob and any(
            sum(line.startswith("当前更新后的上层规则：") for line in row["prompt"].splitlines()) != 1
            for row in train_rows):
        raise ValueError("Rule dropout requires exactly one explicit updated-rule line in every train case prompt")
    calibration = calibrate(train_rows)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {args.output}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GateDistilledGRPO requires CUDA BF16")
    random.seed(args.seed)
    rng = random.Random(args.seed)
    args.rng = rng
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
    policy_adapter = args.policy_adapter or args.dpo_adapter
    reference_adapter = args.reference_adapter or args.dpo_adapter
    model = PeftModel.from_pretrained(base, str(policy_adapter), is_trainable=True,
                                      adapter_name="policy", local_files_only=True)
    model.load_adapter(str(reference_adapter), adapter_name="reference", is_trainable=False, local_files_only=True)
    model.load_adapter(str(args.sft_adapter), adapter_name="sft_teacher", is_trainable=False, local_files_only=True)
    model.load_adapter(str(args.dpo_adapter), adapter_name="dpo_teacher", is_trainable=False, local_files_only=True)
    model.set_adapter("policy")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name and "policy" in name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable policy LoRA parameters")
    model.config.use_cache = True

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "verifier_calibration.json", calibration)
    if args.gate_memory_cache is not None:
        gate_memory, cache_info = load_cached_gate_memory(args.gate_memory_cache, train_rows, args)
        source_audit = json.loads((args.gate_memory_cache / "gate_audit.json").read_text(encoding="utf-8"))
        source_audit["reused_from"] = str(args.gate_memory_cache)
        source_audit["source_gate_audit_sha256"] = cache_info["source_gate_audit_sha256"]
        write_jsonl(args.output / "gate_memory.jsonl", gate_memory)
        write_json(args.output / "gate_audit.json", source_audit)
    else:
        gate_memory = build_gate_memory(model, tok, train_rows, args.output, args, torch)
        cache_info = {"reused": False, "source_dir": None, "rows": len(gate_memory),
                      "train_split_read": True, "test_split_read": False, "test_gold_read": False}
    if args.sft_error_sampling_alpha > 0:
        if args.gate_memory_cache is None:
            raise ValueError("SFT-error sampling requires a validated train-only gate-memory cache")
        if source_audit.get("test_split_read") is not False or source_audit.get("test_gold_read") is not False:
            raise ValueError("SFT-error sampling audit is not explicitly train-only")
        train_weights, sampling_audit = build_sft_error_sampling_weights(
            train_rows, source_audit.get("rows", []), args.sft_error_sampling_alpha)
    else:
        train_weights, sampling_audit = build_train_sampling_weights(
            train_rows, gate_memory, args.hard_sampling_alpha)
    write_json(args.output / "train_sampling_audit.json", sampling_audit)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    manifest = {
        "method": args.method_name, "protocol": "main_experiment_formal_v1_1",
        "training_script_sha256": file_digest(Path(__file__).resolve()),
        "seed": args.seed, "model": str(args.model), "sft_teacher": str(args.sft_adapter),
        "dpo_teacher": str(args.dpo_adapter), "initial_policy_adapter": str(policy_adapter),
        "reference_adapter": str(reference_adapter), "gate_selection": args.gate_selection,
        "sft_adapter_sha256": adapter_digest(args.sft_adapter), "dpo_adapter_sha256": adapter_digest(args.dpo_adapter),
        "policy_adapter_sha256": adapter_digest(policy_adapter),
        "reference_adapter_sha256": adapter_digest(reference_adapter),
        "data_sha256": {n: file_digest(args.data / n) for n in ("train_grpo.jsonl", "dev_closed_book.jsonl", "rule_pairs.jsonl", "prepared_manifest.json")},
        "train_rows": len(train_rows), "train_families": 540, "dev_rows": len(dev_rows), "rule_events": len(rules),
        "gate": ("choose the teacher with the higher train-only verifier reward; ties choose SFT"
                 if args.gate_selection == "verifier_best" else
                 "choose SFT when SFT/DPO decision fields agree; otherwise choose DPO"),
        "replay": "canonical train target replay for rows where selected teacher answer has at least one verifier-passed field",
        "reward": ("base: wrong decision=0; otherwise mean accuracy of seven remaining binary verifier fields; "
                   "optional dense shaping adds a decision-correct bonus and a smaller wrong-decision field score"),
        "strict_reward_weight": args.strict_reward_weight,
        "decision_dense_weight": args.decision_dense_weight,
        "wrong_decision_field_weight": args.wrong_decision_field_weight,
        "rule_dropout_prob": args.rule_dropout_prob,
        "case_prompt_policy": ("randomly remove only the explicit updated-rule line from teacher probes, policy rollouts, and replay prompts; keep rule-QA auxiliary prompts unchanged"
                               if args.rule_dropout_prob else "keep the explicit updated-rule line in train case prompts"),
        "verifier_version": "exact obligation sets/reason basis; normalize explicit permission suffix aliases and direct_dispatch=execute_dispatch",
        "steps": args.steps, "group_size": args.group_size, "temperature": args.temperature,
        "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate, "kl_weight": args.kl_weight,
        "gate_replay_weight": args.gate_replay_weight, "rule_aux_weight": args.rule_aux_weight,
        "hard_sampling_alpha": args.hard_sampling_alpha,
        "sft_error_sampling_alpha": args.sft_error_sampling_alpha,
        "gate_memory_cache": cache_info,
        "train_sampling": {k: v for k, v in sampling_audit.items()
                           if k not in ("family_mean_teacher_rewards", "family_mean_sft_rewards")},
        "early_stop_patience": args.early_stop_patience,
        "checkpoint_selection": "minimum closed-book development final-decision/current-handling NLL",
        "test_split_read": False, "test_gold_read": False,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__, "status": "running",
    }
    write_json(args.output / "run_manifest.json", manifest)
    best_dir = args.output / "best_adapter"
    best_dev = float("inf")
    history = []
    replay_weights = [max(float(item["sample_weight"]), 1e-6) for item in gate_memory]
    train_weight_by_id = {row["id"]: weight for row, weight in zip(train_rows, train_weights)}
    family_teacher_reward = sampling_audit.get("family_mean_teacher_rewards", {})
    started = time.time()
    skipped = 0
    base_skipped = 0
    dense_rescued_groups = 0
    stale_evaluations = 0
    early_stopped = False
    for step in range(1, args.steps + 1):
        row = rng.choices(train_rows, weights=train_weights, k=1)[0]
        case_prompt, dropped_rule = maybe_remove_updated_rule_line(
            row["prompt"], args.rule_dropout_prob, rng)
        model.set_adapter("policy")
        candidates = generate_group(model, tok, case_prompt, args.group_size, args.max_new_tokens,
                                    args.temperature, args.top_p, torch, args.seed + step)
        judged = [score_answer(answer, row, args.strict_reward_weight,
                               args.decision_dense_weight,
                               args.wrong_decision_field_weight) for answer in candidates]
        rewards = [item["score"] for item in judged]
        score_tensor = torch.tensor(rewards, dtype=torch.float32, device="cuda")
        spread = float(score_tensor.std(unbiased=False))
        base_rewards = [item["base_score"] for item in judged]
        base_spread = float(torch.tensor(base_rewards, dtype=torch.float32).std(unbiased=False))
        if base_spread <= 1e-6:
            base_skipped += 1
            if spread > 1e-6:
                dense_rescued_groups += 1
        optimizer.zero_grad(set_to_none=True)
        policy_logps, reference_logps = [], []
        if spread > 1e-6:
            advantages = (score_tensor - score_tensor.mean()) / (score_tensor.std(unbiased=False) + 1e-4)
            for index, answer in enumerate(candidates):
                model.set_adapter("policy")
                model.train()
                policy_logp = mean_sequence_logp(model, tok, case_prompt, answer, torch, args.max_total_tokens)
                with torch.no_grad():
                    model.set_adapter("reference")
                    model.eval()
                    ref_logp = mean_sequence_logp(model, tok, case_prompt, answer, torch, args.max_total_tokens)
                model.set_adapter("policy")
                model.train()
                loss = -advantages[index] * policy_logp + args.kl_weight * (policy_logp - ref_logp).pow(2)
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
                  "family_teacher_reward": family_teacher_reward.get(row["case_family_id"]),
                  "sampling_weight": train_weight_by_id[row["id"]],
                  "mean_reward": sum(rewards) / len(rewards), "max_reward": max(rewards),
                  "reward_spread": spread, "base_reward_spread": base_spread,
                  "group_rewards": rewards, "base_group_rewards": base_rewards,
                  "group_decision_aux_rewards": [item["decision_aux_reward"] for item in judged],
                  "group_wrong_decision_field_aux_rewards": [item["wrong_decision_field_aux_reward"] for item in judged],
                  "dense_rescued_groups": dense_rescued_groups,
                  "strict_passes": sum(int(item["strict_pass"]) for item in judged),
                  "updated_rule_line_dropped": dropped_rule,
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
                stale_evaluations = 0
                if best_dir.exists():
                    import shutil
                    shutil.rmtree(best_dir)
                model.set_adapter("policy")
                model.save_pretrained(best_dir, safe_serialization=True, selected_adapters=["policy"])
                tok.save_pretrained(best_dir)
                write_json(args.output / "selection.json", {"step": step, "dev_nll": dev_nll,
                                                              "adapter_sha256": adapter_digest(best_dir)})
            else:
                stale_evaluations += 1
            print(json.dumps({"step": step, "dev_decision_handling_nll": dev_nll,
                              "best_dev_nll": best_dev,
                              "stale_evaluations": stale_evaluations}, ensure_ascii=False), flush=True)
            if args.early_stop_patience > 0 and stale_evaluations >= args.early_stop_patience:
                early_stopped = True
                print(json.dumps({"early_stop": "development NLL patience reached",
                                  "step": step, "best_dev_nll": best_dev}, ensure_ascii=False), flush=True)
                break
    write_json(args.output / "history.json", history)
    manifest.update({"status": "early_stopped_dev_nll" if early_stopped else "complete",
                     "steps_completed": history[-1]["step"],
                     "elapsed_sec": round(time.time() - started, 2),
                     "best_dev_nll": best_dev, "selection": json.loads((args.output / "selection.json").read_text()),
                     "adapter_sha256": adapter_digest(best_dir), "skipped_policy_updates": skipped,
                     "base_skipped_policy_updates": base_skipped,
                     "dense_rescued_groups": dense_rescued_groups,
                     "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    write_json(args.output / "run_manifest.json", manifest)
    (args.output / "complete.txt").write_text(
        f"{manifest['status']} at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--dpo-adapter", type=Path, required=True)
    parser.add_argument("--policy-adapter", type=Path,
                        help="Trainable initialization; defaults to --dpo-adapter")
    parser.add_argument("--reference-adapter", type=Path,
                        help="Frozen KL reference; defaults to --dpo-adapter")
    parser.add_argument("--gate-selection", choices=("decision_disagreement", "verifier_best"),
                        default="decision_disagreement",
                        help="Train-only teacher selection; verifier_best uses gold only on train rows")
    parser.add_argument("--method-name", default="GateDistilledGRPO",
                        help="Name written to the run manifest for this variant")
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
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after this many consecutive dev NLL evaluations fail to improve; 0 disables")
    parser.add_argument("--rule-dropout-prob", type=float, default=0.0,
                        help="Probability of removing the explicit updated-rule line from train case prompts")
    parser.add_argument("--strict-reward-weight", type=float, default=0.0,
                        help="Blend strict all-fields pass reward with decision-gated partial reward")
    parser.add_argument("--decision-dense-weight", type=float, default=0.0,
                        help="Optional online-rollout bonus for a correct final decision")
    parser.add_argument("--wrong-decision-field-weight", type=float, default=0.0,
                        help="Optional smaller shaping weight on nondecision fields only when the decision is wrong")
    parser.add_argument("--hard-sampling-alpha", type=float, default=0.0,
                        help="Upweight train case families where the selected train-only teacher has verifier reward < 1")
    parser.add_argument("--sft-error-sampling-alpha", type=float, default=0.0,
                        help="Upweight train case families where the frozen LoRA-SFT teacher has verifier reward < 1")
    parser.add_argument("--gate-memory-cache", type=Path,
                        help="Reuse a completed train-only teacher audit after hash/ID/prompt validation")
    args = parser.parse_args()
    if (args.group_size < 2 or args.probe_batch_size < 1 or args.steps < 1
            or args.early_stop_patience < 0 or not 0.0 <= args.rule_dropout_prob <= 1.0
            or not 0.0 <= args.strict_reward_weight <= 1.0
            or not math.isfinite(args.decision_dense_weight) or args.decision_dense_weight < 0.0
            or not math.isfinite(args.wrong_decision_field_weight) or args.wrong_decision_field_weight < 0.0
            or (args.decision_dense_weight == 0.0 and args.wrong_decision_field_weight != 0.0)
            or (args.decision_dense_weight > 0.0
                and args.decision_dense_weight <= args.wrong_decision_field_weight)
            or not math.isfinite(args.hard_sampling_alpha) or args.hard_sampling_alpha < 0.0
            or not math.isfinite(args.sft_error_sampling_alpha) or args.sft_error_sampling_alpha < 0.0
            or (args.hard_sampling_alpha > 0.0 and args.sft_error_sampling_alpha > 0.0)):
        parser.error("group size must be >=2, batch/steps positive, and early-stop patience nonnegative")
    train(args)


if __name__ == "__main__":
    main()
