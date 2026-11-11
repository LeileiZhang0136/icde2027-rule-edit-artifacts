#!/usr/bin/env python3
"""Prepare leakage-audited training/dev/test inputs for the v1.1 Batch-8 run.

The training target is rendered deterministically from the executable v1.1
state transitions. Test gold is written to a separate file and must not be
passed to any model-training or generation command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phrase_list(values: list[str]) -> str:
    # Semicolons make set-valued fields parseable by the frozen exact-set
    # verifier; the scorer still judges semantic constraints, not this string.
    return "；".join(values) if values else "无"


def render_answer(state: dict[str, Any], delta: dict[str, Any]) -> str:
    """Render all requested state fields with stable labels for verifiers."""
    changed = delta.get("obligation_status_changed", {})
    status_text = "；".join(
        f"{name}：{values['old']}→{values['new']}"
        for name, values in sorted(changed.items())
    ) or "无既有义务状态变化"
    statuses = state.get("obligation_status", {})
    active_statuses = "；".join(
        f"{name}（{status}）" for name, status in sorted(statuses.items())
    ) or "无"
    review = state.get("human_review", {})
    if review.get("required"):
        review_text = f"需要，标准为{review.get('reason') or '按上层规则规定'}，状态为{review.get('status', 'pending')}"
    else:
        review_text = "不需要"
    basis = state.get("basis", [])
    # These are rule-engine fact keys, not invented case facts. The prose keeps
    # the evidence trace explicit while the structured answer remains auditable.
    basis_labels = {
        "action_date": "行为发生日期",
        "age_threshold": "适用年龄门槛",
        "birth_date": "出生日期及行为时年龄",
        "domain_rule": "下游领域规则",
        "secondary_gate_met": "领域前置条件是否满足",
        "upper_rule_version": "当前上层规则版本",
        "feature_type": "特征类型",
        "identity_inference": "是否推断身份",
        "identity_feature_generated": "是否生成身份特征",
        "separate_consent": "单独授权状态",
        "encrypted": "加密状态",
        "purpose_matches_domain": "用途是否符合领域规则",
        "retention_days": "保存期限",
        "impact_level": "影响程度",
        "adverse_proposed": "是否提出不利处理",
        "formal_confirmation": "正式确认状态",
        "materials_checked": "材料审阅情况",
        "system_reason_checked": "理由审阅情况",
        "reviewer_can_override": "审查人能否修改建议",
        "review_reason_recorded": "是否记录审查理由",
        "days_since_last_check": "距上次核验的时间",
        "independent_source_count": "独立证据来源数量",
        "core_claim_true": "核心事实是否成立",
        "affirmative_authorization": "是否存在明确授权",
        "explicit_objection": "是否存在明确反对",
        "restricted_condition": "限制条件是否适用",
        "urgent": "是否紧急",
        "immediate_safety_risk": "是否存在即时安全风险",
        "safe_alternative_available": "是否有安全替代方案",
        "upper_safety_restriction_applies": "上层安全限制是否适用",
        "local_exception_allows_action": "下游例外是否允许行动",
        "direct_conflict": "上下层规则是否冲突",
        "independent_safeguard_complete": "独立保障条件是否完成",
        "filed_date": "提交日期",
        "request_status": "申请当前状态",
        "legacy_requirement_met": "旧标准条件是否满足",
    }
    rationale = "、".join(basis_labels.get(key, key) for key in basis) or "案例事实与领域规则"
    result_type = delta.get("result_type", "")
    impact = {
        "unaffected": "新规则更新不改变本案处理结果",
        "decision_changed": "新规则更新改变了本案的实质决策",
        "procedure_only": "实质决策未变，但程序或义务发生变化",
        "exception_preserves": "通常会受更新影响，但本案例外条件阻断了该影响",
    }.get(result_type, "按照当前规则与领域条件处理")
    return (
        f"最终处理判断：{state['final_decision']}。当前处置：{state['current_handling']}。"
        f"新增义务：{phrase_list(delta.get('obligations_added', []))}。"
        f"取消义务：{phrase_list(delta.get('obligations_removed', []))}。"
        f"保持义务：{phrase_list(delta.get('obligations_unchanged', []))}。"
        f"义务状态变化：{status_text}。"
        f"新增程序：{phrase_list(delta.get('procedures_added', []))}。"
        f"取消程序：{phrase_list(delta.get('procedures_removed', []))}。"
        f"当前适用义务及状态：{active_statuses}。"
        f"当前许可：{phrase_list(state.get('permissions', []))}。"
        f"当前禁止：{phrase_list(state.get('prohibitions', []))}。"
        f"人工复核：{review_text}。理由：{impact}；需结合{rationale}以及案例中相应证据判断。"
    )


def stale_delta(old_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_type": "unaffected",
        "obligations_added": [],
        "obligations_removed": [],
        "obligations_unchanged": sorted(old_state.get("obligations", [])),
        "obligation_status_changed": {},
        "procedures_added": [],
        "procedures_removed": [],
    }


def make_rejected(chosen: str, old_state: dict[str, Any], new_state: dict[str, Any], delta: dict[str, Any]) -> tuple[str, str]:
    old_answer = render_answer(old_state, stale_delta(old_state))
    if old_answer != chosen:
        return old_answer, "stale_old_rule_state"
    # For truly unaffected cases the stale answer is valid. Use an explicit
    # false transition claim, which the frozen constraint scorer can reject.
    return chosen + " 错误断言：上层规则更新已经改变了本案的最终决定。", "false_update_effect_claim"


def prompt_rows(path: Path, split: str, metadata_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in read_jsonl(path):
        metadata = metadata_by_id.get(row["request_id"])
        if metadata is None:
            raise ValueError(f"Prompt request has no case metadata: {row['request_id']}")
        result.append({
            "id": row["request_id"],
            "request_id": row["request_id"],
            "case_family_id": metadata["case_family_id"],
            "family_id": metadata["case_family_id"],
            "event_id": metadata["event_id"],
            "domain_id": metadata["domain_id"],
            "difficulty": metadata["difficulty"],
            "split": split,
            "prompt": row["text"],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src = args.dataset
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Prepared directory is not empty: {args.output}")

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != "main-experiment-formal-v1.1" or manifest.get("status") != "frozen_v1.1":
        raise ValueError("Expected frozen main-experiment v1.1 data")
    frozen_hashes = json.loads((src / "SHA256.json").read_text(encoding="utf-8"))
    source_files = (
        "manifest.json", "rules.jsonl", "domains.jsonl", "cases.jsonl", "gold_constraints.jsonl",
        "targets_train_constraints.jsonl", "requests_train_with_rule.jsonl",
        "requests_train_closed_book.jsonl", "requests_development_closed_book.jsonl",
        "requests_development_ice.jsonl", "requests_test_closed_book.jsonl", "requests_test_ice.jsonl",
    )
    for name in source_files:
        if frozen_hashes.get(name) != digest(src / name):
            raise ValueError(f"Frozen v1.1 checksum mismatch: {name}")
    rules = read_jsonl(src / "rules.jsonl")
    case_metadata = {row["request_id"]: row for row in read_jsonl(src / "cases.jsonl")}
    if len(case_metadata) != 7260:
        raise ValueError("Case metadata does not cover the frozen v1.1 requests")
    train_with_rule = read_jsonl(src / "requests_train_with_rule.jsonl")
    train_prompts = {row["request_id"]: row for row in prompt_rows(
        src / "requests_train_closed_book.jsonl", "train", case_metadata)}
    dev_prompts = prompt_rows(src / "requests_development_closed_book.jsonl", "development", case_metadata)
    dev_ice = prompt_rows(src / "requests_development_ice.jsonl", "development", case_metadata)
    test_closed = prompt_rows(src / "requests_test_closed_book.jsonl", "test", case_metadata)
    test_ice = prompt_rows(src / "requests_test_ice.jsonl", "test", case_metadata)
    labels = read_jsonl(src / "targets_train_constraints.jsonl")
    gold = read_jsonl(src / "gold_constraints.jsonl")

    if (len(train_with_rule), len(labels), len(dev_prompts), len(dev_ice), len(test_closed), len(test_ice)) != (1620, 1620, 540, 540, 5100, 5100):
        raise ValueError("Unexpected v1.1 request counts")
    train_by_id = {}
    for row in train_with_rule:
        metadata = case_metadata[row["request_id"]]
        train_by_id[row["request_id"]] = {**row, **{
            key: metadata[key] for key in ("difficulty", "case_family_id", "event_id", "domain_id")
        }}
    label_by_id = {row["request_id"]: row for row in labels}
    if len(train_by_id) != len(train_with_rule) or set(train_by_id) != set(label_by_id):
        raise ValueError("Training request/label IDs do not match exactly")
    if set(train_by_id) != set(train_prompts):
        raise ValueError("Training request IDs differ between with-rule and closed-book files")
    if any("当前更新后的上层规则：" not in row["text"] for row in train_with_rule):
        raise ValueError("Train-with-rule inputs must explicitly contain the updated upper rule")
    if any("当前更新后的上层规则：" in row["prompt"] for row in test_closed):
        raise ValueError("Closed-book test prompts leak the updated rule")

    gold_by_family = {row["case_family_id"]: row for row in gold}
    if len(gold_by_family) != len(gold):
        raise ValueError("Duplicate family in gold constraints")
    train_families = {row["case_family_id"] for row in labels}
    dev_families = {row["family_id"] for row in dev_prompts}
    test_families = {row["family_id"] for row in test_closed}
    if train_families & dev_families or train_families & test_families or dev_families & test_families:
        raise ValueError("Case-family leakage across splits")
    for split, family_ids, expected in (("train", train_families, 540), ("development", dev_families, 180), ("test", test_families, 1700)):
        if len(family_ids) != expected:
            raise ValueError(f"Unexpected {split} family count: {len(family_ids)}")

    train_sft, train_dpo, train_rte, train_grpo = [], [], [], []
    train_constraint_by_id = {}
    for request_id, request in train_by_id.items():
        label = label_by_id[request_id]
        new_state, old_state, delta = label["new_state"], label["old_state"], label["delta"]
        target = render_answer(new_state, delta)
        rejected, rejected_type = make_rejected(target, old_state, new_state, delta)
        common = {
            "id": request_id,
            "request_id": request_id,
            "case_family_id": label["case_family_id"],
            "event_id": label["event_id"],
            "domain_id": label["domain_id"],
            "difficulty": train_prompts[request_id]["difficulty"],
            "split": "train",
            "target": target,
            "sample_weight": 1.0 / 3.0,
            "target_constraints": label["target_constraints"],
            "old_state": old_state,
            "new_state": new_state,
            "delta": delta,
            "reasoning_requirements": label["reasoning_requirements"],
        }
        train_sft.append({**common, "prompt": request["text"], "kind": "case", "weight": 1.0 / 3.0})
        train_dpo.append({**common, "prompt": request["text"], "chosen": target,
                          "rejected": rejected, "rejected_type": rejected_type})
        train_grpo.append({**common, "prompt": request["text"]})
        rte_input = train_prompts[request_id]
        train_rte.append({**common, "prompt": rte_input["prompt"]})
        train_constraint_by_id[request_id] = label

    # 18 rule QA views (direct + rephrase for each of the 9 update events).
    rule_pairs = []
    rule_examples = []
    for row in rules:
        pair = {
            "event_id": row["event_id"],
            "title": row["title"],
            "question": f"在虚构受控环境中，{row['title']}后当前生效的上层规则是什么？",
            "rephrase": f"请准确陈述事件“{row['title']}”对应的更新后规则。",
            "target": row["new_rule"],
            "old_rule": row["old_rule"],
            "new_rule": row["new_rule"],
        }
        rule_pairs.append(pair)
        for variant in ("question", "rephrase"):
            rule_examples.append({
                "id": f"{row['event_id']}::{variant}", "event_id": row["event_id"],
                "kind": "rule", "prompt": pair[variant], "target": row["new_rule"],
                "weight": 7.5,
            })
    train_sft.extend(rule_examples)

    dev_gold = []
    for row in dev_prompts:
        label = gold_by_family[row["family_id"]]
        dev_gold.append({**row, "target": render_answer(label["new_state"], label["delta"]),
                         "target_constraints": label["must_satisfy"],
                         "new_state": label["new_state"], "old_state": label["old_state"],
                         "delta": label["delta"], "reasoning_requirements": label["reasoning_requirements"]})
    dev_ice_gold = []
    ice_by_id = {row["id"]: row for row in dev_ice}
    for row in dev_gold:
        ice = ice_by_id[row["id"]]
        dev_ice_gold.append({**ice, **{k: row[k] for k in ("target", "target_constraints", "new_state", "old_state", "delta", "reasoning_requirements")}})
    dev_closed_inputs = [{key: row[key] for key in (
        "id", "request_id", "case_family_id", "family_id", "event_id", "domain_id", "difficulty", "split", "prompt"
    )} for row in dev_prompts]
    dev_ice_inputs = [{key: row[key] for key in (
        "id", "request_id", "case_family_id", "family_id", "event_id", "domain_id", "difficulty", "split", "prompt"
    )} for row in dev_ice]

    # Test gold is deliberately separated from every model-visible test input.
    test_gold = [row for row in gold if row["case_family_id"] in test_families]
    if len(test_gold) != 1700:
        raise ValueError(f"Test gold coverage mismatch: {len(test_gold)}")

    outputs = {
        "train_sft.jsonl": train_sft,
        "train_dpo.jsonl": train_dpo,
        "train_grpo.jsonl": train_grpo,
        "train_rte.jsonl": train_rte,
        "rule_pairs.jsonl": rule_pairs,
        "dev_closed_book_inputs.jsonl": dev_closed_inputs,
        "dev_ice_inputs.jsonl": dev_ice_inputs,
        "dev_closed_book.jsonl": dev_gold,
        "dev_ice.jsonl": dev_ice_gold,
        "test_closed_book.jsonl": test_closed,
        "test_ice.jsonl": test_ice,
        "test_gold_constraints.jsonl": test_gold,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_jsonl(args.output / name, rows)

    if len({row["id"] for row in test_closed}) != 5100 or len({row["id"] for row in test_ice}) != 5100:
        raise ValueError("Duplicate test request IDs")
    if {row["id"] for row in test_closed} != {row["id"] for row in test_ice}:
        raise ValueError("Closed-book and ICE tests must pair exactly")
    if any("new_rule" in row or "target" in row or "target_constraints" in row for row in test_closed + test_ice):
        raise ValueError("Model-visible test input contains hidden labels")

    manifest_out = {
        "protocol": "main_experiment_formal_v1_1_batch8_seed42",
        "source_version": manifest["version"],
        "source_manifest_sha256": digest(src / "manifest.json"),
        "seed": 42,
        "generation_batch_size": 8,
        "training_case_families": len(train_families),
        "training_variants": len(train_rte),
        "development_case_families": len(dev_families),
        "development_variants": len(dev_gold),
        "test_case_families": len(test_families),
        "test_variants_per_condition": len(test_closed),
        "rule_events": len(rules),
        "rule_qa_views": len(rule_examples),
        "methods": ["Pure-Qwen", "Pure-Qwen+ICE", "LoRA-SFT-RuleCase", "RTE", "DPOTransition", "GateDistilledGRPO", "DecisionDisagreementGate"],
        "excluded_methods": ["GRPO-balanced"],
        "training_target_protocol": "deterministic rendered answer from v1.1 executable state/delta; evaluation judges constraints, not string identity",
        "family_weighting": "three surface variants each weight 1/3; SFT rule-QA views each weight 7.5, approximately 20% of total weighted objective",
        "test_gold_separate": True,
        "test_gold_must_not_be_used_for_training_or_checkpoint_selection": True,
        "source_sha256": {name: digest(src / name) for name in source_files},
        "prepared_sha256": {name: digest(args.output / name) for name in outputs},
        "counts": {name: len(rows) for name, rows in outputs.items()},
        "test_split_read_during_preparation": True,
        "test_gold_used_for_training": False,
    }
    manifest_path = args.output / "prepared_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": str(args.output), "counts": manifest_out["counts"],
                      "train_family_count": len(train_families), "dev_family_count": len(dev_families),
                      "test_family_count": len(test_families),
                      "test_gold_separated": True,
                      "manifest_sha256": digest(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
