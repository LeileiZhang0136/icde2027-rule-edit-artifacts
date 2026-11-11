#!/usr/bin/env python3
"""Sensitivity scorer for explicit empty obligation deltas.

This keeps the frozen v0.5 reader judgements intact and applies only a narrow,
auditable rule: when the executable rubric requires an empty added/removed set
and the answer explicitly labels that exact set as "无" (or an equivalent
literal empty marker), count the claim as stated. It does not infer omissions,
paraphrases, or non-empty obligation sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from score_v05_rubric_judging import aggregate, read_jsonl, score_row, sentence_spans


EMPTY_VALUE = r"(?:无|没有|none|null|暂无)"
LABELS = {
    "obligations_added": (r"新增义务", r"应新增"),
    "obligations_removed": (r"取消义务", r"应取消"),
}
EMPTY_TARGETS = {
    "obligations_added": {"应新增:无", "应新增:没有"},
    "obligations_removed": {"应取消:无", "应取消:没有"},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_empty(row: dict[str, Any], field: str) -> bool:
    claim = next((x for x in row["rubric"]["required_claims"] if x["id"] == field), None)
    if not claim:
        return False
    requirement = re.sub(r"\s+", "", claim.get("requirement", "")).replace("：", ":").rstrip("。")
    return requirement in EMPTY_TARGETS[field]


def explicit_empty_evidence(row: dict[str, Any], field: str) -> list[dict[str, Any]]:
    label_pattern = "(?:" + "|".join(LABELS[field]) + ")"
    pattern = re.compile(
        rf"{label_pattern}\s*[：:]\s*{EMPTY_VALUE}(?=$|[。！？；;\n])",
        re.IGNORECASE,
    )
    return [span for span in sentence_spans(row["answer"]) if pattern.search(span["text"])]


def merge_reasoning(base: dict[str, dict[str, Any]], reasons: dict[str, dict[str, Any]]) -> None:
    keys = ("reasoning_status", "reasoning_evidence_ids", "internal_conflict", "conflict_evidence_ids")
    for row_id, judgment in base.items():
        extra = reasons.get(row_id, {})
        if judgment.get("answer_sha256") != extra.get("answer_sha256"):
            raise ValueError(f"Reasoning-pass answer hash mismatch: {row_id}")
        for variant in ("a", "b"):
            parsed = extra.get("readers", {}).get(variant, {}).get("parsed") or {}
            target = judgment.get("readers", {}).get(variant, {}).get("parsed")
            if isinstance(target, dict):
                for key in keys:
                    target[key] = parsed.get(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--reasoning-judgements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    judgments = {x["id"]: x for x in read_jsonl(args.judgements)}
    reasons = {x["id"]: x for x in read_jsonl(args.reasoning_judgements)}
    merge_reasoning(judgments, reasons)

    repairs: list[dict[str, Any]] = []
    items = []
    for row in rows:
        item = score_row(row, judgments.get(row["id"]))
        for field in LABELS:
            if not required_empty(row, field):
                continue
            evidence = explicit_empty_evidence(row, field)
            if not evidence:
                continue
            previous = item["fields"][field]["consensus"]
            item["fields"][field] = {
                **item["fields"][field],
                "consensus": "satisfied",
                "deterministic_repair": {
                    "rule": "explicit_labeled_empty_delta_v1",
                    "previous_consensus": previous,
                    "evidence_spans": evidence,
                },
            }
            repairs.append({"id": row["id"], "field": field, "previous_consensus": previous,
                            "evidence_spans": evidence})

        checks = [value["consensus"] for value in item["fields"].values()]
        checks.extend((item["reasoning"]["consensus"], item["internal_consistency"]["consensus"]))
        bad = {"contradicted", "not_mentioned", "incorrect"}
        item["status"] = "fail" if any(x in bad for x in checks) else "pass" if all(x == "satisfied" for x in checks) else "uncertain"
        items.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "v06_v05_consensus_plus_explicit_empty_delta_sensitivity",
        "note": "Sensitivity analysis only. The original v0.5 report and judgements are unchanged. Repairs apply only to rubric-required empty sets with an explicit labeled empty marker; no open-ended semantic inference is added.",
        "provenance": {
            "input_sha256": digest(args.input),
            "judgements_sha256": digest(args.judgements),
            "reasoning_judgements_sha256": digest(args.reasoning_judgements),
            "scorer_sha256": digest(Path(__file__)),
        },
        "repair_counts": {
            "total": len(repairs),
            "previously_not_satisfied": sum(x["previous_consensus"] != "satisfied" for x in repairs),
            "by_field": {field: sum(x["field"] == field for x in repairs) for field in LABELS},
            "previous_status_by_field": {
                field: {status: sum(x["field"] == field and x["previous_consensus"] == status for x in repairs)
                        for status in ("satisfied", "not_mentioned", "contradicted", "uncertain", "incorrect")}
                for field in LABELS
            },
        },
        "metrics": aggregate(items, {}),
        "items": items,
        "repairs": repairs,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repair_counts": report["repair_counts"], "metrics": report["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
