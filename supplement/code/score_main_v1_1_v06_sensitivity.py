#!/usr/bin/env python3
"""Conservative v0.6 sensitivity pass for explicit empty obligation deltas.

This does not rerun a judge or replace the frozen v0.5 result. It only repairs
an obligation field when (a) the frozen rubric explicitly defines its gold set
as empty, (b) the answer contains a direct, narrowly whitelisted empty-set
statement, and (c) neither v0.5 reader marked a contradiction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from score_v05_rubric_judging import aggregate, read_jsonl, sentence_spans


FIELDS = ("obligations_added", "obligations_removed")
EMPTY_GOLD_MARKERS = {
    "obligations_added": "新增义务的正确集合为空",
    "obligations_removed": "取消义务的正确集合为空",
}
EMPTY_GOLD_EXACT_REQUIREMENTS = {
    # The small-scale Round34 packet uses these compact, explicit empty-set
    # labels instead of the longer main-experiment rubric wording.
    "obligations_added": {"应新增:无"},
    "obligations_removed": {"应取消:无"},
}

# The natural-language forms below are intentionally narrow. They include the
# explicit empty labels used by earlier v0.6 and the exact accepted phrasings
# in the v1.1 rubric; this is not a semantic paraphrase classifier.
EMPTY_PATTERNS = {
    "obligations_added": tuple(re.compile(p) for p in (
        r"(?<!应)(?<!不)(?:本案|本次|此次|该案)?新增义务[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"(?<!不)应新增义务[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"(?<!不)应新增[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"(?<!不)(?:应新增|需要新增)[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"不应新增[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"(?:没有|无|并无)(?:任何)?(?:新的|新增)义务",
        r"(?:没有|无)(?:任何|其他)?义务需要新增",
        r"(?:无需|不需要|不必)新增(?:任何)?义务",
        r"新增义务(?:的正确集合)?(?:为|是)?空",
    )),
    "obligations_removed": tuple(re.compile(p) for p in (
        r"(?<!应)(?<!不)(?:本案|本次|此次|该案)?取消义务[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"(?<!不)应取消义务[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"(?<!不)应取消[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"(?<!不)(?:应取消|需要取消)[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"不应取消[:：](?:无|没有|暂无|none|null|为空)(?=$|[。！？；;，,、])",
        r"(?:没有|无|并无)(?:任何)?(?:既有|原有)?义务被取消",
        r"(?:没有|无)(?:任何|其他|既有|原有)?义务需要取消",
        r"原有义务均继续适用",
        r"(?:没有|无|并无)取消义务",
        r"(?:无需|不需要|不必)取消(?:任何)?(?:既有|现有|原有)?义务",
        r"取消义务(?:的正确集合)?(?:为|是)?空",
    )),
}
NEGATED_OR_HYPOTHETICAL_PREFIX = re.compile(
    r"(?:并非|并不是|不是|不能(?:说|认为|断言)?|不应(?:说|认为|认定)?|"
    r"不代表|不等于|如果|若|假如|假设|除非|一旦|是否)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[\s*#`_]+", "", text).replace("：", ":").lower()


def rubric_requires_empty(row: dict[str, Any], field: str) -> bool:
    claim = next((x for x in row["rubric"]["required_claims"] if x["id"] == field), None)
    if not claim:
        return False
    requirement = normalize(claim.get("requirement", ""))
    return (EMPTY_GOLD_MARKERS[field] in requirement or
            requirement in EMPTY_GOLD_EXACT_REQUIREMENTS[field])


def direct_empty_evidence(answer: str, field: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for span in sentence_spans(answer):
        text = normalize(span["text"])
        for pattern in EMPTY_PATTERNS[field]:
            for match in pattern.finditer(text):
                prefix = text[max(0, match.start() - 12):match.start()]
                if NEGATED_OR_HYPOTHETICAL_PREFIX.search(prefix):
                    continue
                found.append({**span, "matched_text": match.group(0)})
                break
            if found and found[-1]["id"] == span["id"]:
                break
    return found


def reader_statuses(field_item: dict[str, Any]) -> dict[str, str]:
    readers = field_item.get("readers", {})
    return {name: value.get("status", "uncertain") for name, value in readers.items()
            if name in {"a", "b"} and isinstance(value, dict)}


def recompute_status(item: dict[str, Any]) -> None:
    states = [value.get("consensus", "uncertain") for value in item.get("fields", {}).values()]
    states.extend((item.get("reasoning", {}).get("consensus", "uncertain"),
                   item.get("internal_consistency", {}).get("consensus", "uncertain")))
    bad = {"contradicted", "not_mentioned", "incorrect"}
    item["status"] = ("fail" if any(x in bad for x in states) else
                      "pass" if all(x == "satisfied" for x in states) else "uncertain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v05-report", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet_hash = sha256(args.packet)
    report_hash = sha256(args.v05_report)
    report = json.loads(args.v05_report.read_text(encoding="utf-8"))
    if report.get("protocol") != "v05_field_preserving_consensus":
        raise ValueError(f"Expected frozen v0.5 report, got {report.get('protocol')!r}")
    if report.get("provenance", {}).get("input_sha256") != packet_hash:
        raise ValueError("Packet hash differs from the packet scored by the frozen v0.5 report")

    packet = read_jsonl(args.packet)
    packet_by_id = {row["id"]: row for row in packet}
    items = report.get("items", [])
    if len(items) != len(packet) or len(packet_by_id) != len(packet):
        raise ValueError("Packet/report row count mismatch or duplicate packet IDs")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Duplicate scored item IDs")

    # Build an independent output object so the frozen v0.5 artifact stays intact.
    output_items = items
    repairs: list[dict[str, Any]] = []
    repair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in output_items:
        row = packet_by_id.get(item["id"])
        if row is None or item.get("family_id") != row.get("family_id"):
            raise ValueError(f"Packet/scored item identity mismatch: {item.get('id')}")
        if item.get("metadata") != row.get("metadata"):
            raise ValueError(f"Packet/scored metadata mismatch: {item.get('id')}")
        for field in FIELDS:
            if not rubric_requires_empty(row, field):
                continue
            evidence = direct_empty_evidence(row["answer"], field)
            if not evidence:
                continue
            field_item = item.get("fields", {}).get(field, {})
            previous = field_item.get("consensus", "uncertain")
            if previous == "satisfied":
                continue
            statuses = reader_statuses(field_item)
            if set(statuses) != {"a", "b"} or any(
                status not in {"satisfied", "not_mentioned", "uncertain"}
                for status in statuses.values()
            ):
                continue
            if field_item.get("formal_guard") or any(
                error.get("field") == field for error in item.get("formal_guard_errors", [])
            ):
                continue
            field_item["consensus"] = "satisfied"
            field_item["deterministic_repair"] = {
                "rule": "v1_1_direct_explicit_empty_obligation_v1",
                "previous_consensus": previous,
                "reader_statuses": statuses,
                "evidence_spans": evidence,
            }
            repairs.append({"id": item["id"], "family_id": item["family_id"],
                            "method": item.get("metadata", {}).get("method"),
                            "field": field, "previous_consensus": previous,
                            "reader_statuses": statuses, "evidence_spans": evidence})
            repair_counts[field][previous] += 1
        recompute_status(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": "main_v1_1_v06_explicit_empty_sensitivity",
        "note": ("Post-hoc deterministic sensitivity analysis over the frozen v0.5 judgments. "
                 "Only rubric-required empty obligation sets with a direct whitelisted statement are repaired; "
                 "any reader-level contradiction or matching formal guard prevents repair. "
                 "The original v0.5 report is not modified; no model is rerun."),
        "provenance": {
            "v05_score_report_sha256": report_hash,
            "packet_sha256": packet_hash,
            "v05_provenance": report.get("provenance", {}),
            "scorer_sha256": sha256(Path(__file__)),
        },
        "repair_counts": {
            "total": len(repairs),
            "by_field": dict(Counter(x["field"] for x in repairs)),
            "by_method": dict(Counter(x["method"] for x in repairs)),
            "by_previous_consensus": dict(Counter(x["previous_consensus"] for x in repairs)),
        },
        "metrics": aggregate(output_items, {}),
        "items": output_items,
        "repairs": repairs,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": result["protocol"], "repair_counts": result["repair_counts"],
                      "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
