#!/usr/bin/env python3
"""Prepare leakage-audited inputs for Rule-RAG, Case-RAG and SERAC-style baselines.

Only train supervision, public dev/test prompts, rule text, and development-only
scope labels are exported. Test gold is intentionally never parsed or copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def development_scope_labels(dev_rows: list[dict[str, Any]], dev_families: set[str]) -> list[dict[str, Any]]:
    """Use only development rows to derive the minimal gate-calibration labels."""
    labels: dict[str, dict[str, Any]] = {}
    for row in dev_rows:
        family_id = row.get("case_family_id")
        if family_id not in dev_families or row.get("split") != "development":
            raise ValueError(f"Unexpected development family/split: {family_id}")
        result_type = row["delta"]["result_type"]
        label = {"case_family_id": family_id, "event_id": row["event_id"],
                 "result_type": result_type, "route_to_editor": result_type != "unaffected"}
        previous = labels.setdefault(family_id, label)
        if previous != label:
            raise ValueError(f"Development forms disagree on scope label: {family_id}")
    if set(labels) != dev_families:
        raise ValueError("Development-only gold labels do not exactly cover dev prompt families")
    return list(labels.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rules_path = args.dataset_dir / "rules.jsonl"
    train_path = args.prepared_dir / "train_sft.jsonl"
    dev_path = args.prepared_dir / "dev_closed_book.jsonl"
    test_path = args.prepared_dir / "test_closed_book.jsonl"
    rules = read_jsonl(rules_path)
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    test = read_jsonl(test_path)

    if len(rules) != 9 or len(dev) != 540 or len(test) != 5100:
        raise ValueError(f"Unexpected source counts: rules={len(rules)}, dev={len(dev)}, test={len(test)}")
    if len({row["event_id"] for row in rules}) != 9:
        raise ValueError("Rule event IDs are not unique")
    if any(row.get("kind") not in {"case", "rule"} for row in train):
        raise ValueError("Unknown train row kind")
    case_train = [row for row in train if row.get("kind") == "case"]
    if len(case_train) != 1620 or len([row for row in case_train if row["difficulty"] == "medium"]) != 540:
        raise ValueError("Expected 1,620 case rows and exactly 540 medium canonical exemplars")
    dev_families = {row["case_family_id"] for row in dev}
    train_families = {row["case_family_id"] for row in case_train}
    test_families = {row["case_family_id"] for row in test}
    if (len(dev_families), len(train_families), len(test_families)) != (180, 540, 1700):
        raise ValueError("Unexpected unique family counts")
    if train_families & dev_families or train_families & test_families or dev_families & test_families:
        raise ValueError("Family leakage across splits")
    if any("当前更新后的上层规则：" in row["prompt"] for row in dev + test):
        raise ValueError("Closed-book dev/test prompts expose the updated rule")
    forbidden_test_keys = {"target", "answer", "target_constraints", "old_state", "new_state", "delta"}
    if any(forbidden_test_keys & set(row) for row in test):
        raise ValueError("Test prompt source unexpectedly contains gold/target fields")

    dev_labels = development_scope_labels(dev, dev_families)
    public_fields = ("id", "request_id", "case_family_id", "family_id", "event_id",
                     "domain_id", "difficulty", "split", "prompt")
    dev_public = [{key: row[key] for key in public_fields} for row in dev]
    test_public = [{key: row[key] for key in public_fields} for row in test]
    outputs = {
        "rules.jsonl": rules,
        "train_sft.jsonl": train,
        "dev_closed_book.jsonl": dev_public,
        "test_closed_book.jsonl": test_public,
        "dev_scope_labels.jsonl": dev_labels,
    }
    for name, rows in outputs.items():
        write_jsonl(args.output_dir / name, rows)
    source_hashes = {str(path): sha256(path) for path in (rules_path, train_path, test_path)}
    manifest = {
        "protocol": "memory_baselines_v1_1_seed42",
        "rules": len(rules), "train_case_rows": len(case_train), "case_memory_canonical_rows": 540,
        "train_rule_qa_rows_excluded_from_case_memory": len(train) - len(case_train),
        "dev_rows": len(dev), "dev_families": len(dev_families),
        "test_rows": len(test), "test_families": len(test_families),
        "test_gold_read": False,
        "dev_gold_use": "only development result_type for gate calibration; other gold fields stripped before export",
        "scope_label_definition": "route_to_editor iff result_type != unaffected; exception_preserves is positive",
        "sources_sha256": source_hashes,
        "outputs_sha256": {name: sha256(args.output_dir / name) for name in outputs},
        "rows": {name: len(rows) for name, rows in outputs.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
