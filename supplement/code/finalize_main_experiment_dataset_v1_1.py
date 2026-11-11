#!/usr/bin/env python3
"""Freeze a generated main dataset only after independent checks pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def freeze(data_dir: Path) -> dict[str, Any]:
    manifest = read_json(data_dir / "manifest.json")
    validation = read_json(data_dir / "validation_report.json")
    sealed = read_json(data_dir / "scorer_v09_sealed_report.json")
    v12 = read_json(data_dir / "scorer_v12_free_prose_report.json")
    v12_v05 = read_json(data_dir / "scorer_v12_free_prose_report_v05.json")
    v12_labels_path = data_dir / "scorer_v12_free_prose_labels.jsonl"
    v12_keys_path = data_dir / "scorer_v12_free_prose_private_key.jsonl"
    v12_inputs_path = data_dir / "scorer_v12_free_prose_inputs.jsonl"
    v12_manifest = read_json(data_dir / "scorer_v12_free_prose_readers.manifest.json")
    if manifest.get("status") != "generated_pending_finalization":
        raise ValueError("Dataset is not a fresh generated candidate")
    if manifest.get("version") != "main-experiment-formal-v1.1":
        raise ValueError(f"Unexpected candidate version: {manifest.get('version')}")
    if validation.get("status") != "passed":
        raise ValueError("Generator validation did not pass")

    expected_counts = {"events": 9, "domains": 121, "semantic_case_families": 2420,
                       "train_families": 540, "development_families": 180, "test_families": 1700}
    counts = manifest["counts"]
    for name, expected in expected_counts.items():
        if counts.get(name) != expected:
            raise ValueError(f"Unexpected manifest count {name}: {counts.get(name)} != {expected}")

    facts = read_jsonl(data_dir / "case_facts_gold.jsonl")
    gold = read_jsonl(data_dir / "gold_constraints.jsonl")
    domains = read_jsonl(data_dir / "domains.jsonl")
    family_by_id = {row["case_family_id"]: row for row in facts}
    gold_by_id = {row["case_family_id"]: row for row in gold}
    if len(family_by_id) != counts["semantic_case_families"] or set(family_by_id) != set(gold_by_id):
        raise ValueError("Family/gold join is incomplete or duplicated")

    domain_by_id = {row["domain_id"]: row for row in domains}
    supported_validity_modes = {"validity_hold", "validity_maintenance", "validity_reconsent",
                                "validity_quarantine", "validity_downgrade", "validity_route"}
    validity_domains = [row for row in domains if row["event_id"] == "RTE_VALIDITY_180_TO_90"]
    if not validity_domains or any(row["mode"] not in supported_validity_modes for row in validity_domains):
        raise ValueError("Validity event contains an unsupported or unmapped downstream mode")
    domains_by_event_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in domains:
        domains_by_event_split[(row["event_id"], row["split"])].add(row["domain_id"])
    events = sorted({row["event_id"] for row in domains})
    if len(events) != 9:
        raise ValueError("Expected nine events")
    for event in events:
        expected = {"train": 3, "development": 1}
        for split, n in expected.items():
            if len(domains_by_event_split[(event, split)]) != n:
                raise ValueError(f"Wrong {split} domain quota for {event}")
        if len(domains_by_event_split[(event, "test")]) not in {9, 10}:
            raise ValueError(f"Wrong test-domain quota for {event}")
        split_domains = [domains_by_event_split[(event, s)] for s in ("train", "development", "test")]
        if any(split_domains[i] & split_domains[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Domain leakage for {event}")

    facts_by_event: dict[str, set[str]] = defaultdict(set)
    facts_by_event_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_families = Counter()
    outcomes_by_split: dict[str, Counter] = defaultdict(Counter)
    outcome_by_event_split: dict[tuple[str, str], Counter] = defaultdict(Counter)
    exception_domains: set[str] = set()
    for fid, fact_row in family_by_id.items():
        event, split = fact_row["event_id"], fact_row["split"]
        signature = canonical(fact_row["facts"])
        if signature in facts_by_event[event]:
            raise ValueError(f"Duplicate fact vector in {event}")
        facts_by_event[event].add(signature)
        facts_by_event_split[(event, split)].add(signature)
        split_families[split] += 1
        outcome = gold_by_id[fid]["delta"]["result_type"]
        outcomes_by_split[split][outcome] += 1
        outcome_by_event_split[(event, split)][outcome] += 1
        if split == "test" and outcome == "exception_preserves":
            exception_domains.add(fact_row["domain_id"])
    for event in events:
        split_signatures = [facts_by_event_split[(event, s)] for s in ("train", "development", "test")]
        if any(split_signatures[i] & split_signatures[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Fact-vector leakage for {event}")
    if dict(split_families) != {"train": 540, "development": 180, "test": 1700}:
        raise ValueError(f"Wrong split sizes: {dict(split_families)}")
    if outcomes_by_split["test"]["exception_preserves"] != 30 or len(exception_domains) != 10:
        raise ValueError("Exception-preserving coverage is below the agreed design")

    structured_calibration = sealed["metrics"]["calibration"]
    if not structured_calibration["gate"]["passed"]:
        raise ValueError("Sealed scorer calibration did not pass")
    structured_overall = sealed["metrics"]["overall"]
    if (structured_calibration["positive_accepted"], structured_calibration["positive_n"], structured_calibration["false_accepted"],
            structured_calibration["negative_n"], structured_overall["n"], structured_overall["determinate_coverage"]) != (9, 9, 0, 27, 36, 1.0):
        raise ValueError(f"Unexpected structured sealed calibration result: {structured_calibration}")

    # The scorer must pass a fresh, label-free-in-prompt challenge. v10/v11 are
    # development-only because they were used to revise the reader/guard logic.
    v12_calibration = v12["metrics"]["calibration"]
    v12_overall = v12["metrics"]["overall"]
    if not v12_calibration["gate"]["passed"]:
        raise ValueError("Natural free-prose scorer calibration did not pass")
    if (v12_calibration["positive_accepted"], v12_calibration["positive_n"], v12_calibration["false_accepted"],
            v12_calibration["negative_n"], v12_overall["n"], v12_overall["determinate_coverage"]) != (8, 9, 0, 27, 36, 35 / 36):
        raise ValueError(f"Unexpected natural free-prose calibration result: {v12_calibration}")
    if v12_v05["metrics"]["calibration"] != v12_calibration:
        raise ValueError("v05 and v09 free-prose calibration metrics disagree")
    if v12["provenance"]["private_key_sha256"] != sha(v12_labels_path):
        raise ValueError("Natural free-prose report does not match local private labels")
    if v12["provenance"]["calibration_keys_sha256"] != sha(v12_keys_path):
        raise ValueError("Natural free-prose report does not match local challenge keys")
    if v12["provenance"]["input_sha256"] != sha(v12_inputs_path):
        raise ValueError("Natural free-prose report does not match the scored input packet")
    if v12_manifest.get("input_sha256") != sha(v12_inputs_path):
        raise ValueError("Remote scorer manifest input hash does not match local input")
    if v12_manifest.get("prompt_version") != "v05_sentence_ids_field_consensus_2_emptyset_semantics":
        raise ValueError("Unexpected natural free-prose prompt version")

    if sealed["provenance"]["private_key_sha256"] != sha(data_dir / "scorer_v09_sealed_labels.jsonl"):
        raise ValueError("Sealed scorer report does not match local private labels")
    all_private_keys = []
    for filename in ("scorer_v09_sealed_private_key.jsonl", "scorer_v10_free_prose_private_key.jsonl",
                     "scorer_v11_free_prose_private_key.jsonl", "scorer_v12_free_prose_private_key.jsonl"):
        all_private_keys.extend(read_jsonl(data_dir / filename))
    main_signatures = {canonical({"event_id": row["event_id"], "facts": row["facts"]}) for row in facts}
    calibration_signatures = [canonical({"event_id": row["event_id"], "facts": row["facts"]}) for row in all_private_keys]
    if any(signature in main_signatures for signature in calibration_signatures):
        raise ValueError("Scorer calibration overlaps main dataset facts")
    if len(set(calibration_signatures)) != len(calibration_signatures):
        raise ValueError("Scorer calibration fact vectors overlap across versions")

    manifest["status"] = "frozen_v1.1"
    manifest["version"] = "main-experiment-formal-v1.1"
    manifest["frozen_at"] = "2026-10-12"
    manifest["scorer_status"] = (
        "v0.9 structured calibration passed (9/9 positives, 0/27 false accepts, 36/36 determinate); "
        "a separate no-field-label controlled free-prose v12 calibration passed its preset gate "
        "(8/9 positives accepted, 0/27 false accepts, 35/36 determinate). One positive was conservatively "
        "uncertain due to reader disagreement. Evidence is limited and does not guarantee arbitrary prose. "
        "Both readers are two prompt styles of the same Qwen2.5-7B-Instruct model."
    )
    manifest["freeze_note"] = (
        "Unique unseen-domain test set selected for a 5pp planning target (1,700 families, 85 domains, "
        "20 families/domain, assumed ICC=.05); no additional 360-family exploration set. "
        "Power estimate remains assumption-based, not a formal guarantee."
    )
    (data_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    freeze_record = {
        "status": "frozen_v1.1", "frozen_at": manifest["frozen_at"], "counts": counts,
        "split_family_counts": dict(split_families),
        "test_result_type_counts": dict(outcomes_by_split["test"]),
        "test_exception_preserves_domains": sorted(exception_domains),
        "scorer_calibration": {
            "structured_v09": structured_calibration,
            "free_prose_v12": {"metrics": v12_calibration, "overall": v12_overall,
                                "prompt_version": v12_manifest["prompt_version"],
                                "input_sha256": v12_manifest["input_sha256"],
                                "run_manifest": data_dir.joinpath("scorer_v12_free_prose_readers.manifest.json").name},
            "development_only": {
                "v10": "failed: 0/9 positives accepted; 0/27 false accepts; 3 uncertain",
                "v11": "failed after conservative guards: 7/9 positives accepted; 0/27 false accepts; 2 uncertain",
            },
            "interpretation": "v12 passed only the preset controlled-calibration gate; it contains 9 positive and 27 deliberately corrupted candidates, and does not establish reliability for arbitrary free text. Two prompt styles from one Qwen model are not independent raters.",
        },
        "checks": {"generator_validation": "passed", "audit_notebook": "executed_top_to_bottom_and_passed",
                   "sample_size_notebook": "executed_top_to_bottom; estimates 1,695 strict-score families, rounded to 1,700",
                   "domain_and_fact_overlap": "none", "all_calibration_fact_vectors_disjoint": "passed",
                   "v12_free_prose_scorer_calibration": "passed_preset_gate_with_8_of_9_positive_acceptance"},
        "limitations": [
            "5pp sample-size target relies on a planning ICC and method-selection pilot variance, not an independent power guarantee.",
            "Only 30 exception-preserving test cases, all under the age event; treat this slice as descriptive.",
            "The scorer's natural-language calibration is small and limited to controlled prose with deliberately injected errors; arbitrary natural-language reliability is not guaranteed.",
            "Mechanical oracle agreement does not establish conceptual validity of the fictional rules or domain specifications.",
            "Three surface forms are controlled templates; perceived difficulty requires an independent reader study.",
        ],
        "source_hashes": {
            "generator": sha(data_dir.parent.parent / "scripts" / "build_main_experiment_formal_v1_1.py"),
            "validity_regression_tests": sha(data_dir.parent.parent / "scripts" / "test_main_experiment_formal_v1_1_rules.py"),
            "finalizer": sha(Path(__file__).resolve()),
            "scorer_postprocessor": sha(data_dir.parent.parent / "scripts" / "score_v09_structured_state_alignment.py"),
            "v05_scorer": sha(data_dir.parent.parent / "scripts" / "score_v05_rubric_judging.py"),
            "v05_formal_guards": sha(data_dir.parent.parent / "scripts" / "v05_formal_assertion_guards.py"),
            "free_prose_judge": sha(data_dir.parent.parent / "scripts" / "run_v05_rubric_judge.py"),
            "free_prose_challenge_builder": sha(data_dir.parent.parent / "scripts" / "build_v10_free_prose_calibration.py"),
            "notebook_executor": sha(data_dir.parent.parent / "scripts" / "execute_standard_library_notebook.py"),
        },
    }
    (data_dir / "freeze_record.json").write_text(json.dumps(freeze_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {path.name: sha(path) for path in sorted(data_dir.iterdir()) if path.is_file() and path.name != "SHA256.json"}
    (data_dir / "SHA256.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": manifest["status"], "counts": counts,
            "test_result_type_counts": dict(outcomes_by_split["test"]),
            "scorer_calibration": v12_calibration, "hashed_files": len(hashes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--refresh-hashes", action="store_true",
                        help="refresh SHA256.json after reproducible notebooks or documentation are updated")
    args = parser.parse_args()
    if args.refresh_hashes:
        manifest = read_json(args.data_dir / "manifest.json")
        if manifest.get("status") != "frozen_v1.1" or manifest.get("version") != "main-experiment-formal-v1.1":
            raise ValueError("Hash refresh is only allowed for a frozen v1.1 dataset")
        hashes = {path.name: sha(path) for path in sorted(args.data_dir.iterdir())
                  if path.is_file() and path.name != "SHA256.json"}
        (args.data_dir / "SHA256.json").write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "hashes_refreshed", "hashed_files": len(hashes)}, ensure_ascii=False))
        return
    print(json.dumps(freeze(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
