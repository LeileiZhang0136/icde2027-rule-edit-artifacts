#!/usr/bin/env python3
"""Field-preserving, evidence-bound scoring. Historical v04 files stay intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from score_v04_rubric_judging import evidence_is_grounded, read_jsonl
from v05_formal_assertion_guards import check_assertions

STATES = {"satisfied", "contradicted", "not_mentioned", "uncertain"}
BAD = {"contradicted", "not_mentioned", "incorrect"}
FIELDS = ("decision", "handling", "obligations_added", "obligations_removed",
          "obligations_unchanged", "human_review", "reasoning", "internal_consistency")


def sentence_spans(answer: str) -> list[dict[str, Any]]:
    """Return exact, nonoverlapping source spans; generated text never supplies them."""
    spans = []
    for match in re.finditer(r"[^。！？\n]+[。！？]?", answer):
        start, end = match.span()
        while start < end and answer[start].isspace():
            start += 1
        while end > start and answer[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append({"id": f"S{len(spans):03d}", "start": start, "end": end,
                          "text": answer[start:end]})
    return spans


def resolve_evidence(item: dict[str, Any], answer: str, spans: list[dict[str, Any]],
                     key: str = "evidence_ids", legacy: str = "evidence") -> dict[str, Any]:
    indexed = {part["id"]: part for part in spans}
    refs = item.get(key)
    if refs is not None:
        # Brackets and zero padding carry no semantic information. Normalize only
        # explicit sentence identifiers; never guess from a quote or a bare number.
        if isinstance(refs,list):
            normalized=[]
            for ref in refs:
                match=re.fullmatch(r'\s*\[?(S\d+)\]?\s*',ref) if isinstance(ref,str) else None
                normalized.append(f'S{int(match[1][1:]):03d}' if match else ref)
            refs=normalized
        if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or x not in indexed for x in refs):
            return {"grounded": False, "issue": "invalid_or_empty_evidence_ids"}
        selected = [indexed[x] for x in dict.fromkeys(refs)]
        return {"grounded": True, "spans": selected}
    quote = item.get(legacy, "")
    grounded = isinstance(quote, str) and evidence_is_grounded(quote, answer)
    return {"grounded": grounded, "quote": quote,
            "issue": None if grounded else "unanchored_quote"}


def normalize_reader(reader: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    required = [x["id"] for x in row["rubric"]["required_claims"]]
    requirements={x['id']:x.get('requirement','') for x in row['rubric']['required_claims']}
    answer = row["answer"]
    spans = sentence_spans(answer)
    criteria = {key: {"status": "uncertain", "issue": "missing_criterion"} for key in required}
    result = {"criteria": criteria, "reasoning": {"status": "uncertain", "issue": "missing_reader"},
              "internal_consistency": {"status": "uncertain", "issue": "missing_reader"}}
    parsed = reader.get("parsed")
    if not isinstance(parsed, dict):
        result["issue"] = "invalid_json"
        return result
    entries = parsed.get("criteria", [])
    if not isinstance(entries, list):
        entries = []
    seen = Counter(x.get("id") for x in entries if isinstance(x, dict) and isinstance(x.get("id"), str))
    for item in entries:
        if not isinstance(item, dict) or item.get("id") not in criteria:
            continue
        key = item["id"]
        if seen[key] != 1:
            criteria[key] = {"status": "uncertain", "issue": "duplicate_criterion"}
            continue
        status = {"contradictory": "contradicted", "supported": "satisfied", "none": "not_mentioned"}.get(item.get("status"), item.get("status"))
        if status not in STATES:
            criteria[key] = {"status": "uncertain", "issue": "invalid_status"}
        elif status in {"satisfied", "contradicted"}:
            evidence = resolve_evidence(item, answer, spans)
            # A missing citation for an explicit empty-set sentence can be
            # recovered deterministically. This repairs location only; it never
            # changes a reader's semantic verdict or supplies an omitted claim.
            if status=='satisfied' and item.get('evidence_ids')==[] and key in {'obligations_added','obligations_removed'}:
                direction='新增' if key=='obligations_added' else '取消'
                target=re.sub(r'\s+','',requirements[key]).rstrip('。')
                if target==f'应{direction}：无':
                    templates={f'应{direction}：无',f'应{direction}:无',f'{direction}义务：无',f'{direction}义务:无',
                               f'此次不存在应{direction}的义务',f'此次没有需要{direction}的义务',f'无需{direction}任何义务'}
                    located=[part for part in spans if re.sub(r'\s+','',part['text']).strip('*`#。；;') in templates]
                    if located:
                        evidence={'grounded':True,'spans':located,'recovered_from':'explicit_standalone_empty_delta'}
            criteria[key] = {"status": status if evidence["grounded"] else "uncertain",
                             "evidence": evidence, "issue": evidence.get("issue")}
        else:
            criteria[key] = {"status": status, "issue": "reader_uncertain" if status == "uncertain" else None}
    reasoning = parsed.get("reasoning_status", "uncertain")
    if reasoning in {"supported", "contradictory"}:
        evidence = resolve_evidence(parsed, answer, spans, "reasoning_evidence_ids", "reasoning_evidence")
        result["reasoning"] = {"status": ({"supported": "satisfied", "contradictory": "contradicted"}[reasoning]
                                          if evidence["grounded"] else "uncertain"),
                               "evidence": evidence, "issue": evidence.get("issue")}
    else:
        result["reasoning"] = {"status": "not_mentioned" if reasoning == "not_present" else "uncertain",
                               "issue": None if reasoning == "not_present" else "reader_uncertain"}
    conflict = parsed.get("internal_conflict", "uncertain")
    if conflict == "yes":
        evidence = resolve_evidence(parsed, answer, spans, "conflict_evidence_ids", "conflict_evidence")
        result["internal_consistency"] = {"status": "contradicted" if evidence["grounded"] else "uncertain",
                                           "evidence": evidence, "issue": evidence.get("issue")}
    else:
        result["internal_consistency"] = {"status": "satisfied" if conflict == "no" else "uncertain",
                                           "issue": None if conflict == "no" else "reader_uncertain"}
    return result


def consensus(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = left["status"], right["status"]
    status = a if a == b else "incorrect" if a in BAD and b in BAD else "uncertain"
    return {"consensus": status, "readers": {"a": left, "b": right},
            "issue": None if status != "uncertain" else "reader_disagreement" if a != b else "both_uncertain"}


def score_row(row: dict[str, Any], judgement: dict[str, Any] | None) -> dict[str, Any]:
    basic = {"id": row["id"], "family_id": row["family_id"], "metadata": row.get("metadata", {})}
    required = row["rubric"]["required_claims"]
    if not row["answer"].strip():
        return {**basic, "status": "fail", "reason": "empty_answer",
                "fields": {x["id"]: {"consensus": "not_mentioned", "category": x["category"]} for x in required},
                "reasoning": {"consensus": "not_mentioned"}, "internal_consistency": {"consensus": "uncertain"}}
    actual_hash = hashlib.sha256(row["answer"].encode()).hexdigest()
    valid = judgement and judgement.get("answer_sha256") == actual_hash == row.get("answer_sha256")
    readers = judgement.get("readers", {}) if valid else {}
    a = normalize_reader(readers.get("a", {}), row)
    b = normalize_reader(readers.get("b", {}), row)
    fields = {x["id"]: {**consensus(a["criteria"][x["id"]], b["criteria"][x["id"]]), "category": x["category"]} for x in required}
    reasoning = consensus(a["reasoning"], b["reasoning"])
    internal = consensus(a["internal_consistency"], b["internal_consistency"])
    formal_errors = check_assertions(row, sentence_spans(row['answer']))
    for error in formal_errors:
        # Explicit factual contradictions can veto a pass, but never create one.
        reasoning['consensus'] = 'contradicted'
        reasoning['formal_guard'] = True
        if error['field'] in fields:
            fields[error['field']]['consensus'] = 'contradicted'
            fields[error['field']]['formal_guard'] = True
        elif error['field'] == 'internal_consistency':
            internal['consensus'] = 'contradicted'
            internal['formal_guard'] = True
    checks = [v["consensus"] for v in fields.values()] + [reasoning["consensus"], internal["consensus"]]
    status = "fail" if any(x in BAD for x in checks) else "pass" if all(x == "satisfied" for x in checks) else "uncertain"
    return {**basic, "status": status, "reason": None if valid else "missing_or_mismatched_judgement",
            "fields": fields, "reasoning": reasoning, "internal_consistency": internal,
            "formal_guard_errors": formal_errors}


def counts(values: list[str], overall: bool = False) -> dict[str, Any]:
    c, n = Counter(values), len(values)
    good = c["pass"] if overall else c["satisfied"]
    bad = c["fail"] if overall else sum(c[x] for x in BAD)
    unknown = c["uncertain"]
    return {"n": n, "correct": good, "incorrect": bad, "uncertain": unknown,
            "confirmed_correct_rate": good / n if n else None,
            "determinate_coverage": (good + bad) / n if n else None,
            "accuracy_determinate_only": good / (good + bad) if good + bad else None,
            "status_counts": dict(c)}


def aggregate(items: list[dict[str, Any]], private: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"overall": counts([x["status"] for x in items], True), "by_method": {}, "fields_by_method": {}}
    for method in sorted({x["metadata"].get("method", "calibration") for x in items}):
        rows = [x for x in items if x["metadata"].get("method", "calibration") == method]
        metrics["by_method"][method] = counts([x["status"] for x in rows], True)
        metrics["fields_by_method"][method] = {
            field: counts([x.get(field, {"consensus": "uncertain"})["consensus"] if field in {"reasoning", "internal_consistency"}
                           else x["fields"].get(field, {"consensus": "uncertain"})["consensus"] for x in rows])
            for field in FIELDS}
    if private:
        positives = [x for x in items if private[x["id"]].get("expected") is True]
        negatives = [x for x in items if private[x["id"]].get("expected") is False]
        accept = sum(x["status"] == "pass" for x in positives)
        false = sum(x["status"] == "pass" for x in negatives)
        metrics["calibration"] = {"positive_n": len(positives), "positive_accepted": accept,
                                  "positive_acceptance": accept / max(1, len(positives)),
                                  "negative_n": len(negatives), "false_accepted": false,
                                  "false_accept_rate": false / max(1, len(negatives)),
                                  "negative_detected": sum(x["status"] == "fail" for x in negatives),
                                  "gate": {"min_positive_acceptance": 0.80, "max_false_accept_rate": 0.02,
                                           "min_determinate_coverage": 0.90,
                                           "passed": accept / max(1, len(positives)) >= 0.80 and false / max(1, len(negatives)) <= 0.02
                                                     and metrics["overall"]["determinate_coverage"] >= 0.90}}
    family = defaultdict(list)
    for item in items:
        family[(item["metadata"].get("method", "calibration"), item["family_id"])].append(item["status"])
    metrics["family_by_method"] = {}
    for method in metrics["by_method"]:
        statuses = ["pass" if all(x == "pass" for x in states) else "fail" if "fail" in states else "uncertain"
                    for (m, f), states in family.items() if m == method]
        metrics["family_by_method"][method] = counts(statuses, True)
    metrics["by_difficulty"] = {difficulty: counts([x["status"] for x in items if x["metadata"].get("difficulty") == difficulty], True)
                                for difficulty in sorted({x["metadata"].get("difficulty") for x in items if x["metadata"].get("difficulty")})}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--reasoning-judgements", type=Path, action='append')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    judgements = {x["id"]: x for x in read_jsonl(args.judgements)}
    if args.reasoning_judgements:
        reasons = {x['id']: x for path in args.reasoning_judgements for x in read_jsonl(path)}
        reason_keys = ('reasoning_status', 'reasoning_evidence_ids', 'internal_conflict', 'conflict_evidence_ids')
        for key, judge in judgements.items():
            extra = reasons.get(key, {})
            if judge.get('answer_sha256') != extra.get('answer_sha256'):
                raise ValueError(f'Reasoning-pass answer hash mismatch: {key}')
            for variant in ('a','b'):
                parsed = extra.get('readers',{}).get(variant,{}).get('parsed') or {}
                base = judge.get('readers',{}).get(variant,{}).get('parsed')
                if not isinstance(base,dict):
                    continue
                for field in reason_keys:
                    base[field] = parsed.get(field)
    private = {x["id"]: x for x in read_jsonl(args.private)} if args.private else {}
    items = [score_row(x, judgements.get(x["id"])) for x in rows]
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    provenance={'input_sha256':digest(args.input),'judgements_sha256':digest(args.judgements),
                'reasoning_judgements_sha256':[digest(p) for p in (args.reasoning_judgements or [])],
                'scorer_sha256':digest(__file__),
                'formal_guards_sha256':digest(Path(__file__).with_name('v05_formal_assertion_guards.py'))}
    report = {"protocol": "v05_field_preserving_consensus", "provenance":provenance,
              "metrics": aggregate(items, private), "items": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
