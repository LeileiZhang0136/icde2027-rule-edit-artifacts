#!/usr/bin/env python3
"""Run E5 Rule-RAG, train a SERAC-style scope gate, and run Case-RAG.

The model-visible inputs contain only prompts and retrieved training material.
IDs, labels, event/domain metadata and gate targets are used only by the router,
for output bookkeeping, or for development diagnostics. Test gold is not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_fingerprint(model_dir: Path, weight_glob: str) -> dict[str, Any]:
    config = model_dir / "config.json"
    weights = sorted(model_dir.glob(weight_glob))
    if not config.is_file() or not weights:
        raise FileNotFoundError(f"Missing config or model weights under {model_dir}")
    return {"path": str(model_dir), "config_sha256": digest(config),
            "weights": [{"name": path.name, "size_bytes": path.stat().st_size,
                         "sha256": digest(path)} for path in weights]}


def prompt_sections(prompt: str) -> tuple[str, str, str]:
    """Extract visible historical rule, downstream domain rule, and case facts."""
    history_marker = "历史上层规则（仅用于比较，不代表本案现行版本）："
    domain_marker = "本领域规则："
    case_marker = "案例材料："
    if domain_marker not in prompt or case_marker not in prompt:
        raise ValueError("Prompt is missing expected v1.1 section markers")
    history = ""
    if history_marker in prompt:
        history = prompt.split(history_marker, 1)[1].split("\n", 1)[0].strip()
    after_domain = prompt.split(domain_marker, 1)[1]
    domain = after_domain.split("\n" + case_marker, 1)[0].strip()
    # Training prompts explicitly include the new rule before the case block;
    # remove that line so retrieval and gate features match closed-book queries.
    domain = re.split(r"\n当前更新后的上层规则：", domain, maxsplit=1)[0].strip()
    facts = after_domain.split("\n" + case_marker, 1)[1].strip()
    # RuleQA/test boilerplate is excluded; case facts remain intact.
    facts = re.split(r"\n(?:请判断|请根据|请作出|请说明)", facts, maxsplit=1)[0].strip()
    return history, domain, facts


def split_long_text(text: str, tokenizer: Any, budget: int = 480) -> list[str]:
    """Losslessly split into E5-sized chunks; never rely on silent truncation."""
    text = text.strip()
    if not text:
        return []
    if len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) <= budget:
        return [text]
    atoms = [part for part in re.split(r"(?<=[。！？；\n])", text) if part]
    if len(atoms) <= 1:
        middle = len(text) // 2
        for sep in ("，", "、", "：", " "):
            left = text.rfind(sep, 0, middle)
            right = text.find(sep, middle)
            choices = [x for x in (left, right) if x > 0]
            if choices:
                middle = min(choices, key=lambda x: abs(x - middle)) + 1
                break
        if middle <= 0 or middle >= len(text):
            raise ValueError("Unable to split long retrieval text without dropping content")
        return split_long_text(text[:middle], tokenizer, budget) + split_long_text(text[middle:], tokenizer, budget)
    chunks: list[str] = []
    current = ""
    for atom in atoms:
        candidate = current + atom
        if current and len(tokenizer(candidate, add_special_tokens=True, truncation=False)["input_ids"]) > budget:
            chunks.append(current)
            current = atom
        else:
            current = candidate
        if len(tokenizer(current, add_special_tokens=True, truncation=False)["input_ids"]) > budget:
            chunks.extend(split_long_text(current, tokenizer, budget))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def l2_normalize(x: Any) -> Any:
    import torch
    return torch.nn.functional.normalize(x.float(), p=2, dim=-1)


def encode_documents(documents: list[list[tuple[str, str]]], prefix: str, tokenizer: Any,
                     encoder: Any, device: Any, section_weights: dict[str, float] | None = None,
                     batch_size: int = 48) -> Any:
    """Batch-encode named document sections; every section is chunked, not truncated."""
    import torch
    section_weights = section_weights or {}
    flat: list[tuple[int, str, str]] = []
    for doc_index, sections in enumerate(documents):
        for name, text in sections:
            for chunk in split_long_text(text, tokenizer):
                flat.append((doc_index, name, prefix + chunk))
    vectors: list[Any] = []
    for start in range(0, len(flat), batch_size):
        batch = [item[2] for item in flat[start:start + batch_size]]
        tokens = tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to(device)
        if tokens.input_ids.shape[1] > 512:
            raise ValueError(f"E5 input exceeded 512 tokens: {tokens.input_ids.shape[1]}")
        with torch.inference_mode():
            hidden = encoder(**tokens).last_hidden_state.float()
            mask = tokens.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            vectors.extend(l2_normalize(pooled).cpu().unbind(0))
    # Aggregate chunks within each section, then combine sections with frozen weights.
    grouped: list[dict[str, list[Any]]] = [dict() for _ in documents]
    for item, vector in zip(flat, vectors):
        grouped[item[0]].setdefault(item[1], []).append(vector)
    rows = []
    for section_map in grouped:
        if not section_map:
            raise ValueError("Empty section bundle in retrieval input")
        combined = None
        total_weight = 0.0
        for section_name, chunks in section_map.items():
            section_vector = l2_normalize(torch.stack(chunks).mean(0))
            weight = section_weights.get(section_name, 1.0)
            combined = section_vector * weight if combined is None else combined + section_vector * weight
            total_weight += weight
        rows.append(l2_normalize((combined / max(total_weight, 1e-9)).unsqueeze(0))[0])
    return torch.stack(rows)


def build_query_embeddings(rows: list[dict[str, Any]], include_history: bool,
                           tokenizer: Any, encoder: Any, device: Any) -> Any:
    documents = []
    for row in rows:
        history, domain, facts = prompt_sections(row["prompt"])
        components: list[tuple[str, str]] = []
        if include_history and history:
            components.append(("history", history))
        components.extend((("domain", domain), ("facts", facts)))
        documents.append(components)
    weights = {"history": 0.20, "domain": 0.40, "facts": 0.40} if include_history else {"domain": 0.50, "facts": 0.50}
    return encode_documents(documents, "query: ", tokenizer, encoder, device, weights)


def build_passage_embeddings(rows: list[dict[str, Any]], tokenizer: Any,
                             encoder: Any, device: Any) -> Any:
    documents = []
    for row in rows:
        _, domain, facts = prompt_sections(row["prompt"])
        documents.append([("domain", domain), ("facts", facts)])
    return encode_documents(documents, "passage: ", tokenizer, encoder, device,
                            {"domain": 0.50, "facts": 0.50})


def build_rule_embeddings(rules: list[dict[str, Any]], tokenizer: Any,
                          encoder: Any, device: Any) -> Any:
    docs = []
    for rule in rules:
        old = rule.get("old_rule", "")
        new = rule.get("new_rule", "")
        effective = rule.get("effective_date", "")
        doc = f"历史规则：{old}\n更新后的规则：{new}\n生效时间：{effective}"
        if not old or not new:
            raise ValueError(f"Rule document incomplete: {rule.get('event_id')}")
        docs.append(("rule", doc))
    return encode_documents([[(name, text)] for name, text in docs], "passage: ", tokenizer,
                            encoder, device, {"rule": 1.0})


def pair_features(query: Any, rules: Any) -> Any:
    import torch
    n, dim = query.shape
    r = rules.unsqueeze(0).expand(n, -1, -1)
    q = query.unsqueeze(1).expand_as(r)
    return torch.cat((q, r, torch.abs(q - r), q * r), dim=-1)


def labels_for_pairs(rows: list[dict[str, Any]], rules: list[dict[str, Any]], dev_labels: dict[str, dict[str, Any]] | None = None) -> Any:
    import torch
    event_index = {row["event_id"]: i for i, row in enumerate(rules)}
    labels = []
    for row in rows:
        if dev_labels is None:
            event_id = row["event_id"]
            result_type = row["delta"]["result_type"]
            family_id = row["case_family_id"]
        else:
            family_id = row["case_family_id"]
            label = dev_labels[family_id]
            event_id, result_type = label["event_id"], label["result_type"]
        for rule in rules:
            if rule["event_id"] != event_id:
                labels.append(0)  # unrelated edit
            elif result_type == "unaffected":
                labels.append(1)  # correct edit, but locality says do not route
            else:
                labels.append(2)  # edit affects output or exception-based rationale
    return torch.tensor(labels, dtype=torch.long)


def macro_f1(y_true: Any, y_pred: Any, classes: int = 3) -> float:
    scores = []
    for label in range(classes):
        tp = int(((y_true == label) & (y_pred == label)).sum().item())
        fp = int(((y_true != label) & (y_pred == label)).sum().item())
        fn = int(((y_true == label) & (y_pred != label)).sum().item())
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return sum(scores) / classes


def route_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(a and b for a, b in zip(y_true, y_pred))
    fp = sum((not a) and b for a, b in zip(y_true, y_pred))
    tn = sum((not a) and (not b) for a, b in zip(y_true, y_pred))
    fn = sum(a and (not b) for a, b in zip(y_true, y_pred))
    tpr, tnr = tp / max(1, tp + fn), tn / max(1, tn + fp)
    return {"n": len(y_true), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": tpr, "specificity": tnr, "balanced_accuracy": (tpr + tnr) / 2,
            "precision": tp / max(1, tp + fp), "false_route_rate_on_unaffected": fp / max(1, fp + tn)}


def choose_route_threshold(scores: list[float], truth: list[bool]) -> tuple[float, dict[str, float]]:
    candidates = [i / 100 for i in range(5, 96, 5)]
    evaluated = [(threshold, route_metrics(truth, [score >= threshold for score in scores])) for threshold in candidates]
    threshold, metrics = max(evaluated, key=lambda item: (item[1]["balanced_accuracy"],
                                                           item[1]["specificity"], item[0]))
    return threshold, metrics


def save_gate_predictions(rows: list[dict[str, Any]], logits: Any, rule_events: list[str], threshold: float,
                          output_path: Path) -> None:
    import torch
    probabilities = torch.softmax(logits, dim=-1)[:, :, 2]
    scores, best = probabilities.max(dim=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row, score, index in zip(rows, scores.tolist(), best.tolist()):
            # event_id is copied only as evaluator metadata; it is not an input feature.
            out = {key: row[key] for key in ("id", "case_family_id", "event_id", "domain_id", "difficulty", "split") if key in row}
            out.update({"method": "SERAC-style", "route_to_editor": score >= threshold,
                        "scope_score": score, "selected_memory_event": rule_events[index],
                        "threshold": threshold})
            stream.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")


def make_rule_prompt(prompt: str, rule: dict[str, Any]) -> str:
    marker = "案例材料："
    if marker not in prompt:
        raise ValueError("Cannot insert retrieved rule: case marker missing")
    notice = rule.get("controlled_notice", rule.get("notice", ""))
    text = f"当前经检索匹配的新版上层规则：{rule['new_rule']}"
    if notice:
        text = f"{notice}\n{text}"
    return prompt.replace(marker, text + "\n" + marker, 1)


def make_case_prompt(prompt: str, exemplars: list[dict[str, Any]]) -> str:
    sections = ["以下是从训练领域检索到的相似案例及其训练目标答案。示例仅供参考，需根据待处理案例的事实重新判断。"]
    for i, row in enumerate(exemplars, 1):
        sections.append(f"示例{i}题面：\n{row['prompt']}\n示例{i}训练答案：\n{row['target']}")
    sections.append("待处理案例：\n" + prompt)
    return "\n\n".join(sections)


def generate_method(method: str, rows: list[dict[str, Any]], prompts: list[str], model: Any,
                    tokenizer: Any, output: Path, manifest_path: Path, source_hash: str,
                    model_path: Path, batch_size: int = 8, max_new_tokens: int = 1536) -> None:
    import torch
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate request IDs for {method}")
    config = {"protocol": "memory_baselines_v1_1_batch8_greedy_v1", "method": method,
              "retrieval_prompt_sha256": source_hash, "rows": len(rows), "batch_size": batch_size,
              "max_new_tokens": max_new_tokens, "decode": "greedy", "dtype": "bfloat16",
              "attention": "sdpa", "model_path": str(model_path), "status": "running"}
    if output.exists() or manifest_path.exists():
        if not output.is_file() or not manifest_path.is_file():
            raise FileExistsError(f"Incomplete output/manifest pair for {method}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in config.items() if key != "status"):
            raise ValueError(f"Existing run config differs for {method}")
        if previous.get("status") == "complete":
            print(json.dumps({"method": method, "status": "already_complete", "rows": previous.get("rows")}), flush=True)
            return
    else:
        write_json(manifest_path, config)
    existing = read_jsonl(output) if output.exists() else []
    done = {row.get("id") for row in existing}
    if len(done) != len(existing) or not done.issubset(set(ids)):
        raise ValueError(f"Partial output IDs invalid for {method}")
    pending_idx = [i for i, request_id in enumerate(ids) if request_id not in done]
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending_idx), batch_size):
            batch_idx = pending_idx[start:start + batch_size]
            batch_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompts[i]}],
                                 tokenize=False, add_generation_prompt=True) for i in batch_idx]
            encoded = tokenizer(batch_prompts, padding=True, truncation=False, return_tensors="pt").to("cuda")
            width = encoded.input_ids.shape[1]
            limit = getattr(model.config, "max_position_embeddings", None)
            if limit and width + max_new_tokens > limit:
                raise ValueError(f"{method} prompt+generation exceeds context: {width}+{max_new_tokens}>{limit}")
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                           use_cache=True)
            decoded = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            for idx, answer in zip(batch_idx, decoded):
                row = rows[idx]
                record = {key: row[key] for key in ("id", "request_id", "case_family_id", "family_id", "event_id", "domain_id", "difficulty", "split") if key in row}
                record.update({"method": method, "answer": answer.strip()})
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps({"method": method, "done": len(done) + min(start + len(batch_idx), len(pending_idx)),
                              "total": len(rows), "elapsed_sec": round(time.perf_counter() - started, 2)}, ensure_ascii=False), flush=True)
            del encoded, generated
    final = read_jsonl(output)
    if len(final) != len(rows) or {row.get("id") for row in final} != set(ids) or any(not x.get("answer") for x in final):
        raise RuntimeError(f"Incomplete output for {method}: {len(final)}/{len(rows)}")
    config.update({"status": "complete", "rows": len(final), "output_sha256": digest(output),
                   "completed_at_unix": time.time(), "elapsed_sec": round(time.perf_counter() - started, 3)})
    write_json(manifest_path, config)


def preflight_prompts(prompts: list[str], tokenizer: Any, model: Any, max_new_tokens: int,
                      batch_size: int = 64) -> dict[str, int]:
    """Check every prompt against the context window before generation starts."""
    limit = getattr(model.config, "max_position_embeddings", None)
    if not limit:
        raise ValueError("Model config does not expose max_position_embeddings")
    max_prompt = 0
    for start in range(0, len(prompts), batch_size):
        rendered = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
                    for prompt in prompts[start:start + batch_size]]
        encoded = tokenizer(rendered, padding=False, truncation=False, add_special_tokens=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        max_prompt = max(max_prompt, max(lengths, default=0))
        if max_prompt + max_new_tokens > limit:
            raise ValueError(f"Prompt preflight exceeds context: {max_prompt}+{max_new_tokens}>{limit}")
    return {"prompts": len(prompts), "max_prompt_tokens": max_prompt,
            "max_new_tokens": max_new_tokens, "context_limit": limit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--e5-model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("Use the frozen batch-8 protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = read_jsonl(args.data_dir / "rules.jsonl")
    train_all = read_jsonl(args.data_dir / "train_sft.jsonl")
    train = [row for row in train_all if row.get("kind") == "case"]
    case_memory = [row for row in train if row["difficulty"] == "medium"]
    dev = read_jsonl(args.data_dir / "dev_closed_book.jsonl")
    test = read_jsonl(args.data_dir / "test_closed_book.jsonl")
    dev_scope = {row["case_family_id"]: row for row in read_jsonl(args.data_dir / "dev_scope_labels.jsonl")}
    if (len(rules), len(train), len(case_memory), len(dev), len(test), len(dev_scope)) != (9, 1620, 540, 540, 5100, 180):
        raise ValueError("Unexpected input sizes for memory baselines")
    if any("当前更新后的上层规则：" in row["prompt"] for row in dev + test):
        raise ValueError("Closed-book dev/test prompt leaks an updated-rule marker")
    if any("target" in row or "delta" in row or "new_state" in row for row in dev + test):
        raise ValueError("Dev/test prompt input contains hidden labels")
    source_hash = hashlib.sha256("".join(digest(args.data_dir / name) for name in (
        "rules.jsonl", "train_sft.jsonl", "dev_closed_book.jsonl", "test_closed_book.jsonl", "dev_scope_labels.jsonl"
    )).encode()).hexdigest()

    import numpy as np
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")

    print("Loading E5 encoder", flush=True)
    e5_fingerprint = model_fingerprint(args.e5_model, "model.safetensors")
    qwen_fingerprint = model_fingerprint(args.model, "*.safetensors")
    run_manifest_path = args.output_dir / "run_manifest.json"
    run_manifest = {"protocol": "memory_baselines_v1_1_seed42",
                    "methods": ["SERAC-style", "Rule-RAG", "Case-RAG"],
                    "source_bundle_sha256": source_hash, "e5_model": e5_fingerprint,
                    "qwen_model": qwen_fingerprint, "seed": args.seed,
                    "retrieval": {"rule_top_k": 1, "case_top_k": 3,
                                  "case_memory": "one medium-difficulty case per train family; rule-QA rows excluded"},
                    "generation": {"batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
                                   "decoding": "greedy", "dtype": "bfloat16", "attention": "sdpa"},
                    "gate": {"architecture": "concat(query,rule,absdiff,product)-MLP-256", "classes": [
                        "unrelated_rule", "same_rule_invariant", "same_rule_affected"],
                        "exception_preserves_positive": True, "optimizer": "AdamW",
                        "learning_rate": 0.0003, "weight_decay": 0.01, "max_epochs": 30,
                        "batch_size": 256, "gradient_clip_norm": 1.0,
                        "checkpoint_selection": "development macro-F1"}, "status": "running"}
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in run_manifest.items() if key != "status"):
            raise ValueError("Existing memory-baseline run manifest differs")
    else:
        write_json(run_manifest_path, run_manifest)
    e5_tokenizer = AutoTokenizer.from_pretrained(str(args.e5_model), local_files_only=True)
    e5 = AutoModel.from_pretrained(str(args.e5_model), torch_dtype=torch.bfloat16,
                                   local_files_only=True, attn_implementation="sdpa").to(device).eval()
    rule_vectors = build_rule_embeddings(rules, e5_tokenizer, e5, device)
    rule_events = [row["event_id"] for row in rules]
    memory_vectors = build_passage_embeddings(case_memory, e5_tokenizer, e5, device)
    dev_rule_q = build_query_embeddings(dev, True, e5_tokenizer, e5, device)
    test_rule_q = build_query_embeddings(test, True, e5_tokenizer, e5, device)
    dev_case_q = build_query_embeddings(dev, False, e5_tokenizer, e5, device)
    test_case_q = build_query_embeddings(test, False, e5_tokenizer, e5, device)

    dev_rule_scores = dev_rule_q @ rule_vectors.T
    test_rule_scores = test_rule_q @ rule_vectors.T
    dev_rule_idx = dev_rule_scores.argmax(-1).tolist()
    test_rule_idx = test_rule_scores.argmax(-1).tolist()
    rule_recall = sum(rule_events[i] == row["event_id"] for i, row in zip(dev_rule_idx, dev)) / len(dev)
    dev_case_scores = dev_case_q @ memory_vectors.T
    test_case_scores = test_case_q @ memory_vectors.T
    dev_top3 = dev_case_scores.topk(3, dim=1).indices.tolist()
    test_top3 = test_case_scores.topk(3, dim=1).indices.tolist()
    case_event_recall_at_1 = sum(case_memory[i]["event_id"] == row["event_id"] for i, row in zip([x[0] for x in dev_top3], dev)) / len(dev)
    case_event_recall_at_3 = sum(any(case_memory[i]["event_id"] == row["event_id"] for i in indices) for indices, row in zip(dev_top3, dev)) / len(dev)

    # Train a 3-way scope gate: unrelated rule / same rule but invariant / edited scope.
    train_q = build_query_embeddings(train, True, e5_tokenizer, e5, device)
    x_train = pair_features(train_q, rule_vectors).reshape(-1, rule_vectors.shape[-1] * 4).to(device)
    y_train = labels_for_pairs(train, rules).to(device)
    x_dev = pair_features(dev_rule_q, rule_vectors).reshape(-1, rule_vectors.shape[-1] * 4).to(device)
    y_dev = labels_for_pairs(dev, rules, dev_scope).to(device)
    from torch import nn
    gate = nn.Sequential(nn.Linear(x_train.shape[-1], 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 3)).to(device)
    counts = torch.bincount(y_train, minlength=3).float()
    class_weights = (len(y_train) / (3 * counts.clamp_min(1))).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-4, weight_decay=0.01)
    best_state, best_macro = None, -1.0
    best_epoch = 0
    y_dev_cpu = y_dev.cpu()
    for epoch in range(1, 31):
        gate.train()
        order = torch.randperm(len(y_train), device=device)
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(gate(x_train[idx]), y_train[idx], weight=class_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
        gate.eval()
        with torch.inference_mode():
            dev_logits = gate(x_dev)
            dev_pred = dev_logits.argmax(-1).cpu()
        f1 = macro_f1(y_dev_cpu, dev_pred)
        if f1 > best_macro:
            best_macro, best_epoch = f1, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
        print(json.dumps({"gate_epoch": epoch, "dev_macro_f1": round(f1, 6), "best": round(best_macro, 6)}, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("Scope gate failed to produce a checkpoint")
    gate.load_state_dict(best_state)
    gate.eval()
    with torch.inference_mode():
        dev_logits = gate(x_dev).reshape(len(dev), len(rules), 3)
        dev_scores = torch.softmax(dev_logits, -1)[:, :, 2].max(dim=1).values.cpu().tolist()
        dev_truth = [dev_scope[row["case_family_id"]]["route_to_editor"] for row in dev]
    threshold, threshold_metrics = choose_route_threshold(dev_scores, dev_truth)
    test_query = build_query_embeddings(test, True, e5_tokenizer, e5, device)
    x_test = pair_features(test_query, rule_vectors).reshape(-1, rule_vectors.shape[-1] * 4).to(device)
    with torch.inference_mode():
        test_logits = gate(x_test).reshape(len(test), len(rules), 3).cpu()
    gate_model_path = args.output_dir / "serac_scope_gate.pt"
    torch.save({"state_dict": best_state, "input_dim": x_train.shape[-1], "hidden_dim": 256,
                "classes": ["unrelated_rule", "same_rule_invariant", "same_rule_affected"],
                "seed": args.seed, "best_epoch": best_epoch, "dev_macro_f1": best_macro,
                "threshold": threshold, "source_hash": source_hash}, gate_model_path)
    save_gate_predictions(test, test_logits, rule_events, threshold, args.output_dir / "SERAC-style-gate-test.jsonl")
    write_json(args.output_dir / "development_selection.json", {
        "protocol": "dev_only_selection_v1", "source_hash": source_hash,
        "rule_rag_recall_at_1": rule_recall,
        "case_rag_same_event_recall_at_1": case_event_recall_at_1,
        "case_rag_same_event_recall_at_3": case_event_recall_at_3,
        "serac_gate_best_epoch": best_epoch, "serac_gate_dev_macro_f1": best_macro,
        "serac_scope_threshold": threshold, "serac_route_metrics": threshold_metrics,
        "threshold_objective": "maximum dev balanced accuracy; tie-break specificity, then higher threshold",
        "test_gold_read": False,
    })

    # Persist per-query retrieval trace separately from model-visible prompts.
    trace_path = args.output_dir / "retrieval_trace_test.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for query_i, (row, rule_i, case_is) in enumerate(zip(test, test_rule_idx, test_top3)):
            trace = {key: row[key] for key in ("id", "case_family_id", "event_id", "domain_id", "difficulty", "split") if key in row}
            trace.update({"retrieved_rule_event": rule_events[rule_i],
                          "rule_score": float(test_rule_scores[query_i, rule_i].item()),
                          "retrieved_case_ids": [case_memory[i]["case_family_id"] for i in case_is],
                          "retrieved_case_scores": [float(test_case_scores[query_i, i].item()) for i in case_is]})
            stream.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")

    rule_prompts = [make_rule_prompt(row["prompt"], rules[index]) for row, index in zip(test, test_rule_idx)]
    case_prompts = [make_case_prompt(row["prompt"], [case_memory[i] for i in indices[:3]]) for row, indices in zip(test, test_top3)]
    prompt_dir = args.output_dir / "prompts"
    prompt_hashes = {}
    for method, rows, prompts in (("Rule-RAG", test, rule_prompts), ("Case-RAG", test, case_prompts)):
        prompt_file = prompt_dir / f"{method}.jsonl"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        with prompt_file.open("w", encoding="utf-8") as stream:
            for row, prompt in zip(rows, prompts):
                stream.write(json.dumps({"id": row["id"], "prompt": prompt}, ensure_ascii=False) + "\n")
        prompt_hash = digest(prompt_file)
        prompt_hashes[method] = prompt_hash

    # Release E5 and gate before loading the 7B generator.
    del e5, e5_tokenizer, gate, x_train, x_dev, x_test, train_q, dev_rule_q, test_rule_q, dev_case_q, test_case_q
    torch.cuda.empty_cache()
    del memory_vectors, rule_vectors
    torch.cuda.empty_cache()
    print("Loading Qwen generator", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.bfloat16,
                                                  local_files_only=True, attn_implementation="sdpa",
                                                  low_cpu_mem_usage=True).to("cuda").eval()
    prompt_preflight = {"Rule-RAG": preflight_prompts(rule_prompts, tokenizer, model, args.max_new_tokens),
                        "Case-RAG": preflight_prompts(case_prompts, tokenizer, model, args.max_new_tokens)}
    write_json(args.output_dir / "generation_preflight.json", prompt_preflight)
    for method, prompts in (("Rule-RAG", rule_prompts), ("Case-RAG", case_prompts)):
        output = args.output_dir / "predictions" / f"{method}.jsonl"
        manifest = output.with_suffix(output.suffix + ".run.json")
        generate_method(method, test, prompts, model, tokenizer, output, manifest, prompt_hashes[method],
                        args.model, args.batch_size, args.max_new_tokens)
    run_manifest.update({"status": "complete", "development_selection_sha256": digest(args.output_dir / "development_selection.json"),
                         "generation_preflight": prompt_preflight,
                         "gate_model_sha256": digest(gate_model_path),
                         "gate_predictions_sha256": digest(args.output_dir / "SERAC-style-gate-test.jsonl"),
                         "retrieval_trace_sha256": digest(trace_path),
                         "retrieval_prompt_sha256": prompt_hashes,
                         "prediction_outputs_sha256": {
                             method: digest(args.output_dir / "predictions" / f"{method}.jsonl")
                             for method in ("Rule-RAG", "Case-RAG")},
                         "completed_at_unix": time.time()})
    write_json(run_manifest_path, run_manifest)
    print(json.dumps({"status": "complete", "rule_rag_rows": len(test), "case_rag_rows": len(test),
                      "serac_gate_rows": len(test), "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
