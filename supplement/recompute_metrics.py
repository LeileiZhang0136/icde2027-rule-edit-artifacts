"""Recompute the three headline metrics from frozen item-level judgments."""

import csv
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "handling", "obligations_added", "obligations_removed",
    "obligations_unchanged", "human_review", "reasoning",
    "internal_consistency",
]


def main() -> None:
    path = Path(__file__).resolve().parent / "scores/item_scores.csv"
    grouped = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["method"]].append(row)
    print("method,n,strict_pct,decision_pct,decision_gated_pct,uncertain_pct")
    for method, rows in grouped.items():
        n = len(rows)
        strict = sum(row["strict_status"] == "pass" for row in rows) / n
        decision = sum(row["decision"] == "satisfied" for row in rows) / n
        gated = sum(
            sum(row[field] == "satisfied" for field in FIELDS) / 7
            if row["decision"] == "satisfied" else 0
            for row in rows
        ) / n
        uncertain = sum(row["strict_status"] == "uncertain" for row in rows) / n
        print(f"{method},{n},{strict*100:.2f},{decision*100:.2f},{gated*100:.2f},{uncertain*100:.2f}")


if __name__ == "__main__":
    main()
