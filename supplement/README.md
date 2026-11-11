# ICDE 2027 review artifact

This directory contains the unpacked reproducibility artifact for **High-Level Rule Editing in LLMs with Incomplete Downstream Coverage**. The same files are available in the [downloadable ZIP](../supplement_release.zip).

## Contents

- `data/`: frozen v1.1 executable-rule dataset, prompts, calibration cases, and validation report.
- `code/`: construction, training, inference, and scoring scripts.
- `predictions/`: test answers for eight main systems and five ablations.
- `scores/item_scores.csv`: frozen item-level judgments for the eight evaluated fields.
- `manifest.json`: counts and SHA-256 checksums of packaged files.

## Recompute metrics

Run `python recompute_metrics.py` from this directory. It reports strict complete-answer correctness, final-decision accuracy, decision-gated score, and uncertain rate from the frozen judgments.
