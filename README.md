# doScenes Language-Guided Trajectory Prediction

A compact training and evaluation toolkit for the doScenes Instructed Driving Challenge. The project focuses on the **Language + History** setting: given a short ego-vehicle trajectory history and a natural-language driving instruction, the model predicts the future ego trajectory for 12 time steps.

## Overview

This repository contains an experimental pipeline for instruction-conditioned trajectory prediction:

- text encoding with DistilBERT-style language features;
- history encoding with recurrent trajectory features;
- multimodal fusion for language-conditioned motion prediction;
- dual-head evaluation for instruction and history-only baselines;
- CSV export and precheck utilities for challenge submissions.

The toolkit is designed for reproducible experiments around ADE/FDE-style trajectory metrics and challenge-compatible submission files.

## Task

The challenge input consists of:

- an ego-vehicle history trajectory;
- a natural-language instruction;
- metadata identifiers used for submission alignment.

The expected output is a future trajectory with 12 `(x, y)` points in ego coordinates. Submission files follow the format:

```text
sample_token,instruction,x1,y1,...,x12,y12
```

## Repository Structure

```text
doScenes/
  doScenes-main/
    configs/                 # experiment configurations
    src/doscenes/            # package source code
      data/                  # dataset loading and preprocessing
      evaluation/            # metrics, diagnostics, submission checks
      models/                # trajectory-language model definitions
      training/              # training loop and checkpoint utilities
    scripts/                 # helper scripts for profiling and plotting
    tests/                   # unit tests
    artifacts/               # experiment outputs and submissions
    release/                 # challenge submission snapshots
  models/                    # local model assets
```

Large datasets and checkpoints are intentionally not documented as source artifacts. Keep local data paths outside public documentation when publishing.

## Installation

Python 3.10 or newer is recommended.

```bash
cd doScenes/doScenes-main
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
cd doScenes\doScenes-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Data Setup

Prepare the doScenes/nuScenes-style metadata and update `paths.txt` or the configured path file used by your experiment config. The default config is:

```text
configs/default.json
```

The code expects the data path configuration to point to local dataset metadata. Do not commit private local paths, credentials, or raw personal files.

## Training

Run training with a config file:

```bash
python -m doscenes train --config configs/default.json
```

Example with a tuned config:

```bash
python -m doscenes train --config configs/ade_query_residual.json
```

## Evaluation

Evaluate an instruction-conditioned checkpoint:

```bash
python -m doscenes eval \
  --config configs/ade_query_residual.json \
  --checkpoint artifacts/checkpoints/best.pt \
  --output artifacts/experiments/eval_report.json
```

Evaluate a history-only baseline by ignoring text:

```bash
python -m doscenes eval \
  --config configs/ade_query_residual.json \
  --checkpoint artifacts/checkpoints/best.pt \
  --ignore-text \
  --output artifacts/experiments/eval_baseline_report.json
```

Compare instruction and baseline metrics:

```bash
python -m doscenes compare \
  --baseline artifacts/experiments/eval_baseline_report.json \
  --instruction artifacts/experiments/eval_report.json \
  --output artifacts/experiments/delta_report.json
```

## Language Effect Report

Generate a report comparing instruction and baseline predictions:

```bash
python -m doscenes language-report \
  --config configs/ade_query_residual.json \
  --checkpoint artifacts/checkpoints/best.pt \
  --output-json artifacts/experiments/language_effect_report.json \
  --output-csv artifacts/experiments/language_effect_samples.csv
```

## Submission Export

Export a challenge-style CSV:

```bash
python -m doscenes export-submission \
  --config configs/ade_query_residual.json \
  --checkpoint artifacts/checkpoints/best.pt \
  --output artifacts/submissions/submission.csv
```

For official scene-map based export:

```bash
python -m doscenes export-submission \
  --config configs/ade_query_residual.json \
  --checkpoint artifacts/checkpoints/best.pt \
  --official-scene-map-csv artifacts/submissions/official_scene_instruction_map_127.csv \
  --expected-scenes 127 \
  --output artifacts/submissions/submission.csv
```

Precheck a submission file:

```bash
python -m doscenes precheck-submission \
  --submission artifacts/submissions/submission.csv \
  --expected-steps 12 \
  --output artifacts/experiments/submission_precheck_report.json
```

## Results

Representative local validation metrics from the language-effect report:

| Metric | Instruction | Baseline | Delta |
| --- | ---: | ---: | ---: |
| ADE | 3.4355 | 3.9987 | 0.5631 |
| FDE | 8.2893 | 8.6488 | 0.3595 |

The final submission precheck used 398 rows and 26 columns, with no duplicate rows or invalid numeric cells.

## Notes

- Keep raw datasets, checkpoints, personal documents, generated reports, and private paths out of Git.
- Use `README.md` as the only Markdown file intended for publication.
- Use `--force-with-lease` instead of plain `--force` when overwriting a remote branch.

## License

No license is declared in this repository. Add one before redistributing or reusing the project outside its intended experimental context.
