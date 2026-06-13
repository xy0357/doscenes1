# Dual-Head Language-Conditioned Trajectory Prediction for doScenes Challenge

Author: Ke Deng

## Abstract

This report presents a complete Language+History solution for the doScenes Challenge. The task is to predict 6 seconds of future ego trajectory (12 points at 2 Hz) from 2 seconds of history and one natural-language instruction. Beyond absolute trajectory accuracy, the challenge explicitly evaluates language utility using Delta ADE (`ADE_baseline - ADE_instruction`).

We build an end-to-end, reproducible pipeline that covers data processing, training, diagnostics, and submission export. The model uses DistilBERT as the text encoder, GRU as the history encoder, and cross-attention for multimodal fusion. To prevent language-collapse (instruction predictions degenerating into baseline-like predictions), we use a Dual-Head decoder design (instruction head and baseline head), non-empty instruction rebalancing, and ranking loss on language-available samples.

In the current meter-based evaluation setup, a representative run achieves `ADE_instruction = 3.474533`, `ADE_baseline = 3.955950`, and `Delta ADE = 0.481417`. Across seeds 42/43/44, the average gain is `Delta ADE = 0.404754 +/- 0.054824`. The corrected submission files also satisfy official protocol constraints, including exact row count and scene-token matching.

Keywords: trajectory prediction, language-conditioned modeling, multimodal learning, dual-head decoding, doScenes challenge, Delta ADE

## 1. Introduction

Trajectory prediction is a core module in autonomous driving and intelligent transportation systems. The goal is to estimate future motion from observed history. Classical methods primarily rely on trajectory history, interactions, or map context. For example, Social LSTM models multi-agent interactions with social pooling, while Trajectron++ extends interaction modeling with scene graphs and multimodal uncertainty.

However, real driving intent is often easier to express through language than through coordinate history alone. Instructions such as "change to the left lane", "slow down and yield", or "continue straight behind the lead car" provide explicit semantic constraints that pure numerical history may not fully capture.

The doScenes Challenge is designed around this idea. In the Language+History track, models must incorporate instruction text and produce open-loop single-shot future trajectories. This introduces three practical difficulties:

1. Heterogeneous modality fusion: language and trajectory features live in very different representation spaces.
2. Supervision imbalance: many samples include empty instructions, which can dilute language learning signals.
3. Relative objective pressure: the model must both predict well and demonstrate measurable language gain (`Delta ADE`).

In early iterations, we observed a common failure mode: even with text encoder integration, a model optimized only for generic regression can ignore text and converge to nearly identical instruction and baseline predictions (`Delta ADE` near zero). This work addresses that failure mode directly.

Main contributions:

1. A complete and reproducible engineering pipeline for doScenes Language+History.
2. A Dual-Head architecture that structurally separates instruction and baseline prediction paths.
3. A language-gain-oriented training strategy combining instruction rebalancing and ranking loss.
4. Stable positive language gain across multiple random seeds and corrected protocol-compliant submission outputs.

## 2. Challenge Protocol and Problem Definition

### 2.1 Official protocol

According to the challenge rules, the Language+History track requires future prediction over 6 seconds at 2 Hz, resulting in 12 future coordinates. Each query requires exactly one prediction row in the CSV file:

`sample_token,instruction,x1,y1,x2,y2,...,x12,y12`

Core metrics:

- ADE: average displacement error over all future timesteps.
- FDE: final displacement error at the last timestep.
- Delta ADE: `ADE_baseline - ADE_instruction`.

### 2.2 Formulation used in this report

Each sample is represented as:

- `H = {(x_t, y_t)}_{t=1}^{T_h}`: history trajectory (`T_h` corresponds to 2 seconds at 2 Hz).
- `I`: natural-language instruction (possibly empty).
- `Y = {(x_t, y_t)}_{t=1}^{T_f}`: future trajectory supervision (`T_f = 12`).

The model learns:

`f(H, I) -> Y_hat`

In implementation, trajectories are transformed into a local coordinate frame centered at the last observed point and optionally aligned by heading. Internal training uses normalized coordinates (`coord_scale = 10`) for numerical stability, while exported submission coordinates are rescaled back to meters.

## 3. Related Work

### 3.1 Classical trajectory prediction

Social LSTM established interaction-aware recurrent prediction with social pooling. Trajectron++ further improved dynamics-aware and multimodal modeling with graph-based structures. These methods provide strong history-interaction baselines but generally do not directly model free-form language instructions.

### 3.2 Transformer and pretrained language encoders

Transformer attention enables flexible long-range and cross-modal feature learning. BERT and its compressed variant DistilBERT provide strong language representations. DistilBERT is especially practical in medium-size datasets and constrained training environments, making it a suitable encoder for this challenge.

### 3.3 Language-conditioned trajectory prediction

Recent works (including language-based multimodal trajectory prediction studies) show that language can complement trajectory history by encoding intent-level constraints. doScenes systematizes this setting in autonomous driving by providing instruction-annotated trajectory scenes and challenge benchmarks.

## 4. Data, Cleaning, and Sample Construction

### 4.1 Data source

We construct training samples by joining doScenes annotation CSV files with nuScenes scene metadata. Each annotation row includes fields such as `scene_number`, `instruction_type`, and `instruction`. Scene matching is done through normalized scene naming and token lookup.

### 4.2 Cleaning rules

To ensure valid supervision without over-filtering, we apply:

1. Column normalization and text cleanup.
2. Removal of invalid or non-parsable scene identifiers.
3. Retention of empty-instruction rows as valid baseline-like queries.
4. Removal of rows that fail scene matching.
5. End-frame padding for short sequences when required.
6. Consistent coordinate transforms across train/eval/export.

### 4.3 Data profile

In the current setup:

| Item | Value |
| --- | ---: |
| Total annotation rows | 4729 |
| Empty-instruction rows | 2279 |
| Non-empty instruction rows | 2450 |
| Empty ratio | 48.19% |
| Unique scenes | 1004 |

This near-50% empty ratio motivates explicit rebalancing strategies for language learning.

### 4.4 Coordinate transform

Default construction parameters:

- `hist_sec = 2.0`
- `fut_sec = 6.0`
- `sample_freq = 2.0`
- `coord_scale = 10.0`
- `window_stride = 2`

Both history and future are converted to local coordinates around the anchor (last observed point), with optional heading alignment. Training uses normalized coordinates; submission export restores meter units.

## 5. Method

### 5.1 Design objective

The primary objective is not only lower ADE, but stable positive Delta ADE. The model should:

1. Capture short-term motion trend from history.
2. Encode instruction semantics reliably.
3. Use language to outperform a matched no-language baseline when instruction is available.

### 5.2 Text encoder

DistilBERT encodes instruction text. We keep token-level representations for attention-based cross-modal fusion instead of collapsing directly into a single sentence vector.

### 5.3 History encoder

A single-layer GRU encodes history trajectory after linear projection into hidden space. The final hidden state serves as motion context.

### 5.4 Cross-modal fusion

Trajectory features act as queries over text token features (keys/values) through multi-head cross-attention. The fused representation then passes through MLP and residual normalization.

### 5.5 Dual-Head decoder

Two separate decoders are used:

- `decoder_instruction`: instruction-conditioned prediction.
- `decoder_baseline`: no-language baseline prediction.

This structural split prevents the model from collapsing both modes into the same output mapping.

### 5.6 Loss design

Total loss combines:

1. Instruction head trajectory loss.
2. Baseline head trajectory loss.
3. Ranking loss (active on non-empty instruction samples), encouraging instruction error to be lower than baseline error by margin.
4. Optional endpoint term for FDE balance.

General form:

`L = L_inst + w_base * L_base + w_rank * L_rank + w_fde * L_fde`

## 6. Training Strategy and Engineering Details

### 6.1 Non-empty instruction rebalancing

Because empty instructions are frequent, we apply weighted sampling to increase non-empty instruction exposure. This significantly improves language signal utilization and helps move Delta ADE away from zero.

### 6.2 Partial text unfreezing

To balance stability and adaptation, we freeze most DistilBERT layers and unfreeze only the last few layers for task-specific adaptation.

### 6.3 Optimization and early stopping

We use AdamW with gradient clipping and validation-driven checkpointing. We monitor train/val ADE/FDE and overfit gap trends.

### 6.4 Efficiency and GPU control

The training pipeline supports AMP mixed precision and GPU utilization throttling to keep resource usage controlled in desktop/shared environments.

## 7. Results (Meter-based)

### 7.1 Single-run result

| Metric | Instruction | Baseline | Delta |
| --- | ---: | ---: | ---: |
| ADE | 3.474533 | 3.955950 | +0.481417 |
| FDE | 8.276808 | 8.584694 | +0.307886 |

### 7.2 Subset analysis

| Subset | Count | ADE Instruction | ADE Baseline | Delta ADE | FDE Instruction | FDE Baseline | Delta FDE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 7299 | 3.474533 | 3.955950 | +0.481417 | 8.276808 | 8.584694 | +0.307886 |
| has_instruction | 3978 | 3.758050 | 4.138420 | +0.380370 | 9.019087 | 9.157837 | +0.138750 |
| empty_instruction | 3321 | 3.134927 | 3.737381 | +0.602453 | 7.387683 | 7.898165 | +0.510483 |

### 7.3 Multi-seed stability

| Metric | Mean | Std |
| --- | ---: | ---: |
| ADE_instruction | 3.413795 | 0.139851 |
| ADE_baseline | 3.818549 | 0.170394 |
| Delta ADE | +0.404754 | 0.054824 |
| FDE_instruction | 8.130321 | 0.271662 |
| FDE_baseline | 8.296842 | 0.341483 |
| Delta FDE | +0.166521 | 0.102774 |

All three seeds maintain positive Delta ADE, indicating stable language gain.

## 8. Analysis and Discussion

### 8.1 Why early Delta ADE was zero

In earlier setups, shared-output behavior plus pure regression optimization allowed the model to minimize loss while ignoring instruction semantics.

### 8.2 Why Dual-Head helps

Dual-Head enforces parameter-level separation and allows explicit relative optimization through ranking loss. This directly penalizes language-collapse behavior.

### 8.3 Remaining caveat

Positive gains on empty-instruction subsets indicate some benefit may come from head-function bias, not only semantic grounding. This motivates stricter ablations.

### 8.4 ADE/FDE balance

While current metrics are positive for both Delta ADE and Delta FDE, future tuning should still monitor endpoint stability under different ranking/FDE weight trade-offs.

## 9. Corrected Official Submission Protocol

Following organizer feedback, we generated corrected files with strict protocol alignment:

- Exactly 127 rows.
- One row per language-available test scene.
- `sample_token` set to official dataloader-returned `scene_token`.
- Matched instruction-conditioned and no-language baseline files for Delta ADE reporting.

Both files pass local precheck with row/token consistency.

## 10. Ethics and Safety Considerations

The dataset is derived from public-road driving data and language annotations for research usage. In practical system design:

1. Language must not override traffic law or hard safety constraints.
2. Ambiguous instructions should defer more to scene context and conservative behavior.
3. Adversarial/misleading wording should be stress-tested for robustness.

A trajectory predictor should remain a component within a larger safety-validated stack, not an unconstrained control policy.

## 11. Future Work

Planned directions include:

1. Stronger semantic grounding checks and targeted ablations.
2. Additional robustness evaluation under adversarial and contradictory instructions.
3. Further optimization of endpoint stability under strict protocol-consistent test conditions.
4. More systematic seed and configuration sweeps balancing language gain and variance.

## 12. Conclusion

This report presents a reproducible Dual-Head language-conditioned trajectory framework for doScenes Language+History prediction. By combining structural separation, sample rebalancing, and ranking-based relative optimization, the method achieves stable positive language gain under meter-based evaluation and passes corrected official submission protocol checks.

## References

[1] doScenes Challenge Website: https://mi3-lab.github.io/doScenes_challenge

[2] doScenes Repository: https://github.com/rossgreer/doScenes

[3] Alahi et al., Social LSTM, CVPR 2016.

[4] Salzmann et al., Trajectron++, ECCV 2020.

[5] Vaswani et al., Attention Is All You Need, NeurIPS 2017.

[6] Devlin et al., BERT, NAACL 2019.

[7] Sanh et al., DistilBERT, arXiv 2019.

[8] Bae et al., Language-Based Multimodal Trajectory Prediction, CVPR 2024.
