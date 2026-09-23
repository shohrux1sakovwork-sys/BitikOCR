# Ablation Study for Qwen2.5-VL-7B OCR Fine-Tuning

Run a reproducible screening study on **1,600 training / 400 development samples**, then confirm promising configurations before training on the full 18K training split. Select by development CER, report WER, and preserve the existing 2K evaluation split for final assessment. This sequential search identifies promising configurations; it does not establish a global optimum.

## Current State

- Data: `data/train.jsonl` (18K), `data/eval.jsonl` (2K), and images under `data/images/`.
- Hardware: two NVIDIA L40 GPUs (48 GB each), with shared availability to be checked at execution time.
- Trainer: [bitikocr/train/train.py](bitikocr/train/train.py), supporting single-device LoRA training and generated-text CER/WER evaluation.
- Current adapter: `r=16`, `lora_alpha=32`, `target_modules=["q_proj", "v_proj"]`, implicit `lora_dropout=0.0`.
- Evaluation currently depends on the configured training schedule; there is no explicit final evaluation or dedicated metrics-file contract.

## Data Preparation

### [NEW] scripts/prepare_ablation_data.py

1. Sample and split **only from `data/train.jsonl`** using sampling seed 42. Save `data/ablation_train.jsonl` (1,600) and `data/ablation_eval.jsonl` (400).
2. Keep `data/eval.jsonl` out of hyperparameter selection. Inspect its identifiers and fingerprints only to verify disjointness; do not use its OCR scores during development.
3. Keep records sharing an ID, image path, or nonempty fingerprint in the same group. Reject overlap with the reserved evaluation set. If group constraints prevent the requested split sizes, report that explicitly instead of splitting duplicate groups.
4. Stratify by `document_type`, checking `script` coverage as well. Use deterministic allocation for small strata and report category counts and any missing categories.
5. Persist the selected record IDs, source-file hashes, split hashes, seed, and allocation policy in a manifest. All runs use the same manifest.
6. Verify every referenced image exists. Existing records use `images/...` paths, so set `image_folder="data"` and `eval_image_folder="data"`.

The development set remains fixed through screening and confirmation. After configuration selection is complete, the final model may train on all 18K training records, including these development records; final reporting then uses the untouched 2K evaluation split.

## Controlled Settings

Persist the complete resolved configuration for every run, including model revision, dependency versions, data manifest, and hardware information.

| Setting | Screening value / policy |
|---------|--------------------------|
| Training seed and data seed | 42 |
| Epochs | 1 (200 optimizer updates for 1,600 examples) |
| Train / eval batch size | 1 / 1 |
| Gradient accumulation | 8 |
| Precision | BF16 |
| Gradient checkpointing | Enabled, non-reentrant as supported by the trainer |
| Optimizer | `adamw_torch` |
| Scheduler / warmup ratio | Linear / 0.0 |
| Weight decay | 0.0 |
| Maximum gradient norm | 1.0 |
| LoRA dropout | 0.0, preserving the current baseline |
| LoRA scaling | Standard LoRA with `alpha=2*r`; no rank-stabilized scaling |
| Image bounds | Initial `image_min_pixels=3136`, `image_max_pixels=1003520`; no forced width/height |
| Generation | Greedy, one beam, initial `max_new_tokens=1024` |
| Evaluation | Explicit final evaluation; `eval_strategy="no"` during screening |
| Checkpoint policy | Save final checkpoint; no intermediate checkpoint selection |

Image resolution and generation length are provisional until the resource pilot. Check development reference token lengths and record generation-limit hits. If limits need changing, freeze the revised settings before screening and rerun the baseline under those settings. Never silently reduce resolution or generation length for a single candidate.

Use the existing corpus CER/WER computation consistently. Record its normalization behavior and save per-example predictions and references for inspection, including metrics by document type and script where sample counts permit.

## Ablation Dimensions

### Target Modules

| Config | Adaptation scope |
|--------|------------------|
| A (current) | Language decoder `q_proj`, `v_proj` |
| B | Language decoder `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| C | Language decoder attention and MLP: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| D | Same decoder adapters as B, plus full training of the visual backbone and visual merger |

C means the seven named decoder projection types, not every linear layer in the multimodal model. Resolve and validate the decoder scope against actual model module names, excluding the visual subtree and language output head. Log all matched module names and measured trainable parameter counts; fail on empty or unexpected matches.

D is conditional on a successful resource pilot. Record the exact visual module subtree included in its checkpoint. It compares joint visual training with a frozen visual encoder; it is not a pure decoder-target comparison.

### Rank and Learning Rate

- Ranks: `8`, `16`, `32`, `64`, with alpha `16`, `32`, `64`, `128` respectively. Keeping `alpha/r=2` avoids also changing the explicit adapter scaling factor during the rank sweep.
- Learning rates: `1e-5`, `2e-5`, `5e-5`, `1e-4`.
- Use one learning rate for all trainable parameters, including D's vision parameters. Interpret D's outcome as evidence for this specific training recipe, not proof that vision adaptation is unnecessary.

## Experiment Schedule

A phase is a search stage containing separate runs, not continued training of one model. Each new run starts from the original pretrained weights. Later phases inherit selected configuration values only. Exact reused results do not launch another run.

### Phase 0 — Resource Pilot and Untuned Baseline

- Default to `CUDA_VISIBLE_DEVICES=1`, with a CLI override. Check current free memory before launching; the trainer requires exactly one visible GPU.
- Exercise forward, backward, an optimizer step, checkpoint save/reload, and generation evaluation on representative examples, including the largest processed images and longest sequences.
- Pilot the largest decoder adapter configuration and D separately. Measure peak allocated/reserved memory, wall time, trainable parameters, and checkpoint size. Do not assume D fits based on total GPU capacity.
- If D cannot fit at the common settings, mark it skipped with the reason and continue A–C. Any common-setting revision requires a new pilot and baseline.
- Evaluate the unmodified pretrained model on all 400 development examples, using the same processing and generation settings as trained candidates.
- Estimate screening and confirmation costs from measured training and generation throughput. No fixed 4–8 hour promise before this measurement.

### Phase 1 — Target Screening

Run A, B, C, and, if feasible, D with `r=16`, `alpha=32`, `lr=2e-5`, and the common screening settings. Select the lowest finite development CER; break exact ties by WER, then fewer trainable parameters, then stable configuration name.

### Phase 2 — Rank Screening

Use the winning target configuration and `lr=2e-5`. Compare ranks 8, 16, 32, and 64 with `alpha=2*r`. Reuse the Phase 1 winner for rank 16 when the full configuration, seed, code/model versions, and data manifest match. This adds three training runs.

### Phase 3 — Learning-Rate Screening

Use the winning targets and rank. Compare the four learning rates. Reuse the matching Phase 2 result at `2e-5`; this adds three training runs.

There are **10 unique screening training runs if D fits, or 9 if D is skipped**, plus the untuned evaluation and resource pilots. Every independent training run starts from the same pretrained weights, not the previous phase's trained checkpoint.

### Phase 4 — Confirmation

- Retain the two best distinct configurations observed across screening, plus A with `r=16`, `alpha=32`, `lr=2e-5` if it is not already included.
- Also evaluate the runner-up Phase 1 target configuration with the selected rank and learning rate if that combination is new. This is a limited check for target/rank/LR interactions, not a full interaction study.
- Train each confirmation configuration for three epochs with seeds 42, 43, and 44, keeping the same 1,600/400 split and other controlled settings. Each run starts from pretrained weights and evaluates its final checkpoint. Budget 6–12 confirmation runs after deduplication.
- Compare mean development CER, its standard deviation across seeds, WER, untuned performance, and compute cost. If candidates remain indistinguishable relative to seed variation, report the uncertainty and prefer the smaller configuration instead of claiming a decisive winner.
- If learning curves or seed variability leave selection inconclusive, extend development training or enlarge the training subset from the remaining 18K source records while preserving the fixed development set and duplicate-group boundaries. Reconfirm all finalists under the same revised budget before full training.
- Freeze the configuration and full-training duration policy before consulting the reserved 2K evaluation scores. Pilot the selected configuration on full-training sequence sizes before committing to the full run.

## Project Configuration

[ablation.yaml](ablation.yaml) is the source of truth for study settings, the W&B project name, and run-name templates. All pilots, baseline, screening, confirmation, and eventual full-training runs use the single W&B project **`isakovsh/BitikOCR`**. Group by study ID and tag each run with its phase.

The runner reads this YAML, resolves phase-dependent values, translates settings into trainer arguments, and saves the fully resolved configuration as `resolved_config.yaml` in each run directory and in W&B's run config. Use unique run IDs and expand the configured names with phase, target, rank, alpha, learning rate, seed, and attempt. Reused results retain their original run identity, with a reference in the later phase summary. Confirmation uses each listed seed for training and data order while preserving the fixed split.

## Proposed Implementation Changes

### [MODIFY] bitikocr/params.py

Add `LoraArguments` with:

- `lora_r: int = 16`
- `lora_alpha: int = 32` (the runner explicitly supplies `2*r`)
- `lora_target_modules: str = "q_proj,v_proj"` (comma-separated decoder projection names)
- `lora_dropout: float = 0.0`
- `freeze_vision_encoder: bool = True`

Validate positive rank/alpha, dropout range, and supported nonempty target names. Preserve existing defaults for ordinary training. Provide an explicit evaluation-only mode for the untuned baseline and a final-evaluation option for ablation runs.

### [MODIFY] bitikocr/train/train.py

- Parse the new arguments from CLI and JSON and resolve decoder-only adapter targets.
- For frozen vision runs, assert the visual subtree has no trainable parameters after PEFT wrapping.
- For D, configure the actual visual subtree through PEFT `modules_to_save` so it is both trainable and included in saved adapters. Do not rely on unfreezing before `get_peft_model`, which freezes non-adapter parameters. Verify this path with the installed PEFT/model versions before launching D.
- Assert the intended parameters are trainable after wrapping and included in optimizer groups. Verify nonzero visual gradients and an actual visual weight update in the D pilot.
- Support a fresh-process reload of D's adapter and saved visual weights onto the original base model; compare restored weights and deterministic predictions with the pre-save model.
- Explicitly call `evaluate()` after each ablation training run and persist its returned metrics with `save_metrics("eval", metrics)` as `eval_results.json`. Do not rely on W&B or stdout parsing.
- Persist per-example predictions, references, record IDs, and generation-limit indicators alongside metrics. Mark a run successful only after evaluation and all required artifacts are saved.
- Support evaluation-only baseline execution without training or attaching adapters.

PEFT references: [LoRA configuration](https://huggingface.co/docs/peft/main/package_reference/lora) and [additional trainable modules](https://huggingface.co/docs/peft/main/conceptual_guides/lora). Validate behavior against the installed version.

### [NEW] scripts/run_ablation.py

1. Load and validate `ablation.yaml`, build phase-dependent configurations, and launch subprocesses using the current Python interpreter and `-m bitikocr.train.train`, with the repository root as working directory.
2. Set single-GPU visibility and explicitly pass all controlled settings, including image roots and evaluation options.
3. Save resolved configuration, configuration hash, command, versions, data manifest, stdout/stderr, exit status, timing, and resource measurements under `outputs/ablation/<study-id>/<run-name>/`.
4. Read `eval_results.json`; require successful process completion, expected sample count, and finite CER/WER. Error rates above 1 are valid and must not be rejected.
5. Mark OOM, failed, skipped, and incomplete runs explicitly. Exclude them from ranking. Stop a dependent phase if no valid candidate remains, with an actionable error.
6. Reuse only completed results with matching provenance. Preserve existing outputs on restart; retries use separate attempt directories so stale metrics cannot be mistaken for success.
7. Print and save JSON/CSV summaries with phase, configuration, seed, status, CER/WER, baseline difference, parameter count, time, and peak memory.
8. Use W&B project `BitikOCR` from `ablation.yaml` for every run, including full training, with group and run names resolved from its templates. Log the complete resolved configuration. Online reporting is the default; explicit offline mode must preserve project and run identities for later sync. Local artifacts remain authoritative.

### [MODIFY] Makefile

Include `scripts/` in formatting, lint, and type checks once the scripts exist. Existing targets currently cover only `bitikocr` and, for some checks, `tests`.

## Verification Plan

### Offline Tests and Checks

- Run the existing training, metric, dataset, and data-utility tests.
- Test deterministic split selection, duplicate grouping, reserved-set disjointness, rare strata, and correct image resolution.
- Test CLI/JSON parsing, invalid LoRA arguments, decoder-only module matching, and preserved defaults.
- Use a small model fixture to test frozen/trainable vision behavior, optimizer membership, weight updates, and checkpoint round-trip behavior. Cover the real Qwen model path in the GPU pilot.
- Test explicit final evaluation and baseline-only execution, including metrics/prediction artifact schema and evaluated sample counts.
- Validate YAML loading, numeric types, run-name expansion, the shared W&B project, and saved resolved configurations.
- Test runner phase selection, deduplication, provenance matching, ties, subprocess failures, missing/nonfinite metrics, and restart behavior without training real models.
- Run `make checks` and `make test` after implementation.

### Execution Acceptance Criteria

- The data manifest proves disjointness from the reserved evaluation split.
- Each successful run has a complete configuration, checkpoint (except the untuned baseline), metrics, predictions, and resource report.
- D is eligible for comparison only after its gradient/update and fresh-process checkpoint reload checks pass.
- Screening results are labeled provisional; confirmation results include variation across seeds and the untuned baseline.
- Final full-training evaluation uses the reserved 2K split only after development decisions are frozen.
