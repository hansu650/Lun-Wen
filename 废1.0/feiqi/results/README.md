# Experiment Result Index

This folder stores lightweight result summaries. Checkpoints and TensorBoard
logs are not tracked because they are generated artifacts.

## Result Archives

- [`v0.0-1.0/`](./v0.0-1.0/): experiments from the earliest encoder/fusion
  attempts through the confirmed v1.0 stable baseline.
- [`checkpoint_cleanup_summary.md`](./checkpoint_cleanup_summary.md): record of
  the removed local checkpoint archive and the result files kept as the source
  of truth.

## Current Main Baseline

The current main reference is documented in:

- [`v0.0-1.0/swin_dino_mid_stable_baseline_10runs.md`](./v0.0-1.0/swin_dino_mid_stable_baseline_10runs.md)

This file defines the stable v1.0 baseline:

- encoder: RGB `Swin-B` + depth `DINOv2-B`
- c1: `GatedFusion`
- c2: `GatedFusion`
- c3: `DepthAwareLocalRefineFusion`
- c4: `GatedFusion + DepthPromptTokenBlock`
- decoder: `SimpleFPNDecoder`

Summary:

- 10 repeated runs
- mean validation mIoU: `0.4828`
- best validation mIoU: `0.4900`
- standard deviation: `0.0044`

## v0.0-1.0 Contents

- [`swin_dino_mid_1.md`](./v0.0-1.0/swin_dino_mid_1.md): first
  Swin-B + DINOv2-B mid-fusion result.
- [`swin_dino_mid_ktb_fdam_prompt_c4_1.md`](./v0.0-1.0/swin_dino_mid_ktb_fdam_prompt_c4_1.md):
  KTB/FDAM/prompt attempt.
- [`swin_dino_mid_prompt_c4_clean_ablation_1.md`](./v0.0-1.0/swin_dino_mid_prompt_c4_clean_ablation_1.md):
  clean c4 prompt ablation.
- [`swin_dino_mid_c4_dformer_3runs_6results.md`](./v0.0-1.0/swin_dino_mid_c4_dformer_3runs_6results.md):
  DFormer-style c4 prompt tests.
- [`swin_dino_mid_b2_mcads_decoder_summary.md`](./v0.0-1.0/swin_dino_mid_b2_mcads_decoder_summary.md):
  decoder/head tests inspired by MCADS.
- [`swin_dino_mid_stable_baseline_10runs.md`](./v0.0-1.0/swin_dino_mid_stable_baseline_10runs.md):
  final v1.0 stability check across 10 repeated runs.

## How To Read These Results

Use v1.0 as the main comparison target. A new module should be considered useful
only if it improves the average result across repeated runs, not just one lucky
checkpoint.

Future result folders should follow the same archive style, for example
`v1.1-*`, `v1.2-*`, or another clear version range when several related
experiments are grouped together.
