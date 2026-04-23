# Experiment Result Index

This folder stores lightweight result summaries. Checkpoints and TensorBoard
logs are not tracked because they are generated artifacts.

## Main Baseline

- `swin_dino_mid_stable_baseline_10runs.md`

This is the most important document. It defines the current stable v1.0
baseline:

- c1: `GatedFusion`
- c2: `GatedFusion`
- c3: `DepthAwareLocalRefineFusion`
- c4: `GatedFusion + DepthPromptTokenBlock`
- decoder: `SimpleFPNDecoder`

Summary:

- 10 repeated runs
- mean validation mIoU: `0.4828`
- best validation mIoU: `0.4900`

## Other Ablations

- `swin_dino_mid_1.md`: first Swin-B + DINOv2-B mid-fusion result
- `swin_dino_mid_ktb_fdam_prompt_c4_1.md`: KTB/FDAM/prompt attempt
- `swin_dino_mid_prompt_c4_clean_ablation_1.md`: clean c4 prompt ablation
- `swin_dino_mid_c4_dformer_3runs_6results.md`: DFormer-style c4 prompt tests
- `swin_dino_mid_b2_mcads_decoder_summary.md`: decoder/head tests inspired by MCADS

## How To Read These Results

Use v1.0 as the main comparison target. A new module should be considered useful
only if it improves the average result across repeated runs, not just one lucky
checkpoint.
