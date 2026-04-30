# Model Changes

This file records architecture decisions for the current RGB-D semantic
segmentation project on NYUDepthV2.

## Current Main Line

- Date: 2026-04-29
- Task: RGB-D semantic segmentation
- Dataset: NYUDepthV2
- RGB branch: DINOv2-small
- Depth branch: Swin-Tiny
- Fusion direction: multi-level RGB-D mid-fusion
- Current improvement focus: Context-FPN, ResGamma, Depth Adapter, and fusion
  modules
- Current confirmed best result: Context-FPN ResGamma 7 runs, best mIoU about
  `0.3933`

## Module Tracking

| Area | Current status | Notes |
|---|---|---|
| RGB branch | Active | DINOv2-small is the current usable RGB branch. |
| Depth branch | Active | Swin-Tiny is the current usable depth branch. |
| Fusion | Active | Multi-level RGB-D mid-fusion is the current main direction. |
| Context-FPN | Under improvement | Track whether it improves multi-scale context without destabilizing training. |
| ResGamma | Under improvement | Track impact on repeated-run mIoU and stability. |
| Depth Adapter | Under improvement | Track whether depth features become more useful before fusion. |

## Deprecated / Invalid Attempts

| Attempt | Status | Reason | Usage rule |
|---|---|---|---|
| Swin-B RGB + DINOv2-B Depth | invalid / deprecated | This attempt appeared in the old README, but the pretrained models are too large for the current experiment environment. It cannot be treated as a valid current version. | Do not use it as the paper main line, an effective baseline, or the current best result. |

Any result tied to the deprecated Swin-B RGB + DINOv2-B Depth record must be
treated as invalid unless the user explicitly confirms a current-environment
run with matching configuration, logs, and checkpoints.
