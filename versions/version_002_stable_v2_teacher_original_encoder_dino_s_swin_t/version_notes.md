# v2.0 Stable Snapshot

Date: 2026-04-24

This version uses the teacher original code structure as the baseline.

## Main Structure

- RGB branch: DINOv2-small
- Depth branch: Swin-Tiny
- Fusion: GatedFusion on c1, c2, c3, c4
- Decoder: SimpleFPNDecoder

## Scope

- Only `src/models/encoder.py` is changed from the teacher original code.
- `train.py`, `eval.py`, `infer.py`, data loading, fusion, decoder, metrics, and visualization stay aligned with the teacher original code.
- No SA-Gate, BMP, DFormer-style c4, Prompt Token Block, MCADS decoder, or HHA input code is kept in this snapshot.

## Feature Extraction Note

- DINOv2-small outputs token features from selected hidden layers.
- Tokens are reshaped to feature maps and projected to `[96, 192, 384, 768]`.
- Swin-Tiny outputs four native hierarchical stages: `[96, 192, 384, 768]`.
- These paired stages are fused by the original GatedFusion blocks.

## Confirmed Run

- Experiment name: `teacher_original_only_encoder_dino_s_swin_t_run01`
- Best validation mIoU observed: `0.3361`
