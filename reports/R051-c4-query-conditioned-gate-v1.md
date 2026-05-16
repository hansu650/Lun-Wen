# R051 C4 Query-Conditioned Gate

## Hypothesis

CAFuser motivates condition-aware multimodal fusion, and R050 showed that deleting c4 external depth fusion is harmful. R051 tests whether a compact scene query from DFormerv2 c4 can condition the existing c4 GatedFusion gate logits while preserving the original c4 mixture and refine path.

## Implementation

- Added independent model entry `dformerv2_c4_query_conditioned_gate`.
- Replaced only the c4 fusion block with `QueryConditionedGatedFusion`.
- Kept c1-c3 original `GatedFusion` unchanged.
- Kept DFormerv2-S, DepthEncoder, SimpleFPNDecoder, CE loss, data, eval, and fixed training recipe unchanged.
- Added a zero-initialized query delta to the c4 gate logit and logged `train/qc_c4_delta_abs`, `train/qc_c4_gate_mean`, and `train/qc_c4_gate_std`.
- After the negative result, the code was removed from the active registry and archived under `final daima/feiqi/failed_experiments_r051_20260517/`.

## Evidence

- TensorBoard event: `final daima/checkpoints/R051_c4_query_conditioned_gate_run01/lightning_logs/version_0/events.out.tfevents.1778947060.Administrator.8956.0`
- Best checkpoint: `final daima/checkpoints/R051_c4_query_conditioned_gate_run01/dformerv2_c4_query_conditioned_gate-epoch=45-val_mIoU=0.5367.pt`
- mIoU detail: `final daima/miou_list/R051_c4_query_conditioned_gate_run01.md`
- Training completed with `Trainer.fit` reaching `max_epochs=50`.

## Results

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.536702 |
| best epoch | 46 |
| last val/mIoU | 0.507323 |
| last-5 mean | 0.498532 |
| last-10 mean | 0.514932 |
| best-to-last drop | 0.029379 |
| best val/loss | 0.981511 |
| last val/loss | 1.200262 |
| final qc c4 delta abs | 0.305460 |
| final qc c4 gate mean | 0.550471 |

## Comparison

- vs R016 corrected baseline `0.541121`: `-0.004419`
- vs R036 bounded residual `0.539790`: `-0.003088`
- vs R049 SyncBN norm-eval `0.537890`: `-0.001188`
- vs R041 DiffPixel c4 cue `0.537098`: `-0.000396`
- vs R050 c4 bypass `0.533066`: `+0.003636`

## Decision

R051 is negative as a corrected-baseline improvement. The query-conditioned gate beats R050's c4 bypass, confirming that preserving c4 fusion is useful, but it remains below R016/R036/R049/R041 and has severe late instability.

Do not tune query hidden size or gate delta scale. The next experiment should avoid another c4 gate-conditioning micro-variant and pivot to a distinct mechanism, such as lightweight c4 pre-gate rectification only if it is clearly different from R045, or a broader contract/stability hypothesis.
