# Checkpoint Cleanup Summary

This note records why the local checkpoint archive was removed and which
experiment results should be kept as the lightweight source of truth.

## Cleanup Decision

The checkpoint directory was treated as disposable training output, not as part
of the long-term paper code archive.

- local directory: `framework_download/checkpoints/`
- total files before cleanup: `234`
- total size before cleanup: `136.47 GB`
- checkpoint files: `73`
- checkpoint size: `136.46 GB`
- files larger than `100 MB`: `73`
- files larger than `2 GB`: `61`

The full archive was too large for normal GitHub storage. Most files were
failed, exploratory, or intermediate training outputs. The important paper
evidence is preserved in the result documents under `framework_download/results/`.

No raw checkpoint files were selected for long-term retention in this cleanup.

## Directory Size Before Cleanup

| Directory | Files | Size |
| --- | ---: | ---: |
| `v0.0-1.0` | 201 | `129.67 GB` |
| `v2_cstc_stage1_alpha01_run03` | 3 | `0.64 GB` |
| `v2_cstc_stage1_alpha01_run04` | 3 | `0.64 GB` |
| `v2_cstc_stage1_alpha0_run01` | 3 | `0.64 GB` |
| `v2_cstc_stage1_alpha01_run02` | 3 | `0.64 GB` |
| `v2_cstc_stage1_run01` | 3 | `0.64 GB` |
| `v2_cstc_stage1_alpha0_run02` | 3 | `0.64 GB` |
| `dinov2_s_swin_t_run01` | 3 | `0.63 GB` |
| `dinov2_s_swin_t_hha_run01` | 3 | `0.63 GB` |
| `v2_dino_s_swin_t_run02` | 3 | `0.56 GB` |
| `teacher_original_only_encoder_dino_s_swin_t_run01` | 3 | `0.56 GB` |
| `teacher_clean_dino_s_swin_t_run01` | 3 | `0.56 GB` |

## Important Recorded Results

The most useful result records are:

- `v0.0-1.0/swin_dino_mid_stable_baseline_10runs.md`
  - stable v1.0 baseline
  - mean validation mIoU: `0.4828`
  - best validation mIoU: `0.4900`
- `v0.0-1.0/swin_dino_mid_b2_mcads_decoder_summary.md`
  - MCADS-inspired reduced decoder head
  - mean validation mIoU: `0.4842`
  - best validation mIoU: `0.4902`
- `v0.0-1.0/swin_dino_mid_prompt_c4_clean_ablation_1.md`
  - clean c4 prompt ablation
  - best validation mIoU: `0.4807`
- `v0.0-1.0/swin_dino_mid_c4_dformer_3runs_6results.md`
  - DFormerv2-style c4 replacement study
  - best-per-run mean mIoU: `0.4790`
  - best validation mIoU: `0.4812`
- `v0.0-1.0/swin_dino_mid_ktb_fdam_prompt_c4_1.md`
  - KTB/FDAM plus c4 prompt attempt
  - best validation mIoU: `0.4741`
- `v0.0-1.0/swin_dino_mid_1.md`
  - first Swin-B plus DINOv2-B mid-fusion run
  - best validation mIoU: `0.4663`

## Top Checkpoints Before Cleanup

These filenames were recorded for traceability only. The actual `.ckpt` files
were not retained.

| val/mIoU | Epoch | Size | Checkpoint |
| ---: | ---: | ---: | --- |
| `0.4902` | 23 | `2.09 GB` | `v0.0-1.0/swin_dino_mid_b2_mcads_decoder_reduced_run06/mid_fusion-epoch=23-val_mIoU=0.4902.ckpt` |
| `0.4900` | 21 | `2.08 GB` | `v0.0-1.0/swin_dino_mid_stable_baseline_run08/mid_fusion-epoch=21-val_mIoU=0.4900.ckpt` |
| `0.4891` | 24 | `2.08 GB` | `v0.0-1.0/swin_dino_mid_v1_baseline_cmp_run01/mid_fusion-epoch=24-val_mIoU=0.4891.ckpt` |
| `0.4885` | 25 | `2.09 GB` | `v0.0-1.0/swin_dino_mid_b2_mcads_decoder_reduced_run03/mid_fusion-epoch=25-val_mIoU=0.4885.ckpt` |
| `0.4875` | 17 | `2.08 GB` | `v0.0-1.0/swin_dino_mid_stable_baseline_run06/mid_fusion-epoch=17-val_mIoU=0.4875.ckpt` |

## Interpretation

The checkpoints themselves are not needed for the paper narrative. The useful
information is:

- which model variants were tested
- which ones failed because of memory or weak performance
- which baseline became stable enough to report
- which variant produced the best observed validation score

For future work, use the result markdown files as the experiment archive and
only save a checkpoint when it is needed for a final demo, reproduction package,
or external review.
