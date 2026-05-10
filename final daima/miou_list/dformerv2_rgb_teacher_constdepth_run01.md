# dformerv2_rgb_teacher_constdepth_run01

## Settings

- model: `dformerv2_rgb_teacher_constdepth`
- purpose: Phase 0 RGB-only teacher sanity check for PMAD / PrimKD logit distillation
- architecture: `DFormerv2_S + SimpleFPNDecoder`; depth input is replaced by constant zero depth inside the model
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- checkpoint: `checkpoints/dformerv2_rgb_teacher_constdepth_run01/dformerv2_rgb_teacher_constdepth-epoch=42-val_mIoU=0.4885.pt`

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.488489` at epoch `43`
- last val/mIoU: `0.456266`
- best val/loss: `1.102055` at epoch `7`
- last val/loss: `1.454060`
- mean val/mIoU over last 10 epochs: `0.468347`
- teacher usability threshold: `0.515000`
- clean 10-run RGB-D baseline mean best: `0.517397`
- clean 10-run RGB-D baseline std: `0.004901`
- clean 10-run RGB-D baseline best single: `0.524425`
- repeat5 RGB-D baseline mean best: `0.511893`
- delta vs teacher usability threshold: `-0.026511`
- delta vs clean baseline mean: `-0.028908`
- delta vs clean baseline mean in std units: `-5.898`

## Per-Epoch Metrics

| epoch | step | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|---:|
| 1 | 396 | 0.131116 | 1.825073 | 2.431177 |
| 2 | 793 | 0.187730 | 1.492651 | 1.712008 |
| 3 | 1190 | 0.229981 | 1.337191 | 1.407186 |
| 4 | 1587 | 0.257354 | 1.313472 | 1.205865 |
| 5 | 1984 | 0.308767 | 1.200804 | 1.062142 |
| 6 | 2381 | 0.331744 | 1.168601 | 0.908083 |
| 7 | 2778 | 0.388447 | 1.102055 | 0.804094 |
| 8 | 3175 | 0.395275 | 1.149008 | 0.729713 |
| 9 | 3572 | 0.413295 | 1.152907 | 0.640328 |
| 10 | 3969 | 0.428695 | 1.134478 | 0.567014 |
| 11 | 4366 | 0.426292 | 1.176411 | 0.550307 |
| 12 | 4763 | 0.428656 | 1.168591 | 0.516669 |
| 13 | 5160 | 0.447564 | 1.135222 | 0.467943 |
| 14 | 5557 | 0.454512 | 1.139473 | 0.419501 |
| 15 | 5954 | 0.461781 | 1.176805 | 0.388840 |
| 16 | 6351 | 0.454427 | 1.186033 | 0.371354 |
| 17 | 6748 | 0.454475 | 1.188186 | 0.367058 |
| 18 | 7145 | 0.430659 | 1.284104 | 0.373608 |
| 19 | 7542 | 0.412439 | 1.335129 | 0.360307 |
| 20 | 7939 | 0.463473 | 1.198397 | 0.348931 |
| 21 | 8336 | 0.470223 | 1.237233 | 0.318350 |
| 22 | 8733 | 0.464123 | 1.244597 | 0.303350 |
| 23 | 9130 | 0.472826 | 1.239773 | 0.282683 |
| 24 | 9527 | 0.466194 | 1.249319 | 0.275578 |
| 25 | 9924 | 0.452245 | 1.318203 | 0.288494 |
| 26 | 10321 | 0.472000 | 1.263644 | 0.286238 |
| 27 | 10718 | 0.468782 | 1.250750 | 0.253590 |
| 28 | 11115 | 0.463546 | 1.286782 | 0.259917 |
| 29 | 11512 | 0.468484 | 1.272837 | 0.249348 |
| 30 | 11909 | 0.467112 | 1.304334 | 0.245218 |
| 31 | 12306 | 0.457635 | 1.321785 | 0.275616 |
| 32 | 12703 | 0.474720 | 1.293453 | 0.241000 |
| 33 | 13100 | 0.476150 | 1.302826 | 0.225128 |
| 34 | 13497 | 0.482002 | 1.302580 | 0.220363 |
| 35 | 13894 | 0.445782 | 1.431791 | 0.213220 |
| 36 | 14291 | 0.447926 | 1.369632 | 0.250909 |
| 37 | 14688 | 0.467736 | 1.308031 | 0.247857 |
| 38 | 15085 | 0.478865 | 1.331274 | 0.210437 |
| 39 | 15482 | 0.487399 | 1.319804 | 0.202087 |
| 40 | 15879 | 0.484314 | 1.337620 | 0.193438 |
| 41 | 16276 | 0.486635 | 1.355018 | 0.191277 |
| 42 | 16673 | 0.482404 | 1.356886 | 0.189569 |
| 43 | 17070 | 0.488489 | 1.362785 | 0.204959 |
| 44 | 17467 | 0.464221 | 1.433588 | 0.188522 |
| 45 | 17864 | 0.452445 | 1.425141 | 0.240957 |
| 46 | 18261 | 0.437098 | 1.458986 | 0.227095 |
| 47 | 18658 | 0.470453 | 1.382727 | 0.208320 |
| 48 | 19055 | 0.478862 | 1.367102 | 0.181878 |
| 49 | 19452 | 0.466601 | 1.393486 | 0.177015 |
| 50 | 19849 | 0.456266 | 1.454060 | 0.182097 |

## Interpretation

- The RGB-only constant-depth teacher is **not usable for PMAD distillation** in its current form.
- Best val/mIoU `0.488489` is below the minimum teacher gate `0.515000` by `0.026511` and below the clean RGB-D baseline mean by `0.028908`.
- Validation mIoU peaks at epoch 43 and then declines to `0.456266`, while train/loss_epoch remains very low, indicating overfitting and insufficient teacher quality.
- Do not use this checkpoint as a teacher for `dformerv2_primkd_logit_only`; KD from a weak teacher would likely anchor the RGB-D student to a worse decision boundary.

## Decision

- Stop Phase 1 PMAD formal training with this teacher.
- First fix or replace the teacher. Candidate fixes: train a proper RGB-only DFormerV2 segmentation model that uses the original depth geometry path only as zero prior, add a small RGB-only head without DepthEncoder/GatedFusion if the current wrapper is too weak, or abandon teacher-based KD unless a teacher reaches at least `0.515`.
