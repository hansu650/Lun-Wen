# R046 DGFusion c4 Depth-Token Lite

- Branch: `exp/R046-dgfusion-c4-depth-token-v1`
- Model: `dformerv2_dgfusion_c4_depth_token`
- Run: `R046_dgfusion_c4_depth_token_run01`
- Status: `completed_negative_depth_token_below_corrected_baseline`

## Hypothesis

DGFusion-style c4 local depth-token interaction can provide local geometry-conditioned feature correction on top of the existing R016 GatedFusion c4 output, without replacing GatedFusion or changing decoder/loss/training recipe.

## Implementation

- Added independent model entry `dformerv2_dgfusion_c4_depth_token`.
- c1-c3 stay exactly on the original R016 `GatedFusion` path.
- c4 first uses original `GatedFusion`; then `DGFusionC4DepthTokenLite` computes a local pooled depth token from aligned c4 DepthEncoder features.
- The module computes normalized fused-query/depth-token key affinity, multiplies it into a projected depth value, and applies a zero-initialized output projection as a feature residual.
- This is a minimal idea transfer from DGFusion `depth_token_guided_pca.py`; no Detectron2/OneFormer framework, global condition token, auxiliary depth head, sampling offset, raw-depth Sobel cue, gate-logit diff cue, or self-adapter was added.
- Did not change dataset split, eval, mIoU, loaders, augmentation, optimizer, scheduler, batch size, max epochs, lr, workers, early stopping, DFormerv2-S level, pretrained loading, loss, `GatedFusion` class, or `SimpleFPNDecoder`.

## Smoke Test

- `py_compile train.py src\models\mid_fusion.py`: passed.
- `train.py --help`: passed and exposed `dformerv2_dgfusion_c4_depth_token`.
- Random tensor forward/backward: logits shape `(1, 40, 128, 128)`; initial `c4_token_delta_abs` was `0.0`; zero-init out projection received nonzero gradient.

## Evidence

- TensorBoard event: `final daima/checkpoints/R046_dgfusion_c4_depth_token_run01/lightning_logs/version_0/events.out.tfevents.1778913804.Administrator.33260.0`
- Best checkpoint: `final daima/checkpoints/R046_dgfusion_c4_depth_token_run01/dformerv2_dgfusion_c4_depth_token-epoch=43-val_mIoU=0.5318.pt`
- Saved command: `final daima/checkpoints/R046_dgfusion_c4_depth_token_run01/run_r046.ps1`
- mIoU detail: `final daima/miou_list/R046_dgfusion_c4_depth_token_run01.md`
- Archived failed code: `final daima/feiqi/failed_experiments_r046_20260516/R046_dgfusion_c4_depth_token_code.md`

## Metrics

- Best val/mIoU: `0.531838` at validation epoch `44`
- Last val/mIoU: `0.527239`
- Last-5 mean val/mIoU: `0.510100`
- Last-10 mean val/mIoU: `0.514172`
- Best-to-last drop: `0.004599`
- Best val/loss: `0.961911` at validation epoch `10`
- Last val/loss: `1.208670`
- Final train/loss_epoch: `0.054207`

## Depth-Token Diagnostics

- `c4_token_delta_abs` first/last/min/max: `0.009067` / `0.273799` / `0.009067` / `0.289744`
- `c4_token_affinity_mean` first/last/min/max: `0.623793` / `0.287895` / `0.287302` / `0.623793`
- `c4_token_affinity_std` first/last/min/max: `0.026211` / `0.028371` / `0.026211` / `0.048227`

## Decision

R046 is negative relative to the corrected R016 baseline. It crosses `0.53`, but remains below R016 `0.541121` by `-0.009283`, below R036 `0.539790` by `-0.007952`, and below R041 `0.537098` by `-0.005260`. The final drop is small (`0.004599`), but the late-window means remain weak.

The depth-token path clearly opened (`c4_token_delta_abs` max `0.289744`), while affinity mean contracted to `0.287895`. That did not translate into a stronger fixed-recipe peak. Do not tune window size/token dim/scale; archive this exact module and pivot to a distinct R047 hypothesis.

## Command

This command reproduces the experiment-time R046 implementation. Because R046 was rejected, the active mainline registry no longer exposes `dformerv2_dgfusion_c4_depth_token`; re-running this exact failed variant requires restoring the archived code diff from `final daima/feiqi/failed_experiments_r046_20260516/R046_dgfusion_c4_depth_token_code.md` or checking out the experiment branch state before post-run archival cleanup.

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_dgfusion_c4_depth_token `
  --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --early_stop_patience 30 `
  --accelerator gpu `
  --devices 1 `
  --dformerv2_pretrained "C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth" `
  --loss_type ce `
  --checkpoint_dir ".\checkpoints\R046_dgfusion_c4_depth_token_run01"
```
