# R043 DepthGeo c4 Cue

## Dry Check

- branch: `exp/R043-depthgeo-c4-cue-v1`
- model: `dformerv2_depthgeo_c4_cue`
- run: `R043_depthgeo_c4_cue_run01`
- checkpoint_dir: `checkpoints/R043_depthgeo_c4_cue_run01`
- hypothesis: a c4-only raw-depth Sobel/normal-like geometry cue can improve high-level gate decisions more safely than RGB-depth feature disagreement.
- main code changes: `src/models/mid_fusion.py`, `train.py`
- fixed recipe: unchanged batch size, epochs, lr, optimizer, scheduler, workers, early stopping, data split, augmentation, loader, eval, metric, DFormerv2-S level, and pretrained loading.

## Implementation

R043 preserved the original c1-c3 `GatedFusion` outputs and replaced only c4 with `DepthGeometryC4CueFusion`. The c4 block computes the original `[rgb, depth_proj]` gate logit, adds a zero-initialized logit correction from a raw-depth cue, then applies the original sigmoid fusion and refine path.

The raw-depth cue is computed on the fly inside the model from the already-loaded depth tensor: Sobel `dx/dy`, magnitude, and normal-like `nx/ny/nz`. This did not change dataloader preprocessing or dataset files.

## Verification Before Full Train

- `py_compile`: passed
- `train.py --help`: passed and exposed `dformerv2_depthgeo_c4_cue`
- real NYU CUDA batch forward/backward: passed
- smoke logits: `(2, 40, 480, 640)`
- smoke CE: `3.770108`
- DFormerv2 pretrained load stats: `loaded_keys=774`, `missing_keys=6`, `unexpected_keys=11`
- smoke gradients: geo gate, c4 depth projection, base gate, and refine path all received nonzero gradients
- archival note: after the result was recorded, the R043 registry entry was removed and current `train.py --help` no longer exposes `dformerv2_depthgeo_c4_cue`.

## Full Train Evidence

- status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`
- TensorBoard event: `final daima/checkpoints/R043_depthgeo_c4_cue_run01/lightning_logs/version_0/events.out.tfevents.1778893712.Administrator.9528.0`
- checkpoint: `final daima/checkpoints/R043_depthgeo_c4_cue_run01/dformerv2_depthgeo_c4_cue-epoch=41-val_mIoU=0.5356.pt`
- saved command: `final daima/checkpoints/R043_depthgeo_c4_cue_run01/run_r043.cmd`
- mIoU detail: `final daima/miou_list/R043_depthgeo_c4_cue_run01.md`

## Result

- recorded validation epochs: `50`
- best val/mIoU: `0.535592` at validation epoch `42`
- last val/mIoU: `0.522214`
- last-5 mean val/mIoU: `0.518946`
- last-10 mean val/mIoU: `0.522097`
- best-to-last drop: `0.013378`
- best val/loss: `0.961905` at validation epoch `11`
- last val/loss: `1.208118`
- final train/loss_epoch: `0.052872`

## Diagnostics

- `train/depthgeo_c4_geo_logit_abs`: `0.015203 -> 0.153483`, max `0.153980`
- `train/depthgeo_c4_gate_mean`: `0.499898 -> 0.512162`, max `0.512432`
- `train/depthgeo_c4_gate_std`: `0.207853 -> 0.212319`, max `0.213092`
- `train/depthgeo_c4_edge_mean`: `0.145911 -> 0.145908`
- `train/depthgeo_c4_edge_std`: `0.491999 -> 0.491720`

## Comparison

- vs R042 c3-to-c4 DiffPixel cue: `+0.004863` best and far less late collapse.
- vs R041 DiffPixel c4 cue: `-0.001506`.
- vs R036 c3/c4 bounded residual: `-0.004198`.
- vs R016 corrected baseline: `-0.005529`.

## Conclusion

R043 is a partial positive geometry-cue diagnostic below the corrected baseline. The explicit raw-depth geometry cue is safer than c3-propagated feature disagreement, and the c4 gate stays conservative instead of exploding, but the peak remains below R041/R036/R016 and the late-window means are still weak.

Decision: do not promote the code to active mainline. Archive the implementation under `final daima/feiqi/failed_experiments_r043_20260516/`, keep all evidence records, and move to a distinct R044 hypothesis rather than tuning Sobel/normal cue hidden size or scale.
