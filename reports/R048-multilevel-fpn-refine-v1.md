# R048 Multi-Level FPN Refinement Decoder

- Branch: `exp/R048-multilevel-fpn-refine-v1`
- Model: `dformerv2_refined_fpn_decoder`
- Run: `R048_refined_fpn_decoder_run01`
- Status: `completed_stable_but_below_corrected_baseline`

## Hypothesis

The R016 `SimpleFPNDecoder` may be too shallow because it smooths only final `p1`. R048 tests whether per-level FPN smoothing plus p1-p4 concatenation and lightweight refinement improves semantic/detail aggregation while leaving DFormerv2-S, DepthEncoder, GatedFusion, loss, data, and fixed training recipe unchanged.

## Paper-Code Evidence

- FPN arXiv `1612.03144` supports top-down lateral fusion with per-level smoothing.
- PANet arXiv `1803.01534` and UPerNet-style segmentation heads support explicit multilevel aggregation rather than only final-level smoothing.
- CGRSeg arXiv `2405.06228` / `nizhenliang/CGRSeg` supports decoder/refinement-layer context reconstruction as a distinct line from RGB-D fusion tweaks. Only the minimal multilevel decoder refinement idea was ported.

## Implementation

- Added independent model entry `dformerv2_refined_fpn_decoder`.
- Added `RefinedFPNDecoder`: lateral p4-p1, smooth each level with `ConvBNReLU`, upsample p2/p3/p4 to p1, concatenate p1-p4, fuse with one `ConvBNReLU`, then classify.
- Added `DFormerV2RefinedFPNDecoderSegmentor` that inherits the original R016 path and replaces only `self.decoder`.
- Did not change dataset split, eval, mIoU, loaders, augmentation, optimizer, scheduler, batch size, max epochs, lr, workers, early stopping, DFormerv2-S level, pretrained loading, loss, DepthEncoder, or GatedFusion.

## Smoke Test

- `py_compile train.py src\models\decoder.py src\models\mid_fusion.py`: passed.
- `train.py --help`: passed and exposed `dformerv2_refined_fpn_decoder`.
- Random tensor forward/backward passed: logits `(1, 40, 128, 128)`, finite CE, decoder instance was `RefinedFPNDecoder`.
- Static code reviewer: PASS.

## Evidence

- TensorBoard event: `checkpoints\R048_refined_fpn_decoder_run01\lightning_logs\version_0\events.out.tfevents.1778926901.Administrator.39660.0`
- Best checkpoint: `checkpoints\R048_refined_fpn_decoder_run01\dformerv2_refined_fpn_decoder-epoch=41-val_mIoU=0.5342.pt`
- Saved command: `checkpoints\R048_refined_fpn_decoder_run01\run_r048.ps1` and `checkpoints\R048_refined_fpn_decoder_run01\launch_command.txt`
- mIoU detail: `miou_list\R048_refined_fpn_decoder_run01.md`
- Archived failed code: `feiqi\failed_experiments_r048_20260516\R048_refined_fpn_decoder_code.md`

## Metrics

- Best val/mIoU: `0.534154` at validation epoch `42`
- Last val/mIoU: `0.530318`
- Last-5 mean val/mIoU: `0.522281`
- Last-10 mean val/mIoU: `0.522532`
- Best-to-last drop: `0.003837`
- Best val/loss: `0.977211` at validation epoch `7`
- Last val/loss: `1.234661`
- Final train/loss_epoch: `0.058174`

## Decision

R048 is stable but negative below the corrected baseline. It stays below R016 `0.541121` by `-0.006967`, below R036 `0.539790` by `-0.005636`, and below R041 `0.537098` by `-0.002944`. The small best-to-last drop (`0.003837`) is useful evidence that this decoder is less collapse-prone than many fusion variants, but the peak is too low for the 0.56 path.

Do not promote `dformerv2_refined_fpn_decoder` to active mainline. Archive the exact implementation and pivot to a distinct R049 hypothesis. Do not tune decoder width, smooth depth, or add dropout as a micro-search; R048 already tested the main decoder-refinement hypothesis.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_refined_fpn_decoder `
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
  --checkpoint_dir ".\checkpoints\R048_refined_fpn_decoder_run01"
```
