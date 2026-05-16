# R047 GatedFusion Local GroupNorm

- Branch: `exp/R047-gatedfusion-local-gn-v1`
- Model: `dformerv2_gatedfusion_gn`
- Run: `R047_gatedfusion_local_gn_run01`
- Status: `completed_negative_gn_below_corrected_baseline`

## Hypothesis

R016 late instability may partly come from batch-dependent BatchNorm statistics inside the randomly initialized local `GatedFusion` blocks under fixed `batch_size=2`. Replacing only the fusion-local `BatchNorm2d` layers with `GroupNorm(32, C)` tests whether a BN-free fusion path improves stability while preserving the original fusion equation and all fixed training settings.

## Paper-Code Evidence

- Group Normalization, ECCV 2018, arXiv `1803.08494`: GN does not rely on batch-dimension statistics and is a standard small-batch normalization alternative.
- PyTorch `torch.nn.GroupNorm` standard implementation was used directly; no custom normalization math was introduced.
- MoBaNet 2026 multimodal gated fusion uses GroupNorm inside a gated fusion block, supporting BN-free gated fusion as a recent multimodal implementation pattern. ICCV 2025 unimodal-bias work supports modality imbalance as motivation, but was not ported as a loss or regularizer.

## Implementation

- Added independent model entry `dformerv2_gatedfusion_gn`.
- Added `GatedFusionGN` matching original `GatedFusion` except gate/refine `BatchNorm2d` became `GroupNorm(32, C)`.
- Added `DFormerV2GatedFusionGNSegmentor` that reuses DFormerv2-S, DepthEncoder, original four-stage fusion structure, and SimpleFPNDecoder, replacing only `self.fusions` with GN variants.
- Did not change dataset split, eval, mIoU, loaders, augmentation, optimizer, scheduler, batch size, max epochs, lr, workers, early stopping, DFormerv2-S level, pretrained loading, loss, DepthEncoder, or SimpleFPNDecoder.

## Smoke Test

- `py_compile train.py src\models\mid_fusion.py`: passed.
- `train.py --help`: passed and exposed `dformerv2_gatedfusion_gn`.
- Random tensor forward/backward passed: logits `(1, 40, 128, 128)`, finite CE, `8` GroupNorm layers across four fusion blocks, and `0` fusion-local BatchNorm layers.
- Static code reviewer: PASS.

## Evidence

- TensorBoard event: `checkpoints\R047_gatedfusion_local_gn_run01\lightning_logs\version_0\events.out.tfevents.1778920890.Administrator.5584.0`
- Best checkpoint: `checkpoints\R047_gatedfusion_local_gn_run01\dformerv2_gatedfusion_gn-epoch=24-val_mIoU=0.5283.pt`
- Saved command: `checkpoints\R047_gatedfusion_local_gn_run01\run_r047.ps1` and `checkpoints\R047_gatedfusion_local_gn_run01\launch_command.txt`
- mIoU detail: `miou_list\R047_gatedfusion_local_gn_run01.md`
- Archived failed code: `feiqi\failed_experiments_r047_20260516\R047_gatedfusion_gn_code.md`

## Metrics

- Best val/mIoU: `0.528301` at validation epoch `25`
- Last val/mIoU: `0.472746`
- Last-5 mean val/mIoU: `0.509970`
- Last-10 mean val/mIoU: `0.513930`
- Best-to-last drop: `0.055555`
- Best val/loss: `0.971623` at validation epoch `15`
- Last val/loss: `1.294296`
- Final train/loss_epoch: `0.103899`

## Decision

R047 is negative. It stays below R016 `0.541121` by `-0.012820`, below R036 `0.539790` by `-0.011489`, and below R041 `0.537098` by `-0.008797`. The final drop is severe (`0.055555`), so full BN-to-GN replacement inside `GatedFusion` does not solve the fixed-recipe late-instability problem and lowers the peak.

Do not promote `dformerv2_gatedfusion_gn` to active mainline. Archive the exact implementation and pivot to a distinct R048 hypothesis. A much narrower gate-only GN ablation should not be the immediate next step unless later evidence specifically implicates the gate normalization alone, because full GN already damaged the fusion path.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_gatedfusion_gn `
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
  --checkpoint_dir ".\checkpoints\R047_gatedfusion_local_gn_run01"
```
