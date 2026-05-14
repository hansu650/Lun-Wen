# R017 RGB/BGR Official Channel Contract

## Dry Check

- Branch: `exp/R017-rgb-bgr-contract-v1`
- Model: `dformerv2_mid_fusion`
- Planned run: `R017_rgb_bgr_official_contract`
- Hypothesis: after R015/R016 align labels and depth normalization, RGB channel order should match the official DFormer NYUDepthV2 input contract. Official DFormer uses OpenCV BGR mode for non-SUNRGBD datasets, while the local data module was converting BGR to RGB.
- Official source: `ref_codes/DFormer/utils/dataloader/RGBXDataset.py` sets `rgb_mode = "BGR"` except for the SUNRGBD + DFormerv2 special case, and `_open_image(..., "BGR")` returns `cv2.imread(..., IMREAD_UNCHANGED)` without `BGR2RGB`.
- Local minimal change: in `final daima/src/data_module.py`, remove `cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)` and keep OpenCV BGR channel order before the existing Normalize/ToTensor path.
- No model, decoder, backbone, split, label mapping, depth normalization, metric, mIoU, optimizer, scheduler, batch size, epoch count, lr, num_workers, early stopping, or pretrained-loading logic is changed.
- Claim boundary: this is another official input-contract alignment, not a novel method contribution.

## Smoke Evidence

- Real-batch loader sanity passed after removing `BGR2RGB`.
- RGB tensor kept shape `(2, 3, 480, 640)` after Normalize/ToTensor.
- Depth remained on the R016 DFormer normalization contract.
- Labels remained canonical train IDs plus `255`.
- CUDA forward sanity passed with logits `(2, 40, 480, 640)` and CE loss `3.714051`.

## Full Train Command

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\nyu056-mainline\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_mid_fusion `
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
  --checkpoint_dir ".\checkpoints\R017_rgb_bgr_official_contract"
```

## Result

Full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.

- TensorBoard event: `checkpoints/R017_rgb_bgr_official_contract/lightning_logs/version_0/events.out.tfevents.1778750750.Administrator.38244.0`
- Best checkpoint: `checkpoints/R017_rgb_bgr_official_contract/dformerv2_mid_fusion-epoch=37-val_mIoU=0.5291.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.529090` at validation epoch `38`
- Last val/mIoU: `0.523078`
- Last-5 mean val/mIoU: `0.494251`
- Last-10 mean val/mIoU: `0.506949`
- Best-to-last drop: `0.006011`
- Best val/loss: `0.973518` at validation epoch `9`
- Last val/loss: `1.228286`
- Final train/loss_epoch: `0.063107`
- Evidence table: `final daima/miou_list/R017_rgb_bgr_official_contract.md`

Comparison against the R016 official-label-and-depth baseline:

- R016 best val/mIoU: `0.541121`
- R017 best val/mIoU: `0.529090`
- Delta: `-0.012031`

Interpretation:

- R017 is a negative contract-alignment test. Keeping OpenCV BGR input, although consistent with the official DFormer NYUDepthV2 dataloader path, substantially underperforms the R016 RGB input baseline in this local adaptation.
- Do not merge the active BGR code into main.
- Keep R016 as the current corrected baseline and continue with the next contract gate, likely DFormerv2-S `drop_path_rate=0.25`.
