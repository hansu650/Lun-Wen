# R022 Ham Dropout Parity

- branch: `exp/R022-ham-dropout-parity-v1`
- model: `dformerv2_ham_decoder`
- run: `R022_ham_dropout_parity_run01`
- status: completed partial positive below corrected baseline
- evidence: `final daima/miou_list/R022_ham_dropout_parity_run01.md`

## Hypothesis

R021 may underperform because it omitted the official `BaseDecodeHead.cls_seg()` dropout before the segmentation classifier. R022 adds only `Dropout2d(0.1)` before the Ham classifier to test strict classifier parity.

## Implementation

R022 changes only `OfficialHamDecoder` in `src/models/decoder.py`:

- add `self.dropout = nn.Dropout2d(0.1)`
- replace `self.classifier(x)` with `self.classifier(self.dropout(x))`

No model registry, encoder, backbone, pretrained loading, data, split, loader, augmentation, eval metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, or early stopping changed.

## Fixed Recipe

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\nyu056-mainline\final daima"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_ham_decoder `
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
  --checkpoint_dir ".\checkpoints\R022_ham_dropout_parity_run01"
```

## Evidence

- full train exit code: `0`
- recorded validation epochs: `50`
- TensorBoard event: `final daima/checkpoints/R022_ham_dropout_parity_run01/lightning_logs/version_0/events.out.tfevents.1778782418.Administrator.36360.0`
- best checkpoint: `final daima/checkpoints/R022_ham_dropout_parity_run01/dformerv2_ham_decoder-epoch=49-val_mIoU=0.5343.pt`

## Result

- best val/mIoU: `0.534332` at validation epoch `50`
- last val/mIoU: `0.534332`
- last-5 mean val/mIoU: `0.527687`
- last-10 mean val/mIoU: `0.512629`
- best-to-last drop: `0.000000`
- best val/loss: `1.106345` at validation epoch `21`
- final train/loss_epoch: `0.059158`

## Comparison

- R021 no-dropout Ham: `0.527353`; R022 is higher by `0.006979`.
- R020 branch depth blend adapter: `0.532924`; R022 is higher by `0.001408`.
- R016 corrected baseline: `0.541121`; R022 is lower by `0.006790`.

## Decision

R022 is a partial-positive parity fix but not a new corrected baseline. It shows the official classifier dropout mattered, yet Ham decoder parity still does not close the gap to R016 or the final `0.56` target. The next highest-value step is the corrected-contract geometry-primary teacher refresh, because old PMAD used a weak pre-corrected teacher.
