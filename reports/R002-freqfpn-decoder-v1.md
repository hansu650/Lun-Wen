# R002 Frequency-Aware FPN Decoder

## Verdict

- status: `completed_negative`
- model: `dformerv2_freqfpn_decoder`
- branch: `exp/R002-freqfpn-decoder-v1`
- best val/mIoU: `0.516915` at epoch `44`
- last val/mIoU: `0.486524`
- target: `>= 0.530000`
- result: target not reached; slightly below clean 10-run baseline mean `0.517397`

## Hypothesis

Frequency-aware top-down decoder fusion would improve boundary/detail recovery by correcting low-frequency context and high-frequency detail mismatch during FPN upsampling, while leaving the DFormerv2 encoder, DepthEncoder, GatedFusion, loss, data, and fixed training recipe unchanged.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONIOENCODING = "utf-8"
& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_freqfpn_decoder `
  --data_root C:/Users/qintian/Desktop/qintian/data/NYUDepthv2 `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --checkpoint_dir checkpoints/dformerv2_freqfpn_decoder_run01 `
  --early_stop_patience 30 `
  --devices 1 `
  --accelerator auto `
  --dformerv2_pretrained C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth `
  --loss_type ce
```

## Evidence

- mIoU detail: `final daima/miou_list/dformerv2_freqfpn_decoder_run01.md`
- TensorBoard event: `final daima/checkpoints/dformerv2_freqfpn_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778607762.Administrator.22656.0`
- best checkpoint: `final daima/checkpoints/dformerv2_freqfpn_decoder_run01/dformerv2_freqfpn_decoder-epoch=43-val_mIoU=0.5169.pt`
- process note: `Trainer.fit` reached `max_epochs=50`.

## Comparisons

- delta vs clean 10-run baseline mean `0.517397`: `-0.000482`
- delta vs clean baseline mean + 1 std `0.522298`: `-0.005383`
- delta vs PMAD logit-only w0.15/T4 five-run mean `0.520795`: `-0.003880`
- last-5 mean val/mIoU: `0.498475`
- last-10 mean val/mIoU: `0.504222`

## Diagnostics

- The run peaked late at epoch 44 with `0.516915`, then fell to `0.486524` by epoch 50.
- The best value is near the clean baseline mean but does not beat it, so the decoder did not create decision-value improvement.
- The implementation keeps the baseline `SimpleFPNDecoder` unchanged and exposes the decoder experiment only as `dformerv2_freqfpn_decoder`.

## Forbidden-Change Audit

- No dataset split, dataloader, validation/test loader, data augmentation, eval metric, or mIoU calculation changed.
- No optimizer, scheduler, batch size, epoch count, learning rate, worker count, or early-stop setting changed.
- Checkpoints, TensorBoard event logs, pretrained weights, datasets, and large logs are evidence only and must not be committed.

## Decision

Reject this exact decoder. It is a useful decoder ablation, but it should not be claimed as an improvement and should not be repeated unchanged.

## Audit

- code review: `approved`
- reproducer/report audit: `audit_passed_no_rerun`
- metric audit: TensorBoard event contains 50 `val/mIoU` records; best, last, last-5 mean, last-10 mean, best val/loss, final train loss, and all per-epoch table rows match the report.
- commit hygiene requirement: stage only explicit R002 code/docs/metrics/report files; do not stage tracked checkpoint deletions or ignored checkpoint evidence.
