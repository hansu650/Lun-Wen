# R038 DSCF C4 Lite Run01

## Summary

- Branch: exp/R038-dscf-c4-lite-v1
- Model: dformerv2_dscf_c4_lite
- Run: R038_dscf_c4_lite_run01
- Hypothesis: KTB/CVPR 2025 DSCF-style dynamic sparse cross-modal sampling at c4 may suppress depth noise better than dense c4 GatedFusion while leaving c1-c3, decoder, loss, and training recipe unchanged.
- Status: completed full train; `Trainer.fit` reached `max_epochs=50`; exit code `0`.
- TensorBoard event: `checkpoints\R038_dscf_c4_lite_run01\lightning_logs\version_0\events.out.tfevents.1778863547.Administrator.8372.0`
- Best checkpoint: `checkpoints\R038_dscf_c4_lite_run01\dformerv2_dscf_c4_lite-epoch=37-val_mIoU=0.5308.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.530810` at validation epoch `38`
- Last val/mIoU: `0.530308`
- Last-5 mean val/mIoU: `0.526104`
- Last-10 mean val/mIoU: `0.522189`
- Best-to-last drop: `0.000502`
- Best val/loss: `0.936448` at validation epoch `10`
- Last val/loss: `1.218423`
- Final train/loss_epoch: `0.056458`
- DSCF c4 offset_abs first/last: `0.961821` / `1.675011`
- DSCF c4 weight_entropy first/last: `1.376656` / `1.336370`
- Delta vs R016 corrected baseline best `0.541121`: `-0.010311`
- Decision: negative below R016; do not promote as active mainline.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_dscf_c4_lite `
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
  --checkpoint_dir ".\checkpoints\R038_dscf_c4_lite_run01"
```

## Smoke Evidence

- `py_compile` and `train.py --help` passed.
- Real NYU CUDA smoke confirmed c1-c3 are original `GatedFusion`, c4 is `DSCFC4LiteFusion`, logits `(1, 40, 480, 640)`, finite CE `3.778212`, unchanged DFormerv2 pretrained load stats, initial offset_abs `0.244919`, initial entropy `1.386294`, nonzero offset/sample-weight/gate/refine gradients, and non-identical K-branch offset gradients.
- Saved launch command: `checkpoints/R038_dscf_c4_lite_run01/run_r038.cmd`.

## Per-Epoch Validation Metrics

| Val Epoch | Global Step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.166513 | 1.672458 |
| 2 | 793 | 0.236868 | 1.366824 |
| 3 | 1190 | 0.302596 | 1.197978 |
| 4 | 1587 | 0.350634 | 1.111161 |
| 5 | 1984 | 0.420840 | 1.006343 |
| 6 | 2381 | 0.437734 | 0.994010 |
| 7 | 2778 | 0.472226 | 0.960653 |
| 8 | 3175 | 0.477698 | 0.985579 |
| 9 | 3572 | 0.474038 | 0.999135 |
| 10 | 3969 | 0.502431 | 0.936448 |
| 11 | 4366 | 0.486001 | 0.979735 |
| 12 | 4763 | 0.479699 | 1.008198 |
| 13 | 5160 | 0.492012 | 1.003205 |
| 14 | 5557 | 0.494064 | 0.991791 |
| 15 | 5954 | 0.498149 | 0.980988 |
| 16 | 6351 | 0.499892 | 1.016057 |
| 17 | 6748 | 0.495776 | 1.031453 |
| 18 | 7145 | 0.516575 | 0.996456 |
| 19 | 7542 | 0.504849 | 1.043235 |
| 20 | 7939 | 0.505743 | 1.052472 |
| 21 | 8336 | 0.499743 | 1.089497 |
| 22 | 8733 | 0.487665 | 1.119700 |
| 23 | 9130 | 0.503509 | 1.077438 |
| 24 | 9527 | 0.524466 | 1.047839 |
| 25 | 9924 | 0.499241 | 1.129589 |
| 26 | 10321 | 0.516174 | 1.076602 |
| 27 | 10718 | 0.520841 | 1.108333 |
| 28 | 11115 | 0.496660 | 1.171909 |
| 29 | 11512 | 0.495533 | 1.158864 |
| 30 | 11909 | 0.508787 | 1.133187 |
| 31 | 12306 | 0.519103 | 1.142141 |
| 32 | 12703 | 0.525069 | 1.144751 |
| 33 | 13100 | 0.513761 | 1.149701 |
| 34 | 13497 | 0.491335 | 1.190529 |
| 35 | 13894 | 0.484330 | 1.173769 |
| 36 | 14291 | 0.521114 | 1.130118 |
| 37 | 14688 | 0.523558 | 1.143810 |
| 38 | 15085 | 0.530810 | 1.120253 |
| 39 | 15482 | 0.526406 | 1.194961 |
| 40 | 15879 | 0.501904 | 1.219700 |
| 41 | 16276 | 0.496780 | 1.225666 |
| 42 | 16673 | 0.524283 | 1.165431 |
| 43 | 17070 | 0.516916 | 1.166013 |
| 44 | 17467 | 0.526515 | 1.160588 |
| 45 | 17864 | 0.526877 | 1.168193 |
| 46 | 18261 | 0.527288 | 1.172868 |
| 47 | 18658 | 0.523076 | 1.181257 |
| 48 | 19055 | 0.522378 | 1.196132 |
| 49 | 19452 | 0.527470 | 1.214553 |
| 50 | 19849 | 0.530308 | 1.218423 |
