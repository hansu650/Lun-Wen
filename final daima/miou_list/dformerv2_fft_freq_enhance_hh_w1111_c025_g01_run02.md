# dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run02

## Settings

- model: `dformerv2_fft_freq_enhance`
- batch_size: 2
- max_epochs: 50
- lr: 6e-5
- num_workers: 4
- early_stop_patience: 30
- cutoff_ratio: 0.25
- gamma_init: 0.1
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- checkpoint_dir: `checkpoints/dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run02`

## Model Summary

- Total params: 43.1M
- Trainable params: 43.1M

## Summary

- recorded validation epochs: 50
- best val/mIoU: 0.5159 at epoch 38
- last val/mIoU: 0.443
- best val/loss: 1.032717 at epoch 8
- train/loss_epoch: first 2.293756, last 0.352748

## Baseline Comparison

- clean 10-run GatedFusion baseline mean best: 0.517397
- delta vs baseline mean: -0.001497
- clean 10-run baseline population std: 0.004901
- delta in baseline std units: -0.305
- clean 10-run baseline best single run: 0.524425
- delta vs baseline best single: -0.008525

## Comparison with run01

- run01 best: 0.522688
- delta vs run01: -0.006788

## Conclusion

Negative result. Best val/mIoU 0.5159 is below the clean 10-run baseline mean by -0.001497. The last-epoch mIoU 0.443 shows severe late collapse, dropping 0.073 from the best. This is significantly worse than run01 (0.522688), suggesting run01 was a high-variance outlier.
