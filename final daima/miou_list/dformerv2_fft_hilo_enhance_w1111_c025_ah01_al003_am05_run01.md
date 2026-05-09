# dformerv2_fft_hilo_enhance_w1111_c025_ah01_al003_am05_run01

## Settings

- model: `dformerv2_fft_hilo_enhance`
- batch_size: 2
- max_epochs: 50
- lr: 6e-5
- num_workers: 4
- early_stop_patience: 30
- cutoff_ratio: 0.25
- alpha_high_init: 0.10
- alpha_low_init: 0.03
- alpha_max: 0.5
- hilo_stage_weights: 1,1,1,1
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- checkpoint_dir: `checkpoints/dformerv2_fft_hilo_enhance_w1111_c025_ah01_al003_am05_run01`

## Model Summary

- Total params: 46.6M
- Trainable params: 46.6M
- Non-trainable params: 0

## Validation Results (50 epochs)

| Epoch | val/mIoU | val/loss | train/loss_epoch |
|------:|---------:|---------:|-----------------:|
|     0 | 0.160945 | 1.684600 |         2.251478 |
|     1 | 0.230663 | 1.366606 |         1.586882 |
|     2 | 0.298275 | 1.208566 |         1.279054 |
|     3 | 0.343607 | 1.139121 |         1.067391 |
|     4 | 0.377689 | 1.063359 |         0.911780 |
|     5 | 0.416845 | 1.061592 |         0.774494 |
|     6 | 0.445437 | 1.038171 |         0.665433 |
|     7 | 0.442101 | 1.055323 |         0.589965 |
|     8 | 0.457389 | 1.031858 |         0.529374 |
|     9 | 0.476303 | 1.017970 |         0.472786 |
|    10 | 0.474974 | 1.021638 |         0.428472 |
|    11 | 0.483548 | 1.030388 |         0.393883 |
|    12 | 0.459148 | 1.089961 |         0.375330 |
|    13 | 0.483880 | 1.048253 |         0.361294 |
|    14 | 0.485662 | 1.064247 |         0.330625 |
|    15 | 0.479537 | 1.079144 |         0.299769 |
|    16 | 0.492931 | 1.052433 |         0.288977 |
|    17 | 0.483238 | 1.124329 |         0.277148 |
|    18 | 0.483648 | 1.099097 |         0.274080 |
|    19 | 0.501521 | 1.085413 |         0.272594 |
|    20 | 0.481752 | 1.149653 |         0.248840 |
|    21 | 0.457302 | 1.274583 |         0.259126 |
|    22 | 0.489488 | 1.151080 |         0.283326 |
|    23 | 0.508950 | 1.121000 |         0.225456 |
|    24 | 0.508862 | 1.104517 |         0.208564 |
|    25 | 0.513597 | 1.131022 |         0.197730 |
|    26 | 0.511003 | 1.134998 |         0.195549 |
|    27 | 0.511277 | 1.127617 |         0.194093 |
|    28 | 0.516428 | 1.141593 |         0.195543 |
|    29 | 0.477776 | 1.186051 |         0.212601 |
|    30 | 0.477392 | 1.249399 |         0.256234 |
|    31 | 0.485180 | 1.159011 |         0.235306 |
|    32 | 0.497212 | 1.165783 |         0.197064 |
|    33 | 0.509231 | 1.166318 |         0.173769 |
|    34 | 0.510593 | 1.175962 |         0.166574 |
|    35 | 0.511148 | 1.186530 |         0.160694 |
|    36 | 0.476215 | 1.287107 |         0.173898 |
|    37 | 0.497509 | 1.200780 |         0.217121 |
|    38 | 0.503388 | 1.171931 |         0.175455 |
|    39 | 0.510792 | 1.181824 |         0.157309 |
|    40 | 0.511187 | 1.191054 |         0.150430 |
|    41 | 0.519128 | 1.212880 |         0.146578 |
|    42 | 0.499360 | 1.285506 |         0.171874 |
|    43 | 0.478204 | 1.275159 |         0.240125 |
|    44 | 0.486255 | 1.236343 |         0.189908 |
|    45 | 0.496692 | 1.228916 |         0.178786 |
|    46 | 0.513026 | 1.203458 |         0.152048 |
|    47 | 0.512780 | 1.216256 |         0.145168 |
|    48 | 0.516373 | 1.219154 |         0.136789 |
|    49 | 0.518313 | 1.230155 |         0.134208 |

## Summary

- recorded validation epochs: 50
- best val/mIoU: 0.519128 at epoch 41
- last val/mIoU: 0.518313
- best val/loss: 1.017970 at epoch 9
- train/loss_epoch: first 2.251478, last 0.134208
- mean val/mIoU over last 10 epochs (40-49): 0.508414

## Baseline Comparison

- clean 10-run GatedFusion baseline mean best: 0.517397
- delta vs baseline mean: +0.001731
- clean 10-run baseline population std: 0.004901
- delta in baseline std units: +0.353
- clean 10-run baseline best single run: 0.524425
- delta vs baseline best single: -0.005297

## Comparison with FFT Freq Enhance (gamma=0.1)

- dformerv2_fft_freq_enhance g01 best: 0.522688
- delta vs freq_enhance g01: -0.003560

## Conclusion

Positive single-run signal but marginal. The best val/mIoU 0.519128 beats the clean 10-run baseline mean by +0.001731 (0.35 std), but does not beat the baseline best single run 0.524425. The last-epoch mIoU 0.518313 is very close to the best, indicating stable late training without collapse. However, this result is weaker than the original dformerv2_fft_freq_enhance (gamma=0.1) single run 0.522688. The HiLo dual-band design with alpha_low=0.03, alpha_high=0.10 did not outperform the simpler high-frequency-only freq_enhance design.

Training shows significant oscillation in val/mIoU throughout (drops at epochs 21, 29-30, 36, 42-43), suggesting the dual-band enhancement introduces more instability than the single-band design.
