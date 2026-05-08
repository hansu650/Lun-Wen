# dformerv2_depth_fft_select_c030_run01 mIoU

- Model: `dformerv2_depth_fft_select`
- Setting: `cutoff_ratio=0.30`, depth-only FFT low/high frequency selection at c2/c3/c4

```
epoch,val/mIoU,val/loss
1,0.171203,1.639639
2,0.227809,1.357210
3,0.280382,1.277787
4,0.313773,1.173552
5,0.369829,1.087763
6,0.384670,1.128722
7,0.429062,1.060113
8,0.441161,1.061418
9,0.430230,1.101675
10,0.444357,1.076152
11,0.469476,1.025829
12,0.447976,1.167819
13,0.472524,1.058373
14,0.470953,1.078237
15,0.457010,1.111292
16,0.484351,1.065532
17,0.482387,1.061961
18,0.495158,1.080611
19,0.487893,1.084851
20,0.497599,1.073493
21,0.452764,1.161310
22,0.484336,1.120742
23,0.491227,1.115963
24,0.504667,1.083269
25,0.506076,1.101505
26,0.509864,1.080277
27,0.508247,1.115965
28,0.474708,1.193904
29,0.468521,1.204842
30,0.503418,1.152146
31,0.490320,1.190455
32,0.505365,1.172747
33,0.511487,1.152397
34,0.509454,1.178153
35,0.503166,1.176916
36,0.494621,1.198439
37,0.470100,1.304421
38,0.492197,1.211043
39,0.499020,1.216799
40,0.487826,1.244884
41,0.444970,1.532920
42,0.480108,1.282770
43,0.513871,1.197768
44,0.513498,1.209292
45,0.513441,1.236527
46,0.504636,1.252870
47,0.507341,1.245470
48,0.493978,1.269835
49,0.486476,1.288779
50,0.482797,1.290829
```

- Best val/mIoU: `0.513871` at epoch 43
- Best val/loss: `1.025829` at epoch 11
- Last val/mIoU: `0.482797`
- Mean last 10 epochs: `0.494112`
- Clean 10-run GatedFusion baseline mean best: `0.517397`
- Delta vs clean 10-run baseline mean: `-0.003526`
- Clean 10-run GatedFusion baseline std: `0.004901`
- Delta in baseline std units: `-0.719`
- Clean 10-run GatedFusion baseline best single run: `0.524425`
- Delta vs clean baseline best single run: `-0.010554`

## Checkpoint gate diagnostics

Evidence checkpoint: `checkpoints/dformerv2_depth_fft_select_c030_run01/dformerv2_depth_fft_select-epoch=42-val_mIoU=0.5139.pt`

The depth FFT selection gates stayed very close to identity. Values below are the bias-implied average selection weights, computed as `sigmoid(bias) * 2`, plus depthwise convolution weight norms.

| Stage | Branch | Avg gate from bias | Weight norm | Weight abs mean |
|---|---|---:|---:|---:|
| c2 | low | 0.997994 | 0.258338 | 0.00609283 |
| c2 | high | 1.007397 | 0.393132 | 0.00910004 |
| c3 | low | 0.993755 | 0.522824 | 0.00890125 |
| c3 | high | 1.011458 | 0.737450 | 0.01320340 |
| c4 | low | 0.993354 | 0.691974 | 0.00842417 |
| c4 | high | 1.007414 | 0.940259 | 0.01144233 |

Conclusion: negative single-run result. The run stays below the clean 10-run GatedFusion baseline mean, and the checkpoint diagnostics show that the internal depth FFT selection modules remained near identity. This supports treating `dformerv2_depth_fft_select` as a negative/low-impact encoder-internal frequency selection ablation rather than a main candidate.
