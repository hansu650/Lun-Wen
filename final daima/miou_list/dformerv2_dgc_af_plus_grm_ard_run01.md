# dformerv2_dgc_af_plus_grm_ard_run01 mIoU Detail

- model: `dformerv2_dgc_af_plus_grm_ard`
- source event: `checkpoints/dformerv2_dgc_af_plus_grm_ard_run01/lightning_logs/version_0/events.out.tfevents.1777886471.Administrator.70540.0`
- best checkpoint: `checkpoints/dformerv2_dgc_af_plus_grm_ard_run01/dformerv2_dgc_af_plus_grm_ard-epoch=44-val/mIoU=0.5077.ckpt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- comparison baseline mean best: `0.513406`
- comparison DGC-AF++ run01 best: `0.513584`
- comparison DGC-AF Full run01 best: `0.512766`
- comparison PG-SparseComp run01 best: `0.511478`

| recorded epoch | event step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.121340 | 1.754442 |
| 2 | 793 | 0.187607 | 1.441613 |
| 3 | 1190 | 0.254570 | 1.279157 |
| 4 | 1587 | 0.300063 | 1.188700 |
| 5 | 1984 | 0.339016 | 1.134001 |
| 6 | 2381 | 0.384611 | 1.081419 |
| 7 | 2778 | 0.403629 | 1.091012 |
| 8 | 3175 | 0.430073 | 1.085847 |
| 9 | 3572 | 0.442409 | 1.109940 |
| 10 | 3969 | 0.470812 | 1.043510 |
| 11 | 4366 | 0.438525 | 1.120050 |
| 12 | 4763 | 0.474851 | 1.057017 |
| 13 | 5160 | 0.481893 | 1.055616 |
| 14 | 5557 | 0.477650 | 1.084576 |
| 15 | 5954 | 0.472929 | 1.115434 |
| 16 | 6351 | 0.482990 | 1.106762 |
| 17 | 6748 | 0.483783 | 1.115729 |
| 18 | 7145 | 0.490010 | 1.123615 |
| 19 | 7542 | 0.484577 | 1.148134 |
| 20 | 7939 | 0.476980 | 1.177783 |
| 21 | 8336 | 0.469394 | 1.204393 |
| 22 | 8733 | 0.482122 | 1.165059 |
| 23 | 9130 | 0.480249 | 1.173223 |
| 24 | 9527 | 0.473407 | 1.226356 |
| 25 | 9924 | 0.501369 | 1.182032 |
| 26 | 10321 | 0.500216 | 1.167609 |
| 27 | 10718 | 0.506377 | 1.180902 |
| 28 | 11115 | 0.500181 | 1.204064 |
| 29 | 11512 | 0.466269 | 1.337515 |
| 30 | 11909 | 0.487907 | 1.245619 |
| 31 | 12306 | 0.497757 | 1.227022 |
| 32 | 12703 | 0.498803 | 1.229092 |
| 33 | 13100 | 0.472091 | 1.324255 |
| 34 | 13497 | 0.445063 | 1.393375 |
| 35 | 13894 | 0.478755 | 1.246849 |
| 36 | 14291 | 0.492175 | 1.206226 |
| 37 | 14688 | 0.483231 | 1.253656 |
| 38 | 15085 | 0.503442 | 1.238971 |
| 39 | 15482 | 0.459045 | 1.362269 |
| 40 | 15879 | 0.503928 | 1.227020 |
| 41 | 16276 | 0.503560 | 1.240065 |
| 42 | 16673 | 0.497939 | 1.239988 |
| 43 | 17070 | 0.502473 | 1.231904 |
| 44 | 17467 | 0.504878 | 1.241078 |
| 45 | 17864 | 0.507743 | 1.266918 |
| 46 | 18261 | 0.504504 | 1.274146 |
| 47 | 18658 | 0.499552 | 1.272980 |
| 48 | 19055 | 0.484542 | 1.335756 |
| 49 | 19452 | 0.471298 | 1.363302 |
| 50 | 19849 | 0.501644 | 1.249933 |

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.507743` at recorded epoch `45`
- last val/mIoU: `0.501644`
- best val/loss: `1.043510` at recorded epoch `10`
- delta vs baseline mean best: `-0.005663`
- delta vs DGC-AF++ run01 best: `-0.005841`
- delta vs DGC-AF Full run01 best: `-0.005023`
- delta vs PG-SparseComp run01 best: `-0.003735`
- conclusion: negative result. Adding GRM/ARD on top of DGC-AF++ reduced performance substantially, so this exact heavier residual-control design should not replace DGC-AF++.
