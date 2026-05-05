# dformerv2_dgc_af_plus_run01 mIoU Detail

- model: `dformerv2_dgc_af_plus`
- source event: `checkpoints/dformerv2_dgc_af_plus_run01/lightning_logs/version_0/events.out.tfevents.1777879824.Administrator.72520.0`
- best checkpoint: `checkpoints/dformerv2_dgc_af_plus_run01/dformerv2_dgc_af_plus-epoch=49-val/mIoU=0.5136.ckpt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- comparison baseline mean best: `0.513406`
- comparison DGC-AF Full run01 best: `0.512766`
- comparison PG-SparseComp run01 best: `0.511478`
- comparison SA-Gate five-run mean best: `0.513216`

| recorded epoch | event step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.146774 | 1.750034 |
| 2 | 793 | 0.204122 | 1.429851 |
| 3 | 1190 | 0.269562 | 1.248187 |
| 4 | 1587 | 0.320683 | 1.156784 |
| 5 | 1984 | 0.356896 | 1.115620 |
| 6 | 2381 | 0.385787 | 1.103325 |
| 7 | 2778 | 0.409021 | 1.071163 |
| 8 | 3175 | 0.438091 | 1.034468 |
| 9 | 3572 | 0.447860 | 1.058611 |
| 10 | 3969 | 0.462398 | 1.059623 |
| 11 | 4366 | 0.450928 | 1.085301 |
| 12 | 4763 | 0.467804 | 1.039806 |
| 13 | 5160 | 0.479100 | 1.041325 |
| 14 | 5557 | 0.482321 | 1.063764 |
| 15 | 5954 | 0.447310 | 1.189104 |
| 16 | 6351 | 0.458007 | 1.148798 |
| 17 | 6748 | 0.473978 | 1.154269 |
| 18 | 7145 | 0.484006 | 1.127892 |
| 19 | 7542 | 0.483179 | 1.132053 |
| 20 | 7939 | 0.482369 | 1.167466 |
| 21 | 8336 | 0.492437 | 1.140839 |
| 22 | 8733 | 0.476640 | 1.182579 |
| 23 | 9130 | 0.481295 | 1.189001 |
| 24 | 9527 | 0.467641 | 1.214452 |
| 25 | 9924 | 0.481636 | 1.165395 |
| 26 | 10321 | 0.491491 | 1.182314 |
| 27 | 10718 | 0.447872 | 1.262819 |
| 28 | 11115 | 0.486720 | 1.175457 |
| 29 | 11512 | 0.492940 | 1.207121 |
| 30 | 11909 | 0.502310 | 1.164846 |
| 31 | 12306 | 0.502521 | 1.179664 |
| 32 | 12703 | 0.507147 | 1.198095 |
| 33 | 13100 | 0.498852 | 1.215745 |
| 34 | 13497 | 0.498100 | 1.238456 |
| 35 | 13894 | 0.502809 | 1.249421 |
| 36 | 14291 | 0.510920 | 1.228323 |
| 37 | 14688 | 0.486311 | 1.234528 |
| 38 | 15085 | 0.486905 | 1.234750 |
| 39 | 15482 | 0.494991 | 1.217246 |
| 40 | 15879 | 0.499012 | 1.246348 |
| 41 | 16276 | 0.501565 | 1.253708 |
| 42 | 16673 | 0.473377 | 1.328167 |
| 43 | 17070 | 0.484310 | 1.277497 |
| 44 | 17467 | 0.508494 | 1.246683 |
| 45 | 17864 | 0.509302 | 1.228393 |
| 46 | 18261 | 0.511507 | 1.251661 |
| 47 | 18658 | 0.490755 | 1.279769 |
| 48 | 19055 | 0.506811 | 1.262890 |
| 49 | 19452 | 0.503402 | 1.277149 |
| 50 | 19849 | 0.513584 | 1.274342 |

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.513584` at recorded epoch `50`
- last val/mIoU: `0.513584`
- best val/loss: `1.034468` at recorded epoch `8`
- delta vs baseline mean best: `+0.000178`
- delta vs DGC-AF Full run01 best: `+0.000818`
- delta vs PG-SparseComp run01 best: `+0.002105`
- delta vs SA-Gate five-run mean best: `+0.000368`
- conclusion: DGC-AF++ run01 is the first recent DFormerv2-primary residual compensation run to slightly exceed the repeated baseline mean, but the margin is very small. Treat as a promising single-run result that needs repeated-run validation before being used as a stable paper claim.
