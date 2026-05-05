# dformerv2_dgc_af_full_run01 mIoU Detail

- model: `dformerv2_dgc_af_full`
- source event: `checkpoints/dformerv2_dgc_af_full_run01/lightning_logs/version_0/events.out.tfevents.1777874267.Administrator.70456.0`
- best checkpoint: `checkpoints/dformerv2_dgc_af_full_run01/dformerv2_dgc_af_full-epoch=46-val/mIoU=0.5128.ckpt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- comparison baseline mean best: `0.513406`
- comparison PG-SparseComp run01 best: `0.511478`
- comparison gated co-attention run01 partial best: `0.483357`

| recorded epoch | event step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.137689 | 1.764504 |
| 2 | 793 | 0.198406 | 1.451239 |
| 3 | 1190 | 0.275634 | 1.276065 |
| 4 | 1587 | 0.308709 | 1.207725 |
| 5 | 1984 | 0.366203 | 1.109153 |
| 6 | 2381 | 0.395235 | 1.064313 |
| 7 | 2778 | 0.424447 | 1.036298 |
| 8 | 3175 | 0.443840 | 1.049360 |
| 9 | 3572 | 0.442122 | 1.063847 |
| 10 | 3969 | 0.468957 | 1.041144 |
| 11 | 4366 | 0.472574 | 1.065022 |
| 12 | 4763 | 0.453786 | 1.127577 |
| 13 | 5160 | 0.466359 | 1.082658 |
| 14 | 5557 | 0.476962 | 1.078286 |
| 15 | 5954 | 0.482687 | 1.089984 |
| 16 | 6351 | 0.493164 | 1.069240 |
| 17 | 6748 | 0.496250 | 1.089648 |
| 18 | 7145 | 0.485916 | 1.147475 |
| 19 | 7542 | 0.499353 | 1.118506 |
| 20 | 7939 | 0.450060 | 1.202946 |
| 21 | 8336 | 0.472911 | 1.146242 |
| 22 | 8733 | 0.496941 | 1.131862 |
| 23 | 9130 | 0.479013 | 1.194447 |
| 24 | 9527 | 0.501100 | 1.150337 |
| 25 | 9924 | 0.501142 | 1.146708 |
| 26 | 10321 | 0.502186 | 1.172444 |
| 27 | 10718 | 0.497120 | 1.192922 |
| 28 | 11115 | 0.473103 | 1.269523 |
| 29 | 11512 | 0.465358 | 1.260963 |
| 30 | 11909 | 0.467047 | 1.203015 |
| 31 | 12306 | 0.492174 | 1.172385 |
| 32 | 12703 | 0.503493 | 1.196949 |
| 33 | 13100 | 0.507464 | 1.191559 |
| 34 | 13497 | 0.509855 | 1.201670 |
| 35 | 13894 | 0.508007 | 1.211604 |
| 36 | 14291 | 0.501783 | 1.261717 |
| 37 | 14688 | 0.478394 | 1.381485 |
| 38 | 15085 | 0.480086 | 1.321565 |
| 39 | 15482 | 0.499497 | 1.267552 |
| 40 | 15879 | 0.508651 | 1.238564 |
| 41 | 16276 | 0.485901 | 1.301734 |
| 42 | 16673 | 0.469052 | 1.306420 |
| 43 | 17070 | 0.502848 | 1.204112 |
| 44 | 17467 | 0.473469 | 1.277863 |
| 45 | 17864 | 0.496330 | 1.243536 |
| 46 | 18261 | 0.508180 | 1.248584 |
| 47 | 18658 | 0.512766 | 1.266010 |
| 48 | 19055 | 0.502492 | 1.293911 |
| 49 | 19452 | 0.483644 | 1.414239 |
| 50 | 19849 | 0.477881 | 1.354696 |

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.512766` at recorded epoch `47`
- last val/mIoU: `0.477881`
- best val/loss: `1.036298` at recorded epoch `7`
- delta vs baseline mean best: `-0.000640`
- delta vs PG-SparseComp run01 best: `+0.001288`
- delta vs gated co-attention run01 partial best: `+0.029409`
- conclusion: DGC-AF is the strongest single run among the recent primary-preserving residual compensation variants, but it still does not beat the repeated DFormerv2 mid-fusion baseline mean. Treat as near-baseline neutral unless repeated runs improve the mean.
