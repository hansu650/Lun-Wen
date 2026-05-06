# dformerv2_pg_sparse_comp_fusion_run01 mIoU Detail

- model: `dformerv2_pg_sparse_comp_fusion`
- source event: `checkpoints/dformerv2_pg_sparse_comp_fusion_run01/lightning_logs/version_0/events.out.tfevents.1777868589.Administrator.61364.0`
- best checkpoint: `checkpoints/dformerv2_pg_sparse_comp_fusion_run01/dformerv2_pg_sparse_comp_fusion-epoch=43-val/mIoU=0.5115.ckpt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- comparison baseline mean best: `0.513406`

| recorded epoch | event step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.133359 | 1.797739 |
| 2 | 793 | 0.197566 | 1.436347 |
| 3 | 1190 | 0.239875 | 1.287642 |
| 4 | 1587 | 0.288865 | 1.239696 |
| 5 | 1984 | 0.324898 | 1.154359 |
| 6 | 2381 | 0.370500 | 1.107104 |
| 7 | 2778 | 0.399081 | 1.039683 |
| 8 | 3175 | 0.441072 | 1.051733 |
| 9 | 3572 | 0.457668 | 1.041010 |
| 10 | 3969 | 0.465458 | 1.028691 |
| 11 | 4366 | 0.459157 | 1.045949 |
| 12 | 4763 | 0.471785 | 1.032781 |
| 13 | 5160 | 0.474901 | 1.072483 |
| 14 | 5557 | 0.464340 | 1.083591 |
| 15 | 5954 | 0.463359 | 1.122132 |
| 16 | 6351 | 0.478190 | 1.089921 |
| 17 | 6748 | 0.489735 | 1.101168 |
| 18 | 7145 | 0.497226 | 1.096078 |
| 19 | 7542 | 0.500495 | 1.113296 |
| 20 | 7939 | 0.474808 | 1.204950 |
| 21 | 8336 | 0.492944 | 1.136706 |
| 22 | 8733 | 0.464919 | 1.208025 |
| 23 | 9130 | 0.480627 | 1.163003 |
| 24 | 9527 | 0.495901 | 1.140576 |
| 25 | 9924 | 0.469510 | 1.234713 |
| 26 | 10321 | 0.500674 | 1.136849 |
| 27 | 10718 | 0.470366 | 1.224426 |
| 28 | 11115 | 0.503745 | 1.167511 |
| 29 | 11512 | 0.496306 | 1.207765 |
| 30 | 11909 | 0.504879 | 1.173022 |
| 31 | 12306 | 0.497839 | 1.199052 |
| 32 | 12703 | 0.501246 | 1.214654 |
| 33 | 13100 | 0.444079 | 1.295445 |
| 34 | 13497 | 0.491393 | 1.165547 |
| 35 | 13894 | 0.497727 | 1.176997 |
| 36 | 14291 | 0.488335 | 1.221517 |
| 37 | 14688 | 0.504220 | 1.207440 |
| 38 | 15085 | 0.504223 | 1.218955 |
| 39 | 15482 | 0.508860 | 1.201370 |
| 40 | 15879 | 0.500892 | 1.223466 |
| 41 | 16276 | 0.491681 | 1.251265 |
| 42 | 16673 | 0.497104 | 1.256751 |
| 43 | 17070 | 0.503466 | 1.246970 |
| 44 | 17467 | 0.511478 | 1.257681 |
| 45 | 17864 | 0.498037 | 1.275478 |
| 46 | 18261 | 0.508004 | 1.268305 |
| 47 | 18658 | 0.509999 | 1.240740 |
| 48 | 19055 | 0.496718 | 1.300335 |
| 49 | 19452 | 0.495338 | 1.301982 |
| 50 | 19849 | 0.471010 | 1.335675 |

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.511478` at recorded epoch `44`
- last val/mIoU: `0.471010`
- best val/loss: `1.028691` at recorded epoch `10`
- delta vs baseline mean best: `-0.001928`
- conclusion: close to the repeated DFormerv2 mid-fusion baseline, but not an improvement in this run. Treat as a near-baseline negative/neutral result unless repeated runs show a higher mean.
