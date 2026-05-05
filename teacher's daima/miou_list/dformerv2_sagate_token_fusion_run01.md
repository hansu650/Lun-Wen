# dformerv2_sagate_token_fusion_run01

- model: `dformerv2_sagate_token_fusion`
- checkpoint_dir: `checkpoints/dformerv2_sagate_token_fusion_run01`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- event_log: `C:\Users\qintian\Desktop\qintian\teacher's daima\checkpoints\dformerv2_sagate_token_fusion_run01\lightning_logs\version_0\events.out.tfevents.1777855075.Administrator.54468.0`
- epochs recorded: 50
- best epoch: 48
- best val/mIoU: 0.509558
- last val/mIoU: 0.501725
- best val/loss epoch: 7
- best val/loss: 1.048825
- baseline dformerv2_mid_fusion mean best: 0.513406
- delta vs baseline mean: -0.003848
- SA-Gate V1 five-run mean best: 0.513216
- delta vs SA-Gate V1 mean: -0.003658

## val/mIoU by epoch

| epoch | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|
| 1 | 0.148075 | 1.774693 | 2.376760 |
| 2 | 0.202298 | 1.416254 | 1.660024 |
| 3 | 0.249747 | 1.274845 | 1.325651 |
| 4 | 0.300757 | 1.184719 | 1.130410 |
| 5 | 0.347076 | 1.144144 | 0.965408 |
| 6 | 0.389491 | 1.077344 | 0.832937 |
| 7 | 0.436429 | 1.048825 | 0.726523 |
| 8 | 0.437932 | 1.049477 | 0.642599 |
| 9 | 0.456144 | 1.051350 | 0.575484 |
| 10 | 0.464805 | 1.053434 | 0.501967 |
| 11 | 0.442756 | 1.077825 | 0.472600 |
| 12 | 0.460063 | 1.075266 | 0.432112 |
| 13 | 0.449844 | 1.128012 | 0.430277 |
| 14 | 0.462349 | 1.085259 | 0.409295 |
| 15 | 0.476713 | 1.115565 | 0.362900 |
| 16 | 0.485425 | 1.095565 | 0.332303 |
| 17 | 0.491014 | 1.097917 | 0.320863 |
| 18 | 0.446489 | 1.184572 | 0.319708 |
| 19 | 0.456402 | 1.220249 | 0.339373 |
| 20 | 0.475192 | 1.163939 | 0.293131 |
| 21 | 0.490791 | 1.100717 | 0.265557 |
| 22 | 0.490424 | 1.145611 | 0.252467 |
| 23 | 0.499097 | 1.123587 | 0.245486 |
| 24 | 0.497389 | 1.140141 | 0.241232 |
| 25 | 0.495873 | 1.159009 | 0.227847 |
| 26 | 0.487115 | 1.154601 | 0.227383 |
| 27 | 0.477329 | 1.210001 | 0.227809 |
| 28 | 0.464582 | 1.213862 | 0.248764 |
| 29 | 0.493906 | 1.182216 | 0.236377 |
| 30 | 0.483982 | 1.189657 | 0.215774 |
| 31 | 0.485765 | 1.180071 | 0.222452 |
| 32 | 0.496383 | 1.187480 | 0.213373 |
| 33 | 0.500921 | 1.195275 | 0.190739 |
| 34 | 0.504017 | 1.193081 | 0.181863 |
| 35 | 0.493764 | 1.219414 | 0.181070 |
| 36 | 0.502553 | 1.228709 | 0.180866 |
| 37 | 0.500746 | 1.216097 | 0.184394 |
| 38 | 0.500232 | 1.250013 | 0.174488 |
| 39 | 0.438994 | 1.435256 | 0.181749 |
| 40 | 0.482986 | 1.230543 | 0.229589 |
| 41 | 0.499241 | 1.214357 | 0.177885 |
| 42 | 0.498281 | 1.247815 | 0.164807 |
| 43 | 0.504078 | 1.229225 | 0.159062 |
| 44 | 0.503146 | 1.262816 | 0.152888 |
| 45 | 0.474734 | 1.340888 | 0.188729 |
| 46 | 0.487913 | 1.280393 | 0.168964 |
| 47 | 0.502862 | 1.242807 | 0.156904 |
| 48 | 0.509558 | 1.246989 | 0.148239 |
| 49 | 0.500334 | 1.272742 | 0.163063 |
| 50 | 0.501725 | 1.270427 | 0.172354 |

## Conclusion

TokenFusion-style selector with primary/aux/diff input underperformed the repeated DFormerv2 mid-fusion baseline in this run. It also did not improve over SA-Gate V1. Do not continue this exact branch as a main line unless a very specific instability hypothesis is tested separately.
