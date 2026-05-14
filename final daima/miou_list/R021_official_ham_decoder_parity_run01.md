# R021 official Ham decoder parity run01 mIoU

- branch: `exp/R021-official-ham-decoder-parity-v1`
- model: `dformerv2_ham_decoder`
- run: `R021_official_ham_decoder_parity_run01`
- hypothesis: after R015/R016 align label and depth contracts, the remaining gap to DFormerv2-S reference performance may come from the decoder/head contract; test a self-contained c2-c4 LightHam-like decoder instead of SimpleFPNDecoder.
- important audit note: the implemented decoder matches c2/c3/c4 inputs, NMF defaults, `align_corners=False`, and BN eps/momentum, but it omits official `BaseDecodeHead.cls_seg()` `Dropout2d(0.1)`. Therefore this run is LightHam-like rather than strict official Ham parity.
- fixed recipe: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, `AdamW(weight_decay=0.01)`.
- checkpoint dir: `checkpoints/R021_official_ham_decoder_parity_run01`
- TensorBoard event: `checkpoints/R021_official_ham_decoder_parity_run01/lightning_logs/version_0/events.out.tfevents.1778777116.Administrator.12456.0`
- best checkpoint: `checkpoints/R021_official_ham_decoder_parity_run01/dformerv2_ham_decoder-epoch=38-val_mIoU=0.5274.pt`
- exit code: `0`
- recorded validation epochs: `50`

## Summary

- best val/mIoU: `0.527353` at validation epoch `39`
- last val/mIoU: `0.501377`
- last-5 mean val/mIoU: `0.503158`
- last-10 mean val/mIoU: `0.506140`
- best-to-last drop: `0.025976`
- best val/loss: `1.121119` at validation epoch `7`
- final train/loss_epoch: `0.177592`
- comparison to R016 corrected baseline: `-0.013768`
- comparison to R020: `-0.005571`
- comparison to R010 PMAD logit-only: `-0.000116`

## Per-Epoch Metrics

| epoch | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|
| 1 | 0.182980 | 1.640220 | 2.151835 |
| 2 | 0.273507 | 1.387113 | 1.416262 |
| 3 | 0.327667 | 1.248575 | 1.080948 |
| 4 | 0.360401 | 1.218090 | 0.845319 |
| 5 | 0.384952 | 1.242744 | 0.704987 |
| 6 | 0.407790 | 1.156899 | 0.593056 |
| 7 | 0.442230 | 1.121119 | 0.501960 |
| 8 | 0.449830 | 1.132221 | 0.421275 |
| 9 | 0.433992 | 1.165958 | 0.374812 |
| 10 | 0.446982 | 1.123686 | 0.328084 |
| 11 | 0.425961 | 1.226269 | 0.314948 |
| 12 | 0.467944 | 1.128881 | 0.287605 |
| 13 | 0.477089 | 1.131229 | 0.268071 |
| 14 | 0.480246 | 1.135337 | 0.229131 |
| 15 | 0.476901 | 1.137985 | 0.207487 |
| 16 | 0.496356 | 1.125877 | 0.203677 |
| 17 | 0.468188 | 1.226879 | 0.201391 |
| 18 | 0.480846 | 1.204569 | 0.223869 |
| 19 | 0.494244 | 1.152534 | 0.193280 |
| 20 | 0.491112 | 1.124324 | 0.161675 |
| 21 | 0.498226 | 1.151809 | 0.153367 |
| 22 | 0.499157 | 1.156109 | 0.143830 |
| 23 | 0.494242 | 1.181663 | 0.144131 |
| 24 | 0.491720 | 1.200408 | 0.135395 |
| 25 | 0.490376 | 1.186324 | 0.173073 |
| 26 | 0.514476 | 1.152694 | 0.121152 |
| 27 | 0.499191 | 1.203987 | 0.120210 |
| 28 | 0.509605 | 1.156079 | 0.119848 |
| 29 | 0.510593 | 1.166909 | 0.106682 |
| 30 | 0.519888 | 1.162255 | 0.095578 |
| 31 | 0.525046 | 1.129859 | 0.087648 |
| 32 | 0.517577 | 1.186478 | 0.091941 |
| 33 | 0.480920 | 1.225184 | 0.216412 |
| 34 | 0.495025 | 1.238642 | 0.130697 |
| 35 | 0.513611 | 1.177512 | 0.102356 |
| 36 | 0.511273 | 1.184434 | 0.083118 |
| 37 | 0.523064 | 1.193856 | 0.076760 |
| 38 | 0.523028 | 1.216623 | 0.078356 |
| 39 | 0.527353 | 1.172570 | 0.072032 |
| 40 | 0.517514 | 1.216467 | 0.071457 |
| 41 | 0.490809 | 1.332187 | 0.075106 |
| 42 | 0.503957 | 1.265406 | 0.148813 |
| 43 | 0.510088 | 1.250828 | 0.090396 |
| 44 | 0.519097 | 1.224871 | 0.077172 |
| 45 | 0.521663 | 1.202442 | 0.066112 |
| 46 | 0.512950 | 1.237750 | 0.069116 |
| 47 | 0.521707 | 1.243237 | 0.061741 |
| 48 | 0.523929 | 1.220772 | 0.057776 |
| 49 | 0.455828 | 1.484175 | 0.081253 |
| 50 | 0.501377 | 1.266909 | 0.177592 |

## Decision

R021 is a negative decoder result relative to the corrected baseline. It peaks below R016 and shows late instability/overfitting. Because the implementation omitted official `Dropout2d(0.1)` before classification, a single minimal R022 parity fix is justified before retiring the Ham direction.
