# R022 Ham Dropout Parity Run01 mIoU

- branch: `exp/R022-ham-dropout-parity-v1`
- model: `dformerv2_ham_decoder`
- run: `R022_ham_dropout_parity_run01`
- hypothesis: R021 may underperform because it omitted official `BaseDecodeHead.cls_seg()` `Dropout2d(0.1)`; add only that dropout before the Ham classifier.
- fixed recipe: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, `AdamW(weight_decay=0.01)`.
- checkpoint dir: `checkpoints/R022_ham_dropout_parity_run01`
- TensorBoard event: `checkpoints/R022_ham_dropout_parity_run01/lightning_logs/version_0/events.out.tfevents.1778782418.Administrator.36360.0`
- best checkpoint: `checkpoints/R022_ham_dropout_parity_run01/dformerv2_ham_decoder-epoch=49-val_mIoU=0.5343.pt`
- exit code: `0`
- recorded validation epochs: `50`

## Summary

- best val/mIoU: `0.534332` at validation epoch `50`
- last val/mIoU: `0.534332`
- last-5 mean val/mIoU: `0.527687`
- last-10 mean val/mIoU: `0.512629`
- best-to-last drop: `0.000000`
- best val/loss: `1.106345` at validation epoch `21`
- final train/loss_epoch: `0.059158`
- comparison to R021: `+0.006979`
- comparison to R020: `+0.001408`
- comparison to R016 corrected baseline: `-0.006790`

## Per-Epoch Metrics

| epoch | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|
| 1 | 0.188680 | 1.598136 | 2.167334 |
| 2 | 0.279509 | 1.338915 | 1.426115 |
| 3 | 0.335900 | 1.265078 | 1.104432 |
| 4 | 0.376687 | 1.203551 | 0.884779 |
| 5 | 0.388948 | 1.205813 | 0.735177 |
| 6 | 0.421214 | 1.154702 | 0.621312 |
| 7 | 0.435615 | 1.120134 | 0.524354 |
| 8 | 0.450910 | 1.120132 | 0.446876 |
| 9 | 0.450504 | 1.207253 | 0.398107 |
| 10 | 0.458924 | 1.148026 | 0.393512 |
| 11 | 0.463619 | 1.155629 | 0.335206 |
| 12 | 0.462192 | 1.161822 | 0.288485 |
| 13 | 0.479690 | 1.126648 | 0.283307 |
| 14 | 0.466550 | 1.186893 | 0.258972 |
| 15 | 0.480721 | 1.150859 | 0.245326 |
| 16 | 0.487445 | 1.120476 | 0.224106 |
| 17 | 0.495290 | 1.129263 | 0.204236 |
| 18 | 0.482371 | 1.122579 | 0.224014 |
| 19 | 0.470553 | 1.236612 | 0.198005 |
| 20 | 0.485540 | 1.153303 | 0.210742 |
| 21 | 0.501543 | 1.106345 | 0.176825 |
| 22 | 0.503651 | 1.124963 | 0.156186 |
| 23 | 0.508970 | 1.132117 | 0.141643 |
| 24 | 0.502065 | 1.198251 | 0.130692 |
| 25 | 0.506355 | 1.194493 | 0.124633 |
| 26 | 0.508779 | 1.181991 | 0.128899 |
| 27 | 0.513570 | 1.185369 | 0.129461 |
| 28 | 0.489138 | 1.210781 | 0.174957 |
| 29 | 0.496063 | 1.263070 | 0.147326 |
| 30 | 0.507997 | 1.181566 | 0.146272 |
| 31 | 0.506044 | 1.214014 | 0.137675 |
| 32 | 0.516278 | 1.189696 | 0.105611 |
| 33 | 0.521363 | 1.185783 | 0.096691 |
| 34 | 0.528624 | 1.162326 | 0.088405 |
| 35 | 0.510874 | 1.242851 | 0.085201 |
| 36 | 0.513720 | 1.241338 | 0.093767 |
| 37 | 0.515103 | 1.227862 | 0.084889 |
| 38 | 0.512581 | 1.247793 | 0.089152 |
| 39 | 0.516235 | 1.253372 | 0.080209 |
| 40 | 0.525611 | 1.249354 | 0.081537 |
| 41 | 0.524717 | 1.254573 | 0.073825 |
| 42 | 0.475998 | 1.355188 | 0.086530 |
| 43 | 0.464345 | 1.345981 | 0.174395 |
| 44 | 0.503353 | 1.265158 | 0.137724 |
| 45 | 0.519444 | 1.252790 | 0.081845 |
| 46 | 0.522386 | 1.231069 | 0.081215 |
| 47 | 0.524322 | 1.238373 | 0.063857 |
| 48 | 0.531649 | 1.226704 | 0.061596 |
| 49 | 0.525749 | 1.221233 | 0.064059 |
| 50 | 0.534332 | 1.245189 | 0.059158 |

## Decision

R022 is a partial-positive parity fix. Adding official classifier dropout improves R021 by `+0.006979` and makes the Ham decoder path the strongest retained method variant, but it remains below the R016 corrected baseline by `0.006790`. Do not stop the loop; next highest decision-value step is the corrected-contract geometry-primary teacher refresh.
