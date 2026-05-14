# dformerv2_lmlp_decoder_run01

- run_id: `R013-lmlp-decoder-v1`
- model: `dformerv2_lmlp_decoder`
- branch: `exp/R013-lmlp-decoder-v1`
- purpose: test whether a DFormer/SegFormer-style c2-c4 LMLP decoder head improves fused DFormerv2 RGB-D features over the active SimpleFPN decoder.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + LMLPDecoder`
- decoder: c2/c3/c4 MLP projections to `embed_dim=768`, upsample c3/c4 to c2 resolution, concatenate, `1x1` fuse + BN + ReLU + dropout + classifier, then upsample logits to input size.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.517981` at epoch `41`
- last val/mIoU: `0.490231`
- last-5 mean val/mIoU: `0.505172`
- last-10 mean val/mIoU: `0.508065`
- best val/loss: `1.018381` at epoch `4`
- final train/loss: `0.208969`
- checkpoint: `checkpoints/dformerv2_lmlp_decoder_run01/dformerv2_lmlp_decoder-epoch=40-val_mIoU=0.5180.pt`
- TensorBoard event: `checkpoints/dformerv2_lmlp_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778682038.Administrator.26812.0`
- process note: training completed with exit code `0`; `Trainer.fit` stopped because `max_epochs=50` was reached.

## Comparison

- clean 10-run baseline mean best: `0.517397`
- clean 10-run baseline std: `0.004901`
- clean 10-run baseline mean + 1 std: `0.522298`
- clean 10-run baseline best single: `0.524425`
- R010 PMAD run06_retry1 best: `0.527469`
- R004 TGGA c4-only best: `0.522849`
- delta vs clean baseline mean: `+0.000584` (`+0.119` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.004317`
- delta vs clean baseline best single: `-0.006444`
- delta vs R010 run06_retry1: `-0.009488`
- delta vs R004 TGGA c4-only: `-0.004868`
- best-to-last delta: `-0.027750`
- gap to `0.53` goal: `-0.012019`

## Epoch Metrics

| Epoch | Step | val/mIoU | val/loss | train/loss |
|---:|---:|---:|---:|---:|
| 1 | 396 | 0.237852 | 1.285832 | 1.774698 |
| 2 | 793 | 0.336791 | 1.118210 | 1.159701 |
| 3 | 1190 | 0.392144 | 1.086920 | 0.927939 |
| 4 | 1587 | 0.436638 | 1.018381 | 0.756856 |
| 5 | 1984 | 0.437892 | 1.027971 | 0.633162 |
| 6 | 2381 | 0.455448 | 1.041125 | 0.564703 |
| 7 | 2778 | 0.461157 | 1.039378 | 0.502576 |
| 8 | 3175 | 0.464076 | 1.088486 | 0.450765 |
| 9 | 3572 | 0.477323 | 1.063894 | 0.404784 |
| 10 | 3969 | 0.464359 | 1.070307 | 0.374382 |
| 11 | 4366 | 0.468476 | 1.102613 | 0.342947 |
| 12 | 4763 | 0.463729 | 1.137102 | 0.322888 |
| 13 | 5160 | 0.489178 | 1.101310 | 0.327798 |
| 14 | 5557 | 0.489254 | 1.120829 | 0.292199 |
| 15 | 5954 | 0.482675 | 1.135821 | 0.309610 |
| 16 | 6351 | 0.501238 | 1.081037 | 0.267533 |
| 17 | 6748 | 0.489661 | 1.167839 | 0.249736 |
| 18 | 7145 | 0.499932 | 1.138346 | 0.249325 |
| 19 | 7542 | 0.504679 | 1.135040 | 0.236837 |
| 20 | 7939 | 0.488258 | 1.168159 | 0.232000 |
| 21 | 8336 | 0.498928 | 1.201114 | 0.235046 |
| 22 | 8733 | 0.470799 | 1.248604 | 0.283391 |
| 23 | 9130 | 0.496572 | 1.167058 | 0.242950 |
| 24 | 9527 | 0.501139 | 1.229668 | 0.211650 |
| 25 | 9924 | 0.470849 | 1.271493 | 0.200619 |
| 26 | 10321 | 0.474824 | 1.270171 | 0.241769 |
| 27 | 10718 | 0.479844 | 1.271069 | 0.206409 |
| 28 | 11115 | 0.500071 | 1.207022 | 0.205379 |
| 29 | 11512 | 0.498655 | 1.217003 | 0.208836 |
| 30 | 11909 | 0.510887 | 1.212486 | 0.199159 |
| 31 | 12306 | 0.509517 | 1.240574 | 0.176016 |
| 32 | 12703 | 0.476946 | 1.379783 | 0.182866 |
| 33 | 13100 | 0.500535 | 1.244991 | 0.212968 |
| 34 | 13497 | 0.512875 | 1.243845 | 0.179781 |
| 35 | 13894 | 0.517070 | 1.238681 | 0.165884 |
| 36 | 14291 | 0.512211 | 1.280378 | 0.160578 |
| 37 | 14688 | 0.492355 | 1.311512 | 0.200429 |
| 38 | 15085 | 0.504349 | 1.282518 | 0.190170 |
| 39 | 15482 | 0.506239 | 1.274692 | 0.167601 |
| 40 | 15879 | 0.517822 | 1.284879 | 0.154650 |
| 41 | 16276 | 0.517981 | 1.273575 | 0.149103 |
| 42 | 16673 | 0.516174 | 1.331184 | 0.147756 |
| 43 | 17070 | 0.494203 | 1.315316 | 0.166350 |
| 44 | 17467 | 0.511442 | 1.304499 | 0.162189 |
| 45 | 17864 | 0.514988 | 1.318625 | 0.145624 |
| 46 | 18261 | 0.514999 | 1.332272 | 0.144647 |
| 47 | 18658 | 0.505583 | 1.380261 | 0.144592 |
| 48 | 19055 | 0.506552 | 1.397090 | 0.141403 |
| 49 | 19452 | 0.508496 | 1.424749 | 0.142934 |
| 50 | 19849 | 0.490231 | 1.376081 | 0.208969 |

## Conclusion

R013 is a weak near-baseline decoder result and not a goal path. The best val/mIoU `0.517981` is only `+0.000584` above the clean baseline mean and below baseline mean + 1 std, R004 c4-only TGGA, R010 PMAD run06_retry1, and the required `0.53`. The late drop to `0.490231` suggests the LMLP head does not solve the project's stability or peak-performance bottleneck.
