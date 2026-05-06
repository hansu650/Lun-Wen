# dformerv2_gated_coattn_res_fusion_run01 early-stopped manually

- model: `dformerv2_gated_coattn_res_fusion`
- checkpoint_dir: `checkpoints/dformerv2_gated_coattn_res_fusion_run01`
- status: manually terminated before 50 epochs
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- event_log: `C:\Users\qintian\Desktop\qintian\teacher's daima\checkpoints\dformerv2_gated_coattn_res_fusion_run01\lightning_logs\version_0\events.out.tfevents.1777861643.Administrator.61576.0`
- checkpoint files: 1
  - `C:\Users\qintian\Desktop\qintian\teacher's daima\checkpoints\dformerv2_gated_coattn_res_fusion_run01\dformerv2_gated_coattn_res_fusion-epoch=16-val\mIoU=0.4834.ckpt`
- epochs recorded: 20
- best recorded epoch: 17
- best recorded val/mIoU: 0.483357
- last recorded val/mIoU: 0.480325
- best recorded val/loss epoch: 11
- best recorded val/loss: 1.007498
- baseline dformerv2_mid_fusion mean best: 0.513406
- delta vs baseline mean: -0.030049

## val/mIoU by epoch

| epoch | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|
| 1 | 0.153763 | 1.664298 | 2.216042 |
| 2 | 0.221161 | 1.389394 | 1.562570 |
| 3 | 0.277806 | 1.209479 | 1.264261 |
| 4 | 0.328346 | 1.156587 | 1.060788 |
| 5 | 0.376147 | 1.093154 | 0.900667 |
| 6 | 0.392846 | 1.099175 | 0.778500 |
| 7 | 0.432494 | 1.051025 | 0.666290 |
| 8 | 0.449803 | 1.034858 | 0.585244 |
| 9 | 0.462704 | 1.027113 | 0.527780 |
| 10 | 0.455637 | 1.067636 | 0.483859 |
| 11 | 0.469607 | 1.007498 | 0.440926 |
| 12 | 0.463508 | 1.028753 | 0.397814 |
| 13 | 0.470531 | 1.044465 | 0.393777 |
| 14 | 0.479825 | 1.027560 | 0.349192 |
| 15 | 0.468041 | 1.084935 | 0.328128 |
| 16 | 0.467392 | 1.077845 | 0.312223 |
| 17 | 0.483357 | 1.061727 | 0.318433 |
| 18 | 0.480658 | 1.080457 | 0.283358 |
| 19 | 0.469422 | 1.092752 | 0.282470 |
| 20 | 0.480325 | 1.096710 | 0.273193 |

## Conclusion

The run was manually terminated after 20 recorded validation epochs. The best recorded val/mIoU is far below the repeated baseline mean, so this branch should not be continued in its current form.
