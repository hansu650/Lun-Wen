# R013 LMLP Decoder Head

- branch: `exp/R013-lmlp-decoder-v1`
- model: `dformerv2_lmlp_decoder`
- checkpoint_dir: `final daima/checkpoints/dformerv2_lmlp_decoder_run01`
- TensorBoard event: `final daima/checkpoints/dformerv2_lmlp_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778682038.Administrator.26812.0`
- best checkpoint: `final daima/checkpoints/dformerv2_lmlp_decoder_run01/dformerv2_lmlp_decoder-epoch=40-val_mIoU=0.5180.pt`
- mIoU detail: `final daima/miou_list/dformerv2_lmlp_decoder_run01.md`

## Hypothesis

SimpleFPN's top-down additive decoder may be a bottleneck for the fused DFormerv2 RGB-D features. A DFormer/SegFormer-style c2-c4 LMLP head tests direct multi-scale MLP projection and concatenation without changing the encoder, fusion path, data/eval path, loss, or fixed training recipe.

## Paper/Code Basis

- SegFormer: `https://arxiv.org/abs/2105.15203`
- Official SegFormer repo: `https://github.com/NVlabs/SegFormer`
- DFormer repo: `https://github.com/VCIP-RGBD/DFormer`
- Local reference: `ref_codes/DFormer/models/decoders/LMLPDecoder.py`

## Fixed Recipe Check

- data root: `C:\Users\qintian\Desktop\qintian\data\NYUDepthv2`
- pretrained DFormerv2-S: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`
- `batch_size=2`
- `max_epochs=50`
- `lr=6e-5`
- `num_workers=4`
- `early_stop_patience=30`
- `loss_type=ce`
- `accelerator=gpu`, `devices=1`

No dataset split, dataloader, validation/test loader, augmentation, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, backbone, encoder, PMAD, or TGGA code was changed.

## Training Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"
& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_lmlp_decoder `
  --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --early_stop_patience 30 `
  --accelerator gpu `
  --devices 1 `
  --dformerv2_pretrained "C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth" `
  --loss_type ce `
  --checkpoint_dir ".\checkpoints\dformerv2_lmlp_decoder_run01"
```

## Smoke Test

- `py_compile` passed for `train.py`, `src/models/decoder.py`, and `src/models/mid_fusion.py`.
- `train.py --help` lists `dformerv2_lmlp_decoder`.
- One real NYU train batch produced logits `(2, 40, 480, 640)`, CE loss `3.821265`, and optimizer `AdamW(lr=6e-5, weight_decay=0.01)`.

## Result

- recorded validation epochs: `50`
- best val/mIoU: `0.517981` at epoch `41`
- last val/mIoU: `0.490231`
- last-5 mean val/mIoU: `0.505172`
- last-10 mean val/mIoU: `0.508065`
- best val/loss: `1.018381` at epoch `4`
- final train/loss: `0.208969`
- process status: exit code `0`; `Trainer.fit` reached `max_epochs=50`.

## Comparison

- clean 10-run baseline mean best: `0.517397`
- clean 10-run baseline std: `0.004901`
- clean 10-run baseline mean + 1 std: `0.522298`
- clean baseline best single: `0.524425`
- R010 PMAD run06_retry1 best: `0.527469`
- R004 TGGA c4-only best: `0.522849`
- delta vs clean baseline mean: `+0.000584` (`+0.119` baseline std units)
- delta vs baseline mean + 1 std: `-0.004317`
- delta vs clean baseline best single: `-0.006444`
- delta vs R010 PMAD run06_retry1: `-0.009488`
- delta vs R004 TGGA c4-only: `-0.004868`
- best-to-last delta: `-0.027750`
- gap to `0.53`: `-0.012019`

## Decision

R013 is a weak near-baseline decoder result and negative for the `0.53` goal. It does not justify promoting or repeating this exact LMLP decoder. The late drop also suggests this head does not solve the project's stability problem.

## Audit Notes

- TensorBoard event contains 50 `val/mIoU` scalar points.
- Best checkpoint filename matches the TensorBoard best to four decimals.
- Checkpoints and TensorBoard events are evidence only and must not be staged.
- `agent_workspace` run scripts/logs are temporary and must not be staged.
