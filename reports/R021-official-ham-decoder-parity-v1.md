# R021 Official Ham Decoder Parity

- branch: `exp/R021-official-ham-decoder-parity-v1`
- model: `dformerv2_ham_decoder`
- run: `R021_official_ham_decoder_parity_run01`
- status: completed negative below corrected baseline
- evidence: `final daima/miou_list/R021_official_ham_decoder_parity_run01.md`

## Hypothesis

After R015/R016 align NYUDepthV2 label and depth contracts, the remaining gap toward DFormerv2-S reference performance may come from the decoder/head contract. R021 tests a self-contained c2-c4 LightHam-like decoder in place of `SimpleFPNDecoder`.

## Implementation

R021 adds a separate `dformerv2_ham_decoder` entry. It keeps DFormerv2-S, pretrained loading, the external ResNet-18 DepthEncoder, GatedFusion, label mapping, depth normalization, loader, metric, optimizer, scheduler, batch size, epoch count, learning rate, worker count, and early stopping unchanged.

The decoder uses c2/c3/c4 fused features, upsamples c3/c4 to c2, concatenates them, applies `squeeze -> NMF Hamburger -> align -> classifier`, and upsamples logits to the input resolution.

Audit caveat: the implementation aligns with the official Ham head on inputs, NMF defaults, `align_corners=False`, and BN eps/momentum, but it omits the official `BaseDecodeHead.cls_seg()` `Dropout2d(0.1)`. Therefore R021 should be described as LightHam-like, not strict official Ham parity.

## Fixed Recipe

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\nyu056-mainline\final daima"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_ham_decoder `
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
  --checkpoint_dir ".\checkpoints\R021_official_ham_decoder_parity_run01"
```

## Evidence

- full train exit code: `0`
- recorded validation epochs: `50`
- TensorBoard event: `final daima/checkpoints/R021_official_ham_decoder_parity_run01/lightning_logs/version_0/events.out.tfevents.1778777116.Administrator.12456.0`
- best checkpoint: `final daima/checkpoints/R021_official_ham_decoder_parity_run01/dformerv2_ham_decoder-epoch=38-val_mIoU=0.5274.pt`

## Result

- best val/mIoU: `0.527353` at validation epoch `39`
- last val/mIoU: `0.501377`
- last-5 mean val/mIoU: `0.503158`
- last-10 mean val/mIoU: `0.506140`
- best-to-last drop: `0.025976`
- best val/loss: `1.121119` at validation epoch `7`
- final train/loss_epoch: `0.177592`

## Comparison

- R016 corrected baseline: `0.541121`; R021 is lower by `0.013768`.
- R020 branch depth blend adapter: `0.532924`; R021 is lower by `0.005571`.
- R010 PMAD logit-only: `0.527469`; R021 is lower by `0.000116`.

## Decision

R021 is negative for the current corrected pipeline. It does not beat R016 and shows late instability/overfitting. Since one official decoder detail was missing, the next Ham-related step should be exactly one minimal R022 dropout parity fix. If that also fails to approach R016/R020, stop Ham decoder work and move to corrected-contract PMAD teacher refresh.
