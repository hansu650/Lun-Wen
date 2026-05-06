# `swin_dino_mid_ktb_fdam_prompt_c4_1` Result Record

## Experiment Summary

- Experiment name: `swin_dino_mid_ktb_fdam_prompt_c4_1`
- Main route: `mid_fusion`
- RGB encoder: local `Swin-B` from `./pretrained/swin_base`
- Depth encoder: local `DINOv2-B` from `./pretrained/dinov2_base`
- c1/c2: original lightweight `GatedFusion`
- c3/c4: block1 `KTB-style` cross-modal fusion
- c3/c4: block2 `FDAM-style` local/frequency enhancement
- c4 only: block3 `DepthPromptTokenBlock`
- Decoder: original `SimpleFPNDecoder`

## Why This Run Exists

- The previous `swin_dino_mid_ktb_fdam_c34_1` run only inserted block1 + block2
- This run adds block3 on top of that baseline
- The goal is not to reproduce DA-VPT or DFormerv2 training systems
- The goal is to verify that a lightweight depth-prompt token block can be inserted after fusion and before the existing decoder

## Training Command

```powershell
cd C:\Users\qintian\Desktop\qintian\framework_download
python train.py --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --max_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 0 --devices 1 --accelerator gpu --checkpoint_dir ".\checkpoints\swin_dino_mid_ktb_fdam_prompt_c4_1"
```

## Best Result

- Best validation mIoU during training: `0.4741`
- Best checkpoint:

```text
C:\Users\qintian\Desktop\qintian\framework_download\checkpoints\swin_dino_mid_ktb_fdam_prompt_c4_1\mid_fusion-epoch=23-val_mIoU=0.4741.ckpt
```

## Comparison Against Previous Runs

- `swin_dino_mid_1`: `0.4663`
- `swin_dino_mid_ktb_fdam_c34_1`: `0.4664`
- `swin_dino_mid_ktb_fdam_prompt_c4_1`: `0.4741`

Improvement:

- Compared with `swin_dino_mid_1`: `+0.0078`
- Compared with `swin_dino_mid_ktb_fdam_c34_1`: `+0.0077`

This is a real gain, even though it is still smaller than the large gain that came from replacing the encoder pair with `Swin-B + DINOv2-B`.

## Evidence That Block 3 Was Really Inserted

This run is not using the original DA-VPT / DFormerv2 source code directly, so it is important to record direct evidence from the current project:

1. The current model contains `c4_prompt_block`
2. The inserted block type is `DepthPromptTokenBlock`
3. The prompt generator type is `DepthPromptGenerator`
4. Prompt grid is `(2, 4)`, so prompt count `K = 8`
5. Prompt token shape is `(B, 8, 1024)`
6. The c4 output shape after the block stays `(B, 1024, 15, 20)` for a `480x640` input
7. Total parameter count increased to about `230.5M`
8. Prompt block parameters alone are about `10.53M`

These checks confirm that block3 is not only described in notes; it is really part of the current forward graph.

## Block 3 Data Flow

Input on c4:

- `fused_feat_c4`: `(B, C, H, W)`
- `depth_feat_c4`: `(B, C, H, W)`

Flow:

1. Flatten `fused_feat_c4` into image tokens `(B, H*W, C)`
2. Build depth prior from `depth_feat_c4`
3. Pool the depth prior into a small prompt grid `(2, 4)`
4. Convert it into prompt tokens `(B, 8, C)`
5. Concatenate prompt tokens after image tokens
6. Run one lightweight token mixer (`MultiheadAttention + MLP`)
7. Remove prompt tokens
8. Reshape updated image tokens back to `(B, C, H, W)`
9. Add residual refinement and send the result to the original decoder

## Why Block 3 Was Placed Only On c4

- c4 has the smallest token count, so attention cost is much lower
- c4 carries the strongest semantic information, so depth-as-geometry-prior is easier to use there
- c1/c2 are more local and noisy, and are better left to lightweight low-level fusion in the first version
- This keeps the change small and makes the effect of the prompt block easier to isolate

## Current Interpretation

- Block1 + block2 alone produced almost no visible gain over `swin_dino_mid_1`
- Adding block3 on c4 brings the model from `0.4664` to `0.4741`
- So in the current engineering skeleton, the prompt-style c4 token block seems more useful than the previous plug-in blocks alone
- This does not prove that the current design is fully optimal, but it does show that the token/prompt direction is worth continuing

## Useful Commands

### Eval

```powershell
python eval.py --checkpoint "C:\Users\qintian\Desktop\qintian\framework_download\checkpoints\swin_dino_mid_ktb_fdam_prompt_c4_1\mid_fusion-epoch=23-val_mIoU=0.4741.ckpt" --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --batch_size 2 --num_workers 0
```

### Inference / Visualization

```powershell
python infer.py --checkpoint "C:\Users\qintian\Desktop\qintian\framework_download\checkpoints\swin_dino_mid_ktb_fdam_prompt_c4_1\mid_fusion-epoch=23-val_mIoU=0.4741.ckpt" --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --num_vis 10 --save_dir ".\visualizations\swin_dino_mid_ktb_fdam_prompt_c4_1"
```

## Current Conclusion

- Current strongest run in this project is now `swin_dino_mid_ktb_fdam_prompt_c4_1`
- Current best validation mIoU is `0.4741`
- The c4-only depth prompt token block is worth keeping as the new strongest variant before trying decoder/head changes
