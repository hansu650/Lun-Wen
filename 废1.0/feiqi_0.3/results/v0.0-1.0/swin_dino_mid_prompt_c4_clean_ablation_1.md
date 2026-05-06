# swin_dino_mid_prompt_c4_clean_ablation_1

## Experiment Goal

Run a clean ablation where the main path is simplified as much as possible:

- c1/c2/c3/c4: all return to the same simple `GatedFusion`
- c4 only: keep `DepthPromptTokenBlock`
- decoder: unchanged `SimpleFPNDecoder`
- encoder: unchanged `Swin-B + DINOv2-B`
- train / eval / infer flow: unchanged

The only experimental variable is:

- replace the previous stacked high-level modules with
- `simple fusion everywhere + c4 prompt block only`

---

## Model Structure Used in This Run

### c1 / c2 / c3

- `rgb_feat + depth_feat -> GatedFusion -> fused_feat`

### c4

- `rgb_feat + depth_feat -> GatedFusion -> fused_feat_c4`
- `fused_feat_c4 + depth_feat_c4 -> DepthPromptTokenBlock -> enhanced_feat_c4`
- `enhanced_feat_c4 -> SimpleFPNDecoder`

---

## Checkpoint / Result

- Run name: `swin_dino_mid_prompt_c4_clean_ablation_1`
- Best checkpoint:
  `C:\Users\qintian\Desktop\qintian\framework_download\checkpoints\swin_dino_mid_prompt_c4_clean_ablation_1\mid_fusion-epoch=36-val_mIoU=0.4807.ckpt`
- Best val/mIoU: `0.4807`

---

## Comparison to Previous Runs

- `swin_dino_mid_1`: `0.4663`
- `swin_dino_mid_ktb_fdam_c34_1`: `0.4664`
- `swin_dino_mid_ktb_fdam_prompt_c4_1`: `0.4741`
- `swin_dino_mid_prompt_c4_clean_ablation_1`: `0.4807`

### Gain

- vs `swin_dino_mid_1`: `+0.0144`
- vs `swin_dino_mid_ktb_fdam_prompt_c4_1`: `+0.0066`

---

## Interpretation

This run suggests three things:

1. The strong encoder upgrade (`Swin-B + DINOv2-B`) is still the main source of gain.
2. Stacking extra KTB/FDAM-style blocks in the main path did not help much in this project skeleton.
3. A cleaner structure works better:
   - keep the main path simple
   - keep only the c4 prompt-token refinement
   - let the decoder stay unchanged

In other words:

`simple fusion backbone path + c4 depth prompt refinement`

is currently stronger than:

`simple fusion + KTB + FDAM + prompt`

for this codebase.

---

## Why This Run Is Important

This is a cleaner ablation than the previous prompt run because:

- no KTB block in the active main path
- no FDAM block in the active main path
- no decoder change
- no encoder change
- no train-strategy change

So the result is easier to interpret:

the gain is mainly attributable to the `DepthPromptTokenBlock` on c4.

---

## Runtime Sanity Check

The current model still includes the prompt block in the forward path:

- `has_c4_prompt_block = True`
- `c4_prompt_block_type = DepthPromptTokenBlock`
- `num_simple_fusions = 4`
- `num_prompts = 8`

This confirms the clean ablation is not a no-op:

- all four stages still fuse
- only c4 receives the extra token/prompt refinement

---

## Next Practical Direction

If continuing this line, the most reasonable next step is:

- keep the clean main path
- keep c4 prompt block
- try a stronger decoder/head

rather than re-introducing heavy c3/c4 plug-in blocks into the main path.
