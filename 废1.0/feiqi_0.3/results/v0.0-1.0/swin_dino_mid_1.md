# `swin_dino_mid_1` Result Record

## Experiment Summary

- Experiment name: `swin_dino_mid_1`
- Fusion mode: `mid_fusion`
- RGB encoder: local `Swin-B` from `./pretrained/swin_base`
- Depth encoder: local `DINOv2-B` from `./pretrained/dinov2_base`
- Depth input handling: first copy 1-channel depth to 3 channels, then feed into DINOv2
- Decoder: existing `SimpleFPNDecoder`
- Training goal: keep the original engineering skeleton and only upgrade the encoders

## Why We Keep `mid_fusion`

- Earlier experiments in this project showed that `mid_fusion` was clearly stronger than `early fusion`
- For this round we intentionally kept the fusion path fixed and only replaced the encoder pair
- This makes the gain easier to attribute to `Swin-B + DINOv2-B`

## Training Command

```powershell
cd C:\Users\qintian\Desktop\qintian\framework_download
python train.py --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --max_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 0 --devices 1 --accelerator gpu --checkpoint_dir ".\checkpoints\swin_dino_mid_1"
```

## Best Result

- Best validation mIoU during training: `0.4663`
- Best validation PixelAcc from eval: `0.7119`
- Best checkpoint (renamed to carry the real score):

```text
C:\Users\qintian\Desktop\qintian\framework_download\checkpoints\swin_dino_mid_1\mid_fusion-epoch=22-val_mIoU=0.4663.ckpt
```

## Result Interpretation

- This run is much stronger than the old encoder version of `mid_fusion`
- It also confirms that in the current project setting, encoder upgrade alone can bring a large gain
- At this stage we continue to use `mid_fusion` as the main route, not `early fusion`

## Problems Encountered

### 1. Checkpoint filename showed `val_mIoU=0.0000`

Reason:

- Training originally logged `val/mIoU`
- Checkpoint filename formatting used `val_mIoU`
- The displayed metric and the filename metric name did not match

Fix:

- Log both `val/mIoU` and `val_mIoU`
- Let checkpoint monitor `val_mIoU`
- This avoids Windows path issues and keeps the score visible in the filename

### 2. `eval.py` originally gave only about `0.15` mIoU

Reason:

- Training used a full-validation confusion matrix and then computed global mIoU
- Old `eval.py` computed mIoU batch by batch and then averaged the batch scores
- The two statistics were not the same, so the eval result looked much lower

Fix:

- Rewrite `eval.py` to accumulate one confusion matrix over the whole validation set
- After the fix, eval matches training:

```text
mIoU=0.4663, PixelAcc=0.7119
```

### 3. `SwinModel LOAD REPORT` showed `classifier.weight` and `classifier.bias` as `UNEXPECTED`

Reason:

- The local Swin checkpoint is a classification-style pretrained model
- The segmentation encoder only needs the backbone, not the classification head

Conclusion:

- This warning can be ignored for the current use case

## Useful Commands

### Eval

```powershell
python eval.py --checkpoint "C:\Users\qintian\Desktop\qintian\framework_download\checkpoints\swin_dino_mid_1\mid_fusion-epoch=22-val_mIoU=0.4663.ckpt" --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --batch_size 2 --num_workers 0
```

### Inference / Visualization

```powershell
python infer.py --checkpoint "C:\Users\qintian\Desktop\qintian\framework_download\checkpoints\swin_dino_mid_1\mid_fusion-epoch=22-val_mIoU=0.4663.ckpt" --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --num_vis 10 --save_dir ".\visualizations\swin_dino_mid_1"
```

## Current Conclusion

- Current strongest usable run in this project: `swin_dino_mid_1`
- Recommended main line going forward: continue with `mid_fusion`
- If we make the next structural change, we should compare against this run instead of older baselines
