# RGB-D Semantic Segmentation Experiments

This repository records an RGB-D semantic segmentation project on NYUDepthv2.
The current main line is a stable mid-fusion baseline built around pretrained
RGB and depth encoders, several fusion ablations, and reproducible experiment
summaries.

## Current Stable Version

The confirmed stable baseline is `v1.0`.

Code snapshot:

- `versions/version_001_stable_baseline_mid_fusion/`

Active project:

- `framework_download/`

Main model structure:

- RGB encoder: Swin-B, loaded from local `pretrained/swin_base`
- Depth encoder: DINOv2-B, loaded from local `pretrained/dinov2_base`
- c1: `GatedFusion`
- c2: `GatedFusion`
- c3: `DepthAwareLocalRefineFusion`
- c4: `GatedFusion + DepthPromptTokenBlock`
- decoder: `SimpleFPNDecoder`

The important code files are:

- `framework_download/src/models/encoder.py`
- `framework_download/src/models/mid_fusion.py`
- `framework_download/src/models/decoder.py`
- `framework_download/train.py`
- `framework_download/eval.py`
- `framework_download/infer.py`

## Result Summary

The most reliable result is the 10-run v1.0 baseline:

- mean validation mIoU: `0.4828`
- best validation mIoU: `0.4900`
- standard deviation: `0.0044`

Main result documents:

- `framework_download/results/swin_dino_mid_stable_baseline_10runs.md`
- `framework_download/results/swin_dino_mid_1.md`
- `framework_download/results/swin_dino_mid_c4_dformer_3runs_6results.md`
- `framework_download/results/swin_dino_mid_b2_mcads_decoder_summary.md`

Interpretation:

- Early old baseline reached about `0.3249`.
- Old mid-fusion baseline reached about `0.3744`.
- Swin-B + DINOv2-B stage-feature mid fusion raised performance to about `0.4663`.
- The stable v1.0 fusion design is reproducible around `0.48-0.49`.
- Later SA-Gate, DFormer-style c4, and decoder/head variants did not show a
  stable improvement over v1.0, so v1.0 remains the main reference baseline.

## How To Train

Run from `framework_download/`:

```powershell
python train.py --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --max_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 0 --devices 1 --accelerator gpu --checkpoint_dir ".\checkpoints\example_run"
```

Run five repeated experiments:

```powershell
cd C:\Users\qintian\Desktop\qintian\framework_download; 1..5 | ForEach-Object { $run = '{0:D2}' -f $_; python train.py --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --max_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 0 --devices 1 --accelerator gpu --checkpoint_dir ".\checkpoints\swin_dino_mid_v10_repeat_$run" }
```

## How To Evaluate And Visualize

Evaluate a checkpoint:

```powershell
python eval.py --checkpoint ".\checkpoints\RUN_NAME\CHECKPOINT.ckpt" --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --batch_size 2 --num_workers 0
```

Generate prediction visualizations:

```powershell
python infer.py --checkpoint ".\checkpoints\RUN_NAME\CHECKPOINT.ckpt" --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --num_vis 10 --save_dir ".\visualizations\RUN_NAME"
```

## What Is Not Tracked

The following files are intentionally not pushed:

- NYUDepthv2 data under `data/`
- pretrained weights under `pretrained/`
- training checkpoints under `framework_download/checkpoints/`
- visualization outputs under `framework_download/visualizations/`
- Lightning logs under `framework_download/lightning_logs/`
- local zip backups such as `framework_download/src.zip`

This keeps the GitHub repository focused on code, experiment records, and
reference notes rather than large generated artifacts.

## Reference Material

Reference papers and external code snapshots are organized by topic:

- `encoder_论文/`
- `fusion_论文/`
- `decoder_论文/`
- `老师论文/RGBD/README.md`
- `306整理/`

Some third-party repositories are included only as reference material. The
project's runnable training code lives in `framework_download/`.
