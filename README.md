# RGB-D Semantic Segmentation Experiments

This repository records an RGB-D semantic segmentation project on NYUDepthV2.
The current usable main line is a lightweight dual-branch model with multi-level
RGB-D mid-fusion.

## Current Main Line

- Task: RGB-D semantic segmentation
- Dataset: NYUDepthV2
- Active project: `framework_download/`
- RGB branch: DINOv2-small
- Depth branch: Swin-Tiny
- Fusion: multi-level RGB-D mid-fusion
- Current improvement focus: Context-FPN, ResGamma, Depth Adapter, and fusion
  modules

## Current Confirmed Best Result

The current confirmed best result is:

- Experiment: Context-FPN ResGamma 7 runs
- Best validation mIoU: about `0.3933`

Only results that were actually run in the current environment with clear
configuration, logs, and checkpoints should be treated as valid experiment
results. Deprecated or invalid records are tracked in `docs/model_changes.md`
and `docs/experiment_log.md`.

## Important Code Files

- `framework_download/src/models/encoder.py`
- `framework_download/src/models/mid_fusion.py`
- `framework_download/src/models/decoder.py`
- `framework_download/train.py`
- `framework_download/eval.py`
- `framework_download/infer.py`

## How To Train

Run from `framework_download/`:

```powershell
python train.py --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --max_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 0 --devices 1 --accelerator gpu --checkpoint_dir ".\checkpoints\example_run"
```

Run seven repeated experiments by changing the last segment of
`--checkpoint_dir`:

```powershell
cd C:\Users\qintian\Desktop\qintian\framework_download
1..7 | ForEach-Object { $run = '{0:D2}' -f $_; python train.py --model mid_fusion --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --max_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 0 --devices 1 --accelerator gpu --checkpoint_dir ".\checkpoints\context_fpn_resgamma_repeat_$run" }
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

## Experiment Documentation

- `docs/experiment_log.md`: experiment configuration, mIoU, conclusions,
  failed attempts, and next steps
- `docs/model_changes.md`: architecture changes, module decisions, and
  deprecated attempts
- `docs/paper_notes.md`: paper motivation, related work, and writing notes

## What Is Not Tracked

The following files are intentionally not pushed:

- NYUDepthV2 data under `data/`
- pretrained weights under `pretrained/`
- training checkpoints under `framework_download/checkpoints/`
- visualization outputs under `framework_download/visualizations/`
- Lightning logs under `framework_download/lightning_logs/`
- local zip backups such as `framework_download/src.zip`

This keeps the GitHub repository focused on code, experiment records, and
reference notes rather than large generated artifacts.

## Reference Material

Reference papers and external code snapshots are organized by topic:

- `fusion论文/`
- `decoder论文/`
- `head 论文/`
- `老师论文/RGBD/README.md`
- `306整理/`

Some third-party repositories are included only as reference material. The
project's runnable training code lives in `framework_download/`.
