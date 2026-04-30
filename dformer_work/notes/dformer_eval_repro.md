# DFormer Official Eval Reproduction Notes

Date: 2026-04-30

Purpose: reproduce the official DFormerv2-Small evaluation result on NYUDepthV2 without modifying official DFormer model code.

## Paths

- Official code: `C:\Users\qintian\Desktop\qintian\dformer_work\DFormer`
- Checkpoint root: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints`
- Dataset root: `C:\Users\qintian\Desktop\qintian\dformer_work\datasets`
- NYUDepthV2 data root used by config: `datasets\NYUDepthv2`
- DFormerv2-Small config: `local_configs\NYUDepthv2\DFormerv2_S.py`

## Local Links

These local links make the official relative paths work from `DFormer\`.

- `DFormer\checkpoints` is a junction to `..\checkpoints`
- `DFormer\datasets` is a junction to `..\datasets`
- `checkpoints\pretrained\DFormerv2_Small_pretrained.pth` is a hard link to the downloaded pretrained file.
- `checkpoints\trained\DFormerv2_Small_NYU.pth` is a hard link to the downloaded NYU trained checkpoint.

Original downloaded files remain under:

- `checkpoints\pretrained\DFormerv2\pretrained\DFormerv2_Small_pretrained.pth`
- `checkpoints\trained\NYUDepthv2\DFormerv2\NYU\DFormerv2_Small_NYU.pth`

## Dataset Format Check

The current `datasets\NYUDepthv2` layout matches the official DFormer loader expectation:

- `RGB\*.jpg`
- `Depth\*.png`
- `Label\*.png`
- `train.txt`
- `test.txt`

Observed counts on 2026-04-30:

- `train.txt`: 795 lines
- `test.txt`: 654 lines
- `RGB`: 1449 files
- `Depth`: 1449 files
- `Label`: 1449 files

The split files use entries like `RGB/2.jpg Label/2.png`; the official loader derives `Depth/2.png` from the RGB stem.

## Clean Environment Plan

Use a dedicated `dformer` conda environment. Do not install DFormer dependencies into `qintian-rgbd`.

The local PyTorch wheels shown by the user are `cp311`, so the clean environment should use Python 3.11:

```bat
conda create -n dformer python=3.11 -y
conda activate dformer

cd /d C:\Users\qintian\Downloads\torch_whl
pip install torch-2.7.1+cu126-cp311-cp311-win_amd64.whl
pip install torchvision-0.22.1+cu126-cp311-cp311-win_amd64.whl
pip install torchaudio-2.7.1+cu126-cp311-cp311-win_amd64.whl

pip install timm opencv-python scipy tensorboardX tabulate easydict ftfy regex mmengine mmcv-lite
```

Verify:

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

cd /d C:\Users\qintian\Desktop\qintian\dformer_work\DFormer
python -c "import torch, cv2, timm, easydict, tensorboardX, mmengine, tabulate, mmcv; print('ok')"
```

## Single-GPU Windows Eval Command

Run from `C:\Users\qintian\Desktop\qintian\dformer_work\DFormer`.

PowerShell:

```powershell
cd C:\Users\qintian\Desktop\qintian\dformer_work\DFormer

$env:CUDA_VISIBLE_DEVICES="0"
$env:LOCAL_RANK="0"
$env:PYTHONPATH="C:\Users\qintian\Desktop\qintian\dformer_work\DFormer"

python utils/eval.py `
  --config=local_configs.NYUDepthv2.DFormerv2_S `
  --gpus=1 `
  --sliding `
  --no-compile `
  --syncbn `
  --mst `
  --compile_mode="reduce-overhead" `
  --amp `
  --pad_SUNRGBD `
  --continue_fpath="checkpoints/trained/DFormerv2_Small_NYU.pth"
```

cmd or Anaconda Prompt:

```bat
cd /d C:\Users\qintian\Desktop\qintian\dformer_work\DFormer

set CUDA_VISIBLE_DEVICES=0
set LOCAL_RANK=0
set PYTHONPATH=C:\Users\qintian\Desktop\qintian\dformer_work\DFormer

python utils\eval.py --config=local_configs.NYUDepthv2.DFormerv2_S --gpus=1 --sliding --no-compile --syncbn --mst --compile_mode=reduce-overhead --amp --pad_SUNRGBD --continue_fpath=checkpoints\trained\DFormerv2_Small_NYU.pth
```

Do not edit official `eval.sh` for the Windows single-GPU run.

On Windows, the messages about `rm` and `ln` not being recognized come from log-link helper commands. They do not prevent the evaluation metric from being computed.

## Clean Environment Evidence

The official eval was repeated in the clean `dformer` conda environment on 2026-04-30.

- Environment: `dformer`, Python 3.11, torch `2.7.1+cu126`, RTX 4090.
- Command style: direct single-GPU `python utils\eval.py`, not `torchrun`.
- Result: `miou:50.55`, `macc:65.32`, `mf1:64.95`
- Evidence log: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\NYUDepthv2_DFormerv2_S_20260430-214400\log_2026_04_30_21_44_00.log`

## Evidence From Temporary Validation Run

A temporary validation run was completed on 2026-04-30 before restoring `qintian-rgbd` to its previous dependency state.

- Environment at run time: `qintian-rgbd`, Python 3.11, torch `2.7.1+cu126`, RTX 4090, with transient DFormer dependencies installed.
- Command style: direct single-GPU `python utils\eval.py`, not `torchrun`.
- Result: `miou:50.52`, `macc:65.31`, `mf1:64.92`
- Evidence log: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\NYUDepthv2_DFormerv2_S_20260430-212149\log_2026_04_30_21_21_49.log`

This confirms the paths, checkpoint, dataset, and single-GPU command can work. The clean `dformer` environment result above is the final official baseline reproduction record.

## Environment Cleanup

The following transient packages were removed from `qintian-rgbd` on 2026-04-30:

- `easydict`
- `tensorboardX`
- `mmengine`
- `tabulate`
- `mmcv-lite`
- `ftfy`
- `addict`
- `termcolor`
- `yapf`
- `platformdirs`
- `wcwidth`

Post-cleanup check showed `qintian-rgbd` still has torch `2.7.1+cu126`, CUDA available, one RTX 4090, and cv2 `4.13.0`.
