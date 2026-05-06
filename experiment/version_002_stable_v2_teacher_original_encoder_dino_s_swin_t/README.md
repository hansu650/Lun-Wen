# v2.0 Experiment

当前版本：

- RGB branch: DINOv2-small
- Depth branch: Swin-Tiny
- Fusion: GatedFusion x4
- Decoder: SimpleFPNDecoder
- 训练入口保持老师原版结构

## 运行一次实验

先进入项目目录：

```powershell
cd C:\Users\qintian\Desktop\qintian\framework_download
```

然后运行：

```powershell
D:\Anaconda\envs\qintian-rgbd\python.exe train.py --model mid_fusion --data_root C:\Users\qintian\Desktop\qintian\data\NYUDepthv2 --max_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 0 --accelerator gpu --devices 1 --checkpoint_dir .\checkpoints\v2_dino_s_swin_t_run02
```

## 常用参数

- `--checkpoint_dir`：实验名称写在最后，例如 `.\checkpoints\v2_dino_s_swin_t_run02`。
- `--max_epochs`：训练轮数，现在默认写 `50`。
- `--batch_size`：batch size，现在建议先用 `2`，更稳。
- `--lr`：学习率，例如 `1e-4` 或 `5e-5`。
- `--num_workers`：DataLoader workers，现在先用 `0`。
- `--devices`：GPU 数量，现在用 `1`。
- `--accelerator`：现在用 `gpu`。

改实验名时只改 `--checkpoint_dir`：

```powershell
D:\Anaconda\envs\qintian-rgbd\python.exe train.py --model mid_fusion --data_root C:\Users\qintian\Desktop\qintian\data\NYUDepthv2 --max_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 0 --accelerator gpu --devices 1 --checkpoint_dir .\checkpoints\v2_dino_s_swin_t_run03
```

改学习率时同步改实验名，方便之后看结果：

```powershell
D:\Anaconda\envs\qintian-rgbd\python.exe train.py --model mid_fusion --data_root C:\Users\qintian\Desktop\qintian\data\NYUDepthv2 --max_epochs 50 --batch_size 2 --lr 5e-5 --num_workers 0 --accelerator gpu --devices 1 --checkpoint_dir .\checkpoints\v2_dino_s_swin_t_lr5e-5_run01
```
