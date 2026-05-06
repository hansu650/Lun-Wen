# Experiment Commands

这个目录只放每个版本的实验命令说明。

每个稳定版本单独一个文件夹，例如：

- `version_002_stable_v2_teacher_original_encoder_dino_s_swin_t/`

规则：

- 每个版本写一个简单 `README.md`。
- 实验名写在 `--checkpoint_dir` 最后一段。
- checkpoint 保存到 `framework_download/checkpoints/<实验名>/`。
- 不在这里放权重、数据集、checkpoint 大文件。
