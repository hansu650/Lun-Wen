# Claude Status

## Current Round

- round: post-ARIS manual iteration (ARIS round 1 completed, auto-iteration paused)
- current_stage: **结果验证与方向评估 — 已完成**
- ARIS 状态: round1 已完成，未进入 round2

## Baseline

- model: `dformerv2_mid_fusion` (DFormerv2_S + DepthEncoder + GatedFusion + SimpleFPNDecoder)
- clean 10-run mean best val/mIoU: **0.517397**
- population std: 0.004901
- best single run: **0.524425** (run05)
- 训练设置: batch_size=2, max_epochs=50, lr=6e-5, early_stop_patience=30

## 已完成实验结果汇总

### 融合架构方向（全部负结果）

| 模型 | Run数 | Mean Best | vs Baseline |
|---|---|---|---|
| dformerv2_sagate_fusion | 5 | 0.513216 | -0.000190 |
| dformerv2_sagate_token_fusion | 1 | 0.509558 | -0.003848 |
| dformerv2_gated_coattn_res_fusion | 1(partial) | 0.483357 | -0.030049 |
| dformerv2_pg_sparse_comp_fusion | 1 | 0.511478 | -0.001928 |
| dformerv2_dgc_af_full | 1 | 0.512766 | -0.000640 |
| dformerv2_dgc_af_plus | 4 | 0.511418 | -0.001988 |
| dformerv2_dgc_af_plus_grm_ard | 1 | 0.507743 | -0.005663 |
| dformerv2_dgc_af_plus_csg | 1 | 0.506402 | -0.007004 |
| dformerv2_guided_depth_comp_fusion | 5 | 0.511379 | -0.002027 |
| dformerv2_guided_depth_adapter_simple | 6 | 0.512316 | -0.001090 |

### 辅助损失方向（全部负结果，已归档至 feiqi/losses/）

| 模型 | Best | vs Baseline Mean |
|---|---|---|
| dformerv2_ms_freqcov (7组sweep) | 0.520539 | sweep mean -0.001700 |
| dformerv2_feat_maskrec_c34 | 0.515327 | -0.002070 |
| dformerv2_cm_infonce | 0.514461 | -0.002936 |

### FFT 频率增强方向（全部无效）

| 模型 | Run数 | Best | 3-run Mean | 判定 |
|---|---|---|---|---|
| dformerv2_fft_freq_enhance g=0.1 | 3 | 0.522688 | 0.517696 | **run01是离群值，不成立** |
| dformerv2_fft_hilo_enhance | 1 | 0.519128 | — | 弱正，不优于freq_enhance |
| dformerv2_fft_freq_enhance g=0.2 | 1 | 0.515696 | — | 负 |
| dformerv2_depth_fft_select | 1 | 0.513871 | — | 负 |

## 关键结论

1. **所有测试过的方向均未超越 baseline mean**。GatedFusion baseline 对此架构已接近最优。
2. FFT freq_enhance run01 的 0.522688 是高方差离群值（3-run mean = 0.517696，仅 +0.06 std）。
3. 融合架构替换类（10个模型）全部负结果，复杂度越高结果越差。
4. 辅助损失类（3个方向）全部负结果，已归档。
5. FFT inference-path 类（4个配置）全部无效。

## 活跃模型

`train.py` MODEL_REGISTRY 中 6 个：
- `early`, `mid_fusion`, `dformerv2_mid_fusion`（基线）
- `dformerv2_depth_fft_select`（负结果，保留记录）
- `dformerv2_fft_freq_enhance`（3-run 验证不成立，保留记录）
- `dformerv2_fft_hilo_enhance`（弱正，不优于 freq_enhance）

## 下一步（待用户决定）

方向已基本穷尽。需要用户重新评估：
- 是否尝试全新方向（如数据增强、loss 设计、学习率调度等非架构改动）
- 是否接受当前 baseline 作为最终结果
- 是否转向论文写作（整理已有消融实验作为 negative ablation）

## 不能忘记的限制

1. **不推 checkpoints 到 GitHub** — 只推 code/miou_list/docs
2. **不用 git restore** — 只手动编辑删除
3. **结果直接写入文件** — 不用先问用户确认
4. **训练完成后自动读 TensorBoard、写文档、推送** — 不用问
5. **不叠 auxiliary loss** — 已全部负结果并归档
6. **不改 fusion 架构** — 已全部负结果
7. **GPU 只有 1 张** — 训练必须串行
8. **Python 环境: qintian-rgbd** — 所有训练/推理用此环境
9. **Pretrained 路径**: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`

## 关键文件位置

- 实验日志: `docs/experiment_log.md`
- 模型变更记录: `docs/model_changes.md`
- 论文笔记: `docs/paper_notes.md`
- mIoU 列表: `miou_list/`
- 归档代码: `feiqi/` (fusion blocks), `feiqi/losses/` (auxiliary losses)
- 活跃模型代码: `src/models/mid_fusion.py`, `src/models/fft_hilo_enhance.py`, `src/models/freq_enhance.py`, `src/models/depth_fft_select.py`
- 训练脚本: `train.py`
- GitHub: `git@github.com:hansu650/Lun-Wen.git`
