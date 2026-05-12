# Prompt For Pro: TGGA No-Aux Run01 Discussion

```text
你作为严谨的 RGB-D semantic segmentation 实验设计顾问，帮我判断 TGGA no-aux 诊断结果，以及下一步是否还值得继续 TGGA。

项目背景：
- 数据集：NYUDepthV2 40-class semantic segmentation。
- Clean baseline：dformerv2_mid_fusion。
- Clean 10-run baseline mean best val/mIoU = 0.517397，population std = 0.004901，mean + 1 std = 0.522298，best single = 0.524425。
- PMAD logit-only w0.15/T4：5-run mean best val/mIoU = 0.520795。

原 TGGA 模型：
- model = dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2
- 结构：DFormerv2_S + TGGA(c3,c4) + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder。
- loss：CE(final_logits, label) + 0.03 * CE(aux_c3, label) + 0.03 * CE(aux_c4, label)。
- Run01：best 0.522206 at epoch 48，last 0.489865，last10 mean 0.510627。
- Run02：best 0.517437 at epoch 49，last 0.486566，last10 mean 0.501329。
- 两 run mean best = 0.519822，比 baseline mean 高 +0.002425，但低于 PMAD mean，且两次 late collapse。
- 原判断：weak positive but unstable，不能 claim stable improvement。

新 no-aux 诊断：
- model = dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1
- 结构：仍然 TGGA(c3,c4) 插在 DFormerV2 c3/c4 后。
- loss：只保留 CE(final_logits, label)，移除 aux CE。
- 注意：因为原版 detach-semantic 如果去掉 aux CE 会让 aux head 变成随机固定 cue，所以 no-aux 版本使用 semantic-gradient gating：semantic cue 不 detach，让 final CE 通过 gate 路径训练 semantic cue。

No-aux run01 结果：
- best val/mIoU = 0.512152 at epoch 48。
- last val/mIoU = 0.492633。
- best val/loss = 1.032364 at epoch 15。
- last val/loss = 1.275699。
- last10 mean = 0.501366。
- last5 mean = 0.498370。
- post-best mean = 0.481050。
- best-to-last drop = 0.019520。
- largest epoch drop = -0.053409 from epoch 37 to 38。
- epoch41-50 val/mIoU =
  0.495870, 0.503853, 0.504099, 0.508151, 0.509840,
  0.510360, 0.507238, 0.512152, 0.469468, 0.492633
- final beta_c3 = 0.035324，beta_c4 = 0.025326。
- final gate_c3_mean = 0.474472，gate_c4_mean = 0.230513。
- final gate_c3_std = 0.311781，gate_c4_std = 0.135297。
- final diagnostic aux CE c3 = 3.880328，c4 = 3.911476。
- delta vs baseline mean = -0.005245，约 -1.070 baseline std。
- delta vs PMAD mean = -0.008643。
- delta vs original TGGA aux run01 = -0.010054。
- delta vs original TGGA aux run02 = -0.005285。
- epochs above baseline mean = 0/50。

请回答：
1. 这个 no-aux 结果应该如何定性？是否足以说明 aux CE 不是唯一不稳定源？
2. 原 TGGA 的高 peak 是否主要依赖 aux CE？no-aux 是否说明 TGGA gate/residual 本体没有足够有效 signal？
3. final gate_c3_mean=0.474、gate_c4_mean=0.231、c4 gate std=0.135 说明什么？
4. 下一步只允许推荐一个实验：应该跑 weak-c3、c4-only、aux schedule、还是停止 TGGA 转向 PMAD/保底路线？
5. 给保守论文策略：TGGA 这条线现在最多能怎么写？什么绝对不能写？

请用中文回答，风格要严谨保守，不要为了“继续做”而继续做。
```
