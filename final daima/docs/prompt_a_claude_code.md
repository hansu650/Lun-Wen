# RGB-D 语义分割自动迭代研究 — Claude Code 主控提示词

## 【安全规则，优先级最高】

1. 每轮最多跑 1 个 50 epoch 实验，同时只能跑 1 个 GPU 实验。
2. 所有训练命令必须使用 Windows CMD 可执行格式，不要使用 bash 反斜杠换行。
3. 实现前必须检查 train.py 是否已有目标 argparse 参数。如果没有，只允许新增必要的 argparse 参数，并只传给新 model。
4. 不要修改 backbone、dataset、dataloader、decoder、base_lit.py。
5. 不要把 single-run positive 写成 stable improvement。
6. 删除操作必须先问，其他操作直接执行。
7. 每轮流程：实现代码 → smoke test → 本地 commit（不 push）→ 50 epoch 训练 → 训练成功后才 push；训练失败只记录，不 push。只推 final daima 下的代码/miou_list/docs，不推 checkpoints/weights/data/ref_codes/笔记/。
8. ref_codes/、笔记/、data/、pretrained/ 只读，不允许修改、不允许推送。

---

## 模式：目标驱动持续自动迭代

当前模式是目标驱动持续自动迭代，不设置固定轮数上限。

### 主目标

在第五次课 4+1 大方向内，自动搜索能提高 NYUDepthV2 val/mIoU 的轻量改进。以提升 mIoU 为核心目标，但不能追随机高点。

当前 baseline：dformerv2_mid_fusion clean 10-run mean best mIoU = 0.517397，std = 0.004901，best single = 0.524425。

### 主方向

以第五次课 4+1 为主方向：
- 优先围绕方向 1：映射、对齐、筛选。
- 可以结合 +1 辅助 loss，如 alignment loss、InfoNCE、轻量 contrastive loss。
- 可以微微发散到 frequency / reliability / selective fusion，但必须和 4+1 主方向有明确关系。
- 不要完全跳出 RGB-D 中间融合和轻量辅助约束。
- 不要改 backbone、dataset、dataloader、decoder。

---

## 文件托管交接协议

Claude 和 Codex 通过以下共享目录交接信息，不需要人工复制粘贴。

### 协作目录

```
final daima/docs/agent_handoff/
├── requests/          # Claude 写 request 到这里
├── reviews/           # Codex 写 review 到这里
├── status/            # Claude 和 Codex 写状态文件
└── archive/           # 已完成的 request/review 归档（不删除原始记录）
```

如果目录不存在，允许创建。

### 文件命名规范

每一轮 Round N 使用固定文件名：

**Claude 负责写：**
- `requests/roundN_idea_request.md` — idea 提案
- `requests/roundN_result_request.md` — 训练结果
- `status/claude_status.md` — Claude 当前状态

**Codex 负责写：**
- `reviews/roundN_idea_review.md` — idea 审查
- `reviews/roundN_result_review.md` — 结果审查
- `status/codex_status.md` — Codex 当前状态

不要使用模糊的 roundN.md。必须区分 idea_request、idea_review、result_request、result_review。

### Claude 写 claude_status.md

每轮开始前，Claude 必须更新 `status/claude_status.md`，内容必须包含：

```markdown
# Claude Status

- current_round: N
- current_stage: Exploration / Validation
- current_model_name: xxx
- current_run_name: xxx
- last_action: (描述上一步做了什么)
- last_train_result: (best mIoU / 失败原因)
- codex_review_status: (ok / pending / quota_unknown / stop)
- whether_waiting_for_codex: yes / no
- whether_safe_to_continue: yes / no
```

### Claude 写 idea_request

每轮提出 idea 后，Claude 必须写 `requests/roundN_idea_request.md`，必须包含：

```markdown
# Claude Idea Request — Round N

## Round 编号
N

## 当前阶段
Exploration / Validation

## 3 个候选 idea
1. ...
2. ...
3. ...

## 最终选择
选择的 idea：...
选择理由：...

## 与第五次课 4+1 方向的关系
...

## 与已有结果的关系
与 negative 结果的关系：...
与 promising 结果的关系：...

## 代码修改计划
- 修改哪些文件：...
- 新增哪些类/函数：...
- 是否新增 model name：yes / no
- 是否修改 backbone/dataset/dataloader/decoder/base_lit.py：NO

## Smoke test 计划
- train.py --help 检查新参数：...
- MODEL_REGISTRY 识别检查：...
- 1 epoch forward/backward 测试：...

## 训练命令草案
(Windows CMD 格式)

## 希望 Codex 审查的问题
1. ...
2. ...

## ALLOW_CODE_CHANGE_BY_CODEX
no
```

### Claude 读取 idea_review

写完 idea_request 后，Claude 必须检查是否存在 `reviews/roundN_idea_review.md`。

如果 review 文件存在：
- 必须读取
- 如果 Codex 建议 STOP，必须停止
- 如果 Codex 建议 MODIFY，必须先修改计划再实现
- 如果 Codex 建议 CONTINUE，可以继续实现

如果 review 文件不存在：
- Claude 可以 self-review 并继续
- 但必须在所有记录中写：`Codex review: pending` 和 `pending file: reviews/roundN_idea_review.md`

### Claude 写 result_request

每轮训练完成后，Claude 必须写 `requests/roundN_result_request.md`，必须包含：

```markdown
# Claude Result Request — Round N

## 基本信息
- model name: ...
- run name: ...
- full training command: (Windows CMD 格式)

## 训练结果
- best val/mIoU: ...
- best epoch: ...
- last val/mIoU: ...
- delta vs baseline mean (0.517397): ...
- delta vs baseline std (0.004901): ...
- delta vs baseline best single (0.524425): ...

## 证据
- TensorBoard event 提取证据: (逐 epoch val/mIoU)
- miou_list 文件路径: ...
- experiment_log 更新摘要: ...

## Claude 自己的结果判断
NEGATIVE / WEAK_SINGLE_RUN / PROMISING_SINGLE_RUN / VALIDATION_RESULT

## paper_notes 拟写表述
...

## 希望 Codex 审查的问题
1. ...
2. ...
```

### Claude 读取 result_review 并最终更新 paper_notes.md

写完 result_request 后，Claude 必须检查是否存在 `reviews/roundN_result_review.md`。

如果 result_review 存在：
- 必须先读取 Codex 对 paper_notes 拟写表述的审查意见
- 然后才将最终措辞写入 paper_notes.md（如 Codex 认为措辞过强，必须改弱后再写入）
- 如果 Codex 输出 STOP_DUE_TO_CODEX_QUOTA，必须停止自动迭代
- 如果 Codex 输出 STOP_DUE_TO_RESULT_RISK，必须停止并总结

如果 result_review 不存在：
- 可以自行更新 paper_notes.md，但必须在记录中标注 "Codex result review: pending"

如果 result_review 不存在：
- 可以继续下一步
- 但必须记录：`Codex result review: pending`

### Claude 如何处理 Codex 停止信号

每轮正式实现前必须检查 Codex idea review。如果出现以下任意信号，必须停止或修改：

- `STOP_DUE_TO_CODEX_QUOTA`：立即停止自动迭代
- `STOP`：停止并总结
- `MODIFY`：修改计划后重新写 request，不直接训练
- 不允许进入实现：不得实现
- 违反禁改规则：不得实现

每轮写 paper_notes 前必须检查 Codex result review。如果出现以下任意信号：

- `MODIFY_PAPER_NOTES`：必须按 Codex 建议改弱
- `STOP_DUE_TO_RESULT_RISK`：停止并总结
- 认为 single-run 被过度 claim：必须改成 weaker wording
- 认为需要 repeat：进入 Validation，不继续乱换 idea

---

## 当前项目状态

- 工作区：C:\Users\qintian\Desktop\qintian
- Active project：C:\Users\qintian\Desktop\qintian\final daima
- Baseline model：dformerv2_mid_fusion（DFormerV2_S + DepthEncoder + GatedFusion + SimpleFPNDecoder）
- Clean 10-run baseline mean best val/mIoU：0.517397
- Baseline population std：0.004901
- Baseline best single run：0.524425
- 训练设置：batch_size=2, max_epochs=50, lr=6e-5, AdamW(weight_decay=0.01), early_stop_patience=30
- 预训练权重：C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth
- 数据集：NYUDepthV2（795 train / 654 test，40 类）

## 已探索方向及结论

| 方向 | 结果 | 状态 |
|---|---|---|
| freqcov 辅助 loss（7 个设置 sweep） | sweep mean 0.515697 < baseline，best single 0.520539 | 有限信号，不 stable |
| maskrec 辅助 loss（c3+c4） | best 0.515327 < baseline mean | negative |
| FFT freq enhance（cutoff=0.25, gamma=0.1） | best 0.522688 > baseline mean | promising single-run，未 repeat |
| FFT freq enhance（gamma=0.2） | best 0.515696 < baseline mean | negative |
| depth FFT select（内部选择） | best 0.513871 < baseline mean | negative |
| DGC-AF++ / Full++ / SA-Gate 等 fusion 变体 | 多次 negative | 已 deprecated |

## 第五次课 4+1 方向（未充分探索的部分）

1. 映射、对齐、筛选 — 加 projection head + alignment loss。尚未实现。
2. Encoder 频域增强 — 已部分实现（FFT freq enhance 有 positive signal）。
3. RGB 高频与 Depth 几何比较 — 尚未实现，类似 residual 思路多次 negative。
4. Depth 频域增强 — 已实现，negative。
5. +1 辅助 Loss — freqcov/maskrec 已做；InfoNCE 对比 loss 尚未实现。

## 初始主方向

首选：Cross-modal Feature Alignment Loss（方向 1 的轻量化实现）

设计：
- 4 个 stage 各加 RGB/Depth projection head（1×1 conv → align_proj_dim 维共享空间）
- 计算 stage-wise cosine alignment loss：L_align_i = spatial_mean(1 - cosine_sim(Z_r_i, Z_d_i))
- 总 loss：L_total = L_seg + lambda_align * sum(w_i * L_align_i) / sum(w_i)
- 默认：lambda_align=0.01, align_proj_dim=64, align_stage_weights=0,0,1,1
- 推理路径不变
- 注册为新 model name：dformerv2_cross_align

备选方向（如果 alignment negative，按顺序尝试）：
1. InfoNCE 对比 loss（投影 + 对比学习，不需 reconstruction head）
2. FFT freq enhance 精细 sweep（利用已有 positive signal，测 gamma=0.05/0.075, cutoff=0.20/0.30）

---

## 排列组合规则

允许在 4+1 大方向内自由排列组合或轻微发散，但每轮必须控制变量：

1. 每轮最多引入 1 个主要新机制。
2. 每轮最多调 1 组关键超参。
3. 不允许一次同时新增 alignment + InfoNCE + FFT + gate 四种东西。
4. 如果要组合两个机制，必须满足以下至少一条：
   - 至少一个机制已有 non-negative 或 promising signal；
   - 或者组合有明确互补逻辑；
   - 并且必须在记录里写清楚组合理由。
5. 每个新 idea 必须注册成新 model name 或清晰参数开关，不覆盖 baseline。

---

## 结果策略：Exploration / Validation 两阶段

### Exploration 阶段

目标是快速寻找可能超过 baseline mean 的候选。

- best_mIoU < 0.517397 → NEGATIVE：记录，换方向或换关键超参。
- 0.517397 ≤ best_mIoU < 0.522000 → WEAK_SINGLE_RUN：可以围绕该方向微调一次，但不能说 stable。
- best_mIoU ≥ 0.522000 → PROMISING_SINGLE_RUN：进入 validation 阶段。

### Validation 阶段

不再继续乱换新 idea。对同一 setting 跑 run02/run03。

- 如果 repeat mean > 0.517397 且接近或超过 baseline mean + 0.004901（1σ），可以作为 potential improvement。
- 如果 repeat mean 没有超过 baseline mean，回到 exploration。
- 没有 3-run 或更多重复前，不能说 stable improvement。

---

## 硬性规则（必须严格遵守）

### 每轮流程

1. 读取当前实验状态（experiment_log.md, paper_notes.md, miou_list/）
2. 更新 `status/claude_status.md`
3. 检查上一轮 Codex review 状态（例如当前 Round 2 检查 `reviews/round1_result_review.md`）：
   - 如果 `reviews/round{N-1}_result_review.md` 存在且含 STOP_DUE_TO_CODEX_QUOTA，立即停止。
   - 如果连续 2 轮没有 Codex review（只有 self-review），停止并等待人工。
   - 如果 Codex review pending 只有 1 次，可以继续，但必须在记录里标注。
4. 提出最多 3 个 idea，自评后选 1 个
5. 写 `requests/roundN_idea_request.md`
6. 检查是否存在 `reviews/roundN_idea_review.md`：
   - 存在：读取，按 Codex 建议处理（STOP/MODIFY/CONTINUE）
   - 不存在：self-review 并继续，记录 "Codex review: pending"
7. 实现代码（最多改 3 个文件，必须新建 model name）
8. Smoke test：
   - 检查 `train.py --help` 是否包含新增参数
   - 检查新 model name 能被 MODEL_REGISTRY 识别
   - 跑 1 个 epoch 或最小可行训练，确认 forward/backward 不报错
   - smoke test 结果和正式 run 分开保存
9. Smoke test 通过后，git add 具体改动的文件 + 本地 commit（不 push）。commit message: `round{N}: {model_name} - {简述}`
10. 跑 1 个 50 epoch 正式实验（同时只能跑 1 个 GPU 实验）
11. 训练完成后提取 TensorBoard event 里的逐 epoch val/mIoU
12. 在 miou_list/ 下生成该实验的 .md 文件
13. 更新 experiment_log.md
14. 写 `requests/roundN_result_request.md`（其中包含 paper_notes 拟写表述，供 Codex 审查措辞）
15. 检查是否存在 `reviews/roundN_result_review.md`：
    - 存在：读取，按 Codex 建议处理（改弱措辞 / 停止 / 继续）
    - 不存在：记录 "Codex result review: pending"
16. 更新 paper_notes.md（如有 Codex 反馈则按其建议调整措辞后再写入）和 model_changes.md
17. 判断结果：
    - 如果训练成功：git push
    - 如果训练失败：只记录失败原因，不 push
18. 根据结果策略决定下一步（exploration 继续换方向 / validation 跑 repeat）

### Git 推送规则

**可以推的文件：**
- final daima/src/models/ 下的模型代码
- final daima/train.py
- final daima/miou_list/*.md
- final daima/docs/*.md（包括 agent_handoff/ 下的文件）

**只读，不允许修改、不允许推送：**
- ref_codes/（参考论文代码）
- 笔记/（课程笔记）
- data/（数据集）
- pretrained/（预训练权重）

**不要推的文件：**
- final daima/checkpoints/（已在 .gitignore）
- *.pth、*.ckpt、*.pt（模型权重）
- .agents/、.aris/、.claude/

### 代码限制
- 可改文件：final daima/src/models/mid_fusion.py、final daima/src/models/ 下新文件、final daima/train.py（仅添加 MODEL_REGISTRY 条目和 build_model 分支）
- 不可改：backbone（dformerv2_encoder.py）、dataset/dataloader（data_module.py）、decoder（decoder.py）、base_lit.py 的核心训练逻辑
- 只读目录：ref_codes/、data/、pretrained/、笔记/（不允许修改、不允许推送）
- 不要写兜底代码、不要 try/except 兜底、不要自动猜路径
- 实现前必须检查 train.py 是否已有目标 argparse 参数，如果没有才新增

### Codex 代码修改权限

Codex 默认只允许写 `reviews/` 和 `status/` 目录。除非 Claude 的 idea_request 中明确写 `ALLOW_CODE_CHANGE_BY_CODEX: yes` 并同时指定唯一允许修改的文件和唯一允许完成的任务，否则 Codex 永远只做 reviewer。

### 停止条件（任一触发即停止自动迭代并总结）

1. Codex 或 Claude 明确提示额度接近上限、不可继续，停止。
2. Codex reviewer 无法继续工作，且 Claude 已连续 2 轮只能 self-review，停止并等待人工。
3. GPU 不可用，或训练无法启动，停止。
4. 同一轮代码报错超过 2 次仍无法修复，停止。
5. 连续 3 个正式实验 best_mIoU < 0.517397，停止并总结失败方向。
6. 连续 4 个正式实验没有刷新当前 best_mIoU，停止并总结平台期。
7. 磁盘空间不足、checkpoint 写入失败、TensorBoard event 无法读取，停止。
8. 用户手动要求停止，立即停止。
9. 同时只能跑 1 个 GPU 实验，发现已有训练进程正在跑时，不启动新训练。

### 不可以做的事
- 不要把 single-run positive 写成 stable improvement
- 不要追随机高点
- 不要重复已确认 negative 的方向（除非有新变量）
- 不要同时改 backbone + fusion + loss
- 不要同时跑多个 GPU 实验

## 训练命令模板（Windows CMD 格式）

```
cd /d "C:\Users\qintian\Desktop\qintian\final daima" && python train.py --model dformerv2_cross_align --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --dformerv2_pretrained "C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth" --batch_size 2 --max_epochs 50 --lr 6e-5 --num_workers 4 --early_stop_patience 30 --checkpoint_dir ".\checkpoints\dformerv2_cross_align_run01" --lambda_align 0.01 --align_proj_dim 64 --align_stage_weights "0,0,1,1"
```

## 提取 mIoU 的方法

```
python -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; ea = EventAccumulator('<checkpoint_dir>'); ea.Reload(); miou_events = ea.Scalars('val/mIoU'); [print(f'epoch={e.step}, val/mIoU={e.value:.6f}') for e in miou_events]"
```

## 记录格式

### miou_list/{model}_{run}.md 必须包含
- model name, run name
- 所有超参数
- 每个 epoch 的 val/mIoU
- best val/mIoU 及对应 epoch
- last val/mIoU
- delta vs baseline mean, delta vs baseline best single
- 结论（negative/weak/promising）

### experiment_log.md 必须包含
- 完整配置
- 结果摘要
- 与 baseline 的对比
- 结论和下一步

### paper_notes.md 必须包含
- 该实验的 paper boundary
- 措辞限制

---

## 现在开始

请先阅读以下文件确认当前状态：
1. final daima/docs/experiment_log.md
2. final daima/docs/paper_notes.md
3. final daima/docs/model_changes.md
4. final daima/miou_list/ 下最新的几个文件
5. final daima/docs/agent_handoff/status/codex_status.md（如果存在）
6. final daima/docs/agent_handoff/reviews/ 下所有文件（如果存在）

然后更新 claude_status.md，提出 Round 1 的 3 个 idea，写 round1_idea_request.md，检查是否有 round1_idea_review.md，开始实现。
