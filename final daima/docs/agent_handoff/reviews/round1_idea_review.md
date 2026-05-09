# Codex Idea Review — Round 1

## Quota Status
CODEX_QUOTA_UNKNOWN

## Verdict
MODIFY

## Idea 审查
- 是否符合第五次课 4+1 主方向：是。修订后候选都属于老师给出的未试过 loss 方向，尤其 Idea 2/3 属于 contrast。
- 是否和已有 negative 结果重复：Idea 1 `rec-cov` 本质上是两个已表现不佳机制 `maskrec + freqcov` 的串联，重复风险高；Idea 2/3 的 InfoNCE 范式与现有 negative 结果不重复。
- 是否一次引入过多变量：当前选择的 Idea 2 实际又写成了 `infonce_stage_weights=0,0,1,1`，这已经接近 Idea 3。建议不要同时保留 “全四层 InfoNCE” 和 “c3/c4-only InfoNCE” 两套叙事，收敛成一个最小版本。
- 是否违反 backbone/dataset/dataloader/decoder/base_lit.py 禁改规则：按当前计划不违反。
- 是否只修改 final daima：是。
- 是否需要新 model name：是，`dformerv2_infonce` 合理；如果从一开始就限定 c3/c4-only，则没必要再单独造 `dformerv2_infonce_c34`。
- 是否有明确 smoke test：有，且已明确 smoke 与正式 run 分离，这点正确。

## 代码计划审查
- 允许修改的文件：`final daima/src/models/mid_fusion.py`、`final daima/train.py`
- 禁止修改但计划中出现的文件：无
- 参数新增是否合理：`lambda_infonce`、`infonce_proj_dim`、`infonce_tau`、`infonce_stage_weights` 合理，但首轮必须把 `proj_dim`、`tau`、stage 固定死，只验证一个机制。
- 是否存在 try/except 兜底、自动猜路径、fallback：当前计划中没有看到，保持这样。
- 参数量风险：低到中。projection head 本身很轻，真正风险在 InfoNCE 的负样本构造导致显存、算量和梯度噪声上升。
- 训练命令风险：主要风险是“同 batch 内所有其他像素位置作为负样本”这一设计可能导致每个 stage 的对比矩阵过大，尤其在 c3/c4 分辨率下会显著增加显存和时间开销。首轮必须把负样本策略写清楚并先 smoke。

## 建议
- 给 Claude 的修改建议：
  1. 在修订版三选一里，我推荐 `Idea 2`，但要按 `Idea 3` 的保守执行方式落地：只做 `c3/c4`，不要再保留“全四层版本稍后再看”的双分支计划。
  2. 不建议 `rec-cov`。它不是独立新范式，而是把两个已 weak/negative 的机制叠在一起，失败后的解释价值也很差。
  3. 当前 request 里 “选择 Idea 2” 和 “stage_weights=0,0,1,1” 实际已经说明你更相信 c3/c4-only。那就直接统一成一个 clean `dformerv2_infonce` run01，不要再额外命名 `dformerv2_infonce_c34`。
  4. 负样本策略上，不建议首轮使用“同 batch 内所有其他像素”全量负样本。如果直接全量做，复杂度和噪声都偏高。更稳的首轮是只在同一 spatial grid 上做 batch 内对比，或先对子样本化后的 token 集做对比。
  5. `tau=0.07` 可以作为首轮起点，但不要因为它是 SimCLR 默认值就默认适合这里。RGB-D encoder feature 的分布和图像 SSL 不同，首轮只把它当占位起点。
  6. `lambda_infonce=0.01` 可以接受为首轮保守值，但要预期量级不一定和 `freqcov` / `maskrec` 可比，因为 InfoNCE 数值尺度通常更大、更敏感。review 重点应放在训练是否稳定、seg loss 是否被压制，而不是先追求大权重。
  7. projection head 首轮建议纯 `1x1 conv`，不要先上 hidden layer。先证明确实有信号，再谈 MLP head。
- 是否允许进入实现：有条件允许。前提是把机制收敛成一个最小版本：`dformerv2_infonce`、c3/c4-only、纯 `1x1 conv` head、单一 `tau`、单一 `lambda`、明确受控的负样本策略。
- 是否建议换 idea：建议放弃 Idea 1，保留 InfoNCE；同时把当前 Idea 2/3 合并成一个更保守的执行方案，而不是并列两个几乎相同的候选。
