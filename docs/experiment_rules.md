# Experiment Rules

Goal: raise NYUDepthV2 validation mIoU to `>= 0.53`.

Current setup phase:

- Only orchestration files may be created or edited.
- No full training.
- No model-code changes.
- No training-parameter changes.

Later experiment phase:

- Every accepted experiment must run a full train.
- Every experiment must test exactly one main hypothesis.
- Every experiment must use a feature branch or independent worktree.
- Every experiment must write a report under `reports/`.
- Every experiment must update `metrics/runs.jsonl` and `metrics/leaderboard.csv`.
- Negative and failed experiments must still be reported.

Evidence:

- Use TensorBoard event logs or explicit checkpoint-backed logs.
- Do not claim improvement from terminal snippets alone.
- Do not claim success unless `val/mIoU >= 0.53` and the forbidden-change audit passes.

Windows notes:

- Quote paths containing spaces, especially `"final daima"`.
- Use PowerShell commands by default.
- Do not run concurrent full trains on the same local GPU.
