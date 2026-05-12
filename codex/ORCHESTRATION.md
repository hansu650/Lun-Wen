# Codex RGB-D Experiment Orchestration

This document defines the Goal-Driven experiment loop for the local Windows workspace at `C:\Users\qintian\Desktop\qintian`.

Current phase: architecture and file initialization only. Do not train, edit model code, edit training parameters, edit data augmentation, or change evaluation behavior in this phase.

## Goal

Raise NYUDepthV2 validation mIoU to `>= 0.53`.

## Success Criteria

- A completed full training run reports `val/mIoU >= 0.53`.
- The result is backed by TensorBoard event logs and a checkpoint path.
- The experiment did not modify dataset split, eval metric, mIoU calculation, validation/test loader, data augmentation, optimizer, scheduler, batch size, epoch count, learning rate, or other fixed training recipe parameters.
- No checkpoint, dataset, pretrained weight, or large log file is committed.
- The run has a report under `reports/` and records in `metrics/runs.jsonl` and `metrics/leaderboard.csv`.
- A reviewer and reproducer have signed off.

## Fixed Active Project

- Repo root: `C:\Users\qintian\Desktop\qintian`
- Active code: `C:\Users\qintian\Desktop\qintian\final daima`
- Training entry: `final daima\train.py`
- Eval entry: `final daima\eval.py`
- Existing result history: `final daima\docs\experiment_log.md`
- Existing model history: `final daima\docs\model_changes.md`
- Existing paper boundary: `final daima\docs\paper_notes.md`

## Scheme A: Multiple Codex Conversations + Independent Worktrees/Branches

Roles:

- Orchestrator conversation: reads queue, reports, and metrics; chooses the next experiment priority.
- Literature/Idea conversation: reads papers/reference code and proposes candidate experiments. It does not edit project code.
- Experimenter conversation: owns one feature branch/worktree, implements one approved hypothesis, later runs full train, writes report, and pushes only the feature branch.
- Reviewer conversation: reviews code/report only. It does not invent a new experiment.
- Reproducer conversation: checks whether the report command and metric evidence can be reproduced or audited.

Windows branch/worktree pattern:

```powershell
cd "C:\Users\qintian\Desktop\qintian"
git fetch origin
git worktree add "..\qintian_exp_<round_id>" -b "exp/<round_id>" main
cd "..\qintian_exp_<round_id>\final daima"
```

Rules:

- One experiment branch per round.
- One Experimenter owns one branch/worktree.
- Do not run two full trains on the same local GPU at the same time.
- Do not write directly to `main`.
- Reports and metrics updates happen on the experiment branch, then reviewer and reproducer inspect that branch.

Strengths:

- Best stability for long local full train.
- Easy recovery after interruption because work is on disk in a branch/worktree.
- Lower risk of cross-experiment contamination.
- Natural audit trail through branches and reports.

Weaknesses:

- More manual coordination.
- Metrics files can conflict if many branches update them concurrently.

## Scheme B: One Codex Main Conversation + Explicit Subagents

Roles:

- Main conversation acts as Orchestrator.
- Subagents: Experiment Planner, Code Reviewer, Reproducer, Report Auditor.
- Subagents communicate only through repo files.

Rules:

- The main conversation remains responsible for final verification.
- Subagents may draft plans/reviews but should not write overlapping files.
- Full train should still be run by one active worker at a time.

Strengths:

- Faster for planning, review, and report auditing.
- Lower setup overhead than worktrees.

Weaknesses:

- Less robust for multi-hour local training.
- Harder to recover if the main conversation loses context.
- Higher risk of file overlap if subagent write scopes are not strict.

## Recommendation

Start with Scheme A.

Reason: local full training is long-running and GPU-bound. Independent worktrees/branches give the best recovery story, clearer ownership, and lower contamination risk. Scheme B can still be used inside one planning session, but the default long-running experiment loop should be Scheme A.

## Shared File Contract

- `experiments/queue.jsonl`: proposed or approved experiments waiting to run.
- `experiments/completed.jsonl`: completed experiments after reviewer/reproducer resolution.
- `experiments/rejected.jsonl`: ideas rejected before training or rejected after review.
- `reports/EXPERIMENT_LOG.md`: human-readable top-level loop ledger.
- `metrics/runs.jsonl`: machine-readable per-run records.
- `metrics/leaderboard.csv`: sorted or append-only leaderboard.

## Startup Order

1. Orchestrator: read `AGENTS.md`, this file, `metrics/`, `experiments/`, and existing `final daima/docs/`.
2. Literature/Idea: propose candidates into `experiments/queue.jsonl`.
3. Orchestrator: approve exactly one next experiment.
4. Experimenter: create a feature branch/worktree and execute one approved round.
5. Reviewer: inspect code changes and report on the experiment branch.
6. Reproducer: verify command, evidence paths, and metric extraction.
7. Orchestrator: update priority and stop if `val/mIoU >= 0.53`.

Parallelism:

- Literature/Idea can run while no full train is active.
- Reviewer and Reproducer must wait for Experimenter report.
- Two Experimenters must not train simultaneously on the local GPU.
- Two agents must not edit the same coordination file at the same time; append only after pulling/rebasing the current branch.
