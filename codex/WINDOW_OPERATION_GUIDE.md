# Multi-Window Operation Guide

This guide tells you exactly how to run the RGB-D mIoU Goal-Driven loop with multiple Codex windows.

Current branch for review: `orchestration/experiment-loop`.

Main goal: push NYUDepthV2 validation mIoU to `>= 0.53`.

Current phase: orchestration setup only. Do not run full train until the user explicitly starts the experiment phase.

## 0. One-Time Preparation

Open PowerShell:

```powershell
cd "C:\Users\qintian\Desktop\qintian"
git fetch origin
git switch orchestration/experiment-loop
git pull
```

Before this branch is merged, all role windows should inspect this branch so they can see the orchestration files.

After this branch is merged into `main`, future experiment branches should start from `main`.

## 1. Recommended Window Layout

Open these Codex windows:

1. Orchestrator
2. Literature/Idea Agent
3. Experimenter-1
4. Reviewer
5. Reproducer, optional but recommended after a promising run

Only one Experimenter should run a full train at a time on the local GPU.

## 2. File Ownership

Use this ownership table to avoid conflicts:

| File or folder | Writer | Readers |
|---|---|---|
| `experiments/candidates.jsonl` | Literature/Idea | Orchestrator |
| `experiments/queue.jsonl` | Orchestrator only | Experimenter, Reviewer |
| `experiments/completed.jsonl` | Experimenter after run, Orchestrator after final decision | Reviewer, Reproducer |
| `experiments/rejected.jsonl` | Orchestrator | All roles |
| `metrics/runs.jsonl` | One active Experimenter | Orchestrator, Reviewer, Reproducer |
| `metrics/leaderboard.csv` | One active Experimenter | Orchestrator, Reviewer, Reproducer |
| `reports/` | Each Experimenter writes its own report file | All roles |
| `final daima/src/` | Experimenter only, after approval | Reviewer |
| `final daima/train.py` | Do not edit unless Orchestrator explicitly approves model registry work | Reviewer |
| `final daima/checkpoints/` | Never commit | All roles read evidence only |

If two windows need to update the same file, stop and let Orchestrator choose one writer.

## 3. Startup Sequence

### Step 1: Orchestrator Window

Paste the full content of:

```text
codex/prompts/00_orchestrator_start.md
```

The Orchestrator should read:

- `AGENTS.md`
- `codex/ORCHESTRATION.md`
- `codex/WINDOW_OPERATION_GUIDE.md`
- `reports/EXPERIMENT_LOG.md`
- `metrics/leaderboard.csv`
- `metrics/runs.jsonl`
- `experiments/candidates.jsonl`
- `experiments/queue.jsonl`
- `experiments/completed.jsonl`
- `experiments/rejected.jsonl`
- `final daima/docs/experiment_log.md`
- `final daima/docs/paper_notes.md`

The Orchestrator should not train or edit model code.

Orchestrator output:

- Select one approved experiment.
- Append one queue record to `experiments/queue.jsonl`.
- Write a short decision note in `reports/EXPERIMENT_LOG.md`.

### Step 2: Literature/Idea Window

Paste the full content of:

```text
codex/prompts/04_literature_ideas.md
```

The Literature/Idea Agent should:

- Search recent relevant papers and source code only when useful.
- Prefer recent CCF-A/B or SCI Q1/Q2 evidence when discussing papers.
- Read current negative results before proposing new ideas.
- Append candidate records to `experiments/candidates.jsonl`.

It must not edit code, train, or write `experiments/queue.jsonl`.

This window can run in parallel with Orchestrator while no full train is active.

### Step 3: Orchestrator Approval

The Orchestrator reads `experiments/candidates.jsonl`, rejects low-value ideas into `experiments/rejected.jsonl`, and approves exactly one experiment into `experiments/queue.jsonl`.

Approved queue record should include:

```json
{
  "round_id": "R001",
  "status": "approved",
  "hypothesis": "...",
  "owner": "Experimenter-1",
  "branch": "exp/R001-short-name",
  "allowed_files": ["..."],
  "forbidden_changes": "See docs/forbidden_changes.md",
  "priority": 1,
  "notes": "..."
}
```

### Step 4: Experimenter Window

Paste the full content of:

```text
codex/prompts/01_experimenter_round.md
```

Before full train is allowed, the Experimenter may only do planning/sanity checks. During the later experiment phase, the Experimenter uses a new branch or worktree.

Recommended worktree pattern:

```powershell
cd "C:\Users\qintian\Desktop\qintian"
git fetch origin
git switch main
git pull
git worktree add "..\qintian_exp_R001" -b "exp/R001-short-name" main
cd "..\qintian_exp_R001\final daima"
```

Experimenter responsibilities:

- Implement exactly one approved hypothesis.
- Keep fixed training recipe unchanged.
- Run the approved full train only in the experiment phase.
- Extract real TensorBoard metrics.
- Write a report under `reports/`.
- Update `metrics/runs.jsonl`.
- Update `metrics/leaderboard.csv`.
- Push only the experiment feature branch.

Never commit:

- `final daima/checkpoints/`
- datasets
- pretrained weights
- TensorBoard event logs
- large raw logs

### Step 5: Reviewer Window

Paste the full content of:

```text
codex/prompts/02_reviewer_check.md
```

Reviewer waits until Experimenter has pushed a branch and report.

Reviewer checks:

- Did the branch change dataset split?
- Did it change metric or mIoU calculation?
- Did it change validation/test dataloader?
- Did it change data augmentation?
- Did it change optimizer, scheduler, batch size, epoch count, learning rate, or fixed training parameters?
- Does the report cite real evidence paths?
- Does the claim match the numbers?

Reviewer output status:

- `approved`
- `changes_requested`
- `rejected`

### Step 6: Reproducer Window

Paste the full content of:

```text
codex/prompts/03_reproducer_check.md
```

Reproducer waits until Experimenter has a report.

Reproducer checks:

- Windows command is complete.
- Branch and commit hash are listed.
- Checkpoint path is listed.
- TensorBoard event path is listed.
- `val/mIoU` can be traced to evidence.
- Metrics in `metrics/runs.jsonl` and `metrics/leaderboard.csv` match the report.

Reproducer output status:

- `reproduced`
- `audit_passed_no_rerun`
- `blocked`
- `failed`

## 4. First Real Experiment Sanity Check

Before allowing the first full train, Orchestrator should ask Experimenter to do a dry audit only:

- Confirm branch/worktree path.
- Confirm the approved queue record.
- Confirm no forbidden files are modified.
- Confirm the planned command still uses the fixed recipe.
- Confirm checkpoint output directory is unique.
- Confirm no large files are staged.

Only after this sanity check should full train begin.

## 5. What Can Run In Parallel

Can run in parallel:

- Orchestrator reading metrics/reports.
- Literature/Idea reading papers and writing `experiments/candidates.jsonl`.
- Reviewer reading an already pushed branch while Literature is searching.

Must wait:

- Experimenter waits for Orchestrator approval.
- Reviewer waits for Experimenter branch/report.
- Reproducer waits for Experimenter report and evidence.
- A second Experimenter waits until the local GPU is free.

## 6. Conflict Avoidance Rules

- One writer per shared file at a time.
- Prefer append-only JSONL records.
- Pull/rebase before appending to shared files.
- Never let Reviewer rewrite Experimenter code.
- Never let Literature edit code.
- Never let Experimenter approve its own claim.

## 7. Minimal Daily Loop

1. Orchestrator reads current metrics and queue.
2. Literature proposes new candidates if queue is empty or weak.
3. Orchestrator approves one experiment.
4. Experimenter runs one full train on one branch.
5. Reviewer audits branch and report.
6. Reproducer verifies evidence.
7. Orchestrator updates decision:
   - stop if `val/mIoU >= 0.53` and audit passes
   - otherwise select the next highest-value hypothesis

## 8. Current Stop Point

This setup creates the operating manual only. Stop here until the user explicitly starts the experiment phase.
