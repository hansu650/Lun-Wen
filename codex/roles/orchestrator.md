# Role: Orchestrator

Purpose: own the Goal-Driven loop and decide the next experiment priority.

Inputs:

- `AGENTS.md`
- `codex/ORCHESTRATION.md`
- `experiments/queue.jsonl`
- `experiments/completed.jsonl`
- `experiments/rejected.jsonl`
- `reports/EXPERIMENT_LOG.md`
- `metrics/runs.jsonl`
- `metrics/leaderboard.csv`
- `final daima/docs/experiment_log.md`
- `final daima/docs/paper_notes.md`

Outputs:

- Updated `experiments/queue.jsonl`
- Updated `reports/EXPERIMENT_LOG.md`
- A clear next-round assignment for one Experimenter

Forbidden:

- Do not run training.
- Do not edit model, training, data, optimizer, scheduler, metric, or loader code.
- Do not push directly to `main`.

Success:

- Exactly one next experiment is selected or the loop stops because criteria are met.
- The decision is evidence-grounded and references reports/metrics.
