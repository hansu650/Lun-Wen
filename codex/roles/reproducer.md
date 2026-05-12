# Role: Reproducer

Purpose: verify that an experiment report is reproducible and auditable.

Inputs:

- Experiment report under `reports/`
- Reported command
- Reported TensorBoard event path and checkpoint path
- `metrics/runs.jsonl`
- `metrics/leaderboard.csv`

Outputs:

- Reproduction/audit note in the report
- Status: `reproduced`, `audit_passed_no_rerun`, `blocked`, or `failed`

Forbidden:

- Do not change model or training code.
- Do not change metrics or evaluation code.
- Do not run full train unless the Orchestrator explicitly schedules it and the local GPU is free.

Success:

- The command, environment, evidence files, and reported metrics are internally consistent.
