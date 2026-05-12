# Experiment Result Workflow

Use this default workflow whenever a training run finishes and the user asks to summarize, discuss, commit, or push the result.

## Required Steps

1. Read the active project context first:
   - `final daima/README.md`
   - `final daima/docs/experiment_log.md`
   - `final daima/docs/model_changes.md`
   - `final daima/docs/paper_notes.md`
2. Extract real metrics from TensorBoard event logs or explicit run logs. Do not infer final results only from terminal text.
3. Create one Markdown result file under `final daima/miou_list/` for the run. It must include every recorded epoch's `val/mIoU`.
4. Compare against the current clean baseline and any active candidate means using the same metric definition.
5. Update `final daima/docs/experiment_log.md` with configuration, result, evidence path, conclusion, and next step.
6. Update `final daima/docs/paper_notes.md` if the paper boundary or claim status changes.
7. If the result changes active/archived status, update `final daima/docs/ACTIVE_STATUS.md` and the status note in `final daima/README.md`.
8. Send the result to a separate GPT/subagent discussion when the user asks for GPT discussion or when the result is ambiguous enough to risk overclaiming.
9. Do not save one-off GPT/Pro prompts or discussion transcripts as standalone Markdown files unless the user explicitly asks. Put durable conclusions directly into `experiment_log.md` and `paper_notes.md`.
10. Commit and push only the relevant code/docs/results. Avoid staging unrelated deleted checkpoints, ignored checkpoint outputs, personal notes, reference-code folders, or unrelated untracked files.

## Interpretation Rules

- A high best epoch is not enough. Check the late curve, final epoch, last-5/last-10 means, and post-best behavior.
- Results below baseline mean + 1 std should not be called strong improvements.
- A run with high best mIoU and poor final mIoU should be called unstable unless repeat runs show clean convergence.
- Repeated-run means matter more than single-run peaks.
- If repeated runs expose instability, prefer diagnostic experiments over more blind repeats.
