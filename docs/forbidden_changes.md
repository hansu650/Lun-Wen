# Forbidden Changes

These changes are forbidden for the `val/mIoU >= 0.53` experiment loop unless the user explicitly changes the contract.

Evaluation/data integrity:

- Do not modify dataset split.
- Do not modify validation/test loader behavior.
- Do not modify eval metric.
- Do not modify mIoU calculation.
- Do not introduce data leakage.

Training recipe:

- Do not modify data augmentation.
- Do not modify optimizer.
- Do not modify scheduler.
- Do not modify batch size.
- Do not modify epoch count.
- Do not modify learning rate.
- Do not modify fixed experiment parameters such as worker count or early-stopping patience.

Repository hygiene:

- Do not push directly to `main`.
- Do not commit checkpoints.
- Do not commit datasets.
- Do not commit pretrained weights.
- Do not commit TensorBoard event logs or large raw logs.

Claim discipline:

- No evidence means no improvement claim.
- Single high epochs must be reported with final epoch, last-5/last-10 behavior, and repeat status when available.
