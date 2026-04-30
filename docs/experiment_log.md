# Experiment Log

This file records only experiment information that should survive across
sessions. Valid results require a current-environment run with clear
configuration, logs, and checkpoints.

## Valid Experiment Records

| Date | Experiment | Configuration | Result | Status | Notes |
|---|---|---|---|---|---|
| 2026-04-29 | Context-FPN ResGamma 7 runs | RGB branch: DINOv2-small; Depth branch: Swin-Tiny; Dataset: NYUDepthV2; Fusion: multi-level RGB-D mid-fusion | best mIoU about `0.3933` | valid current best | Current confirmed best result provided by the user. |

## Invalid Or Deprecated Records

| Date recorded | Record | Claimed result | Status | Reason |
|---|---|---|---|---|
| 2026-04-29 | Swin-B RGB + DINOv2-B Depth from old README | about `0.48-0.49` mIoU | invalid / not adopted / not reproducible | The pretrained models are too large for the current environment, and this is not a current reproducible experiment result. |

## Next Logging Rule

Each new experiment should record:

- Date
- Experiment name
- Branch and module configuration
- Dataset and training settings
- Checkpoint path
- Log path
- Best mIoU and repeated-run stability
- Module changes
- Failed attempts
- Next step
