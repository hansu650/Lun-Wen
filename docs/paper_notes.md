# Paper Notes

This file records paper-facing facts and writing constraints for the RGB-D
semantic segmentation project.

## Current Paper Direction

- Task: RGB-D semantic segmentation
- Dataset: NYUDepthV2
- RGB branch: DINOv2-small
- Depth branch: Swin-Tiny
- Fusion direction: multi-level RGB-D mid-fusion
- Current modules under improvement: Context-FPN, ResGamma, Depth Adapter, and
  fusion modules
- Current confirmed best result: Context-FPN ResGamma 7 runs, best mIoU about
  `0.3933`

## Result Citation Rule

Do not cite old README results as paper results unless they have current
configuration, log, and checkpoint support. The old Swin-B RGB + DINOv2-B Depth
record is deprecated and must not be used as the paper main line, a valid
baseline, or the current best result.
