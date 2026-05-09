# feiqi/losses — archived auxiliary loss modules

These files are deprecated training-only auxiliary loss experiments.

They have been removed from the active MODEL_REGISTRY and mid_fusion.py imports.

## Archived modules

- `freq_cov_loss.py` — MultiScaleFrequencyCovarianceLoss
  - Model: `dformerv2_ms_freqcov`
  - Status: 7-run sweep mean 0.515697 < baseline mean 0.517397; best single 0.520539
  - Conclusion: marginal/negative

- `mask_reconstruction_loss.py` — FeatureMaskReconstructionLoss
  - Model: `dformerv2_feat_maskrec_c34`
  - Status: best 0.515327 < baseline mean 0.517397
  - Conclusion: negative

- `contrastive_loss.py` — CrossModalInfoNCELoss
  - Model: `dformerv2_cm_infonce`
  - Status: contrast loss decreased 45% but best mIoU 0.514461 < baseline mean 0.517397
  - Conclusion: loss converges but does not help segmentation

## How to reproduce

To reproduce these experiments, you need to:

1. Copy the loss .py file back to `src/models/`
2. Add the import back to `src/models/mid_fusion.py`
3. Add the Lit wrapper class back to `src/models/mid_fusion.py`
4. Add the import, registry entry, argparse args, and build_model branch back to `train.py`
