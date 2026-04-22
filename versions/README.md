# Versions Folder

This folder stores code snapshots for stable or important experiment stages.

## Naming Rule

- `version_001_...`
- `version_002_...`
- `version_003_...`

Use the suffix to describe the role of that snapshot, for example:

- `version_001_stable_baseline_mid_fusion`
- `version_002_c4_prompt_upgrade`
- `version_003_decoder_ablation`

## What to Save in Each Version

Each version folder should keep:

- `train.py`
- `eval.py`
- `infer.py`
- `README.md`
- `requirements.txt`
- `src/`
- `scripts/`
- `results/` for the related result notes

Do not copy large runtime folders unless they are explicitly needed:

- `checkpoints/`
- `visualizations/`
- `data/`

## Goal

This structure is used to:

- keep a stable code snapshot before the next round of changes
- make rollback easier
- make experiment comparison clearer
