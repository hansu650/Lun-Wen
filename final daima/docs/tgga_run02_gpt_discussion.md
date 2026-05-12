# TGGA Run02 GPT Discussion

## Context Sent For Discussion

- baseline: `dformerv2_mid_fusion` clean 10-run mean best val/mIoU `0.517397`, population std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- PMAD logit-only w0.15 5-run mean best: `0.520795`.
- bounded class-context 5-run mean best: `0.515986`.
- model: `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`.
- structure: `DFormerv2_S + TGGA(c3,c4) + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- loss: `CE(final) + 0.03 * CE(aux_c3) + 0.03 * CE(aux_c4)`.
- run01: best `0.522206` at epoch 48, last `0.489865`, last-10 mean `0.510627`, final `gate_c3_mean=0.351956`, `gate_c3_std=0.305273`.
- run02: best `0.517437` at epoch 49, last `0.486566`, last-10 mean `0.501329`, final `gate_c3_mean=0.409689`, `gate_c3_std=0.346383`.

## GPT Critique

The result should be treated as **not a valid stable improvement**. Run01 is a promising high single run, but run02 is essentially tied with the clean baseline mean. The two-run mean best is `0.519822`, only `+0.002425` over baseline mean, less than half a baseline std, and below PMAD w0.15's 5-run mean.

The more important evidence is instability. Both runs have severe late collapse, ending around `0.49`. This pattern suggests TGGA is not a stable enhancer; it introduces a late-stage perturbation that sometimes creates a high best epoch but does not converge cleanly.

The most likely mechanism is c3-side feature-distribution drift. In both runs the c3 gate opens substantially and becomes high variance, while c4 remains weak and low variance. The auxiliary CE heads may continue pushing mid-level semantic separability after the validation loss has already reached its best point, causing a mismatch between auxiliary supervision and the final decoder objective.

The recommended next experiment is not a blind run03. The highest-value diagnostic is **TGGA c3/c4 without auxiliary CE loss**, keeping the structure unchanged and removing `0.03 * aux_c3 + 0.03 * aux_c4`. This separates "TGGA structure is unstable" from "the auxiliary training constraint is unstable."

## Decision Recorded

- Do not claim TGGA c3/c4 as a stable paper improvement.
- Treat run01-run02 as a weak positive but unstable signal.
- Prefer no-aux TGGA diagnostic next.
- Use c4-only or weak-c3 variants only after the no-aux result clarifies whether the auxiliary loss is the main instability source.
