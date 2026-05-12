# dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2 run01-run02 summary

## Raw Data

| run | best val/mIoU | best epoch | last val/mIoU | last-10 mean | last-5 mean | final beta_c3 | final gate_c3_mean | final gate_c3_std | conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01 | 0.522206 | 48 | 0.489865 | 0.510627 | 0.512473 | 0.035080 | 0.351956 | 0.305273 | strong single-run peak, late collapse |
| run02 | 0.517437 | 49 | 0.486566 | 0.501329 | 0.501959 | 0.039473 | 0.409689 | 0.346383 | baseline-level peak, late collapse |

## Aggregate

- completed runs: `2`
- mean best val/mIoU: `0.519822`
- population std best val/mIoU: `0.002384`
- mean last val/mIoU: `0.488215`
- mean last-10 val/mIoU: `0.505978`
- mean last-5 val/mIoU: `0.507216`
- clean 10-run baseline mean best: `0.517397`
- clean 10-run baseline population std: `0.004901`
- clean 10-run baseline mean + 1 std: `0.522298`
- clean 10-run baseline best single: `0.524425`
- PMAD logit-only w0.15 5-run mean best: `0.520795`
- bounded class-context 5-run mean best: `0.515986`
- delta vs clean baseline mean: `+0.002425` (`+0.495` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.002476`
- delta vs clean baseline best single: `-0.004603`
- delta vs PMAD w0.15 mean: `-0.000973`
- delta vs bounded class-context mean: `+0.003836`

## Interpretation

TGGA c3/c4 has a weak positive repeated signal on best mIoU, but the signal is not strong enough to claim improvement. The two-run mean best is less than half a baseline standard deviation above the clean baseline mean and remains below PMAD w0.15's five-run mean.

The stronger result is the instability finding. Both runs peak very late and then collapse to final val/mIoU around `0.49`. Run02 also shows a larger final c3 gate (`0.409689`) and c3 gate std (`0.346383`) than run01, while c4 remains a low-variance weak gate. This points toward c3-side TGGA modulation and/or the auxiliary semantic heads as the likely source of late feature-distribution drift.

## Decision

- Do not claim TGGA c3/c4 as a stable paper improvement.
- Do not spend the next experiment on a blind run03 unless the goal is only negative archival.
- The highest-value next diagnostic is a no-aux TGGA c3/c4 run: keep the structure, remove `0.03 * aux_c3 + 0.03 * aux_c4`, and test whether late collapse comes from the auxiliary CE constraints or from TGGA gating itself.
- If no-aux still collapses, prefer c4-only or weak-c3 diagnostics over more repeat seeds.

## Evidence

- `miou_list/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run01.md`
- `miou_list/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02.md`
