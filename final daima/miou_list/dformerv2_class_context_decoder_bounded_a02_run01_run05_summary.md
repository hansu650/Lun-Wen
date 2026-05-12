# dformerv2_class_context_decoder_bounded_a02_run01_run05_summary

## Summary

- model: `dformerv2_class_context_decoder`
- loss: `CE(final_logits, label) + 0.2 * CE(aux_logits, label)`
- class-context settings: `class_context_channels=64`, `class_context_aux_weight=0.2`, `class_context_alpha_init=0.1`, `class_context_alpha_max=0.2`
- purpose: five-run repeat for bounded-alpha CGCD / OCR-lite decoder.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + ClassContextFPNDecoder`
- completed runs: `5`
- dformerv2_class_context_decoder_bounded_a02_run01: best val/mIoU `0.525156` at epoch 46, last `0.469674`, final alpha `0.134989`
- dformerv2_class_context_decoder_bounded_a02_run02: best val/mIoU `0.511353` at epoch 34, last `0.498829`, final alpha `0.134747`
- dformerv2_class_context_decoder_bounded_a02_run03: best val/mIoU `0.511318` at epoch 49, last `0.505855`, final alpha `0.133999`
- dformerv2_class_context_decoder_bounded_a02_run04: best val/mIoU `0.514017` at epoch 43, last `0.488192`, final alpha `0.135494`
- dformerv2_class_context_decoder_bounded_a02_run05: best val/mIoU `0.518087` at epoch 50, last `0.518087`, final alpha `0.135144`
- mean best val/mIoU: `0.515986`
- population std best val/mIoU: `0.005208`
- mean last val/mIoU: `0.496127`
- mean last-10 val/mIoU: `0.504101`
- best single run: `0.525156` (dformerv2_class_context_decoder_bounded_a02_run01)
- worst single run: `0.511318` (dformerv2_class_context_decoder_bounded_a02_run03)
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison unbounded class-context run01 best: `0.519807`
- comparison PMAD logit-only w0.15 5-run mean best: `0.520795`
- mean delta vs clean baseline mean: `-0.001411` (`-0.288` baseline std units)
- mean delta vs clean baseline mean + 1 std: `-0.006312`
- mean delta vs PMAD w0.15 mean: `-0.004809`
- mean delta vs unbounded class-context run01: `-0.003821`
- runs above baseline mean: `2/5`
- runs above baseline mean + 1 std: `1/5`
- runs above baseline best single: `1/5`
- evidence: `miou_list/dformerv2_class_context_decoder_bounded_a02_run01.md`, `miou_list/dformerv2_class_context_decoder_bounded_a02_run02.md`, `miou_list/dformerv2_class_context_decoder_bounded_a02_run03.md`, `miou_list/dformerv2_class_context_decoder_bounded_a02_run04.md`, `miou_list/dformerv2_class_context_decoder_bounded_a02_run05.md`
- conclusion: mixed repeated-run result. Bounded alpha fixes the runaway-alpha failure mode and produces one strong run, but the five-run mean remains below the clean baseline mean and below PMAD. This should not be claimed as a stable improvement.

## Run Table

| Run | Best mIoU | Best Epoch | Last mIoU | Last-10 Mean | Final Alpha | Delta vs Baseline Mean |
|---|---:|---:|---:|---:|---:|---:|
| dformerv2_class_context_decoder_bounded_a02_run01 | 0.525156 | 46 | 0.469674 | 0.516609 | 0.134989 | +0.007759 |
| dformerv2_class_context_decoder_bounded_a02_run02 | 0.511353 | 34 | 0.498829 | 0.500426 | 0.134747 | -0.006044 |
| dformerv2_class_context_decoder_bounded_a02_run03 | 0.511318 | 49 | 0.505855 | 0.498185 | 0.133999 | -0.006079 |
| dformerv2_class_context_decoder_bounded_a02_run04 | 0.514017 | 43 | 0.488192 | 0.500927 | 0.135494 | -0.003380 |
| dformerv2_class_context_decoder_bounded_a02_run05 | 0.518087 | 50 | 0.518087 | 0.504359 | 0.135144 | +0.000690 |
