# R017 RGB/BGR Contract Negative Archive

R017 tested the official DFormer NYUDepthV2 RGB channel-order contract by removing the local `BGR2RGB` conversion:

```diff
diff --git a/final daima/src/data_module.py b/final daima/src/data_module.py
@@
-        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
```

Result:

- Run: `R017_rgb_bgr_official_contract`
- Best val/mIoU: `0.529090` at validation epoch `38`
- Last val/mIoU: `0.523078`
- Delta vs R016 corrected baseline `0.541121`: `-0.012031`
- Evidence: `final daima/miou_list/R017_rgb_bgr_official_contract.md`

Decision:

- Do not keep this active in `src/data_module.py`.
- Keep R016 RGB input plus official label/depth contracts as the corrected baseline.
- Treat this only as a negative official-contract gate, not as a method contribution.
