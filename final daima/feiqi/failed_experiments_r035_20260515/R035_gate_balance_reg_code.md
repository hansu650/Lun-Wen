# R035 Gate Balance Regularizer Archived Code

R035 tested a training-only global gate-balance regularizer. The experiment completed a full train but fell below the 0.53 stage threshold, so the active registry was cleaned after recording evidence.

## Result

- Run: `R035_gate_balance_reg_run01`
- Model during experiment: `dformerv2_gate_balance_reg`
- Best val/mIoU: `0.529498` at validation epoch `38`
- Last val/mIoU: `0.521308`
- Best-to-last drop: `0.008190`
- Decision: negative; do not tune gate-balance lambda

## Archived Implementation

```python
class GatedFusionWithBalanceStats(GatedFusion):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__(rgb_channels=rgb_channels, depth_channels=depth_channels)
        self.last_gate = None

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        g = self.gate(torch.cat([rgb_feat, d], dim=1))
        self.last_gate = g
        fused = g * rgb_feat + (1 - g) * d
        return self.refine(fused)
```

```python
gate_balance = logits.new_zeros(())
for gate in self.model.gate_stats():
    gate_balance = gate_balance + (gate.mean() - 0.5).pow(2)
gate_balance = gate_balance / len(gates)
loss = ce_loss + 0.01 * gate_balance
```

`LitDFormerV2GateBalanceReg` followed the same optimizer contract as `LitDFormerV2MidFusion` and logged `train/gate_balance_loss`, `train/gate_mean_c1..c4`, and `train/gate_std_c1..c4`.
