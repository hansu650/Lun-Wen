"""Smoke checks for the default active training registry."""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import ACTIVE_MODEL_REGISTRY, MODEL_REGISTRY  # noqa: E402


EXPECTED_ACTIVE = {
    "dformerv2_mid_fusion",
    "dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1",
    "dformerv2_geometry_primary_teacher",
    "dformerv2_primkd_logit_only",
}

ARCHIVED_MODELS = {
    "dformerv2_sgbr_decoder",
    "dformerv2_class_context_decoder",
    "dformerv2_context_decoder",
    "dformerv2_depth_fft_select",
    "dformerv2_fft_freq_enhance",
    "dformerv2_fft_hilo_enhance",
    "dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2",
    "dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1",
    "dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1",
}


def main():
    active_names = set(ACTIVE_MODEL_REGISTRY.keys())
    all_names = set(MODEL_REGISTRY.keys())
    print("active models:", ", ".join(sorted(active_names)))

    missing = EXPECTED_ACTIVE - active_names
    if missing:
        raise AssertionError(f"Missing active models: {sorted(missing)}")

    leaked = ARCHIVED_MODELS & all_names
    if leaked:
        raise AssertionError(f"Archived models leaked into default registry: {sorted(leaked)}")

    baseline_cls = ACTIVE_MODEL_REGISTRY["dformerv2_mid_fusion"]
    c4only_cls = ACTIVE_MODEL_REGISTRY["dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1"]
    teacher_cls = ACTIVE_MODEL_REGISTRY["dformerv2_geometry_primary_teacher"]

    baseline = baseline_cls(num_classes=40, lr=1e-4, dformerv2_pretrained=None, loss_type="ce").eval()
    c4only = c4only_cls(num_classes=40, lr=1e-4, dformerv2_pretrained=None, loss_type="ce").eval()
    teacher = teacher_cls(num_classes=40, lr=1e-4, dformerv2_pretrained=None, loss_type="ce").eval()

    print("instantiated:", baseline_cls.__name__, c4only_cls.__name__, teacher_cls.__name__)
    print("skipped PMAD instantiation: teacher_ckpt is intentionally required")

    rgb = torch.randn(1, 3, 240, 320)
    depth = torch.randn(1, 1, 240, 320)
    with torch.no_grad():
        baseline_logits = baseline(rgb, depth)
        c4only_logits = c4only(rgb, depth)
        c4only_final, c4only_aux = c4only.model(rgb, depth, return_aux=True)

    assert baseline_logits.shape == (1, 40, 240, 320), tuple(baseline_logits.shape)
    assert c4only_logits.shape == (1, 40, 240, 320), tuple(c4only_logits.shape)
    assert c4only_final.shape == (1, 40, 240, 320), tuple(c4only_final.shape)
    assert "aux_logits_c3" not in c4only_aux
    assert "aux_logits_c4" in c4only_aux
    assert not hasattr(c4only.model, "tgga_c3")
    assert c4only.model.tgga_c4.forward_calls > 0

    print("baseline logits:", tuple(baseline_logits.shape))
    print("c4-only logits:", tuple(c4only_logits.shape))
    print("c4-only aux c4:", tuple(c4only_aux["aux_logits_c4"].shape))
    print("registry smoke passed")


if __name__ == "__main__":
    main()
