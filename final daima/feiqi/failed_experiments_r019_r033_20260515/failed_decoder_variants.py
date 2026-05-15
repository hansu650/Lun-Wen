"""Archived decoder variants removed from the active registry after R033.

These snippets are kept for inspection only. They are not imported by the
active training entrypoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPNDecoderWithClassifierDropout:
    """R031: rejected SimpleFPN classifier dropout variant."""

    summary = {
        "run": "R031",
        "model": "dformerv2_simplefpn_classifier_dropout",
        "best_val_miou": 0.531544,
        "decision": "archive; below R016 corrected baseline",
    }

    @staticmethod
    def forward_template(self, features, input_size):
        c1, c2, c3, c4 = features
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.smooth(p1)
        p1 = F.interpolate(p1, size=input_size, mode="bilinear", align_corners=False)
        return self.classifier(nn.Dropout2d(0.1)(p1))


class SimpleFPNDecoderC1DetailGate:
    """R032: rejected c1 detail strength variant."""

    summary = {
        "run": "R032",
        "model": "dformerv2_simplefpn_c1_detail_gate",
        "best_val_miou": 0.536603,
        "decision": "archive; below R016 and alpha barely moved",
    }

    c1_detail_logit_init = 6.906755

    @staticmethod
    def forward_template(self, features, input_size):
        c1, c2, c3, c4 = features
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(torch.tensor(SimpleFPNDecoderC1DetailGate.c1_detail_logit_init, device=p2.device, dtype=p2.dtype))
        p1 = alpha * self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.smooth(p1)
        p1 = F.interpolate(p1, size=input_size, mode="bilinear", align_corners=False)
        return self.classifier(p1)


class SimpleFPNHamLogitFusionDecoder:
    """R033: rejected SimpleFPN + Ham logit residual variant."""

    summary = {
        "run": "R033",
        "model": "dformerv2_simplefpn_ham_logit_fusion",
        "best_val_miou": 0.533020,
        "decision": "archive; Ham logits opened but stayed below R016",
    }

    ham_logit_logit_init = -2.944439

    @staticmethod
    def forward_template(simple_fpn, ham_decoder, features, input_size):
        simple_logits = simple_fpn(features, input_size)
        ham_logits = ham_decoder(features, input_size)
        alpha = torch.sigmoid(torch.tensor(SimpleFPNHamLogitFusionDecoder.ham_logit_logit_init, device=simple_logits.device, dtype=simple_logits.dtype))
        return simple_logits + alpha * ham_logits
