"""Active decoder modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPNDecoder(nn.Module):
    """Minimal FPN decoder used by the active baseline and TGGA/PMAD paths."""

    def __init__(self, in_channels, out_channels=128, num_classes=40):
        super().__init__()
        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, 1)

        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features, input_size):
        c1, c2, c3, c4 = features

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        p1 = self.smooth(p1)
        p1 = F.interpolate(p1, size=input_size, mode="bilinear", align_corners=False)
        return self.classifier(p1)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, use_relu=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class NMF2D(nn.Module):
    """NMF block used by the official DFormer LightHam decoder."""

    def __init__(self, md_s=1, md_r=64, train_steps=6, eval_steps=7):
        super().__init__()
        self.md_s = md_s
        self.md_r = md_r
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.inv_t = 1.0

    def _build_bases(self, batch_size, channels_per_s, x):
        bases = torch.rand(
            batch_size * self.md_s,
            channels_per_s,
            self.md_r,
            device=x.device,
            dtype=x.dtype,
        )
        return F.normalize(bases, dim=1)

    @staticmethod
    def _local_step(x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    @staticmethod
    def _compute_coef(x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        return coef * numerator / (denominator + 1e-6)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        channels_per_s = channels // self.md_s
        x = x.view(batch_size * self.md_s, channels_per_s, height * width)

        bases = self._build_bases(batch_size, channels_per_s, x)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self._local_step(x, bases, coef)

        coef = self._compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))
        return x.view(batch_size, channels, height, width)


class Hamburger(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ham_in = nn.Conv2d(channels, channels, 1)
        self.ham = NMF2D(md_s=1, md_r=64, train_steps=6, eval_steps=7)
        self.ham_out = ConvBNReLU(channels, channels, 1, use_relu=False)

    def forward(self, x):
        enjoy = F.relu(self.ham_in(x), inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        return F.relu(x + enjoy, inplace=True)


class OfficialHamDecoder(nn.Module):
    """Self-contained c2-c4 LightHam decoder matching the DFormerv2-S NYU config."""

    def __init__(self, in_channels, channels=512, num_classes=40):
        super().__init__()
        ham_in_channels = sum(in_channels[1:])
        self.squeeze = ConvBNReLU(ham_in_channels, channels, 1)
        self.hamburger = Hamburger(channels)
        self.align = ConvBNReLU(channels, channels, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(channels, num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.1
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features, input_size):
        c2, c3, c4 = features[1], features[2], features[3]
        target_size = c2.shape[-2:]
        c3 = F.interpolate(c3, size=target_size, mode="bilinear", align_corners=False)
        c4 = F.interpolate(c4, size=target_size, mode="bilinear", align_corners=False)
        x = torch.cat([c2, c3, c4], dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        x = self.classifier(self.dropout(x))
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
