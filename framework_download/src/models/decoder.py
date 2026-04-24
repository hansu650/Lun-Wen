"""解码器模块"""
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPNDecoder(nn.Module):
    """简化版 FPN 解码器"""
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
