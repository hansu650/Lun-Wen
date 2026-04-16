"""编码器模块"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class RGBEncoder(nn.Module):
    """基于 ResNet-18 的 RGB 编码器"""
    # 直接调用他们的layer1,2,3,4层输出特征图的通道数
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)# 默认参数的resnet
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):
        feats = []
        x = self.layer0(x)
        x = self.layer1(x); feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        x = self.layer4(x); feats.append(x)
        return feats


class DepthEncoder(nn.Module):
    """基于 ResNet-18 的 Depth 编码器，输入通道为 1"""
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        old_conv = resnet.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        resnet.conv1 = new_conv
        # 我们只有一个通道所以要自己设置一个
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):
        feats = []
        x = self.layer0(x)
        x = self.layer1(x); feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        x = self.layer4(x); feats.append(x)
        return feats


class EarlyFusionEncoder(nn.Module):
    """4通道输入的 ResNet-18 编码器"""
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        old_conv = resnet.conv1
        new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old_conv.weight)
            new_conv.weight[:, 3].copy_(old_conv.weight.mean(dim=1))
        resnet.conv1 = new_conv
       # 4通道
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = [64, 128, 256, 512]
    
    def forward(self, x):
        feats = []
        x = self.layer0(x)
        x = self.layer1(x); feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        x = self.layer4(x); feats.append(x)
        return feats
