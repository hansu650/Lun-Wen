"""Mid Fusion 模型"""
import torch
import torch.nn.functional as F
import torch.nn as nn

from .encoder import RGBEncoder, DepthEncoder
from .decoder import SimpleFPNDecoder
from .base_lit import BaseLitSeg


class GatedFusion(nn.Module):
    """Gated fusion anchor."""
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        g = self.gate(torch.cat([rgb_feat, d], dim=1))
        fused = g * rgb_feat + (1 - g) * d
        return self.refine(fused)


class CSTCModule(nn.Module):
    """Cross-Selective Token Cleaning Module"""
    def __init__(self, channels, rgb_cls_dim=384):
        super().__init__()
        self.rgb_base = nn.Linear(rgb_cls_dim, channels)
        self.rgb_delta = nn.Sequential(
            nn.Linear(rgb_cls_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.rgb_norm = nn.LayerNorm(channels)
        # beta 初始化为 0，训练初期等价于普通 Linear。
        self.beta = nn.Parameter(torch.zeros(1))

        self.depth_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.depth_max_pool = nn.AdaptiveMaxPool2d(1)
        self.depth_token = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )

        hidden_channels = max(channels // 4, 16)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels * 4, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
            nn.LayerNorm(channels),
        )
        self.rgb_gate = nn.Linear(channels, channels)
        self.depth_gate = nn.Linear(channels, channels)
        # alpha 初始化为 0，训练初期 CSTC 不破坏预训练特征。
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_feat, depth_feat, rgb_cls):
        rgb_base = self.rgb_base(rgb_cls)
        rgb_delta = self.rgb_delta(rgb_cls)
        rgb_token = self.rgb_norm(rgb_base + self.beta * rgb_delta)

        # Depth 分支没有 CLS，用 AvgPool + MaxPool 提取 summary token。
        depth_avg = self.depth_avg_pool(depth_feat).flatten(1)
        depth_max = self.depth_max_pool(depth_feat).flatten(1)
        depth_token = self.depth_token(torch.cat([depth_avg, depth_max], dim=1))

        diff_token = torch.abs(rgb_token - depth_token)
        agree_token = rgb_token * depth_token
        cross_token = torch.cat([rgb_token, depth_token, diff_token, agree_token], dim=1)

        shared = self.shared_mlp(cross_token)
        rgb_gate = self.rgb_gate(shared + depth_token)
        depth_gate = self.depth_gate(shared + rgb_token)

        # CSTC 只做通道清洗，不改变 C 和 H×W。
        rgb_clean = rgb_feat * (1 + self.alpha * torch.tanh(rgb_gate).view(rgb_feat.shape[0], -1, 1, 1))
        depth_clean = depth_feat * (1 + self.alpha * torch.tanh(depth_gate).view(depth_feat.shape[0], -1, 1, 1))
        return rgb_clean, depth_clean


class SCFABlock(nn.Module):
    """Spatial Contrast-guided Fusion Attention"""
    def __init__(self, channels):
        super().__init__()
        hidden_channels = max(channels // 4, 16)
        # SCFA 负责空间位置级融合注意力，不负责 token 清洗。
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_clean, depth_clean):
        # diff_map 表示 RGB 和 Depth 的冲突区域。
        diff_map = torch.abs(rgb_clean - depth_clean)
        # agree_map 表示 RGB 和 Depth 的一致区域。
        agree_map = rgb_clean * depth_clean
        gate_input = torch.cat([rgb_clean, depth_clean, diff_map, agree_map], dim=1)
        gate_logits = self.gate(gate_input)
        # softmax 让同一空间位置的 RGB / Depth 权重相加为 1。
        weights = torch.softmax(gate_logits, dim=1)
        w_rgb = weights[:, 0:1, :, :]
        w_depth = weights[:, 1:2, :, :]
        scfa_out = w_rgb * rgb_clean + w_depth * depth_clean
        return self.refine(scfa_out)


class CSREBlock(nn.Module):
    """Context-guided Structure Residual Enhancement Block"""
    def __init__(self, channels):
        super().__init__()
        # CSRE 只做融合后结构增强，不再做 RGB-D 融合。
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # 方向卷积增强水平和垂直上下文。
        self.h_context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.v_context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # 从 fused feature 自身提取结构响应，不引入额外边缘图。
        self.structure_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # gamma 初始化为 0，训练初期等价于当前模块二主线。
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, fused):
        local_feat = self.local(fused)
        h_feat = self.h_context(local_feat)
        v_feat = self.v_context(local_feat)
        structure_feat = local_feat + h_feat + v_feat
        enhance = self.out_proj(structure_feat)

        detail = torch.abs(fused - F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1))
        detail_avg = detail.mean(dim=1, keepdim=True)
        detail_max = detail.max(dim=1, keepdim=True)[0]
        structure_gate = torch.sigmoid(self.structure_gate(torch.cat([detail_avg, detail_max], dim=1)))

        return fused + self.gamma * structure_gate * enhance


class CSTCResidualSCFAFusion(nn.Module):
    def __init__(self, channels, rgb_cls_dim=384):
        super().__init__()
        self.cleaner = CSTCModule(channels, rgb_cls_dim=rgb_cls_dim)
        self.old_fusion = GatedFusion(channels, channels)
        self.scfa = SCFABlock(channels)
        self.csre = CSREBlock(channels)
        # rho 初始化为 0，训练初期等价于原 CSTC + GatedFusion。
        self.rho = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_feat, depth_feat, rgb_cls):
        rgb_clean, depth_clean = self.cleaner(rgb_feat, depth_feat, rgb_cls)
        old_out = self.old_fusion(rgb_clean, depth_clean)
        scfa_out = self.scfa(rgb_clean, depth_clean)
        fused = old_out + self.rho * (scfa_out - old_out)
        return self.csre(fused)


class MidFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.rgb_encoder = RGBEncoder()
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList([
            CSTCResidualSCFAFusion(channels)
            for channels in self.rgb_encoder.out_channels
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)
    
    def forward(self, rgb, depth):
        rgb_feats, rgb_cls_tokens = self.rgb_encoder(rgb, return_cls=True)
        depth_feats = self.depth_encoder(depth)
        
        aligned_depth = []
        for rf, df in zip(rgb_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)
        
        fused_feats = [
            fusion(rgb_feat, depth_feat, rgb_cls)
            for fusion, rgb_feat, depth_feat, rgb_cls in zip(
                self.fusions, rgb_feats, aligned_depth, rgb_cls_tokens
            )
        ]
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitMidFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = MidFusionSegmentor(num_classes=num_classes)
