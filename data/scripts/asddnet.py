from typing import List

import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int] = [1, 6, 12, 18]):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in atrous_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3 if r > 1 else 1, padding=r if r > 1 else 0, dilation=r, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class ASDDNet(nn.Module):
    """ASPP + Dual Attention (channel SE + spatial) on top of a ResNet18 backbone.

    Designed for classification; replaces final pooling/head with attention-augmented features.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Keep feature extractor up to layer4
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        in_ch = backbone.layer4[-1].conv2.out_channels  # 512 for resnet18
        aspp_out = 256
        self.aspp = ASPP(in_ch, aspp_out)
        self.se = SEBlock(aspp_out)
        self.spatial = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(aspp_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.aspp(x)
        x = self.se(x)
        x = self.spatial(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x


