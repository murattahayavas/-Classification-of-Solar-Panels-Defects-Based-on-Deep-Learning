from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import timm


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1280, bottleneck_channels: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


class BackboneWithAE(nn.Module):
    """Wrap a feature-only EfficientNet backbone and attach an autoencoder on the last feature map.

    Returns a dict with a single feature map key '0' for FasterRCNN.
    """

    def __init__(self, model_name: str = "tf_efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.body = timm.create_model(model_name, features_only=True, pretrained=pretrained)
        feat_info = self.body.feature_info
        self.out_channels = feat_info[-1]['num_chs']
        self.autoencoder = ConvAutoencoder(in_channels=self.out_channels, bottleneck_channels=256)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: List[torch.Tensor] = self.body(x)
        last = feats[-1]
        _, rec = self.autoencoder(last)
        # Store reconstruction for external loss via hook attribute
        self.last_feature = last
        self.reconstruction = rec
        return {"0": last}


class FasterRCNNWithAE(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = "tf_efficientnet_b0", pretrained_backbone: bool = True, ae_loss_weight: float = 0.1):
        super().__init__()
        self.backbone = BackboneWithAE(backbone_name, pretrained=pretrained_backbone)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        self.detector = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_nms_thresh=0.5,
        )
        self.ae_loss_weight = ae_loss_weight
        self.ae_criterion = nn.SmoothL1Loss(beta=0.1)

    def forward(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]] = None):
        if self.training:
            losses: Dict[str, torch.Tensor] = self.detector(images, targets)
            # AE loss on last feature
            last = self.backbone.last_feature
            rec = self.backbone.reconstruction
            ae_loss = self.ae_criterion(rec, last)
            losses["loss_autoenc"] = ae_loss * self.ae_loss_weight
            total = sum(losses.values())
            return losses, total
        else:
            return self.detector(images)


