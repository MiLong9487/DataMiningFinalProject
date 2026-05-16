from __future__ import annotations

from typing import Any

import timm
import torch
from torch import Tensor, nn


class DefectClassifier(nn.Module):
    """Binary defect classifier built on top of a timm backbone."""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name: str = backbone
        self.backbone: nn.Module = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim: int = int(self.backbone.num_features)
        self.head: nn.Module = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        feats: Tensor = self.backbone(x)
        return self.head(feats).squeeze(-1)


def build_model(
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.2,
) -> DefectClassifier:
    return DefectClassifier(backbone=backbone, pretrained=pretrained, dropout=dropout)


def load_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[DefectClassifier, dict[str, Any]]:
    """Restore a model + the metadata bundle saved by train.py."""
    bundle: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        backbone=str(bundle["backbone"]),
        pretrained=False,
        dropout=float(bundle.get("dropout", 0.2)),
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()
    return model, bundle
