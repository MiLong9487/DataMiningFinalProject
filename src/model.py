from __future__ import annotations

from typing import Any

import timm
import torch
from torch import Tensor, nn


class DefectClassifier(nn.Module):
    """Shared backbone with automatic asset selection and per-asset defect heads."""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
        num_asset_heads: int = 5,
    ) -> None:
        super().__init__()
        self.backbone_name: str = backbone
        self.num_asset_heads: int = num_asset_heads
        self.backbone: nn.Module = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim: int = int(self.backbone.num_features)
        self.asset_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_asset_heads),
        )
        self.defect_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(feat_dim, 1),
                )
                for _ in range(num_asset_heads)
            ]
        )

    def forward(
        self,
        x: Tensor,
        asset_idx: Tensor | None = None,
        return_asset_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        feats: Tensor = self.backbone(x)
        asset_logits: Tensor = self.asset_head(feats)
        selected_asset_idx = (
            asset_logits.argmax(dim=1)
            if asset_idx is None
            else asset_idx.to(feats.device).long().view(-1)
        )
        all_defect_logits = torch.cat([head(feats) for head in self.defect_heads], dim=1)
        defect_logits = all_defect_logits.gather(
            1, selected_asset_idx.view(-1, 1)
        ).squeeze(1)
        if return_asset_logits:
            return defect_logits, asset_logits
        return defect_logits


def build_model(
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.2,
    num_asset_heads: int = 5,
) -> DefectClassifier:
    return DefectClassifier(
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        num_asset_heads=num_asset_heads,
    )


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
        num_asset_heads=len(bundle.get("asset_types", range(5))),
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()
    return model, bundle
