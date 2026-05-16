from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


ASSET_TYPES: tuple[str, ...] = (
    "glass-insulator",
    "lightning-rod-suspension",
    "polymer-insulator-upper-shackle",
    "vari-grip",
    "yoke-suspension",
)

# good → 0 (normal), bad → 1 (defective)
LABEL_GOOD: int = 0
LABEL_BAD: int = 1


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    asset_type: str


def scan_train_dir(root: Path) -> list[Sample]:
    """Walk train_dataset/<asset>/{good,bad}/*.jpg and collect samples."""
    samples: list[Sample] = []
    for asset in ASSET_TYPES:
        for sub, label in (("good", LABEL_GOOD), ("bad", LABEL_BAD)):
            folder = root / asset / sub
            if not folder.is_dir():
                continue
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    samples.append(Sample(path=img_path, label=label, asset_type=asset))
    return samples


class InsPLADDataset(Dataset[tuple[Tensor, int]]):
    """Binary defect-classification dataset for InsPLAD."""

    def __init__(
        self,
        samples: list[Sample],
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ) -> None:
        self.samples: list[Sample] = samples
        self.transform: Optional[Callable[[Image.Image], Tensor]] = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            img = img.convert("RGB")
        if self.transform is None:
            raise RuntimeError("transform is required")
        return self.transform(img), sample.label


class TestImageDataset(Dataset[tuple[Tensor, str]]):
    """Flat folder of unlabeled test images; returns (tensor, filename)."""

    def __init__(
        self,
        image_dir: Path,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.paths: list[Path] = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.transform: Callable[[Image.Image], Tensor] = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), path.name
