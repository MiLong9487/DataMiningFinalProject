from __future__ import annotations

from torchvision import transforms as T
from torchvision.transforms import Compose


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int) -> Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def build_eval_transform(img_size: int) -> Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
