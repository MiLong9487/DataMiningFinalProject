from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import InsPLADDataset, Sample, scan_train_dir
from metrics import find_threshold_for_precision
from model import build_model
from transforms import build_eval_transform, build_train_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(
    samples: list[Sample], val_ratio: float, seed: int,
) -> tuple[list[Sample], list[Sample]]:
    # stratify on (asset_type, label) so both splits look the same
    keys: list[str] = [f"{s.asset_type}|{s.label}" for s in samples]
    train_s, val_s = train_test_split(
        samples, test_size=val_ratio, random_state=seed, stratify=keys,
    )
    return train_s, val_s


def build_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    counts: Counter[int] = Counter(s.label for s in samples)
    weights: list[float] = [1.0 / counts[s.label] for s in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train: bool = optim is not None
    model.train(is_train)
    total_loss: float = 0.0
    seen: int = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    bar = tqdm(loader, desc=desc, leave=False)
    for imgs, labels in bar:
        imgs = imgs.to(device, non_blocking=True)
        labels_t: Tensor = labels.to(device, non_blocking=True).float()
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                logits: Tensor = model(imgs)
                loss: Tensor = loss_fn(logits, labels_t)
            if is_train:
                assert optim is not None
                optim.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()
        bs: int = imgs.size(0)
        total_loss += float(loss.item()) * bs
        seen += bs
        all_probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        all_labels.append(labels_t.detach().cpu().numpy())
        bar.set_postfix(loss=f"{total_loss / max(seen, 1):.4f}")
    return (
        total_loss / max(seen, 1),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/train_dataset"))
    parser.add_argument("--out_dir", type=Path, default=Path("models"))
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--img_size", type=int, default=288)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_precision", type=float, default=0.90)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    samples: list[Sample] = scan_train_dir(args.data_root)
    if not samples:
        raise SystemExit(f"No samples found under {args.data_root}")
    print(f"[train] total samples: {len(samples)}")
    print(f"[train] label counts: {Counter(s.label for s in samples)}")

    train_s, val_s = stratified_split(samples, args.val_ratio, args.seed)
    print(f"[train] train={len(train_s)} val={len(val_s)}")

    train_ds = InsPLADDataset(train_s, transform=build_train_transform(args.img_size))
    val_ds = InsPLADDataset(val_s, transform=build_eval_transform(args.img_size))

    sampler = build_sampler(train_s)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    model = build_model(backbone=args.backbone, pretrained=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type) if (not args.no_amp and device.type == "cuda") else None

    best_score: float = -1.0
    best_meta: dict[str, object] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, optim, device, scaler, f"epoch {epoch} train")
        val_loss, val_probs, val_labels = run_epoch(model, val_loader, loss_fn, None, device, None, f"epoch {epoch} val")
        scheduler.step()

        thr, m = find_threshold_for_precision(val_probs, val_labels, args.target_precision)
        score: float = m.recall if m.precision >= args.target_precision else m.precision * 0.5 + m.recall * 0.5
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"thr={thr:.3f} P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f} acc={m.accuracy:.4f}"
        )

        if score > best_score:
            best_score = score
            best_meta = {
                "epoch": epoch,
                "threshold": thr,
                "metrics": asdict(m),
                "backbone": args.backbone,
                "img_size": args.img_size,
                "dropout": 0.2,
                "state_dict": model.state_dict(),
            }
            torch.save(best_meta, args.out_dir / "best_model.pth")
            (args.out_dir / "best_metrics.json").write_text(
                json.dumps({k: v for k, v in best_meta.items() if k != "state_dict"}, indent=2)
            )
            print(f"[epoch {epoch:02d}] -> new best (score={score:.4f}) saved")

    print(f"[train] done. best_score={best_score:.4f}")
    if best_meta:
        m = best_meta["metrics"]
        print(f"[train] best metrics: {m}")


if __name__ == "__main__":
    main()
