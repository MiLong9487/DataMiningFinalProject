from __future__ import annotations

import argparse
import csv
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

from dataset import ASSET_TYPES, InsPLADDataset, Sample, scan_train_dir
from metrics import BinaryMetrics, compute_metrics, find_threshold_for_precision
from model import build_model
from transforms import build_eval_transform, build_train_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(
    samples: list[Sample],
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    keys: list[str] = [f"{s.asset_type}|{s.label}" for s in samples]
    train_s, val_s = train_test_split(
        samples,
        test_size=val_ratio,
        random_state=seed,
        stratify=keys,
    )
    return train_s, val_s


def build_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    counts: Counter[str] = Counter(f"{s.asset_type}|{s.label}" for s in samples)
    weights: list[float] = [1.0 / counts[f"{s.asset_type}|{s.label}"] for s in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def find_asset_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    asset_types: np.ndarray,
    target_precision: float,
) -> tuple[dict[str, float], dict[str, BinaryMetrics]]:
    thresholds: dict[str, float] = {}
    metrics: dict[str, BinaryMetrics] = {}
    for asset_type in ASSET_TYPES:
        mask = asset_types == asset_type
        if not mask.any():
            continue
        threshold, metric = find_threshold_for_precision(
            probs[mask],
            labels[mask],
            target_precision,
        )
        thresholds[asset_type] = threshold
        metrics[asset_type] = metric
    return thresholds, metrics


def compute_metrics_with_asset_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    asset_types: np.ndarray,
    asset_thresholds: dict[str, float],
    fallback_threshold: float,
) -> BinaryMetrics:
    thresholds = np.array(
        [asset_thresholds.get(str(asset_type), fallback_threshold) for asset_type in asset_types]
    )
    preds = (probs >= thresholds).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max(len(labels), 1)
    return BinaryMetrics(
        threshold=-1.0,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def compute_asset_classification_metrics(
    asset_labels: np.ndarray,
    asset_preds: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for idx, asset_type in enumerate(ASSET_TYPES):
        mask = asset_labels == idx
        count = int(mask.sum())
        if count == 0:
            continue
        correct = int((asset_preds[mask] == idx).sum())
        incorrect = count - correct
        metrics[asset_type] = {
            "count": count,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": correct / count,
            "error_rate": incorrect / count,
        }
    return metrics


def print_asset_classification_metrics(
    metrics: dict[str, dict[str, float | int]],
) -> None:
    print("[train] validation asset-type classification")
    for asset_type in ASSET_TYPES:
        item = metrics.get(asset_type)
        if item is None:
            continue
        print(
            f"  {asset_type}: "
            f"correct={item['correct']}/{item['count']} "
            f"wrong={item['incorrect']}/{item['count']} "
            f"acc={item['accuracy']:.4f} err={item['error_rate']:.4f}"
        )


def print_asset_thresholds(asset_thresholds: dict[str, float]) -> None:
    print("[train] best validation thresholds by asset type:")
    for asset_type in ASSET_TYPES:
        threshold = asset_thresholds.get(asset_type)
        if threshold is not None:
            print(f"  {asset_type}: threshold={threshold:.4f}")


def print_asset_metrics(asset_metrics: dict[str, dict[str, object]]) -> None:
    print("[train] best validation metrics by asset type:")
    for asset_type in ASSET_TYPES:
        metrics = asset_metrics.get(asset_type)
        if metrics is None:
            continue
        print(
            f"  {asset_type}: "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f} "
            f"tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']}"
        )


def metric_score(metric: BinaryMetrics, target_precision: float) -> float:
    if metric.precision >= target_precision:
        return metric.recall
    return metric.precision * 0.5 + metric.recall * 0.5


def write_validation_error_report(
    path: Path,
    samples: list[Sample],
    probs: np.ndarray,
    labels: np.ndarray,
    true_asset_idxs: np.ndarray,
    pred_asset_idxs: np.ndarray,
    asset_thresholds: dict[str, float],
    fallback_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for i, sample in enumerate(samples):
        selected_asset_type = ASSET_TYPES[int(pred_asset_idxs[i])]
        threshold = asset_thresholds.get(selected_asset_type, fallback_threshold)
        prob = float(probs[i])
        pred_label = int(prob >= threshold)
        true_label = int(labels[i])
        asset_correct = int(true_asset_idxs[i] == pred_asset_idxs[i])
        defect_correct = int(pred_label == true_label)
        if asset_correct and defect_correct:
            continue
        defect_error = ""
        if not defect_correct:
            defect_error = "FP" if pred_label == 1 else "FN"
        rows.append(
            {
                "path": str(sample.path),
                "filename": sample.path.name,
                "true_asset_type": sample.asset_type,
                "pred_asset_type": selected_asset_type,
                "asset_correct": asset_correct,
                "true_label": true_label,
                "pred_label": pred_label,
                "defect_correct": defect_correct,
                "defect_error": defect_error,
                "prob_defective": prob,
                "threshold_used": threshold,
            }
        )

    fieldnames = [
        "path",
        "filename",
        "true_asset_type",
        "pred_asset_type",
        "asset_correct",
        "true_label",
        "pred_label",
        "defect_correct",
        "defect_error",
        "prob_defective",
        "threshold_used",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[train] wrote validation error report: {path} rows={len(rows)}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    defect_loss_fn: nn.Module,
    asset_loss_fn: nn.Module,
    asset_loss_weight: float,
    optim: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    is_train: bool = optim is not None
    model.train(is_train)
    total_loss: float = 0.0
    seen: int = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_asset_labels: list[np.ndarray] = []
    all_asset_preds: list[np.ndarray] = []
    bar = tqdm(loader, desc=desc, leave=False)
    for imgs, labels, asset_idxs in bar:
        imgs = imgs.to(device, non_blocking=True)
        labels_t: Tensor = labels.to(device, non_blocking=True).float()
        asset_idxs_t: Tensor = asset_idxs.to(device, non_blocking=True).long()
        selected_asset_idxs = asset_idxs_t if is_train else None
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                logits, asset_logits = model(
                    imgs,
                    selected_asset_idxs,
                    return_asset_logits=True,
                )
                loss: Tensor = defect_loss_fn(logits, labels_t)
                loss = loss + asset_loss_weight * asset_loss_fn(asset_logits, asset_idxs_t)
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

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        seen += bs
        asset_preds_t = asset_logits.detach().float().argmax(dim=1)
        all_probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        all_labels.append(labels_t.detach().cpu().numpy())
        all_asset_labels.append(asset_idxs_t.detach().cpu().numpy())
        all_asset_preds.append(asset_preds_t.cpu().numpy())
        bar.set_postfix(loss=f"{total_loss / max(seen, 1):.4f}")

    return (
        total_loss / max(seen, 1),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_asset_labels),
        np.concatenate(all_asset_preds),
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
    parser.add_argument(
        "--asset_loss_weight",
        type=float,
        default=0.3,
        help="Loss weight for automatic asset type classification.",
    )
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    samples = scan_train_dir(args.data_root)
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
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        backbone=args.backbone,
        pretrained=True,
        num_asset_heads=len(ASSET_TYPES),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    defect_loss_fn = nn.BCEWithLogitsLoss()
    asset_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type) if (not args.no_amp and device.type == "cuda") else None

    best_score = -1.0
    best_meta: dict[str, object] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _, _ = run_epoch(
            model,
            train_loader,
            defect_loss_fn,
            asset_loss_fn,
            args.asset_loss_weight,
            optim,
            device,
            scaler,
            f"epoch {epoch} train",
        )
        val_loss, val_probs, val_labels, val_asset_labels, val_asset_preds = run_epoch(
            model,
            val_loader,
            defect_loss_fn,
            asset_loss_fn,
            args.asset_loss_weight,
            None,
            device,
            None,
            f"epoch {epoch} val",
        )
        scheduler.step()

        global_threshold, global_metrics = find_threshold_for_precision(
            val_probs,
            val_labels,
            args.target_precision,
        )
        val_pred_asset_types = np.array([ASSET_TYPES[int(idx)] for idx in val_asset_preds])
        asset_thresholds, asset_metrics = find_asset_thresholds(
            val_probs,
            val_labels,
            val_pred_asset_types,
            args.target_precision,
        )
        metrics = compute_metrics_with_asset_thresholds(
            val_probs,
            val_labels,
            val_pred_asset_types,
            asset_thresholds,
            global_threshold,
        )
        asset_cls_metrics = compute_asset_classification_metrics(
            val_asset_labels,
            val_asset_preds,
        )
        asset_acc = float((val_asset_preds == val_asset_labels).mean())
        score = metric_score(metrics, args.target_precision)
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"P={metrics.precision:.4f} R={metrics.recall:.4f} "
            f"F1={metrics.f1:.4f} acc={metrics.accuracy:.4f} asset_acc={asset_acc:.4f}"
        )
        print_asset_classification_metrics(asset_cls_metrics)

        if score > best_score:
            best_score = score
            best_meta = {
                "epoch": epoch,
                "threshold": global_threshold,
                "asset_thresholds": asset_thresholds,
                "metrics": asdict(metrics),
                "global_threshold_metrics": asdict(global_metrics),
                "asset_metrics": {
                    asset_type: asdict(metric)
                    for asset_type, metric in asset_metrics.items()
                },
                "backbone": args.backbone,
                "img_size": args.img_size,
                "dropout": 0.2,
                "asset_loss_weight": args.asset_loss_weight,
                "asset_types": ASSET_TYPES,
                "asset_classification_metrics": asset_cls_metrics,
                "state_dict": model.state_dict(),
            }
            torch.save(best_meta, args.out_dir / "best_model.pth")
            (args.out_dir / "best_metrics.json").write_text(
                json.dumps(
                    {k: v for k, v in best_meta.items() if k != "state_dict"},
                    indent=2,
                )
            )
            write_validation_error_report(
                args.out_dir / "validation_error_report.csv",
                val_s,
                val_probs,
                val_labels,
                val_asset_labels,
                val_asset_preds,
                asset_thresholds,
                global_threshold,
            )
            print(f"[epoch {epoch:02d}] -> new best (score={score:.4f}) saved")

    print(f"[train] done. best_score={best_score:.4f}")
    if best_meta:
        print(f"[train] best metrics: {best_meta['metrics']}")
        print_asset_thresholds(best_meta.get("asset_thresholds", {}))
        print_asset_metrics(best_meta["asset_metrics"])
        print_asset_classification_metrics(best_meta["asset_classification_metrics"])


if __name__ == "__main__":
    main()
