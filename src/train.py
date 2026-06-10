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
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def stratified_folds(
    samples: list[Sample],
    folds: int,
    seed: int,
    val_ratio: float,
) -> list[tuple[int, list[Sample], list[Sample]]]:
    if folds <= 1:
        train_s, val_s = stratified_split(samples, val_ratio, seed)
        return [(1, train_s, val_s)]

    keys = np.array([f"{s.asset_type}|{s.label}" for s in samples])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_splits: list[tuple[int, list[Sample], list[Sample]]] = []
    indices = np.arange(len(samples))
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(indices, keys), start=1):
        train_s = [samples[int(i)] for i in train_idx]
        val_s = [samples[int(i)] for i in val_idx]
        fold_splits.append((fold_idx, train_s, val_s))
    return fold_splits


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


def summarize_cross_validation(
    fold_results: list[dict[str, object]],
    epoch_results: list[dict[str, object]],
) -> dict[str, object]:
    metric_keys = ("precision", "recall", "f1", "accuracy")
    summary: dict[str, object] = {
        "folds": len(fold_results),
        "metrics": {},
        "asset_thresholds": {},
        "epoch_selection": {},
    }
    for key in metric_keys:
        values = np.array([float(result["metrics"][key]) for result in fold_results])
        summary["metrics"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "values": values.tolist(),
        }

    for asset_type in ASSET_TYPES:
        values = [
            float(result["asset_thresholds"][asset_type])
            for result in fold_results
            if asset_type in result["asset_thresholds"]
        ]
        if values:
            arr = np.array(values)
            summary["asset_thresholds"][asset_type] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "values": arr.tolist(),
            }

    epoch_stats: list[dict[str, object]] = []
    if epoch_results:
        epochs = sorted({int(result["epoch"]) for result in epoch_results})
        for epoch in epochs:
            scores = np.array(
                [
                    float(result["score"])
                    for result in epoch_results
                    if int(result["epoch"]) == epoch
                ]
            )
            epoch_stats.append(
                {
                    "epoch": epoch,
                    "mean_score": float(scores.mean()),
                    "std_score": float(scores.std(ddof=0)),
                    "scores": scores.tolist(),
                }
            )
        best_epoch_item = sorted(
            epoch_stats,
            key=lambda item: (
                -float(item["mean_score"]),
                float(item["std_score"]),
                int(item["epoch"]),
            ),
        )[0]
        summary["epoch_selection"] = {
            "selected_epoch": int(best_epoch_item["epoch"]),
            "rule": "highest mean score, then lowest std score, then earliest epoch",
            "epoch_stats": epoch_stats,
        }
    return summary


def print_cross_validation_summary(summary: dict[str, object]) -> None:
    print("[train] cross-validation summary:")
    metrics = summary["metrics"]
    for key in ("precision", "recall", "f1", "accuracy"):
        item = metrics[key]
        print(f"  {key}: mean={item['mean']:.4f} std={item['std']:.4f}")

    print("[train] cross-validation asset thresholds:")
    thresholds = summary["asset_thresholds"]
    for asset_type in ASSET_TYPES:
        item = thresholds.get(asset_type)
        if item is not None:
            print(
                f"  {asset_type}: mean={item['mean']:.4f} "
                f"std={item['std']:.4f}"
            )
    epoch_selection = summary.get("epoch_selection", {})
    if isinstance(epoch_selection, dict) and epoch_selection:
        print(
            "[train] selected final epoch: "
            f"{epoch_selection['selected_epoch']} "
            f"({epoch_selection['rule']})"
        )


def mean_asset_thresholds(summary: dict[str, object]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    summary_thresholds = summary.get("asset_thresholds", {})
    if not isinstance(summary_thresholds, dict):
        return thresholds
    for asset_type in ASSET_TYPES:
        item = summary_thresholds.get(asset_type)
        if isinstance(item, dict) and "mean" in item:
            thresholds[asset_type] = float(item["mean"])
    return thresholds


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


def train_final_model(
    samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
    asset_thresholds: dict[str, float],
    cv_summary: dict[str, object],
    final_epochs: int,
) -> dict[str, object]:
    print("[final] training final model on all training data")
    set_seed(args.seed)
    train_ds = InsPLADDataset(samples, transform=build_train_transform(args.img_size))
    sampler = build_sampler(samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    model = build_model(
        backbone=args.backbone,
        pretrained=True,
        num_asset_heads=len(ASSET_TYPES),
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=final_epochs)
    defect_loss_fn = nn.BCEWithLogitsLoss()
    asset_loss_fn = nn.CrossEntropyLoss()
    scaler = (
        torch.amp.GradScaler(device.type)
        if (not args.no_amp and device.type == "cuda")
        else None
    )

    last_train_loss = 0.0
    for epoch in range(1, final_epochs + 1):
        last_train_loss, _, _, _, _ = run_epoch(
            model,
            train_loader,
            defect_loss_fn,
            asset_loss_fn,
            args.asset_loss_weight,
            optim,
            device,
            scaler,
            f"final epoch {epoch} train",
        )
        scheduler.step()
        print(f"[final epoch {epoch:02d}] train_loss={last_train_loss:.4f}")

    fallback_threshold = (
        float(np.mean(list(asset_thresholds.values())))
        if asset_thresholds
        else 0.5
    )
    final_meta: dict[str, object] = {
        "final_model": True,
        "trained_on_all_data": True,
        "epochs": final_epochs,
        "requested_max_epochs": args.epochs,
        "final_epoch_selection": cv_summary.get("epoch_selection", {}),
        "threshold": fallback_threshold,
        "asset_thresholds": asset_thresholds,
        "crossval_summary": cv_summary,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "dropout": 0.2,
        "asset_loss_weight": args.asset_loss_weight,
        "asset_types": ASSET_TYPES,
        "last_train_loss": last_train_loss,
        "state_dict": model.state_dict(),
    }
    return final_meta


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
        "--folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds. Use 1 for a single split.",
    )
    parser.add_argument(
        "--no_train_final",
        action="store_true",
        help="Skip training the final model on all training data after cross-validation.",
    )
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
    fold_splits = stratified_folds(samples, args.folds, args.seed, args.val_ratio)
    print(f"[train] folds={len(fold_splits)}")

    fold_results: list[dict[str, object]] = []
    epoch_results: list[dict[str, object]] = []
    best_overall_score = -1.0
    best_overall_meta: dict[str, object] = {}
    for fold_idx, train_s, val_s in fold_splits:
        fold_seed = args.seed + fold_idx - 1
        set_seed(fold_seed)
        fold_dir = args.out_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"[fold {fold_idx:02d}] train={len(train_s)} val={len(val_s)}")

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
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        defect_loss_fn = nn.BCEWithLogitsLoss()
        asset_loss_fn = nn.CrossEntropyLoss()
        scaler = (
            torch.amp.GradScaler(device.type)
            if (not args.no_amp and device.type == "cuda")
            else None
        )

        best_fold_score = -1.0
        best_fold_meta: dict[str, object] = {}
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
                f"fold {fold_idx:02d} epoch {epoch} train",
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
                f"fold {fold_idx:02d} epoch {epoch} val",
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
            epoch_results.append(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "score": score,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "accuracy": metrics.accuracy,
                    "asset_accuracy": asset_acc,
                }
            )
            print(
                f"[fold {fold_idx:02d} epoch {epoch:02d}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"P={metrics.precision:.4f} R={metrics.recall:.4f} "
                f"F1={metrics.f1:.4f} acc={metrics.accuracy:.4f} "
                f"asset_acc={asset_acc:.4f}"
            )

            if score > best_fold_score:
                best_fold_score = score
                best_fold_meta = {
                    "fold": fold_idx,
                    "seed": fold_seed,
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
                torch.save(best_fold_meta, fold_dir / "best_model.pth")
                (fold_dir / "best_metrics.json").write_text(
                    json.dumps(
                        {k: v for k, v in best_fold_meta.items() if k != "state_dict"},
                        indent=2,
                    )
                )
                write_validation_error_report(
                    fold_dir / "validation_error_report.csv",
                    val_s,
                    val_probs,
                    val_labels,
                    val_asset_labels,
                    val_asset_preds,
                    asset_thresholds,
                    global_threshold,
                )
                print(
                    f"[fold {fold_idx:02d} epoch {epoch:02d}] "
                    f"-> new fold best (score={score:.4f}) saved"
                )

        fold_result = {k: v for k, v in best_fold_meta.items() if k != "state_dict"}
        fold_results.append(fold_result)
        if best_fold_score > best_overall_score:
            best_overall_score = best_fold_score
            best_overall_meta = best_fold_meta
            torch.save(best_overall_meta, args.out_dir / "best_model.pth")
            (args.out_dir / "best_metrics.json").write_text(
                json.dumps(
                    {k: v for k, v in best_overall_meta.items() if k != "state_dict"},
                    indent=2,
                )
            )

        print(f"[fold {fold_idx:02d}] best_score={best_fold_score:.4f}")
        print_asset_thresholds(best_fold_meta.get("asset_thresholds", {}))
        print_asset_metrics(best_fold_meta["asset_metrics"])
        print_asset_classification_metrics(best_fold_meta["asset_classification_metrics"])

    summary = summarize_cross_validation(fold_results, epoch_results)
    summary["fold_results"] = fold_results
    (args.out_dir / "crossval_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print_cross_validation_summary(summary)

    if not args.no_train_final:
        final_thresholds = mean_asset_thresholds(summary)
        epoch_selection = summary.get("epoch_selection", {})
        final_epochs = (
            int(epoch_selection["selected_epoch"])
            if isinstance(epoch_selection, dict) and "selected_epoch" in epoch_selection
            else args.epochs
        )
        print("[final] using cross-validation mean asset thresholds:")
        print_asset_thresholds(final_thresholds)
        print(f"[final] using selected final epochs: {final_epochs}")
        final_meta = train_final_model(
            samples,
            args,
            device,
            final_thresholds,
            summary,
            final_epochs,
        )
        torch.save(final_meta, args.out_dir / "best_model.pth")
        (args.out_dir / "best_metrics.json").write_text(
            json.dumps(
                {k: v for k, v in final_meta.items() if k != "state_dict"},
                indent=2,
            )
        )
        print("[final] wrote final all-data model to models/best_model.pth")

    print(
        f"[train] done. best_overall_score={best_overall_score:.4f} "
        f"best_fold={best_overall_meta.get('fold')}"
    )


if __name__ == "__main__":
    main()
