from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> BinaryMetrics:
    preds: np.ndarray = (probs >= threshold).astype(np.int32)
    tp: int = int(((preds == 1) & (labels == 1)).sum())
    fp: int = int(((preds == 1) & (labels == 0)).sum())
    tn: int = int(((preds == 0) & (labels == 0)).sum())
    fn: int = int(((preds == 0) & (labels == 1)).sum())
    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1: float = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy: float = (tp + tn) / max(len(labels), 1)
    return BinaryMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def find_threshold_for_precision(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.90,
) -> tuple[float, BinaryMetrics]:
    """Pick the smallest threshold whose precision >= target, maximizing recall.

    Falls back to the threshold with the highest precision if the target
    cannot be reached on the given set.
    """
    candidates: list[float] = sorted(set(np.round(probs, 4).tolist()) | {0.5})
    best: BinaryMetrics | None = None
    fallback: BinaryMetrics | None = None
    for thr in candidates:
        m = compute_metrics(probs, labels, thr)
        if fallback is None or m.precision > fallback.precision or (
            m.precision == fallback.precision and m.recall > fallback.recall
        ):
            fallback = m
        if m.precision >= target_precision:
            if best is None or m.recall > best.recall:
                best = m
    chosen: BinaryMetrics = best if best is not None else fallback  # type: ignore[assignment]
    return chosen.threshold, chosen
