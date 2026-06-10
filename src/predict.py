from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ASSET_TYPES, TestImageDataset
from model import load_checkpoint
from transforms import build_eval_transform


def get_csv_encoding(path: Path) -> str:
    raw_head = path.read_bytes()[:3]
    return "utf-8-sig" if raw_head == b"\xef\xbb\xbf" else "utf-8"


def get_threshold_for_asset(
    bundle: dict[str, object],
    asset_type: str,
    threshold_override: float | None,
) -> float:
    if threshold_override is not None:
        return threshold_override
    raw_thresholds = bundle.get("asset_thresholds", {})
    if isinstance(raw_thresholds, dict) and asset_type in raw_thresholds:
        return float(raw_thresholds[asset_type])
    return float(bundle["threshold"])


def find_ensemble_model_paths(ensemble_dir: Path) -> list[Path]:
    paths = sorted(ensemble_dir.glob("fold_*/best_model.pth"))
    if not paths:
        raise SystemExit(f"No fold checkpoints found under {ensemble_dir}/fold_*/best_model.pth")
    return paths


def get_ensemble_thresholds(
    bundles: list[dict[str, object]],
    asset_types: tuple[str, ...],
    threshold_override: float | None,
) -> dict[str, float]:
    if threshold_override is not None:
        return {asset_type: threshold_override for asset_type in asset_types}

    thresholds: dict[str, float] = {}
    for asset_type in asset_types:
        values: list[float] = []
        for bundle in bundles:
            raw_thresholds = bundle.get("asset_thresholds", {})
            if isinstance(raw_thresholds, dict) and asset_type in raw_thresholds:
                values.append(float(raw_thresholds[asset_type]))
        if values:
            thresholds[asset_type] = float(np.mean(values))
    return thresholds


def predict_ensemble_batch(
    models: list[torch.nn.Module],
    imgs: Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    asset_prob_sum: Tensor | None = None
    asset_logits_by_model: list[Tensor] = []
    with torch.amp.autocast(device_type=imgs.device.type, enabled=imgs.device.type == "cuda"):
        for model in models:
            _, asset_logits = model(imgs, return_asset_logits=True)
            asset_probs = torch.softmax(asset_logits.float(), dim=1)
            asset_prob_sum = asset_probs if asset_prob_sum is None else asset_prob_sum + asset_probs
            asset_logits_by_model.append(asset_logits)

        assert asset_prob_sum is not None
        mean_asset_probs = asset_prob_sum / len(models)
        selected_asset_idxs = mean_asset_probs.argmax(dim=1)

        prob_sum: Tensor | None = None
        for model in models:
            logits, _ = model(
                imgs,
                selected_asset_idxs,
                return_asset_logits=True,
            )
            probs = torch.sigmoid(logits.float())
            prob_sum = probs if prob_sum is None else prob_sum + probs

        assert prob_sum is not None
        mean_probs = prob_sum / len(models)
    return mean_probs.detach().cpu().numpy(), selected_asset_idxs.detach().cpu().numpy()


def predict(
    model_path: Path,
    image_dir: Path,
    template_csv: Path,
    out_csv: Path,
    threshold_override: float | None = None,
    batch_size: int = 64,
    num_workers: int = 6,
    ensemble: bool = False,
    ensemble_dir: Path = Path("models"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] device={device}")

    models: list[torch.nn.Module]
    bundles: list[dict[str, object]]
    if ensemble:
        model_paths = find_ensemble_model_paths(ensemble_dir)
        loaded = [load_checkpoint(str(path), device) for path in model_paths]
        models = [item[0] for item in loaded]
        bundles = [item[1] for item in loaded]
        bundle = bundles[0]
        print(f"[predict] ensemble=true models={len(models)}")
    else:
        model, bundle = load_checkpoint(str(model_path), device)
        models = [model]
        bundles = [bundle]

    img_size = int(bundle["img_size"])
    global_threshold = float(
        threshold_override if threshold_override is not None else bundle["threshold"]
    )
    print(
        f"[predict] backbone={bundle['backbone']} img_size={img_size} "
        f"fallback_threshold={global_threshold:.4f}"
    )

    ds = TestImageDataset(image_dir, transform=build_eval_transform(img_size))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    asset_types = tuple(str(asset) for asset in bundle.get("asset_types", ASSET_TYPES))
    ensemble_thresholds = get_ensemble_thresholds(bundles, asset_types, threshold_override)
    fname_to_prob: dict[str, float] = {}
    fname_to_asset_type: dict[str, str] = {}
    fname_to_threshold: dict[str, float] = {}
    with torch.no_grad():
        for imgs, names in tqdm(loader, desc="predict"):
            imgs = imgs.to(device, non_blocking=True)
            if ensemble:
                probs, pred_asset_idxs = predict_ensemble_batch(models, imgs)
            else:
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits, asset_logits = models[0](imgs, return_asset_logits=True)
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                pred_asset_idxs = asset_logits.detach().float().argmax(dim=1).cpu().numpy()
            for name, prob, pred_asset_idx in zip(names, probs, pred_asset_idxs):
                filename = str(name)
                asset_type = asset_types[int(pred_asset_idx)]
                fname_to_prob[filename] = float(prob)
                fname_to_asset_type[filename] = asset_type
                if ensemble:
                    fname_to_threshold[filename] = ensemble_thresholds.get(
                        asset_type,
                        global_threshold,
                    )
                else:
                    fname_to_threshold[filename] = get_threshold_for_asset(
                        bundle,
                        asset_type,
                        threshold_override,
                    )

    csv_encoding = get_csv_encoding(template_csv)
    template = pd.read_csv(template_csv, encoding=csv_encoding)
    missing = [str(f) for f in template["filename"] if str(f) not in fname_to_prob]
    if missing:
        print(f"[predict] WARN: {len(missing)} files in template not found, e.g. {missing[:3]}")

    template["pred_asset_type"] = template["filename"].map(
        lambda f: fname_to_asset_type.get(str(f), "")
    )
    template["threshold_used"] = template["filename"].map(
        lambda f: fname_to_threshold.get(str(f), global_threshold)
    )
    template["pred_label"] = template["filename"].map(
        lambda f: int(
            fname_to_prob.get(str(f), 0.0)
            >= fname_to_threshold.get(str(f), global_threshold)
        )
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out_csv, index=False, encoding=csv_encoding)
    pos = int(template["pred_label"].sum())
    print(
        f"[predict] wrote {out_csv}  total={len(template)}  defective={pos}  "
        f"normal={len(template) - pos}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--image_dir", type=Path, default=Path("data/test_dataset/images"))
    parser.add_argument("--template", type=Path, default=Path("test_submission_template.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/test_submission.csv"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--ensemble_dir", type=Path, default=Path("models"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    args = parser.parse_args()

    predict(
        model_path=args.model,
        image_dir=args.image_dir,
        template_csv=args.template,
        out_csv=args.out,
        threshold_override=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ensemble=args.ensemble,
        ensemble_dir=args.ensemble_dir,
    )


if __name__ == "__main__":
    main()
