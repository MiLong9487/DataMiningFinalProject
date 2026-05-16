from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestImageDataset
from model import load_checkpoint
from transforms import build_eval_transform


def predict(
    model_path: Path,
    image_dir: Path,
    template_csv: Path,
    out_csv: Path,
    threshold_override: float | None = None,
    batch_size: int = 64,
    num_workers: int = 6,
) -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] device={device}")

    model, bundle = load_checkpoint(str(model_path), device)
    img_size: int = int(bundle["img_size"])
    threshold: float = (
        threshold_override if threshold_override is not None else float(bundle["threshold"])
    )
    print(f"[predict] backbone={bundle['backbone']} img_size={img_size} threshold={threshold:.4f}")

    ds = TestImageDataset(image_dir, transform=build_eval_transform(img_size))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    fname_to_prob: dict[str, float] = {}
    with torch.no_grad():
        for imgs, names in tqdm(loader, desc="predict"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(imgs)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for name, prob in zip(names, probs):
                fname_to_prob[name] = float(prob)

    # preserve a UTF-8 BOM if the template has one
    raw_head: bytes = template_csv.read_bytes()[:3]
    csv_encoding: str = "utf-8-sig" if raw_head == b"\xef\xbb\xbf" else "utf-8"
    template = pd.read_csv(template_csv, encoding=csv_encoding)
    missing: list[str] = [f for f in template["filename"] if f not in fname_to_prob]
    if missing:
        print(f"[predict] WARN: {len(missing)} files in template not found, e.g. {missing[:3]}")

    template["pred_label"] = template["filename"].map(
        lambda f: int(fname_to_prob.get(f, 0.0) >= threshold)
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out_csv, index=False, encoding=csv_encoding)
    pos: int = int(template["pred_label"].sum())
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
    )


if __name__ == "__main__":
    main()
