from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image

# allow `streamlit run src/app.py` to find sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ASSET_TYPES  # noqa: E402
from model import load_checkpoint  # noqa: E402
from transforms import build_eval_transform  # noqa: E402


DEFAULT_MODEL_PATH = Path("models/best_model.pth")


def get_threshold_for_asset(bundle: dict[str, Any], asset_type: str) -> float:
    raw_thresholds = bundle.get("asset_thresholds", {})
    if isinstance(raw_thresholds, dict) and asset_type in raw_thresholds:
        return float(raw_thresholds[asset_type])
    return float(bundle["threshold"])


@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle = load_checkpoint(path, device)
    return model, bundle, device


def run_inference(
    model: torch.nn.Module,
    img: Image.Image,
    img_size: int,
    device: torch.device,
) -> tuple[float, str, float]:
    transform = build_eval_transform(img_size)
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logit, asset_logits = model(tensor, return_asset_logits=True)
        prob_defective = float(torch.sigmoid(logit.float()).item())
        asset_probs = torch.softmax(asset_logits.float(), dim=1)
        asset_idx = int(asset_probs.argmax(dim=1).item())
        asset_type = ASSET_TYPES[asset_idx]
        asset_confidence = float(asset_probs[0, asset_idx].item())
    return prob_defective, asset_type, asset_confidence


def main() -> None:
    st.set_page_config(
        page_title="UAV InsPLAD Defect Inspector",
        page_icon="\u26A1",
        layout="centered",
    )
    st.title("UAV \u96fb\u529b\u8a2d\u5099\u5143\u4ef6\u6b63\u5e38/\u640d\u6bc0\u5224\u65b7 (InsPLAD)")

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Model checkpoint not found: `{DEFAULT_MODEL_PATH}`. "
            "Run `uv run python src/train.py` first."
        )
        st.stop()

    model, bundle, device = get_model(str(DEFAULT_MODEL_PATH))
    img_size = int(bundle["img_size"])

    st.markdown("**\u8acb\u4e0a\u50b3\u55ae\u5f35\u5716\u7247**")
    uploaded = st.file_uploader("image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.divider()
    st.subheader("\u6a21\u578b\u8cc7\u8a0a")
    st.write(f"model path: `{DEFAULT_MODEL_PATH}`")
    st.write(f"backbone: `{bundle['backbone']}`")
    st.write(f"img_size: `{img_size}`")
    st.write(f"device: `{device.type}`")
    st.write(f"fallback/global threshold: `{float(bundle['threshold']):.4f}`")
    st.divider()

    if uploaded is None:
        st.info("\u8acb\u5148\u4e0a\u50b3\u4e00\u5f35 jpg/png \u5716\u7247\u3002")
        return

    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="\u4e0a\u50b3\u5716\u7247", use_container_width=True)

    prob_defective, asset_type, asset_confidence = run_inference(
        model,
        img,
        img_size,
        device,
    )
    default_thr = get_threshold_for_asset(bundle, asset_type)
    st.markdown("**\u5224\u5b9a\u9580\u6abb (threshold)**")
    threshold = st.slider(
        "threshold",
        0.0,
        1.0,
        value=min(max(default_thr, 0.0), 1.0),
        step=0.001,
        format="%.4f",
    )

    pred_label = "defective" if prob_defective >= threshold else "normal"
    confidence = prob_defective if pred_label == "defective" else 1.0 - prob_defective

    st.subheader("\u63a8\u8ad6\u7d50\u679c")
    st.write(f"pred_label: **{pred_label}**")
    st.write(f"auto_asset_type: **{asset_type}**")
    st.write(f"auto_asset_confidence: `{asset_confidence:.4f}`")
    st.write(f"threshold_used: `{threshold:.4f}`")
    st.write(f"confidence: `{confidence:.4f}`")
    st.write(f"prob_defective: `{prob_defective:.4f}`")
    st.progress(min(max(prob_defective, 0.0), 1.0))


if __name__ == "__main__":
    main()
