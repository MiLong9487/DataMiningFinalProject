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


DEFAULT_MODEL_PATH: Path = Path("models/best_model.pth")


@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle = load_checkpoint(path, device)
    return model, bundle, device


def run_inference(
    model: torch.nn.Module,
    img: Image.Image,
    img_size: int,
    device: torch.device,
) -> float:
    transform = build_eval_transform(img_size)
    tensor: torch.Tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logit = model(tensor)
        prob: float = float(torch.sigmoid(logit.float()).item())
    return prob


def main() -> None:
    st.set_page_config(page_title="UAV InsPLAD Defect Inspector", page_icon="\u26A1", layout="centered")
    st.title("UAV \u96fb\u529b\u8a2d\u5099\u5143\u4ef6\u6b63\u5e38/\u640d\u6bc0\u5224\u65b7 (InsPLAD)")

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Model checkpoint not found: `{DEFAULT_MODEL_PATH}`. "
            "Run `uv run python src/train.py` first."
        )
        st.stop()

    model, bundle, device = get_model(str(DEFAULT_MODEL_PATH))
    img_size: int = int(bundle["img_size"])
    default_thr: float = float(bundle["threshold"])

    st.markdown("**\u8acb\u9078\u64c7\u5143\u4ef6\u985e\u578b**")
    asset_type: str = st.selectbox("asset type", ASSET_TYPES, index=ASSET_TYPES.index("vari-grip"))

    st.markdown("**\u5224\u5b9a\u9580\u6abb (threshold)**")
    threshold: float = st.slider("threshold", 0.0, 1.0, value=round(default_thr, 2), step=0.01)

    st.markdown("**\u8acb\u4e0a\u50b3\u55ae\u5f35\u5716\u7247**")
    uploaded = st.file_uploader("image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.divider()
    st.subheader("\u6a21\u578b\u8cc7\u8a0a")
    st.write(f"model path: `{DEFAULT_MODEL_PATH}`")
    st.write(f"backbone: `{bundle['backbone']}`")
    st.write(f"img_size: `{img_size}`")
    st.write(f"device: `{device.type}`")
    st.write(f"default threshold: `{default_thr:.4f}`")
    st.write(f"asset_type (\u4f7f\u7528\u8005\u9078\u64c7): `{asset_type}`")
    st.divider()

    if uploaded is None:
        st.info("\u8acb\u5148\u4e0a\u50b3\u4e00\u5f35 jpg/png \u5716\u7247\u3002")
        return

    img: Image.Image = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="\u4e0a\u50b3\u5716\u7247", use_container_width=True)

    prob_defective: float = run_inference(model, img, img_size, device)
    pred_label: str = "defective" if prob_defective >= threshold else "normal"
    confidence: float = prob_defective if pred_label == "defective" else 1.0 - prob_defective

    st.subheader("\u63a8\u8ad6\u7d50\u679c")
    st.write(f"pred_label: **{pred_label}**")
    st.write(f"confidence: `{confidence:.4f}`")
    st.write(f"prob_defective: `{prob_defective:.4f}`")
    st.progress(min(max(prob_defective, 0.0), 1.0))


if __name__ == "__main__":
    main()
