import os
import sys
import gc
import cv2
import numpy as np
import torch
import gradio as gr
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models import build_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_REPO_ID    = "NarekGabrielyan/steel-defect-detection"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_AREAS     = [300, 300, 1000, 2000]   # per-class noise threshold (pixels)

MODEL_CONFIGS = [
    {"arch": "unet",          "encoder": "resnet34",          "filename": "resnet34_unet_last.pth"},
    {"arch": "fpn",           "encoder": "se_resnext50_32x4d","filename": "fpn_se_resnext50_last.pth"},
    {"arch": "deeplabv3plus", "encoder": "efficientnet-b3",   "filename": "deeplabv3p_efficientnetb3_last.pth"},
]

CLASS_COLORS = {
    0: (255,  50,  50),   # Class 1 - Red
    1: ( 50, 220,  50),   # Class 2 - Green
    2: ( 50, 100, 255),   # Class 3 - Blue
    3: (255, 210,  50),   # Class 4 - Yellow
}
CLASS_NAMES = {
    1: "Class 1 — Surface Cracks",
    2: "Class 2 — Inclusions",
    3: "Class 3 — Patches",
    4: "Class 4 — Scratches",
}

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR     = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

def _ensure(filename: str) -> str:
    local = os.path.join(CKPT_DIR, filename)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    print(f"Downloading {filename} from HF Hub …")
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"checkpoints/{filename}",
        local_dir=BASE_DIR,
    )

print("Loading ensemble models …")
MODELS = []
for cfg in MODEL_CONFIGS:
    try:
        path = _ensure(cfg["filename"])
        m = build_model(cfg["arch"], cfg["encoder"], encoder_weights=None, classes=4)
        ckpt = torch.load(path, map_location=DEVICE)
        m.load_state_dict(ckpt.get("model_state_dict", ckpt))
        m.to(DEVICE).eval()
        MODELS.append(m)
        print(f"  [{cfg['arch']}] loaded ✓")
    except Exception as e:
        print(f"  [{cfg['arch']}] FAILED: {e}")

print(f"Ensemble ready: {len(MODELS)}/3 models loaded on {DEVICE}.")

# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------
TRANSFORM = A.Compose([
    A.Resize(256, 1600),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def predict(pil_image: Image.Image):
    if pil_image is None:
        return None, "Please upload an image."
    if not MODELS:
        return None, "No models loaded. Check Space logs."

    # Pre-process
    img_rgb = np.array(pil_image.convert("RGB"))
    resized = cv2.resize(img_rgb, (1600, 256))
    tensor  = TRANSFORM(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    # Ensemble inference
    with torch.no_grad():
        probs = torch.zeros((4, 256, 1600), device=DEVICE)
        for m in MODELS:
            probs += torch.sigmoid(m(tensor).squeeze(0))
        probs /= len(MODELS)

    raw = (probs > 0.5).cpu().numpy().astype(np.uint8)

    # Min-area noise filter
    filtered = np.zeros_like(raw)
    for c in range(4):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw[c], connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_AREAS[c]:
                filtered[c][labels == i] = 1

    # Build overlay
    overlay = resized.copy().astype(np.float32)
    detected = []
    for c in range(4):
        if filtered[c].sum() == 0:
            continue
        detected.append(c + 1)
        color_layer = np.zeros_like(overlay)
        color_layer[filtered[c] == 1] = CLASS_COLORS[c]
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.55, 0)

    result_img = Image.fromarray(overlay.astype(np.uint8))

    # Summary text
    if detected:
        lines = ["### Detected Anomalies\n"]
        lines += [f"- **{CLASS_NAMES[c]}**" for c in detected]
    else:
        lines = ["### ✅ No defects detected"]

    gc.collect()
    return result_img, "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CSS = """
body { background: #0a0a0f; }
.gradio-container {
    max-width: 900px !important;
    margin: auto;
    font-family: 'Inter', sans-serif;
}
h1 { text-align: center; color: #a5b4fc; font-size: 2.2rem; margin-bottom: 0; }
.subtitle { text-align: center; color: #9ba1a6; margin-top: 0.3rem; margin-bottom: 2rem; font-size: 1rem; }
.gr-button-primary { background: linear-gradient(135deg, #4d7cff, #8b5cf6) !important; border: none !important; }
"""

with gr.Blocks(css=CSS, title="Steel Defect AI") as demo:
    gr.HTML("""
    <h1>🔩 Steel Defect AI</h1>
    <p class="subtitle">
        Upload a steel surface image &mdash; our 3-model ensemble (U-Net + FPN + DeepLabV3+)
        will detect and segment surface anomalies in real-time.
    </p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(
                type="pil",
                label="Upload Steel Surface Image",
                height=300,
            )
            btn = gr.Button("Analyze Surface", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_img  = gr.Image(label="Defect Overlay", height=300)
            out_text = gr.Markdown(value="Results will appear here after analysis.")

    btn.click(fn=predict, inputs=inp, outputs=[out_img, out_text])
    inp.change(fn=predict, inputs=inp, outputs=[out_img, out_text])

    gr.HTML("""
    <div style="text-align:center; color:#555; margin-top:2rem; font-size:0.85rem;">
        <b>Color legend:</b>
        &nbsp; 🔴 Class 1 (Cracks)
        &nbsp; 🟢 Class 2 (Inclusions)
        &nbsp; 🔵 Class 3 (Patches)
        &nbsp; 🟡 Class 4 (Scratches)
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
