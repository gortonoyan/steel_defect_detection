import os
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import build_model

app = FastAPI(title="Steel Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []

# Hugging Face Hub repo where model weights are stored
HF_REPO_ID = "NarekGabrielyan/steel-defect-detection"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

MODEL_CONFIGS = [
    {
        "arch": "unet",
        "encoder": "resnet34",
        "filename": "resnet34_unet_last.pth",
    },
    {
        "arch": "fpn",
        "encoder": "se_resnext50_32x4d",
        "filename": "fpn_se_resnext50_last.pth",
    },
    {
        "arch": "deeplabv3plus",
        "encoder": "efficientnet-b3",
        "filename": "deeplabv3p_efficientnetb3_last.pth",
    },
]

# Per-class minimum area thresholds (pixels) — removes small noise predictions
MIN_AREAS = [300, 300, 1000, 2000]

def ensure_checkpoint(filename: str) -> str:
    """Returns local path to a checkpoint, downloading from HF Hub if absent."""
    local_path = os.path.join(CHECKPOINTS_DIR, filename)
    if os.path.exists(local_path):
        return local_path

    print(f"  Checkpoint '{filename}' not found locally. Downloading from Hugging Face Hub...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"checkpoints/{filename}",
            local_dir=BASE_DIR,
        )
        print(f"  Downloaded to: {downloaded}")
        return downloaded
    except Exception as e:
        raise RuntimeError(
            f"Could not download '{filename}' from HF Hub ({HF_REPO_ID}). "
            f"Error: {e}"
        )


@app.on_event("startup")
async def load_models():
    global models_list
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    print(f"Loading ensemble on device: {device}")

    for cfg in MODEL_CONFIGS:
        try:
            ckpt_path = ensure_checkpoint(cfg["filename"])

            m = build_model(
                model_name=cfg["arch"],
                encoder_name=cfg["encoder"],
                encoder_weights=None,
                classes=4,
            )
            checkpoint = torch.load(ckpt_path, map_location=device)
            state = checkpoint.get("model_state_dict", checkpoint)
            m.load_state_dict(state)
            m.to(device).eval()
            models_list.append(m)
            print(f"  [{cfg['arch']}] loaded ✓")
        except Exception as e:
            print(f"  [{cfg['arch']}] FAILED to load: {e}")

    print(f"Ensemble ready: {len(models_list)}/3 models loaded.")


def preprocess_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid or unreadable image file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(256, 1600),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor = transform(image=img)["image"].unsqueeze(0)
    resized = A.Resize(256, 1600)(image=img)["image"]
    return resized, tensor


def postprocess(preds: np.ndarray) -> np.ndarray:
    """Remove connected components smaller than per-class min_area."""
    filtered = np.zeros_like(preds)
    for c in range(4):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(preds[c], connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_AREAS[c]:
                filtered[c][labels == i] = 1
    return filtered


def apply_overlay(original_img: np.ndarray, preds: np.ndarray):
    COLORS = {
        0: [255,   0,   0, 140],  # Class 1 → Red
        1: [  0, 255,   0, 140],  # Class 2 → Green
        2: [  0,   0, 255, 140],  # Class 3 → Blue
        3: [255, 255,   0, 140],  # Class 4 → Yellow
    }
    rgba = cv2.cvtColor(original_img, cv2.COLOR_RGB2RGBA)
    detected = []
    for i in range(4):
        if preds[i].sum() == 0:
            continue
        detected.append(i + 1)
        overlay = np.zeros_like(rgba)
        overlay[preds[i] == 1] = COLORS[i]
        rgba = cv2.addWeighted(rgba, 1.0, overlay, 0.55, 0)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), detected


class InferenceResponse(BaseModel):
    detected_classes: list[int]
    image_base64: str


@app.post("/api/predict", response_model=InferenceResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not models_list:
        raise HTTPException(status_code=503, detail="No models are loaded. Check server logs.")

    contents = await file.read()
    try:
        original_img, img_tensor = preprocess_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        probs = torch.zeros((4, 256, 1600), device=device)
        for m in models_list:
            probs += torch.sigmoid(m(img_tensor).squeeze(0))
        probs /= len(models_list)

    raw_preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
    filtered_preds = postprocess(raw_preds)

    result_img, detected = apply_overlay(original_img, filtered_preds)

    _, buffer = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return InferenceResponse(
        detected_classes=detected,
        image_base64=f"data:image/jpeg;base64,{img_b64}",
    )


STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
