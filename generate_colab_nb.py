import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Cell 1 - Title
cells.append(nbf.v4.new_markdown_cell("""# 🔩 Steel Defect Detection — Live Demo
**3-model ensemble: U-Net + FPN + DeepLabV3+**

Run all cells to start the public web demo. The URL at the bottom works for anyone in the world.
"""))

# Cell 2 - Install
cells.append(nbf.v4.new_code_cell("""# Install all dependencies
!pip install -q segmentation-models-pytorch albumentations fastapi uvicorn python-multipart huggingface_hub nest_asyncio"""))

# Cell 3 - Download weights
cells.append(nbf.v4.new_code_cell("""import os
from huggingface_hub import hf_hub_download

HF_REPO = "NarekGabrielyan/steel-defect-detection"
os.makedirs("checkpoints", exist_ok=True)

files = [
    "checkpoints/resnet34_unet_last.pth",
    "checkpoints/fpn_se_resnext50_last.pth",
    "checkpoints/deeplabv3p_efficientnetb3_last.pth",
]

for f in files:
    local = hf_hub_download(repo_id=HF_REPO, filename=f, local_dir=".")
    print(f"Downloaded: {local}")
"""))

# Cell 4 - Clone src/ and static/
cells.append(nbf.v4.new_code_cell("""# Get the src/ and static/ packages from GitHub
!git clone --depth 1 https://github.com/gabrielyannarek04-max/steel-defect-detection.git _repo 2>/dev/null || true
import shutil, os
if os.path.exists("_repo/src"):
    shutil.copytree("_repo/src", "src", dirs_exist_ok=True)
    print("src/ package copied.")
if os.path.exists("_repo/static"):
    shutil.copytree("_repo/static", "static", dirs_exist_ok=True)
    print("static/ package copied.")
"""))

# Cell 5 - Write app
cells.append(nbf.v4.new_code_cell("""%%writefile server.py
import os, sys, base64, cv2, numpy as np, torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.insert(0, ".")
from src.models import build_model

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_AREAS = [300, 300, 1000, 2000]
CONFIGS = [
    ("unet",          "resnet34",           "checkpoints/resnet34_unet_last.pth"),
    ("fpn",           "se_resnext50_32x4d", "checkpoints/fpn_se_resnext50_last.pth"),
    ("deeplabv3plus", "efficientnet-b3",    "checkpoints/deeplabv3p_efficientnetb3_last.pth"),
]

MODELS = []
for arch, enc, path in CONFIGS:
    try:
        m = build_model(arch, enc, encoder_weights=None, classes=4)
        ckpt = torch.load(path, map_location=DEVICE)
        m.load_state_dict(ckpt.get("model_state_dict", ckpt))
        m.to(DEVICE).eval()
        MODELS.append(m)
        print(f"[{arch}] loaded")
    except Exception as e:
        print(f"[{arch}] failed: {e}")

TRANSFORM = A.Compose([A.Resize(256,1600), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
COLORS = [(255,50,50),(50,220,50),(50,100,255),(255,210,50)]

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vis  = cv2.resize(img, (1600, 256))
    t    = TRANSFORM(image=img)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = sum(torch.sigmoid(m(t).squeeze(0)) for m in MODELS) / len(MODELS)
    raw = (probs > 0.5).cpu().numpy().astype(np.uint8)
    found = []
    overlay = vis.astype(np.float32)
    for c in range(4):
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(raw[c], connectivity=8)
        mask = np.zeros_like(raw[c])
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_AREAS[c]:
                mask[lbl == i] = 1
        if mask.sum():
            found.append(c+1)
            layer = np.zeros_like(overlay)
            layer[mask==1] = COLORS[c]
            overlay = cv2.addWeighted(overlay, 1.0, layer, 0.55, 0)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return {"detected_classes": found, "image_base64": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
"""))

# Cell 6 - Start server with cloudflared
cells.append(nbf.v4.new_code_cell("""import time, re

print("Starting FastAPI server...")
!nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
time.sleep(3)

print("Starting Cloudflare Tunnel...")
!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://127.0.0.1:8000 > cloudflared.log 2>&1 &

print("Waiting for public URL (can take up to 20 seconds)...")
url = None
for _ in range(15):
    time.sleep(2)
    try:
        with open("cloudflared.log", "r") as f:
            logs = f.read()
            match = re.search(r"https://[-0-9a-zA-Z]+\.trycloudflare\.com", logs)
            if match:
                url = match.group(0)
                break
    except FileNotFoundError:
        pass

if url:
    print("=" * 60)
    print(f"  LIVE PUBLIC URL: {url}")
    print(f"  Share this with anyone!")
    print("=" * 60)
else:
    print("Failed to get Cloudflare URL. Here are the logs:")
    !cat cloudflared.log
"""))

nb["cells"] = cells

with open("notebooks/02_live_demo_colab.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Colab notebook created: notebooks/02_live_demo_colab.ipynb")
