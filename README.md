<div align="center">

# 🔩 Severstal Steel Defect Detection

**Production-ready ensemble segmentation for industrial surface defect analysis**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-FFD21E?style=flat)](https://huggingface.co/spaces/NarekGabrielyan/steel-defect-detection)
[![Kaggle](https://img.shields.io/badge/Kaggle-Severstal-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/severstal-steel-defect-detection)

*Upload a steel surface image → get instant defect segmentation with class-level color masks*

</div>

---

## ✨ Live Demo

👉 **[Try it now on Hugging Face Spaces](https://huggingface.co/spaces/NarekGabrielyan/steel-defect-detection)**

---

## 🏗️ Architecture

This project detects **4 classes of surface defects** on steel using a **3-model ensemble** with post-processing noise filtering.

| Model | Backbone | Weights |
|-------|----------|---------|
| U-Net | ResNet-34 | ImageNet |
| FPN | SE-ResNeXt50-32×4d | ImageNet |
| DeepLabV3+ | EfficientNet-B3 | ImageNet |

**Inference pipeline:**
1. Resize input image → `256×1600` (training resolution)
2. Average sigmoid probability maps across all 3 models
3. Threshold at `0.5`
4. Remove small noise components per class (min-area filter)
5. Overlay color-coded masks on original image

```
Steel_defect_detection/
├── api/
│   └── app.py            # FastAPI backend — ensemble inference, HF Hub weight download
├── src/
│   ├── config.py         # Centralized hyperparameters & paths
│   ├── dataset.py        # SteelDataset, RLE decode, Albumentations pipelines
│   ├── engine.py         # train_one_epoch / validate_one_epoch with memory management
│   ├── metrics.py        # CombinedLoss (BCE + Dice), Dice score, IoU
│   └── models.py         # Model factory (Unet / FPN / DeepLabV3+)
├── static/               # Premium Web UI (dark mode, glassmorphism, drag-and-drop)
├── notebooks/
│   └── 01_EDA.ipynb      # EDA, stratified splits, mask visualization
├── Dockerfile            # HF Spaces & local Docker
├── docker-compose.yml    # Local one-command launch
└── requirements.txt
```

---

## 🚀 Run Locally

### Option A — Docker (recommended)
```bash
git clone https://github.com/gortonoyan/steel_defect_detection.git
cd severstal-steel-defect-detection
docker-compose up --build
```
Open **http://localhost:8000**

> Model weights are automatically downloaded from [Hugging Face Hub](https://huggingface.co/NarekGabrielyan/steel-defect-detection) on first run.

### Option B — Python
```bash
git clone https://github.com/gortonoyan/steel_defect_detection.git
cd severstal-steel-defect-detection
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🎯 Defect Classes

| Class | Color | Description |
|-------|-------|-------------|
| 1 | 🔴 Red | Surface cracks |
| 2 | 🟢 Green | Inclusions |
| 3 | 🔵 Blue | Patches |
| 4 | 🟡 Yellow | Scratches |

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Upload image → returns overlay + detected classes |
| `GET`  | `/docs`        | Interactive Swagger UI |

**Example curl:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@your_steel_image.jpg" | python -m json.tool
```

---

## 🏋️ Training (optional)

If you want to retrain the models:

```bash
# Download Kaggle dataset (requires kaggle.json in ~/.kaggle/)
# Run EDA and generate train_folds.csv
jupyter notebook notebooks/01_EDA.ipynb

# Train with any architecture
python train.py --model_name unet --encoder_name resnet34
python train.py --model_name fpn --encoder_name se_resnext50_32x4d
python train.py --model_name deeplabv3plus --encoder_name efficientnet-b3
```

---

## 🛠️ ML Engineering Highlights

- **Ensemble averaging** across 3 heterogeneous architectures for robust predictions
- **Min-area filtering** via `cv2.connectedComponentsWithStats` removes noise blobs
- **Automatic weight download** from HF Hub — zero manual setup
- **Memory management** — `torch.cuda.empty_cache()` + `gc.collect()` in training loops
- **Multilabel stratified K-Fold** split via `iterative-stratification`
- **Combined Loss** = BCE with class weights + Dice Loss
