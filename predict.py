import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models import build_model
from src.config import cfg

def load_image(image_path):
    """Loads an image and applies validation transformations."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
    return image, image_tensor

@torch.no_grad()
def predict(image_path, model_path, model_name, encoder_name, threshold=0.5):
    """Runs inference on a single image and plots the results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {model_name} with {encoder_name}...")
    model = build_model(
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights=None,
        classes=4
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    # Load Image
    original_image, image_tensor = load_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Predict
    output = model(image_tensor).squeeze(0)  # Shape [4, H, W]
    probs = torch.sigmoid(output)
    preds = (probs > threshold).cpu().numpy().astype(np.uint8)
    
    # Visualize
    plt.figure(figsize=(15, 10))
    plt.imshow(original_image)
    
    colors = {
        0: [1, 0, 0, 0.5],  # Class 1 -> Red
        1: [0, 1, 0, 0.5],  # Class 2 -> Green
        2: [0, 0, 1, 0.5],  # Class 3 -> Blue
        3: [1, 1, 0, 0.5],  # Class 4 -> Yellow
    }
    
    for i in range(4):
        mask = preds[i]
        if mask.sum() > 0:
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask == 1] = colors[i]
            plt.imshow(colored_mask)
            print(f"Detected Defect Class {i+1}")
            
    plt.axis("off")
    plt.title(f"Prediction: {os.path.basename(image_path)}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Steel Defect Detection")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth model weights")
    parser.add_argument("--model_name", type=str, default="unet", help="Model architecture")
    parser.add_argument("--encoder_name", type=str, default="resnet34", help="Backbone encoder")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    predict(args.image_path, args.model_path, args.model_name, args.encoder_name, args.threshold)
