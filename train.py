import argparse
import os
import pandas as pd
import torch
import torch.optim as optim

from src.config import cfg
from src.dataset import get_dataloaders
from src.models import build_model
from src.metrics import CombinedLoss
from src.engine import train_model

def main(args):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DataFrames
    print("Loading data...")
    if not os.path.exists(cfg.paths.TRAIN_CSV) or not os.path.exists(cfg.paths.TRAIN_FOLDS):
        raise FileNotFoundError("Data files not found. Please ensure kaggle data is downloaded and train_folds.csv is generated via the EDA notebook.")

    labels_df = pd.read_csv(cfg.paths.TRAIN_CSV)
    df_folds = pd.read_csv(cfg.paths.TRAIN_FOLDS)

    # We use fold 0 for validation, others for training
    train_df = df_folds[df_folds.fold != 0].copy()
    val_df = df_folds[df_folds.fold == 0].copy()

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(train_df, val_df, labels_df)

    # Build Model
    print(f"Building {args.model_name} with {args.encoder_name} encoder...")
    model = build_model(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights="imagenet",
        classes=4
    )
    model = model.to(device)

    # Optimizer, Criterion, Scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.hyperparams.LEARNING_RATE, 
        weight_decay=cfg.hyperparams.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = CombinedLoss(
        bce_weights=cfg.hyperparams.BCE_WEIGHTS, 
        alpha=cfg.hyperparams.ALPHA
    )

    # Train
    model_save_name = f"{args.model_name}_{args.encoder_name}"
    print(f"Starting training for {cfg.hyperparams.EPOCHS} epochs...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_name=model_save_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation Model for Steel Defect Detection")
    parser.add_argument("--model_name", type=str, default="unet", help="Model architecture: 'unet', 'fpn', 'deeplabv3plus'")
    parser.add_argument("--encoder_name", type=str, default="resnet34", help="Backbone encoder, e.g. 'resnet34', 'se_resnext50_32x4d'")
    args = parser.parse_args()
    
    main(args)
