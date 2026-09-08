import os
import time
import gc
import torch
from tqdm import tqdm
from .config import cfg
from .metrics import compute_dice_score, compute_iou_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})
        
        # Memory management for heavy segmentation models
        del images, masks, outputs, loss
        
    # Free up cache
    torch.cuda.empty_cache()
    gc.collect()
        
    return running_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, threshold):
    """
    Validates the model for one epoch.
    """
    model.eval()
    running_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    
    pbar = tqdm(dataloader, desc="Validation")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item()
        val_dice += compute_dice_score(outputs, masks, threshold=threshold)
        val_iou += compute_iou_score(outputs, masks, threshold=threshold)
        
        # Memory management
        del images, masks, outputs, loss
        
    torch.cuda.empty_cache()
    gc.collect()
        
    return running_loss / len(dataloader), val_dice / len(dataloader), val_iou / len(dataloader)


def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    device, 
    model_name="model"
):
    """
    Full training loop.
    """
    best_dice = 0.0
    best_model_path = os.path.join(cfg.paths.MODELS_DIR, f"{model_name}_best.pth")
    last_model_path = os.path.join(cfg.paths.MODELS_DIR, f"{model_name}_last.pth")
    
    for epoch in range(cfg.hyperparams.EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate_one_epoch(
            model, val_loader, criterion, device, cfg.hyperparams.THRESHOLD
        )
        
        if scheduler:
            scheduler.step()
            
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        
        print(
            f"Epoch [{epoch+1:02d}/{cfg.hyperparams.EPOCHS:02d}] "
            f"({mins:02d}m {secs:02d}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )
        
        # Save checkpoints
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_dice": best_dice,
            "val_dice": val_dice,
            "val_iou": val_iou,
        }
        torch.save(checkpoint, last_model_path)
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(" -> [✓] Best Model Saved!")
            
    print(f"\nTraining Complete. Best Val Dice: {best_dice:.4f}")
    return best_dice
