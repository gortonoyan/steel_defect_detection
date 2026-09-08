import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List, Optional

class CombinedLoss(nn.Module):
    """
    Combined BCE and Dice Loss for segmentation.
    """
    def __init__(self, bce_weights: Optional[List[float]] = None, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        
        pos_weight = None
        if bce_weights is not None:
            pos_weight = torch.tensor(bce_weights, dtype=torch.float32)
            
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = smp.losses.DiceLoss(mode="multilabel")

    def forward(self, logits, targets):
        if self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
            
        return self.bce(logits, targets) + self.alpha * self.dice(logits, targets)

def compute_dice_score(logits, targets, threshold=0.5, eps=1e-7):
    """
    Computes Dice coefficient score for multi-label classification.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    total_area = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2.0 * intersection + eps) / (total_area + eps)

    class_mask = (targets.sum(dim=(2, 3)) > 0).float()
    if class_mask.sum() > 0:
        return (dice * class_mask).sum().item() / class_mask.sum().item()
    return dice.mean().item()

def compute_iou_score(logits, targets, threshold=0.5, eps=1e-7):
    """
    Computes Intersection over Union (IoU) score.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)

    class_mask = (targets.sum(dim=(2, 3)) > 0).float()
    if class_mask.sum() > 0:
        return (iou * class_mask).sum().item() / class_mask.sum().item()
    return iou.mean().item()
