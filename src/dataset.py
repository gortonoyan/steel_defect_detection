import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config import cfg

def rle_decode(mask_rle, shape=(256, 1600)):
    """
    Decodes run-length encoding into a 2D mask.
    
    Args:
        mask_rle (str or float): Run-length encoded string.
        shape (tuple): Output mask shape (Height, Width).
        
    Returns:
        np.ndarray: Decoded binary mask.
    """
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.float32)

    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F').astype(np.float32)


class SteelDataset(Dataset):
    """
    PyTorch Dataset for Severstal Steel Defect Detection.
    """
    def __init__(self, df, labels_df, img_dir, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'ImageId' for the fold.
            labels_df (pd.DataFrame): Main DataFrame with labels and EncodedPixels.
            img_dir (str): Directory with images.
            transforms (albumentations.Compose): Data augmentations.
        """
        self.df = df.reset_index(drop=True)
        self.labels_df = labels_df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['ImageId']
        img_path = os.path.join(self.img_dir, img_id)

        # Load image in RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create 4-channel mask (256, 1600, 4)
        mask = np.zeros((cfg.hyperparams.IMAGE_HEIGHT, cfg.hyperparams.IMAGE_WIDTH, 4), dtype=np.float32)
        img_labels = self.labels_df[self.labels_df['ImageId'] == img_id]

        for _, row in img_labels.iterrows():
            class_idx = int(row['ClassId']) - 1
            mask[:, :, class_idx] = rle_decode(row['EncodedPixels'])

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask is a tensor and formatted correctly (4, H, W)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
            
        if mask.ndim == 3 and mask.shape[-1] == 4:
            mask = mask.permute(2, 0, 1)

        return image, mask

def get_train_transforms():
    """Returns training augmentations that preserve defect shapes."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussNoise(var_limit=(10.0, 40.0)),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms():
    """Returns validation augmentations (only normalization)."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_dataloaders(train_df, val_df, labels_df):
    """
    Creates DataLoaders for training and validation.
    """
    train_dataset = SteelDataset(
        df=train_df, 
        labels_df=labels_df, 
        img_dir=cfg.paths.TRAIN_IMG_DIR, 
        transforms=get_train_transforms()
    )
    
    val_dataset = SteelDataset(
        df=val_df, 
        labels_df=labels_df, 
        img_dir=cfg.paths.TRAIN_IMG_DIR, 
        transforms=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.hyperparams.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.hyperparams.NUM_WORKERS,
        pin_memory=cfg.hyperparams.PIN_MEMORY,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.hyperparams.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.hyperparams.NUM_WORKERS,
        pin_memory=cfg.hyperparams.PIN_MEMORY,
        drop_last=False
    )

    return train_loader, val_loader
