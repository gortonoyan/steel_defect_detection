import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Paths:
    """Directory and file paths used in the project."""
    DATA_DIR: str = os.path.join(os.getcwd(), 'data')
    TRAIN_IMG_DIR: str = os.path.join(DATA_DIR, 'train_images')
    TRAIN_CSV: str = os.path.join(DATA_DIR, 'train.csv')
    TRAIN_FOLDS: str = os.path.join(DATA_DIR, 'train_folds.csv')
    MODELS_DIR: str = os.path.join(os.getcwd(), 'checkpoints')

@dataclass
class Hyperparameters:
    """Hyperparameters for model training."""
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-2
    EPOCHS: int = 15
    THRESHOLD: float = 0.5
    BCE_WEIGHTS: List[float] = field(default_factory=lambda: [2.0, 6.0, 0.5, 2.5])
    ALPHA: float = 1.0
    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 1600
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

@dataclass
class Config:
    """Main configuration class."""
    paths: Paths = field(default_factory=Paths)
    hyperparams: Hyperparameters = field(default_factory=Hyperparameters)
    SEED: int = 42

# Create a global configuration object
cfg = Config()

# Ensure directories exist
os.makedirs(cfg.paths.DATA_DIR, exist_ok=True)
os.makedirs(cfg.paths.MODELS_DIR, exist_ok=True)
