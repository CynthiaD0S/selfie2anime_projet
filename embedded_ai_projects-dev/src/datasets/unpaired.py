#!/usr/bin/env python3
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image

DEFAULT_IMAGE_SIZE = 256

def _load_image(path, size):
    """Charge une image, la convertit en RGB, la normalise en [0, 1] et la redimensionne."""
    image = Image.open(path).convert("RGB")
    # Conversion manuelle en Tensor pour respecter le style du prof [0, 1]
    # (H, W, C) -> (C, H, W)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = data.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0

    # Resize bilinear
    if size is not None and (tensor.shape[1] != size or tensor.shape[2] != size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor

class CycleGANDataset(Dataset):
    def __init__(self, data_root, split, size=DEFAULT_IMAGE_SIZE, augment=False):
        self.data_root = Path(data_root)
        self.size = size
        self.augment = augment
        
        # Pour CycleGAN, on a souvent trainA/trainB ou testA/testB
        # On s'adapte au split (train ou test/val)
        prefix = "train" if split == "train" else "test"
        
        self.dir_A = self.data_root / f"{prefix}A"
        self.dir_B = self.data_root / f"{prefix}B"
        
        self.path_A = sorted(list(self.dir_A.glob("*.jpg")) + list(self.dir_A.glob("*.png")))
        self.path_B = sorted(list(self.dir_B.glob("*.jpg")) + list(self.dir_B.glob("*.png")))

        if not self.path_A or not self.path_B:
            raise FileNotFoundError(f"Dossiers vides dans {self.data_root}")

    def __len__(self):
        # La taille du dataset est celle du plus grand dossier
        return max(len(self.path_A), len(self.path_B))

    def __getitem__(self, idx):
        # Image domaine A
        path_a = self.path_A[idx % len(self.path_A)]
        img_A = _load_image(path_a, self.size)

        # Image domaine B (prise au hasard pour plus de variété)
        path_b = random.choice(self.path_B)
        img_B = _load_image(path_b, self.size)

        # Augmentation : Horizontal Flip
        if self.augment and random.random() < 0.5:
            img_A = torch.flip(img_A, dims=[2])
            img_B = torch.flip(img_B, dims=[2])

        # Retourne (B, C, H, W) une fois mis en batch par le DataLoader
        return {"A": img_A, "B": img_B}

def create_dataloader(data_root, split, batch_size=1, size=DEFAULT_IMAGE_SIZE, shuffle=True):
    augment = (split == "train")
    dataset = CycleGANDataset(data_root, split, size=size, augment=augment)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )