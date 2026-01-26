#!/usr/bin/env python3
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:
    Image = None

DEFAULT_IMAGE_SIZE = 256

def _read_with_pil(path):
    """Lit l'image et la transforme en tenseur [0, 1] (C, H, W)"""
    image = Image.open(path).convert("RGB")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = data.view(image.height, image.width, 3).permute(2, 0, 1)
    return tensor.float() / 255.0

def _load_image(path, size):
    """Charge et redimensionne l'image"""
    if Image is None:
        raise ImportError("Pillow est requis pour charger les images .jpg/.png")
    
    tensor = _read_with_pil(path)

    # Redimensionnement fixe
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
        
        phase = "train" if split == "train" else "val"
        
        self.dir_A = self.data_root / f"{phase}A"
        self.dir_B = self.data_root / f"{phase}B"
        
        self.path_A = sorted(list(self.dir_A.glob("*.jpg")) + list(self.dir_A.glob("*.png")))
        self.path_B = sorted(list(self.dir_B.glob("*.jpg")) + list(self.dir_B.glob("*.png")))

        if not self.path_A or not self.path_B:
            raise FileNotFoundError(f"Erreur : Dossiers vides dans {self.data_root}")

    def __len__(self):
        # On utilise la taille de l'ensemble le plus grand
        return max(len(self.path_A), len(self.path_B))

    def __getitem__(self, idx):
        # Image domaine A (Selfie)
        file_A = self.path_A[idx % len(self.path_A)]
        img_A = _load_image(file_A, self.size)

        # Image domaine B (Anime) - Prise aléatoirement pour l'aspect "Unpaired"
        file_B = random.choice(self.path_B)
        img_B = _load_image(file_B, self.size)

        # Augmentation : Horizontal Flip (Requirement: basic augmentations)
        if self.augment and random.random() < 0.5:
            img_A = torch.flip(img_A, dims=[2])
            img_B = torch.flip(img_B, dims=[2])

        # Retourne un dictionnaire avec les deux images
        return {"A": img_A, "B": img_B}

def create_dataloader(
    data_root,
    split,
    batch_size=1,
    size=DEFAULT_IMAGE_SIZE,
    shuffle=True,
    num_workers=4,
    augment=None,
):
    if augment is None:
        augment = (split == "train")

    dataset = CycleGANDataset(
        data_root=data_root,
        split=split,
        size=size,
        augment=augment,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), # Optimisation pour GPU
    )