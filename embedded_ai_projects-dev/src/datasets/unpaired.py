#!/usr/bin/env python3
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL is optional for the PPM-only example.
    Image = None


DEFAULT_IMAGE_SIZE = 256


def _read_with_pil(path):
    """Read image with PIL and convert to tensor in [0, 1] range"""
    image = Image.open(path).convert("RGB")
    # Convert PIL Image to tensor
    data = torch.ByteTensor(list(image.tobytes()))
    tensor = data.view(image.height, image.width, 3).permute(2, 0, 1)
    return tensor.float() / 255.0


def _load_image(path, size):
    """Load and resize image to specified size"""
    suffix = Path(path).suffix.lower()
    
    # Use PIL for common image formats
    if Image is not None and suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        tensor = _read_with_pil(path)
    else:
        raise ValueError(
            f"Unsupported image format {suffix}. Install Pillow for PNG/JPG support."
        )

    # Resize if needed
    if size is not None and (tensor.shape[1] != size or tensor.shape[2] != size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


class UnpairedDataset(Dataset):
    """Dataset for unpaired image-to-image translation (CycleGAN style)"""
    
    def __init__(self, data_root, split, size=DEFAULT_IMAGE_SIZE, augment=False):
        self.data_root = Path(data_root)
        self.split = split
        self.size = size
        self.augment = augment

        # Determine folders based on split
        if split == "train":
            self.dir_A = self.data_root / "trainA"
            self.dir_B = self.data_root / "trainB"
        elif split == "val":
            # Try val folders first, fallback to train
            val_A = self.data_root / "valA"
            val_B = self.data_root / "valB"
            self.dir_A = val_A if val_A.exists() else self.data_root / "trainA"
            self.dir_B = val_B if val_B.exists() else self.data_root / "trainB"
        else:
            raise ValueError(f"Invalid split: {split}")

        # Collect image paths
        self.paths_A = sorted(list(self.dir_A.glob("*")))
        self.paths_B = sorted(list(self.dir_B.glob("*")))

        # Filter for supported image files
        self.paths_A = [p for p in self.paths_A 
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        self.paths_B = [p for p in self.paths_B 
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        if not self.paths_A or not self.paths_B:
            raise FileNotFoundError(
                f"No images found in {self.dir_A} or {self.dir_B}"
            )

    def __len__(self):
        # Return max length to ensure we can iterate through all images
        return max(len(self.paths_A), len(self.paths_B))

    def __getitem__(self, idx):
        # Get image from domain A (using modulo to handle different lengths)
        path_A = self.paths_A[idx % len(self.paths_A)]
        img_A = _load_image(path_A, self.size)

        # Get image from domain B (random for unpaired, except maybe in validation)
        if self.split == "val" and idx < min(len(self.paths_A), len(self.paths_B)):
            # For validation, use corresponding index when possible
            path_B = self.paths_B[idx % len(self.paths_B)]
        else:
            # For training, random sampling (unpaired)
            path_B = random.choice(self.paths_B)
        
        img_B = _load_image(path_B, self.size)

        # Basic augmentation: horizontal flip
        if self.augment and random.random() < 0.5:
            img_A = torch.flip(img_A, dims=[2])
            img_B = torch.flip(img_B, dims=[2])

        # Return both images
        return img_A, img_B


def create_dataloader(
    data_root,
    split,
    batch_size=4,
    size=DEFAULT_IMAGE_SIZE,
    shuffle=True,
    num_workers=0,
    augment=None,
):
    """Create DataLoader for unpaired dataset"""
    if augment is None:
        augment = split == "train"

    dataset = UnpairedDataset(
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
        pin_memory=torch.cuda.is_available(),
    )