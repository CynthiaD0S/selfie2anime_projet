#!/usr/bin/env python3
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow non installé. Installer avec: pip install pillow")

DEFAULT_IMAGE_SIZE = 256

class CycleGANDataset(Dataset):
    """Dataset unpaired pour CycleGAN (Selfie↔Anime)"""
    
    def __init__(self, data_root, split, size=DEFAULT_IMAGE_SIZE, 
                 augment=False, normalize=True, aligned_val=False):
        """
        Args:
            data_root: chemin vers dataset (trainA/, trainB/, etc.)
            split: 'train' ou 'val'
            size: taille de redimensionnement
            augment: appliquer des augmentations (seulement en train)
            normalize: normaliser de [0,1] à [-1,1] (recommandé pour GANs)
            aligned_val: en validation, prendre A et B alignés si possible
        """
        self.data_root = Path(data_root)
        self.size = size
        self.augment = augment and (split == "train")
        self.normalize = normalize
        self.aligned_val = aligned_val
        
        phase = "train" if split == "train" else "val"
        
        self.dir_A = self.data_root / f"{phase}A"
        self.dir_B = self.data_root / f"{phase}B"
        
        # Chercher images
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
        self.path_A = []
        self.path_B = []
        
        for ext in extensions:
            self.path_A.extend(self.dir_A.glob(ext))
            self.path_B.extend(self.dir_B.glob(ext))
        
        self.path_A = sorted(self.path_A)
        self.path_B = sorted(self.path_B)
        
        if not self.path_A:
            raise FileNotFoundError(f"Aucune image trouvée dans {self.dir_A}")
        if not self.path_B:
            raise FileNotFoundError(f"Aucune image trouvée dans {self.dir_B}")
            
        print(f"[Dataset {phase}] Selfies: {len(self.path_A)}, Anime: {len(self.path_B)}")

    def _load_image(self, path):
        """Charge et redimensionne une image"""
        if not HAS_PIL:
            raise ImportError("Pillow requis: pip install pillow")
            
        # Chargement avec PIL
        with Image.open(path) as img:
            img = img.convert("RGB")
            
            # Convertir en numpy puis tensor (plus rapide que l'ancienne méthode)
            arr = np.array(img, dtype=np.float32) / 255.0
            
            # Redimensionner si nécessaire
            if arr.shape[0] != self.size or arr.shape[1] != self.size:
                from PIL import Image as PILImage
                img_resized = img.resize((self.size, self.size), PILImage.BILINEAR)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
            
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        
        return tensor

    def __len__(self):
        return max(len(self.path_A), len(self.path_B))

    def __getitem__(self, idx):
        # Domaine A (Selfie)
        idx_a = idx % len(self.path_A)
        file_A = self.path_A[idx_a]
        img_A = self._load_image(file_A)
        
        # Domaine B (Anime)
        if not self.augment and self.aligned_val and idx < min(len(self.path_A), len(self.path_B)):
            # Pour validation: prendre même index si possible
            file_B = self.path_B[idx % len(self.path_B)]
        else:
            # Pour entraînement: aléatoire
            file_B = random.choice(self.path_B)
        
        img_B = self._load_image(file_B)
        
        # Augmentations
        if self.augment:
            if random.random() < 0.5:
                img_A = torch.flip(img_A, dims=[2])
                img_B = torch.flip(img_B, dims=[2])
            # Ajouter d'autres augmentations si besoin:
            # if random.random() < 0.3:
            #     brightness = random.uniform(0.8, 1.2)
            #     img_A = torch.clamp(img_A * brightness, 0, 1)
        
        # Normalisation pour GAN
        if self.normalize:
            img_A = img_A * 2 - 1  # [0,1] -> [-1,1]
            img_B = img_B * 2 - 1
        
        return {
            "A": img_A,      # Selfie
            "B": img_B,      # Anime
            "path_A": str(file_A),
            "path_B": str(file_B)
        }

def create_dataloader(
    data_root,
    split="train",
    batch_size=1,
    size=DEFAULT_IMAGE_SIZE,
    shuffle=None,
    num_workers=0,  # Sur Jetson, souvent 0 pour éviter les problèmes
    augment=None,
    normalized=True,
    **kwargs
):
    """
    Crée un DataLoader pour CycleGAN
    """
    if shuffle is None:
        shuffle = (split == "train")
    
    if augment is None:
        augment = (split == "train")
    
    dataset = CycleGANDataset(
        data_root=data_root,
        split=split,
        size=size,
        augment=augment,
        normalize=normalized,
        **kwargs
    )
    
    # Sur Jetson: num_workers=0 souvent plus stable
    # Sur PC: vous pouvez mettre 4-8 selon votre CPU
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train")  # Important pour GANs
    )
    
    return loader


# Test du dataloader
if __name__ == "__main__":
    # Test minimal
    import sys
    
    # Remplacez par votre chemin
    TEST_PATH = "data/selfie2anime"  # ou votre structure
    
    try:
        loader = create_dataloader(
            data_root=TEST_PATH,
            split="train",
            batch_size=4,
            size=256,
            shuffle=True,
            num_workers=0
        )
        
        batch = next(iter(loader))
        print("✓ Dataloader fonctionnel!")
        print(f"  Batch shape A: {batch['A'].shape}")
        print(f"  Batch shape B: {batch['B'].shape}")
        print(f"  Range A: [{batch['A'].min():.3f}, {batch['A'].max():.3f}]")
        print(f"  Range B: [{batch['B'].min():.3f}, {batch['B'].max():.3f}]")
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        print("Structure attendue:")
        print(f"  {TEST_PATH}/trainA/*.jpg")
        print(f"  {TEST_PATH}/trainB/*.jpg")
        print(f"  {TEST_PATH}/valA/*.jpg (optionnel)")
        print(f"  {TEST_PATH}/valB/*.jpg (optionnel)")
        sys.exit(1)
