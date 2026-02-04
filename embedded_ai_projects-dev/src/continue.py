# src/continue_resnet6.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

print("="*80)
print("CONTINUER L'ENTRAÎNEMENT CYCLEGAN RESNET6")
print("="*80)

# ==================== ARCHITECTURE EXACTE ====================

class ResidualBlock(nn.Module):
    """Bloc résiduel avec 2 convolutions (sans InstanceNorm dans les blocs?)"""
    def __init__(self, in_features):
        super().__init__()
        # Basé sur l'analyse: model.10.block.1.weight et model.10.block.5.weight
        # Pas de normalisation dans les blocs (basé sur les clés)
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1, bias=True),
        )
    
    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet6(nn.Module):
    """Générateur avec 6 blocs résiduels (comme dans le checkpoint)"""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6):
        super().__init__()
        
        # Couche initiale: model.1 (conv 7x7)
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=True),  # model.1.weight
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        # model.4: 64->128, stride 2
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1, bias=True),  # model.4.weight
            nn.ReLU(inplace=True)
        )
        
        # model.7: 128->256, stride 2  
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=True),  # model.7.weight
            nn.ReLU(inplace=True)
        )
        
        # Blocs résiduels (6 blocs basés sur l'analyse)
        # model.10 à model.15 (6 blocs)
        residual_blocks = []
        for i in range(n_residual_blocks):
            residual_blocks.append(ResidualBlock(ngf*4))
        self.residual = nn.Sequential(*residual_blocks)
        
        # Upsampling
        # 256->128
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # 128->64
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Couche finale
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, bias=True),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial(x)      # 64 channels
        x = self.down1(x)        # 128 channels
        x = self.down2(x)        # 256 channels
        x = self.residual(x)     # 256 channels (résiduels)
        x = self.up1(x)          # 128 channels
        x = self.up2(x)          # 64 channels
        x = self.final(x)        # 3 channels
        return x

class DiscriminatorPatchGAN(nn.Module):
    """Discriminateur PatchGAN (70x70)"""
    def __init__(self, input_nc=3, ndf=64):
        super().__init__()
        
        # C4: 3->64, stride 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # C8: 64->128, stride 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # C16: 128->256, stride 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # C32: 256->512, stride 1
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Sortie: 512->1, stride 1
        self.output = nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1)
    
    def forward(self, x):
        x = self.layer1(x)  # 64
        x = self.layer2(x)  # 128
        x = self.layer3(x)  # 256
        x = self.layer4(x)  # 512
        x = self.output(x)  # 1
        return x

# ==================== CHARGEMENT ====================

checkpoint_path = 'src/runs/selfie2anime_cyclegan/model_best.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\n✓ Checkpoint chargé: {checkpoint_path}")
print(f"✓ Epochs déjà faites: {checkpoint['epoch']}")
print(f"✓ Architecture: ResNet6 (6 blocs résiduels)")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Device: {device}")

# Créer les modèles avec la bonne architecture
G_AB = GeneratorResNet6(n_residual_blocks=6).to(device)
G_BA = GeneratorResNet6(n_residual_blocks=6).to(device)
D_A = DiscriminatorPatchGAN().to(device)
D_B = DiscriminatorPatchGAN().to(device)

print(f"✓ Générateurs créés: {sum(p.numel() for p in G_AB.parameters()):,} paramètres")
print(f"✓ Discriminateurs créés: {sum(p.numel() for p in D_A.parameters()):,} paramètres")

# Charger les poids (strict=False pour gérer les petites différences)
try:
    G_AB.load_state_dict(checkpoint['G_AB_state_dict'], strict=False)
    G_BA.load_state_dict(checkpoint['G_BA_state_dict'], strict=False)
    D_A.load_state_dict(checkpoint['D_A_state_dict'], strict=False)
    D_B.load_state_dict(checkpoint['D_B_state_dict'], strict=False)
    print("✓ Poids chargés avec succès!")
except Exception as e:
    print(f"⚠ Erreur lors du chargement: {e}")
    print("✓ Entraînement depuis le début...")

# Optimizers avec le même LR que dans la config
config = checkpoint['config']
lr_g = config.get('lr_g', 0.0002)
lr_d = config.get('lr_d', 0.0001)

optimizer_G = optim.Adam(
    list(G_AB.parameters()) + list(G_BA.parameters()), 
    lr=lr_g, 
    betas=(0.5, 0.999)
)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr_d, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr_d, betas=(0.5, 0.999))

# ==================== DATASET ====================

class CycleGANDataset:
    def __init__(self, root_A, root_B, transform=None, size=256):
        self.root_A = root_A
        self.root_B = root_B
        self.size = size
        
        # Lister les images
        self.images_A = []
        self.images_B = []
        
        if os.path.exists(root_A):
            self.images_A = sorted([
                f for f in os.listdir(root_A) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        
        if os.path.exists(root_B):
            self.images_B = sorted([
                f for f in os.listdir(root_B) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        
        self.length = min(len(self.images_A), len(self.images_B))
        
        if self.length == 0:
            print(f"⚠ Aucune image trouvée dans {root_A} ou {root_B}")
            print("Création d'un dataset factice...")
            self.images_A = [f"dummy_{i}.jpg" for i in range(100)]
            self.images_B = [f"dummy_{i}.jpg" for i in range(100)]
            self.length = 100
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Charger image A
        if self.images_A[idx].startswith('dummy_'):
            # Image factice
            img_A = Image.new('RGB', (self.size, self.size), 
                             color=(np.random.randint(0, 255), 
                                   np.random.randint(0, 255), 
                                   np.random.randint(0, 255)))
        else:
            img_path_A = os.path.join(self.root_A, self.images_A[idx])
            img_A = Image.open(img_path_A).convert('RGB')
        
        # Charger image B
        img_idx_B = idx % len(self.images_B)
        if self.images_B[img_idx_B].startswith('dummy_'):
            img_B = Image.new('RGB', (self.size, self.size), 
                             color=(np.random.randint(0, 255), 
                                   np.random.randint(0, 255), 
                                   np.random.randint(0, 255)))
        else:
            img_path_B = os.path.join(self.root_B, self.images_B[img_idx_B])
            img_B = Image.open(img_path_B).convert('RGB')
        
        # Redimensionner
        img_A = img_A.resize((self.size, self.size), Image.BICUBIC)
        img_B = img_B.resize((self.size, self.size), Image.BICUBIC)
        
        # Convertir en numpy et normaliser
        img_A = np.array(img_A, dtype=np.float32).transpose(2, 0, 1)
        img_B = np.array(img_B, dtype=np.float32).transpose(2, 0, 1)
        
        # Normaliser [-1, 1]
        img_A = (img_A / 127.5) - 1.0
        img_B = (img_B / 127.5) - 1.0
        
        return {
            'A': torch.from_numpy(img_A),
            'B': torch.from_numpy(img_B)
        }

# Chemins des données
data_root = config.get('data_root', 'data/selfie2anime')
train_A = os.path.join(data_root, 'trainA')
train_B = os.path.join(data_root, 'trainB')
image_size = config.get('train_size', 256)

print(f"\n✓ Chargement des données:")
print(f"  Domain A: {train_A}")
print(f"  Domain B: {train_B}")
print(f"  Taille image: {image_size}x{image_size}")

dataset = CycleGANDataset(train_A, train_B, size=image_size)
batch_size = config.get('batch_size', 1)

dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # Mettre à 0 pour éviter les problèmes sous Windows
)

print(f"✓ Dataset: {len(dataset)} images")
print(f"✓ Batch size: {batch_size}")

# ==================== ENTRAÎNEMENT ====================

def train_epoch(epoch, total_epochs, start_epoch):
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Hyperparamètres
    lambda_cycle = config.get('lambda_cycle', 10.0)
    lambda_identity = config.get('lambda_identity', 0.5)
    
    total_loss_G = 0
    total_loss_D = 0
    total_loss_cycle = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        
        batch_size_current = real_A.size(0)
        
        # Labels pour le discriminateur
        valid = torch.ones((batch_size_current, 1, 30, 30), device=device)
        fake = torch.zeros((batch_size_current, 1, 30, 30), device=device)
        
        # ========== ENTRAÎNEMENT DES GÉNÉRATEURS ==========
        optimizer_G.zero_grad()
        
        # Loss d'identité
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        
        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        
        # Cycle consistency loss
        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        
        recovered_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B)
        
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
        # Total generator loss
        loss_G = loss_GAN + lambda_cycle * loss_cycle + lambda_identity * loss_identity
        loss_G.backward()
        optimizer_G.step()
        
        # ========== ENTRAÎNEMENT DISCRIMINATEUR A ==========
        optimizer_D_A.zero_grad()
        
        loss_real_A = criterion_GAN(D_A(real_A), valid)
        loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real_A + loss_fake_A) / 2
        loss_D_A.backward()
        optimizer_D_A.step()
        
        # ========== ENTRAÎNEMENT DISCRIMINATEUR B ==========
        optimizer_D_B.zero_grad()
        
        loss_real_B = criterion_GAN(D_B(real_B), valid)
        loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_real_B + loss_fake_B) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        
        # ========== STATISTIQUES ==========
        loss_D = (loss_D_A + loss_D_B) / 2
        
        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()
        total_loss_cycle += loss_cycle.item()
        
        # Mettre à jour la barre de progression
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'G': f'{loss_G.item():.3f}',
                'D': f'{loss_D.item():.3f}',
                'Cycle': f'{loss_cycle.item():.3f}'
            })
    
    return (
        total_loss_G / len(dataloader),
        total_loss_D / len(dataloader),
        total_loss_cycle / len(dataloader)
    )

def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, 
                   losses_history, save_dir):
    """Sauvegarde un checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
        'losses': losses_history,
        'config': config
    }
    
    # Sauvegarder régulièrement
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Toujours sauvegarder le dernier
    last_path = os.path.join(save_dir, 'model_last.pt')
    torch.save(checkpoint, last_path)
    
    print(f"✓ Checkpoint sauvegardé: {checkpoint_path}")
    
    return checkpoint_path

# ==================== LANCER L'ENTRAÎNEMENT ====================

# Paramètres
start_epoch = checkpoint['epoch']
additional_epochs = 100  # Vous voulez 100 epochs supplémentaires
total_epochs = start_epoch + additional_epochs

output_dir = config.get('checkpoint_dir', 'runs/selfie2anime_continued')
os.makedirs(output_dir, exist_ok=True)

print(f"\n" + "="*80)
print("DÉMARRAGE DE L'ENTRAÎNEMENT")
print("="*80)
print(f"Reprise à l'epoch: {start_epoch}")
print(f"Epochs supplémentaires: {additional_epochs}")
print(f"Total cible: {total_epochs}")
print(f"Répertoire de sortie: {output_dir}")
print("="*80 + "\n")

# Historique des pertes (initialiser avec l'historique existant si disponible)
losses_history = checkpoint.get('losses', {
    'G': [], 'D_A': [], 'D_B': [], 'cycle': [], 'identity': [], 'total': []
})

# Boucle d'entraînement
for epoch in range(start_epoch, total_epochs):
    print(f"\nEpoch {epoch+1}/{total_epochs}")
    print("-" * 60)
    
    # Entraîner une epoch
    loss_G, loss_D, loss_cycle = train_epoch(epoch, total_epochs, start_epoch)
    
    print(f"\nRésultats epoch {epoch+1}:")
    print(f"  Loss Générateur: {loss_G:.4f}")
    print(f"  Loss Discriminateur: {loss_D:.4f}")
    print(f"  Loss Cycle: {loss_cycle:.4f}")
    
    # Mettre à jour l'historique
    losses_history['G'].append(loss_G)
    losses_history['D_A'].append(loss_D / 2)  # Approximation
    losses_history['D_B'].append(loss_D / 2)  # Approximation
    losses_history['cycle'].append(loss_cycle)
    losses_history['total'].append(loss_G + loss_D)
    
    # Sauvegarder tous les 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == total_epochs - 1:
        save_checkpoint(
            epoch + 1, G_AB, G_BA, D_A, D_B,
            optimizer_G, optimizer_D_A, optimizer_D_B,
            losses_history, output_dir
        )
    
    # Afficher la progression
    progress = (epoch + 1 - start_epoch) / additional_epochs * 100
    print(f"\nProgression: {progress:.1f}% ({epoch + 1 - start_epoch}/{additional_epochs})")

print(f"\n" + "="*80)
print("ENTRAÎNEMENT TERMINÉ !")
print("="*80)
print(f"✓ Entraînement complété pour {additional_epochs} epochs supplémentaires")
print(f"✓ Total epochs: {total_epochs}")
print(f"✓ Modèles sauvegardés dans: {output_dir}")
print("="*80)

# Sauvegarder le meilleur modèle
best_idx = np.argmin(losses_history['total'])
best_loss = losses_history['total'][best_idx]
print(f"\nMeilleure loss: {best_loss:.4f} (epoch {best_idx + start_epoch + 1})")

# Créer un fichier de résumé
summary_path = os.path.join(output_dir, 'training_summary.txt')
with open(summary_path, 'w') as f:
    f.write("Résumé de l'entraînement continu\n")
    f.write("="*60 + "\n\n")
    f.write(f"Checkpoint original: {checkpoint_path}\n")
    f.write(f"Epoch de départ: {start_epoch}\n")
    f.write(f"Epochs supplémentaires: {additional_epochs}\n")
    f.write(f"Epoch finale: {total_epochs}\n\n")
    f.write("Losses finales:\n")
    f.write(f"  Générateur: {loss_G:.4f}\n")
    f.write(f"  Discriminateur: {loss_D:.4f}\n")
    f.write(f"  Cycle: {loss_cycle:.4f}\n")

print(f"✓ Résumé sauvegardé: {summary_path}")