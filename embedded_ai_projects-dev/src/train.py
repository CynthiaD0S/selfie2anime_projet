#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random
import yaml

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
try:
    from datasets.unpaired import create_dataloader
    from models.gan_generator import create_generator, create_discriminator
except ImportError:
    print("Error: Make sure you have unpaired.py and gan_generator.py in the correct locations")
    sys.exit(1)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_samples(G_AB, G_BA, dataloader, device, epoch, output_dir, num_samples=4):
    """Save sample images to visualize progress"""
    G_AB.eval()
    G_BA.eval()
    
    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        # Get a batch
        for batch in dataloader:
            real_A = batch["A"].to(device)[:num_samples]
            real_B = batch["B"].to(device)[:num_samples]
                
            # Generate translations
            fake_B = G_AB(real_A)  # Selfie -> Anime
            fake_A = G_BA(real_B)  # Anime -> Selfie
            
            # Cycle consistency
            rec_A = G_BA(fake_B)  # Selfie -> Anime -> Selfie
            rec_B = G_AB(fake_A)  # Anime -> Selfie -> Anime
            
            break  # Just one batch
    
    # Save images
    for i in range(min(num_samples, real_A.size(0))):
        # Convert tensors to images
        def tensor_to_image(tensor):
            # Convert from [-1, 1] to [0, 255]
            img = tensor.cpu().numpy()
            img = (img + 1) / 2  # [-1, 1] -> [0, 1]
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            img = (img * 255).astype(np.uint8)
            return Image.fromarray(img)
        
        # Save individual images
        images = {
            'real_A': tensor_to_image(real_A[i]),
            'fake_B': tensor_to_image(fake_B[i]),
            'real_B': tensor_to_image(real_B[i]),
            'fake_A': tensor_to_image(fake_A[i]),
        }
        
        # Create a grid
        grid_width = real_A.shape[3] * 2
        grid_height = real_A.shape[2] * 2
        grid = Image.new('RGB', (grid_width, grid_height))
        
        # Top row: real_A | fake_B
        grid.paste(images['real_A'], (0, 0))
        grid.paste(images['fake_B'], (real_A.shape[3], 0))
        
        # Bottom row: real_B | fake_A
        grid.paste(images['real_B'], (0, real_A.shape[2]))
        grid.paste(images['fake_A'], (real_A.shape[3], real_A.shape[2]))
        
        grid.save(samples_dir / f"epoch_{epoch:03d}_sample_{i}.png")
    
    G_AB.train()
    G_BA.train()


def compute_ssim(img1, img2):
    """Compute SSIM between two images (simplified version)"""
    # Convert from [-1, 1] to [0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = torch.mean(img1, dim=[1, 2, 3], keepdim=True)
    mu2 = torch.mean(img2, dim=[1, 2, 3], keepdim=True)
    
    sigma1_sq = torch.mean((img1 - mu1) ** 2, dim=[1, 2, 3], keepdim=True)
    sigma2_sq = torch.mean((img2 - mu2) ** 2, dim=[1, 2, 3], keepdim=True)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[1, 2, 3], keepdim=True)
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return torch.mean(ssim_map)


class CycleGANLoss:
    """CycleGAN loss functions"""
    
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.5):
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Use MSE loss for LSGAN
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
    
    def gan_loss(self, discriminator_output, target_real):
        """GAN loss: discriminator should output 'real' for real images"""
        if target_real:
            target = torch.ones_like(discriminator_output)
        else:
            target = torch.zeros_like(discriminator_output)
        return self.criterion_gan(discriminator_output, target)
    
    def compute_generator_loss(self, fake_output):
        """Generator tries to fool discriminator"""
        return self.gan_loss(fake_output, True)
    
    def compute_discriminator_loss(self, real_output, fake_output):
        """Discriminator tries to distinguish real from fake"""
        loss_real = self.gan_loss(real_output, True)
        loss_fake = self.gan_loss(fake_output, False)
        return (loss_real + loss_fake) * 0.5


def train_cyclegan(config):
    """Main training function for CycleGAN"""
    
    # Extract config
    data_root = config['data_root']
    output_dir = Path(config.get('checkpoint_dir', 'runs/cyclegan'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 4)
    image_size = config.get('train_size', 256)
    lr_g = config.get('lr_g', 0.0002)
    lr_d = config.get('lr_d', 0.0001)
    beta1 = config.get('beta1', 0.5)
    
    # Loss weights
    lambda_cycle = config.get('lambda_cycle', 10.0)
    lambda_identity = config.get('lambda_identity', 0.5)
    
    # Other parameters
    seed = config.get('seed', 42)
    num_workers = config.get('num_workers', 0)
    save_interval = config.get('save_interval', 10)
    val_interval = config.get('val_interval', 5)
    
    # Set seed
    set_seed(seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(
        data_root=data_root,
        split="train",
        batch_size=batch_size,
        size=image_size,
        shuffle=True,
        num_workers=num_workers,
        augment=config.get('augment', True)
    )
    
    val_loader = create_dataloader(
        data_root=data_root,
        split="val",
        batch_size=min(4, batch_size),
        size=image_size,
        shuffle=False,
        num_workers=num_workers,
        augment=False
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create models
    print("Creating models...")
    generator_type = config.get('generator_type', 'resnet6')
    
    G_AB = create_generator(generator_type).to(device)  # Selfie -> Anime
    G_BA = create_generator(generator_type).to(device)  # Anime -> Selfie
    D_A = create_discriminator().to(device)  # Discriminator for Selfies
    D_B = create_discriminator().to(device)  # Discriminator for Anime
    
    print(f"G_AB parameters: {sum(p.numel() for p in G_AB.parameters()):,}")
    print(f"G_BA parameters: {sum(p.numel() for p in G_BA.parameters()):,}")
    print(f"D_A parameters: {sum(p.numel() for p in D_A.parameters()):,}")
    print(f"D_B parameters: {sum(p.numel() for p in D_B.parameters()):,}")
    
    # Loss functions
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_gan = nn.MSELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=lr_g,
        betas=(beta1, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr_d, betas=(beta1, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - epoch / epochs
    )
    scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - epoch / epochs
    )
    scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - epoch / epochs
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir))
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    
    best_cycle_loss = float('inf')
    losses_history = {
        'G': [], 'D_A': [], 'D_B': [], 'cycle': [], 'identity': [], 'total': []
    }
    
    for epoch in range(1, epochs + 1):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        
        epoch_loss_G = 0
        epoch_loss_D_A = 0
        epoch_loss_D_B = 0
        epoch_loss_cycle = 0
        epoch_loss_identity = 0
        
        for batch_idx, batch in enumerate(train_loader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            
            # ========== Train Generators ==========
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A) * lambda_identity
            loss_id_B = criterion_identity(G_AB(real_B), real_B) * lambda_identity
            
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_gan(D_B(fake_B), torch.ones_like(D_B(fake_B)))
            
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_gan(D_A(fake_A), torch.ones_like(D_A(fake_A)))
            
            # Cycle consistency loss
            rec_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_cycle
            
            rec_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_cycle
            
            # Total generator loss
            loss_G = (loss_GAN_AB + loss_GAN_BA + 
                     loss_cycle_A + loss_cycle_B + 
                     loss_id_A + loss_id_B)
            
            loss_G.backward()
            optimizer_G.step()
            
            # ========== Train Discriminator A ==========
            optimizer_D_A.zero_grad()
            
            loss_real_A = criterion_gan(D_A(real_A), torch.ones_like(D_A(real_A)))
            loss_fake_A = criterion_gan(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A.detach())))
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # ========== Train Discriminator B ==========
            optimizer_D_B.zero_grad()
            
            loss_real_B = criterion_gan(D_B(real_B), torch.ones_like(D_B(real_B)))
            loss_fake_B = criterion_gan(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B.detach())))
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            
            loss_D_B.backward()
            optimizer_D_B.step()
            
            # Accumulate losses
            epoch_loss_G += loss_G.item()
            epoch_loss_D_A += loss_D_A.item()
            epoch_loss_D_B += loss_D_B.item()
            epoch_loss_cycle += (loss_cycle_A.item() + loss_cycle_B.item()) / 2
            epoch_loss_identity += (loss_id_A.item() + loss_id_B.item()) / 2
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"G={loss_G.item():.4f}, "
                      f"D_A={loss_D_A.item():.4f}, "
                      f"D_B={loss_D_B.item():.4f}")
        
        # Average losses for the epoch
        num_batches = len(train_loader)
        avg_loss_G = epoch_loss_G / num_batches
        avg_loss_D_A = epoch_loss_D_A / num_batches
        avg_loss_D_B = epoch_loss_D_B / num_batches
        avg_loss_cycle = epoch_loss_cycle / num_batches
        avg_loss_identity = epoch_loss_identity / num_batches
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        
        # Log losses
        losses_history['G'].append(avg_loss_G)
        losses_history['D_A'].append(avg_loss_D_A)
        losses_history['D_B'].append(avg_loss_D_B)
        losses_history['cycle'].append(avg_loss_cycle)
        losses_history['identity'].append(avg_loss_identity)
        losses_history['total'].append(avg_loss_G + avg_loss_D_A + avg_loss_D_B)
        
        # TensorBoard logging
        writer.add_scalar('Loss/Generator', avg_loss_G, epoch)
        writer.add_scalar('Loss/Discriminator_A', avg_loss_D_A, epoch)
        writer.add_scalar('Loss/Discriminator_B', avg_loss_D_B, epoch)
        writer.add_scalar('Loss/Cycle', avg_loss_cycle, epoch)
        writer.add_scalar('Loss/Identity', avg_loss_identity, epoch)
        writer.add_scalar('LR/Generator', scheduler_G.get_last_lr()[0], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs}: "
              f"G={avg_loss_G:.4f}, "
              f"D_A={avg_loss_D_A:.4f}, "
              f"D_B={avg_loss_D_B:.4f}, "
              f"Cycle={avg_loss_cycle:.4f}, "
              f"Identity={avg_loss_identity:.4f}")
        
        # Save samples
        if epoch % val_interval == 0:
            save_samples(G_AB, G_BA, val_loader, device, epoch, output_dir)
        
        # Save checkpoint
        if epoch % save_interval == 0 or epoch == epochs:
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
            
            checkpoint_path = output_dir / f"epoch_{epoch:03d}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model based on cycle loss
            if avg_loss_cycle < best_cycle_loss:
                best_cycle_loss = avg_loss_cycle
                torch.save(checkpoint, output_dir / "model_best.pt")
                print(f"  ✓ Saved best model (cycle loss: {best_cycle_loss:.4f})")
            
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_checkpoint = {
        'epoch': epochs,
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'config': config
    }
    torch.save(final_checkpoint, output_dir / "model_last.pt")
    
    # Save training history
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(losses_history, f, indent=2)
    
    writer.close()
    
    print(f"\n✅ Training completed!")
    print(f"   Checkpoints saved in: {output_dir}")
    print(f"   Best model: {output_dir / 'model_best.pt'}")
    print(f"   Final model: {output_dir / 'model_last.pt'}")
    
    return {
        'output_dir': str(output_dir),
        'best_path': str(output_dir / 'model_best.pt'),
        'last_path': str(output_dir / 'model_last.pt'),
        'losses': losses_history
    }


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train CycleGAN for Selfie to Anime translation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print("CycleGAN Training Script")
    print("=" * 50)
    
    # Load configuration
    config = load_config(args.config)
    print(f"Configuration loaded from: {args.config}")
    print(f"Project: {config.get('project', 'N/A')}")
    
    # Train
    results = train_cyclegan(config)
    
    print("\nTraining completed successfully!")
    print(f"Results saved in: {results['output_dir']}")


if __name__ == "__main__":
    main()
