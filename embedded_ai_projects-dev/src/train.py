#!/usr/bin/env python3
import argparse
import yaml
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from pathlib import Path

# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import numpy as np

# Modules locaux
from model import Generator, Discriminator
from dataloader import create_dataloader

def compute_metrics(real_img, fake_img):
    """Calcule PSNR et SSIM entre deux tenseurs [0, 1]"""
    # Conversion en numpy CPU format (H, W, C)
    real = real_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    fake = fake_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    
    # Clip pour éviter les erreurs de flottants hors [0, 1]
    real = np.clip(real, 0, 1)
    fake = np.clip(fake, 0, 1)

    psnr = psnr_func(real, fake, data_range=1.0)
    # win_size=3 car les images peuvent être petites en début de projet
    ssim = ssim_func(real, fake, data_range=1.0, channel_axis=-1, win_size=3)
    return psnr, ssim

def train(config_path):
    # --- Chargement de la Config ---
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = cfg["project"]
    run_dir = Path(f"src/runs/{exp_name}")
    samples_dir = run_dir / "samples"
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    # --- Initialisation ---
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    opt_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=cfg["lr"], betas=(0.5, 0.999))
    
    criterion_GAN = nn.MSELoss()
    criterion_Cycle = nn.L1Loss()
    
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=str(run_dir))
    
    train_loader = create_dataloader(cfg["data_root"], "train", batch_size=cfg["batch_size"], size=cfg["train_size"])
    val_loader = create_dataloader(cfg["data_root"], "test", batch_size=1, size=cfg["train_size"])

    best_ssim = -1.0

    # --- Boucle Principale ---
    for epoch in range(cfg["epochs"]):
        G_A2B.train(); G_B2A.train()
        total_loss_G = 0.0
        total_loss_D = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for batch in loop:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Entraînement G
            with torch.amp.autocast('cuda'):
                fake_B = G_A2B(real_A)
                fake_A = G_B2A(real_B)
                
                loss_gan = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B))) + \
                           criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
                
                loss_cycle = (criterion_Cycle(G_B2A(fake_B), real_A) + \
                              criterion_Cycle(G_A2B(fake_A), real_B)) * 10.0
                
                loss_G = loss_gan + loss_cycle

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            total_loss_G += loss_G.item()

            # Entraînement D
            with torch.amp.autocast('cuda'):
                loss_D = (criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A))) + \
                          criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A.detach()))) + \
                          criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B))) + \
                          criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B.detach())))) / 2
            
            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            total_loss_D += loss_D.item()
            
            scaler.update()

        # --- Validation & Metrics ---
        val_psnr, val_ssim = 0.0, 0.0
        G_A2B.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                vA, vB = batch["A"].to(device), batch["B"].to(device)
                fB = G_A2B(vA)
                p, s = compute_metrics(vA, fB) # On compare A et sa version transformée (ou cycle)
                val_psnr += p
                val_ssim += s
                
                # Sauvegarde d'une grille d'exemple (Vrai A, Faux B, Rec A)
                if i == 0:
                    grid = make_grid([vA[0], fB[0], G_B2A(fB)[0]], normalize=False)
                    save_image(grid, samples_dir / f"epoch_{epoch+1}.png")

        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)

        # Print Epoch Loss & Metrics
        print(f"Epoch {epoch+1} - Loss G: {total_loss_G/len(train_loader):.4f} - PSNR: {avg_psnr:.2f} - SSIM: {avg_ssim:.4f}")

        # Save Best
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(G_A2B.state_dict(), run_dir / "best.pt")

        writer.add_scalar("Val/PSNR", avg_psnr, epoch)
        writer.add_scalar("Val/SSIM", avg_ssim, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)