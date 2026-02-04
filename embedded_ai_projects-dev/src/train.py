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
import traceback

# ✅ Progress bar (Windows terminal friendly)
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

try:
    from datasets.unpaired import create_dataloader
    from models.gan_generator import create_generator, create_discriminator
except Exception:
    print("IMPORT ERROR DETAILS:")
    traceback.print_exc()
    sys.exit(1)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """t: (C,H,W) in [-1,1]"""
    img = t.detach().cpu().numpy()
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


@torch.no_grad()
def save_samples_oneway(G, dataloader, device, epoch, output_dir, num_samples=4):
    """
    Save a small grid:
    top row: real_A (selfie) | fake_B (anime)
    """
    G.eval()

    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for real_A, real_B in dataloader:
        real_A = real_A.to(device)[:num_samples]
        fake_B = G(real_A)
        break

    for i in range(min(num_samples, real_A.size(0))):
        im_real = tensor_to_pil(real_A[i])
        im_fake = tensor_to_pil(fake_B[i])

        w, h = im_real.size
        grid = Image.new("RGB", (w * 2, h))
        grid.paste(im_real, (0, 0))
        grid.paste(im_fake, (w, 0))

        grid.save(samples_dir / f"epoch_{epoch:03d}_sample_{i}.png")

    G.train()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train One-Way GAN: Selfie -> Anime")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def train_oneway(config):
    # Required
    data_root = config["data_root"]

    # Output
    output_dir = Path(config.get("checkpoint_dir", "runs/oneway_gan"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Params
    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 4))
    image_size = int(config.get("train_size", 256))
    lr_g = float(config.get("lr_g", 0.0002))
    lr_d = float(config.get("lr_d", 0.0002))
    beta1 = float(config.get("beta1", 0.5))

    seed = int(config.get("seed", 42))
    num_workers = int(config.get("num_workers", 0))

    # ✅ Save every 20 epochs (checkpoints + samples)
    save_interval = 20
    sample_interval = 20

    # Identity loss (stabilize)
    lambda_identity = float(config.get("lambda_identity", 0.2))

    # Device
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader = create_dataloader(
        data_root=data_root,
        split="train",
        batch_size=batch_size,
        size=image_size,
        shuffle=True,
        num_workers=num_workers,
        augment=config.get("augment", True),
    )

    val_loader = create_dataloader(
        data_root=data_root,
        split="val",
        batch_size=min(4, batch_size),
        size=image_size,
        shuffle=False,
        num_workers=num_workers,
        augment=False,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    # Models: ONLY A->B and D_B
    generator_type = config.get("generator_type", "resnet6")
    G = create_generator(generator_type).to(device)       # Selfie -> Anime
    D_B = create_discriminator().to(device)               # Real Anime vs Fake Anime

    print(f"G parameters:   {sum(p.numel() for p in G.parameters()):,}")
    print(f"D_B parameters: {sum(p.numel() for p in D_B.parameters()):,}")

    # Losses (LSGAN)
    criterion_gan = nn.MSELoss()
    criterion_id = nn.L1Loss()

    # Optims
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D_B.parameters(), lr=lr_d, betas=(beta1, 0.999))

    # LR schedulers (linear decay)
    scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda e: 1.0 - e / epochs
    )
    scheduler_D = optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=lambda e: 1.0 - e / epochs
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir))

    losses_history = {"G": [], "D": [], "id": []}

    print(f"\nStarting training for {epochs} epochs (one-way)...")

    # ✅ Progress bar on epochs (Windows terminal friendly)
    pbar_epochs = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch")

    for epoch in pbar_epochs:
        G.train()
        D_B.train()

        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_id = 0.0

        # ✅ Progress bar on batches
        pbar_batches = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            leave=False,  # keep terminal clean on Windows
        )

        for batch_idx, (real_A, real_B) in pbar_batches:
            real_A = real_A.to(device)  # selfies
            real_B = real_B.to(device)  # anime

            # -------------------------
            # Train Generator G (A->B)
            # -------------------------
            optimizer_G.zero_grad()

            fake_B = G(real_A)
            out_fake = D_B(fake_B)
            loss_G_gan = criterion_gan(out_fake, torch.ones_like(out_fake))

            # Identity (optional): G(real_B) should be ~ real_B
            loss_id = torch.tensor(0.0, device=device)
            if lambda_identity > 0:
                same_B = G(real_B)
                loss_id = criterion_id(same_B, real_B) * lambda_identity

            loss_G = loss_G_gan + loss_id
            loss_G.backward()
            optimizer_G.step()

            # -------------------------
            # Train Discriminator D_B
            # -------------------------
            optimizer_D.zero_grad()

            out_real = D_B(real_B)
            loss_D_real = criterion_gan(out_real, torch.ones_like(out_real))

            out_fake_det = D_B(fake_B.detach())
            loss_D_fake = criterion_gan(out_fake_det, torch.zeros_like(out_fake_det))

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            epoch_loss_id += loss_id.item()

            # ✅ Live update on batch bar
            pbar_batches.set_postfix(
                G=f"{loss_G.item():.4f}",
                D=f"{loss_D.item():.4f}",
                id=f"{loss_id.item():.4f}",
            )

        # Averages
        n = max(1, len(train_loader))
        avg_G = epoch_loss_G / n
        avg_D = epoch_loss_D / n
        avg_id = epoch_loss_id / n

        # Schedulers
        scheduler_G.step()
        scheduler_D.step()

        # Logs
        losses_history["G"].append(avg_G)
        losses_history["D"].append(avg_D)
        losses_history["id"].append(avg_id)

        writer.add_scalar("Loss/G", avg_G, epoch)
        writer.add_scalar("Loss/D", avg_D, epoch)
        writer.add_scalar("Loss/Identity", avg_id, epoch)
        writer.add_scalar("LR/G", scheduler_G.get_last_lr()[0], epoch)

        # ✅ Live update on epoch bar
        pbar_epochs.set_postfix(G=f"{avg_G:.4f}", D=f"{avg_D:.4f}", id=f"{avg_id:.4f}")

        print(f"\nEpoch {epoch}/{epochs}: G={avg_G:.4f}, D={avg_D:.4f}, id={avg_id:.4f}")

        # ✅ Save samples every 20 epochs
        if epoch % sample_interval == 0:
            save_samples_oneway(G, val_loader, device, epoch, output_dir, num_samples=4)
            print(f"  ✓ Saved samples for epoch {epoch}")

        # ✅ Save checkpoint every 20 epochs
        if epoch % save_interval == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "G_state_dict": G.state_dict(),
                "D_B_state_dict": D_B.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "losses": losses_history,
                "config": config,
            }
            ckpt_path = output_dir / f"epoch_{epoch:03d}.pth"
            torch.save(ckpt, ckpt_path)

            # Also keep a "latest"
            torch.save(ckpt, output_dir / "model_latest.pt")

            print(f"  ✓ Saved checkpoint: {ckpt_path}")

    # Final
    torch.save({"G_state_dict": G.state_dict(), "config": config}, output_dir / "model_G_last.pt")

    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(losses_history, f, indent=2)

    writer.close()

    print("\n✅ Training completed!")
    print(f"   Output dir: {output_dir}")
    print(f"   Latest:     {output_dir / 'model_latest.pt'}")
    print(f"   Final G:    {output_dir / 'model_G_last.pt'}")


def main():
    args = parse_args()

    print("One-Way GAN Training Script (Selfie -> Anime)")
    print("=" * 60)

    config = load_config(args.config)
    print(f"Configuration loaded from: {args.config}")
    print(f"Project: {config.get('project', 'N/A')}")

    train_oneway(config)


if __name__ == "__main__":
    main()
