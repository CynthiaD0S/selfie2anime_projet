import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import shutil

# Imports locaux
from models import Generator, Discriminator
from datasets import create_dataloader

def train(config_path):
    # --- 1. CHARGEMENT CONFIG ---
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Détection automatique du matériel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Lancement de l'entraînement sur : {device}")
    
    # --- GESTION DU DOSSIER (Solution radicale pour l'erreur FailedPrecondition) ---
    # On définit un chemin complet et propre
    base_run_dir = Path.cwd() / "src" / "runs"
    run_dir = base_run_dir / cfg["project"]

    # Suppression forcée de l'ancien dossier pour éviter le conflit
    if run_dir.exists():
        import shutil
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except:
            pass 

    # Création manuelle du dossier avec les outils Python standards
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Dossier de logs : {run_dir.absolute()}")

    # On essaie de lancer le Writer. Si ça plante encore, on s'en passe pour le test.
    try:
        writer = SummaryWriter(log_dir=str(run_dir.absolute()))
    except Exception as e:
        print(f"⚠️ TensorBoard n'a pas pu démarrer ({e}), on continue sans...")
        writer = None

    # --- 2. MODÈLES & OPTIMISEURS ---
    g_a2b = Generator().to(device)
    g_b2a = Generator().to(device)
    d_a = Discriminator().to(device)
    d_b = Discriminator().to(device)

    opt_g = torch.optim.Adam(list(g_a2b.parameters()) + list(g_b2a.parameters()), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=cfg["lr"], betas=(0.5, 0.999))

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()

    # Mixed Precision : Uniquement si CUDA est présent
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # --- 3. DATALOADER ---
    # Pour le test sur CPU, on utilise num_workers=0 pour éviter des erreurs Windows
    train_loader = create_dataloader(cfg["data_root"], "train", batch_size=cfg["batch_size"], size=cfg["train_size"], num_workers=0)

    # --- 4. BOUCLE D'ENTRAÎNEMENT ---
    epochs = 1 # On force à 1 pour ton test
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(loop):
            real_a = batch["A"].to(device)
            real_b = batch["B"].to(device)

            # --- GÉNÉRATEURS ---
            # On utilise autocast seulement si le GPU est là
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_b = g_a2b(real_a)
                fake_a = g_b2a(real_b)
                
                # Perte GAN
                loss_gan = criterion_gan(d_b(fake_b), torch.ones_like(d_b(fake_b))) + \
                           criterion_gan(d_a(fake_a), torch.ones_like(d_a(fake_a)))
                
                # Perte Cycle
                loss_cycle = (criterion_cycle(g_b2a(fake_b), real_a) + \
                              criterion_cycle(g_a2b(fake_a), real_b)) * 10.0
                
                loss_g = loss_gan + loss_cycle

            opt_g.zero_grad()
            if scaler:
                scaler.scale(loss_g).backward()
                scaler.step(opt_g)
            else:
                loss_g.backward()
                opt_g.step()

            # --- DISCRIMINATEURS ---
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss_d = (criterion_gan(d_a(real_a), torch.ones_like(d_a(real_a))) + \
                          criterion_gan(d_a(fake_a.detach()), torch.zeros_like(d_a(fake_a.detach()))) + \
                          criterion_gan(d_b(real_b), torch.ones_like(d_b(real_b))) + \
                          criterion_gan(d_b(fake_b.detach()), torch.zeros_like(d_b(fake_b.detach())))) / 2
            
            opt_d.zero_grad()
            if scaler:
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
                scaler.update()
            else:
                loss_d.backward()
                opt_d.step()

            # Mise à jour de la barre de progression
            loop.set_postfix(loss_g=loss_g.item(), loss_d=loss_d.item())

        # Sauvegarde de fin de test
        torch.save(g_a2b.state_dict(), run_dir / "last_test.pt")
        print(f"✅ Test réussi ! Modèle sauvegardé dans {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)