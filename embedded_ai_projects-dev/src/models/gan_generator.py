#!/usr/bin/env python3
import torch
from torch import nn

# --- Bloc de base adapté pour les GANs ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_norm)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels)) # Indispensable pour le transfert de style
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_norm))
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# --- Générateur (Basé sur le SmallUNet du prof) ---
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32): # base_channels augmenté pour plus de détails
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bn = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bn)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        # Sortie en [0, 1] comme demandé
        return torch.sigmoid(self.head(dec1))

# --- Discriminateur (PatchGAN) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        def disc_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            disc_block(base_channels, base_channels * 2),
            disc_block(base_channels * 2, base_channels * 4),
            disc_block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        # Le PatchGAN sort une grille de prédictions (ex: 30x30)
        # Chaque pixel de cette grille juge une zone (patch) de l'image
        return self.model(x)