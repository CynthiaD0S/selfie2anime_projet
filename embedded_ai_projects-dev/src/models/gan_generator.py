#!/usr/bin/env python3
import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv block with instance normalization for GANs"""
    def __init__(self, in_channels, out_channels, use_norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))  # Better for style transfer than BatchNorm
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_norm))
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block for ResNet-based generators (better for CycleGAN)"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ==================== GENERATORS ====================

class UNetGenerator(nn.Module):
    """U-Net based generator (similar to prof's but adapted for GANs)"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, use_tanh=True):
        super().__init__()
        self.use_tanh = use_tanh  # tanh for GANs, sigmoid for normal
        
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

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        out = self.head(dec1)
        return torch.tanh(out) if self.use_tanh else torch.sigmoid(out)


class ResNetGenerator(nn.Module):
    """ResNet-based generator (standard for CycleGAN)"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, n_blocks=6):
        super().__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        
        # Downsampling
        in_features = base_channels
        out_features = base_channels * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, 7),
            nn.Tanh()  # Output in [-1, 1] for GANs
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# ==================== DISCRIMINATORS ====================

class Discriminator(nn.Module):
    """PatchGAN discriminator (standard for CycleGAN)"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, base_channels, normalize=False),
            *discriminator_block(base_channels, base_channels * 2),
            *discriminator_block(base_channels * 2, base_channels * 4),
            *discriminator_block(base_channels * 4, base_channels * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(base_channels * 8, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleDiscriminator(nn.Module):
    """Simpler discriminator for testing"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 8, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)


# ==================== FACTORY FUNCTIONS ====================

def create_generator(generator_type="unet_small", **kwargs):
    """Factory function to create generator"""
    if generator_type == "unet_small":
        return UNetGenerator(base_channels=32, **kwargs)
    elif generator_type == "unet_tiny":
        return UNetGenerator(base_channels=16, **kwargs)
    elif generator_type == "resnet6":
        return ResNetGenerator(n_blocks=6, **kwargs)
    elif generator_type == "resnet9":
        return ResNetGenerator(n_blocks=9, **kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def create_discriminator(discriminator_type="patchgan", **kwargs):
    """Factory function to create discriminator"""
    if discriminator_type == "patchgan":
        return Discriminator(**kwargs)
    elif discriminator_type == "simple":
        return SimpleDiscriminator(**kwargs)
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")
