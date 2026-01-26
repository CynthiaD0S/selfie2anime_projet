#!/usr/bin/env python3
"""
GAN models for Selfie2Anime project
Includes: Generator, Discriminator, and CycleGAN wrapper
"""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    """Residual block with instance normalization for CycleGAN"""
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

class ResNetGenerator(nn.Module):
    """ResNet-based generator for CycleGAN (6 or 9 blocks)"""
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
            nn.Tanh()  # Output in [-1, 1]
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class UNetGenerator(nn.Module):
    """U-Net generator (your original architecture)"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        
        # Encoder
        self.enc1 = conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(base_channels * 2, base_channels * 4)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = conv_block(base_channels * 2, base_channels)
        
        # Output
        self.head = nn.Conv2d(base_channels, out_channels, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder with skip connections
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Output with tanh for [-1, 1]
        return torch.tanh(self.head(d1))

class Discriminator(nn.Module):
    """PatchGAN discriminator (70x70)"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        def disc_block(in_c, out_c, stride=2, norm=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not norm)
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            # No norm on first layer
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            disc_block(base_channels, base_channels * 2),
            disc_block(base_channels * 2, base_channels * 4),
            disc_block(base_channels * 4, base_channels * 8, stride=1),
            
            # Output layer
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

def create_generator(generator_type="resnet6", **kwargs):
    """Factory function to create generator"""
    if generator_type == "resnet6":
        return ResNetGenerator(n_blocks=6, **kwargs)
    elif generator_type == "resnet9":
        return ResNetGenerator(n_blocks=9, **kwargs)
    elif generator_type == "unet_small":
        return UNetGenerator(base_channels=32, **kwargs)
    elif generator_type == "unet_tiny":
        return UNetGenerator(base_channels=16, **kwargs)
    else:
        raise ValueError(f"Unknown generator: {generator_type}")

def create_discriminator(**kwargs):
    """Factory function to create discriminator"""
    return Discriminator(**kwargs)

# Test the models
if __name__ == "__main__":
    print("Testing GAN models...")
    
    # Test input
    batch_size = 2
    img_size = 256
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Test generators
    for gen_type in ["resnet6", "unet_small"]:
        print(f"\nTesting {gen_type}:")
        G = create_generator(gen_type)
        y = G(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Output range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  Params: {sum(p.numel() for p in G.parameters()):,}")
    
    # Test discriminator
    print("\nTesting Discriminator:")
    D = create_discriminator()
    d_out = D(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {d_out.shape}")  # Should be [2, 1, 30, 30] for 256x256 input
    print(f"  Params: {sum(p.numel() for p in D.parameters()):,}")
    
    print("\n✓ All models tested successfully!")