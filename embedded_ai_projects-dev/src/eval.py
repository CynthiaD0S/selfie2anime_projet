import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """Charge la configuration depuis un fichier YAML"""
    print(f"Reading config file: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            print(f"File content:\n{content}")
            config = yaml.safe_load(content)
    except Exception as e:
        print(f"Error reading config file: {e}")
        config = {}
    
    # Si le fichier YAML est vide ou mal formaté
    if config is None:
        print("Config file is empty or None")
        config = {}
    elif isinstance(config, str):
        print(f"Warning: Config is a string, not a dict: {config}")
        config = {}
    
    print(f"\nParsed config type: {type(config)}")
    if isinstance(config, dict):
        print(f"Config keys: {list(config.keys())}")
        for key, value in config.items():
            print(f"  {key}: {value} (type: {type(value)})")
    
    # Créer une configuration par défaut robuste
    default_config = {
        'data': {
            'image_size': 256,
            'train_A': 'data/selfie2anime/trainA',
            'train_B': 'data/selfie2anime/trainB',
            'test_A': 'data/selfie2anime/valA',
            'test_B': 'data/selfie2anime/valB'
        },
        'model': {
            'input_channels': 3,
            'output_channels': 3,
            'n_residual_blocks': 9
        },
        'training': {
            'epochs': 100,
            'batch_size': 1,
            'lr': 0.0002,
            'num_workers': 2
        }
    }
    
    # Si config n'est pas un dict, utiliser les valeurs par défaut
    if not isinstance(config, dict):
        print("Warning: Config is not a dictionary, using defaults")
        return default_config
    
    # Nettoyer et valider la configuration
    cleaned_config = {}
    
    # Traiter la section data
    if 'data' in config:
        if isinstance(config['data'], dict):
            cleaned_config['data'] = config['data']
        else:
            print(f"Warning: 'data' section is not a dict, using defaults")
            cleaned_config['data'] = default_config['data']
    else:
        cleaned_config['data'] = default_config['data']
    
    # Traiter la section model
    if 'model' in config:
        if isinstance(config['model'], dict):
            cleaned_config['model'] = config['model']
        elif isinstance(config['model'], str):
            # Essayer de parser la chaîne comme YAML
            try:
                model_config = yaml.safe_load(config['model'])
                if isinstance(model_config, dict):
                    cleaned_config['model'] = model_config
                else:
                    print(f"Warning: Could not parse 'model' string as dict")
                    cleaned_config['model'] = default_config['model']
            except:
                print(f"Warning: 'model' is a string, using defaults")
                cleaned_config['model'] = default_config['model']
        else:
            print(f"Warning: 'model' section is not a dict, using defaults")
            cleaned_config['model'] = default_config['model']
    else:
        cleaned_config['model'] = default_config['model']
    
    # S'assurer que toutes les clés nécessaires existent dans model
    for key in ['input_channels', 'output_channels', 'n_residual_blocks']:
        if key not in cleaned_config['model']:
            cleaned_config['model'][key] = default_config['model'][key]
    
    # Traiter la section training
    if 'training' in config:
        if isinstance(config['training'], dict):
            cleaned_config['training'] = config['training']
        else:
            cleaned_config['training'] = default_config['training']
    else:
        cleaned_config['training'] = default_config['training']
    
    # Afficher la configuration finalisée
    print("\n" + "="*60)
    print("FINAL CONFIGURATION:")
    print("="*60)
    for section in cleaned_config:
        print(f"\n[{section.upper()}]")
        if isinstance(cleaned_config[section], dict):
            for key, value in cleaned_config[section].items():
                print(f"  {key}: {value}")
        else:
            print(f"  {cleaned_config[section]}")
    print("="*60)
    
    return cleaned_config

# [Le reste du code reste inchangé jusqu'à la fonction main...]
# Je vais seulement copier les fonctions essentielles pour gagner de l'espace

class SimpleTransforms:
    """Transformations simples sans torchvision"""
    def __init__(self, size=256):
        self.size = size
    
    def __call__(self, img):
        img = img.resize((self.size, self.size), Image.BICUBIC)
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor

def calculate_psnr(img1, img2):
    img1_norm = (img1 + 1) / 2
    img2_norm = (img2 + 1) / 2
    mse = torch.mean((img1_norm - img2_norm) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def calculate_ssim_simple(img1, img2):
    img1_norm = (img1 + 1) / 2
    img2_norm = (img2 + 1) / 2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = torch.mean(img1_norm)
    mu_y = torch.mean(img2_norm)
    sigma_x = torch.std(img1_norm)
    sigma_y = torch.std(img2_norm)
    sigma_xy = torch.mean((img1_norm - mu_x) * (img2_norm - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)
    return (numerator / denominator).item()

def save_image_tensor(tensor, filename, nrow=3):
    tensor_norm = (tensor + 1) / 2 * 255
    tensor_norm = tensor_norm.clamp(0, 255).byte()
    
    if tensor_norm.dim() == 3:
        tensor_norm = tensor_norm.unsqueeze(0)
    
    batch_size = tensor_norm.size(0)
    height = tensor_norm.size(2)
    width = tensor_norm.size(3)
    channels = tensor_norm.size(1)
    
    rows = (batch_size + nrow - 1) // nrow
    grid_height = rows * height
    grid_width = nrow * width
    
    if channels == 1:
        grid = torch.zeros((grid_height, grid_width), dtype=torch.uint8)
    else:
        grid = torch.zeros((channels, grid_height, grid_width), dtype=torch.uint8)
    
    for i in range(batch_size):
        row = i // nrow
        col = i % nrow
        if channels == 1:
            grid[row*height:(row+1)*height, col*width:(col+1)*width] = tensor_norm[i, 0]
        else:
            grid[:, row*height:(row+1)*height, col*width:(col+1)*width] = tensor_norm[i]
    
    if channels == 1:
        grid_img = Image.fromarray(grid.numpy(), mode='L')
    else:
        grid_img = Image.fromarray(grid.permute(1, 2, 0).numpy(), mode='RGB')
    
    grid_img.save(filename)
    print(f"Saved: {filename}")

def create_generator(input_channels=3, output_channels=3, n_residual_blocks=9):
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(channels)
            )
        
        def forward(self, x):
            return x + self.conv_block(x)
    
    class Generator(nn.Module):
        def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
            super().__init__()
            model = [
                nn.Conv2d(input_channels, 64, kernel_size=7, padding=3, bias=False),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ]
            
            in_channels = 64
            out_channels = 128
            for _ in range(2):
                model += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ]
                in_channels = out_channels
                out_channels *= 2
            
            for _ in range(n_residual_blocks):
                model += [ResidualBlock(in_channels)]
            
            out_channels = in_channels // 2
            for _ in range(2):
                model += [
                    nn.ConvTranspose2d(in_channels, out_channels, 
                                     kernel_size=3, stride=2, 
                                     padding=1, output_padding=1, bias=False),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ]
                in_channels = out_channels
                out_channels = out_channels // 2
            
            model += [
                nn.Conv2d(64, output_channels, kernel_size=7, padding=3),
                nn.Tanh()
            ]
            
            self.model = nn.Sequential(*model)
        
        def forward(self, x):
            return self.model(x)
    
    return Generator(input_channels, output_channels, n_residual_blocks)

def load_model(config, device, weights_path):
    print(f"\nLoading model with config:")
    print(f"  Input channels: {config['model']['input_channels']}")
    print(f"  Output channels: {config['model']['output_channels']}")
    print(f"  Residual blocks: {config['model']['n_residual_blocks']}")
    
    G_AB = create_generator(
        input_channels=config['model']['input_channels'],
        output_channels=config['model']['output_channels'],
        n_residual_blocks=config['model']['n_residual_blocks']
    ).to(device)
    
    G_BA = create_generator(
        input_channels=config['model']['input_channels'],
        output_channels=config['model']['output_channels'],
        n_residual_blocks=config['model']['n_residual_blocks']
    ).to(device)
    
    print(f"\nLoading weights: {weights_path}")
    
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
                
                # Essayer différentes clés
                load_functions = [
                    (['G_AB_state_dict', 'G_BA_state_dict'], "G_AB and G_BA state_dict"),
                    (['generator_AB_state_dict', 'generator_BA_state_dict'], "generator AB/BA state_dict"),
                    (['G_AB', 'G_BA'], "G_AB and G_BA"),
                    (['state_dict'], "single state_dict"),
                ]
                
                loaded = False
                for keys, desc in load_functions:
                    if all(k in checkpoint for k in keys):
                        try:
                            if len(keys) == 2:
                                G_AB.load_state_dict(checkpoint[keys[0]])
                                G_BA.load_state_dict(checkpoint[keys[1]])
                            else:
                                G_AB.load_state_dict(checkpoint[keys[0]])
                            print(f"✓ Loaded from {desc}")
                            loaded = True
                            break
                        except Exception as e:
                            print(f"  Failed to load from {desc}: {e}")
                            continue
                
                if not loaded:
                    print("Warning: Could not load with standard keys")
                    # Essayer la première clé qui ressemble à un modèle
                    for key in checkpoint.keys():
                        if 'state_dict' in key or 'generator' in key.lower():
                            try:
                                G_AB.load_state_dict(checkpoint[key])
                                print(f"✓ Loaded from key: {key}")
                                loaded = True
                                break
                            except:
                                continue
            else:
                G_AB.load_state_dict(checkpoint)
                print("✓ Loaded directly from checkpoint")
            
            if not loaded:
                print("⚠ Using randomly initialized weights")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("⚠ Using randomly initialized weights")
    else:
        print(f"⚠ Weights file not found: {weights_path}")
        print("⚠ Using randomly initialized weights")
    
    G_AB.eval()
    G_BA.eval()
    
    params = sum(p.numel() for p in G_AB.parameters())
    print(f"\nModel loaded:")
    print(f"  Parameters per generator: {params:,}")
    print(f"  Total parameters (both): {params * 2:,}")
    
    return G_AB, G_BA

def create_simple_dataset(root_A, root_B, transform, max_samples=50):
    class SimpleDataset:
        def __init__(self, root_A, root_B, transform, max_samples):
            self.root_A = root_A
            self.root_B = root_B
            self.transform = transform
            
            # Lister les images
            self.images_A = []
            self.images_B = []
            
            if os.path.exists(root_A):
                self.images_A = sorted([f for f in os.listdir(root_A) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            
            if os.path.exists(root_B):
                self.images_B = sorted([f for f in os.listdir(root_B) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            
            # Limiter le nombre d'échantillons
            if max_samples:
                self.images_A = self.images_A[:max_samples]
                self.images_B = self.images_B[:max_samples]
            
            self.length = min(len(self.images_A), len(self.images_B))
            print(f"  Found {len(self.images_A)} images in {root_A}")
            print(f"  Found {len(self.images_B)} images in {root_B}")
            print(f"  Using {self.length} pairs")
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            # Charger l'image A
            img_path_A = os.path.join(self.root_A, self.images_A[idx])
            img_A = Image.open(img_path_A).convert('RGB')
            
            # Charger l'image B (même index modulo nombre d'images B)
            img_idx_B = idx % len(self.images_B)
            img_path_B = os.path.join(self.root_B, self.images_B[img_idx_B])
            img_B = Image.open(img_path_B).convert('RGB')
            
            # Appliquer les transformations
            if self.transform:
                img_A = self.transform(img_A)
                img_B = self.transform(img_B)
            
            return {'A': img_A, 'B': img_B}
    
    return SimpleDataset(root_A, root_B, transform, max_samples)

def evaluate_model(G_AB, G_BA, test_loader, device, output_dir, num_samples=16):
    G_AB.eval()
    G_BA.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    samples_saved = 0
    
    print(f"\nEvaluating on {len(test_loader.dataset)} images...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if samples_saved >= num_samples:
                break
            
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            
            reconstructed_A = G_BA(fake_B)
            reconstructed_B = G_AB(fake_A)
            
            for i in range(real_A.size(0)):
                if samples_saved < num_samples:
                    psnr_val = calculate_psnr(real_A[i:i+1], reconstructed_A[i:i+1])
                    ssim_val = calculate_ssim_simple(real_A[i:i+1], reconstructed_A[i:i+1])
                    
                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    count += 1
                    
                    # Créer la grille 2x3
                    grid_images = []
                    
                    # Ligne 1
                    grid_images.append(real_A[i].cpu())
                    grid_images.append(fake_B[i].cpu())
                    grid_images.append(reconstructed_A[i].cpu())
                    
                    # Ligne 2
                    grid_images.append(real_B[i].cpu())
                    grid_images.append(fake_A[i].cpu())
                    grid_images.append(reconstructed_B[i].cpu())
                    
                    grid_tensor = torch.stack(grid_images, dim=0)
                    
                    save_image_tensor(
                        grid_tensor,
                        os.path.join(output_dir, f'sample_{samples_saved:03d}.png'),
                        nrow=3
                    )
                    
                    print(f"  Sample {samples_saved}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.3f}")
                    samples_saved += 1
    
    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    
    return avg_psnr, avg_ssim, count

def main():
    parser = argparse.ArgumentParser(description='Evaluate CycleGAN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of qualitative samples to save')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--max_test', type=int, default=50, help='Maximum number of test images')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CYCLEGAN EVALUATION")
    print("="*60)
    
    # Étape 1: Charger la configuration
    print(f"\n[1/4] LOADING CONFIGURATION")
    print(f"File: {args.config}")
    config = load_config(args.config)
    
    # Étape 2: Configurer le device
    print(f"\n[2/4] SETTING UP DEVICE")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Étape 3: Charger le modèle
    print(f"\n[3/4] LOADING MODEL")
    print(f"Weights: {args.weights}")
    G_AB, G_BA = load_model(config, device, args.weights)
    
    # Étape 4: Préparer les données
    print(f"\n[4/4] PREPARING DATA")
    transform = SimpleTransforms(size=config['data']['image_size'])
    
    test_dataset = create_simple_dataset(
        root_A=config['data']['test_A'],
        root_B=config['data']['test_B'],
        transform=transform,
        max_samples=args.max_test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Créer le répertoire de sortie
    exp_name = os.path.basename(os.path.dirname(args.weights))
    output_dir = os.path.join('src/runs', exp_name, 'samples')
    
    print(f"\n" + "="*60)
    print("EVALUATION STARTING")
    print("="*60)
    
    # Évaluer le modèle
    avg_psnr, avg_ssim, num_evaluated = evaluate_model(
        G_AB, G_BA, test_loader, device, output_dir, args.num_samples
    )
    
    # Afficher les résultats
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Images evaluated: {num_evaluated}")
    print(f"Average PSNR:    {avg_psnr:.2f} dB")
    print(f"Average SSIM:    {avg_ssim:.4f}")
    print(f"\nSamples saved to: {output_dir}")
    
    # Sauvegarder les métriques
    metrics_dir = os.path.dirname(output_dir)
    metrics_file = os.path.join(metrics_dir, 'evaluation_results.txt')
    
    with open(metrics_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Date: {os.popen('date').read().strip()}\n")
        f.write(f"Model: {args.weights}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Images evaluated: {num_evaluated}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Samples directory: {output_dir}\n")
    
    print(f"\nMetrics saved to: {metrics_file}")
    print("="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
