import torch
import yaml
import json

# Charger le checkpoint
checkpoint = torch.load('src/runs/selfie2anime_cyclegan/model_best.pt', map_location='cpu')

print("="*80)
print("ANALYSE DÉTAILLÉE DU CHECKPOINT")
print("="*80)

# 1. Voir toutes les clés
print("\n1. TOUTES LES CLÉS DU CHECKPOINT:")
for key, value in checkpoint.items():
    print(f"  - {key}: {type(value)}")

# 2. Voir la configuration (peut être une chaîne YAML)
print("\n2. CONFIGURATION SAUVEGARDÉE:")
if 'config' in checkpoint:
    config_data = checkpoint['config']
    if isinstance(config_data, str):
        print("  Config est une chaîne YAML:")
        print("  " + "-"*40)
        print(config_data)
        print("  " + "-"*40)
        # Essayer de parser
        try:
            parsed_config = yaml.safe_load(config_data)
            if isinstance(parsed_config, dict):
                print("\n  Config parsée:")
                for key, value in parsed_config.items():
                    print(f"    {key}: {value}")
        except:
            print("  Impossible de parser le YAML")
    elif isinstance(config_data, dict):
        print("  Config est un dictionnaire:")
        for key, value in config_data.items():
            print(f"    {key}: {value}")
    else:
        print(f"  Type inattendu: {type(config_data)}")

# 3. Analyser l'architecture du générateur en détail
print("\n3. ARCHITECTURE DU GÉNÉRATEUR (G_AB):")
G_AB_state = checkpoint['G_AB_state_dict']
print(f"  Nombre total de clés: {len(G_AB_state.keys())}")

# Organiser par couches
layers = {}
for key in G_AB_state.keys():
    parts = key.split('.')
    if len(parts) >= 2:
        layer_name = parts[0]  # model
        if layer_name not in layers:
            layers[layer_name] = []
        layers[layer_name].append(key)

print("\n  Structure du modèle:")
for layer_name, keys in layers.items():
    print(f"    [{layer_name}] - {len(keys)} paramètres")

# Afficher les 20 premières clés avec leur forme
print("\n  Détail des paramètres (premiers 20):")
for i, (key, tensor) in enumerate(list(G_AB_state.items())[:20]):
    print(f"    {i+1:2d}. {key:40s} : {str(tensor.shape):20s} ({tensor.dtype})")

# Compter les types de couches
conv_layers = sum(1 for k in G_AB_state.keys() if 'conv' in k.lower() or ('weight' in k and G_AB_state[k].dim() == 4))
norm_layers = sum(1 for k in G_AB_state.keys() if 'norm' in k.lower() or 'bn' in k.lower())
bias_layers = sum(1 for k in G_AB_state.keys() if 'bias' in k)

print(f"\n  Statistiques:")
print(f"    - Couches de convolution: {conv_layers}")
print(f"    - Couches de normalisation: {norm_layers}")
print(f"    - Bias: {bias_layers}")

# 4. Voir l'epoch et les pertes
print("\n4. ÉTAT D'ENTRAÎNEMENT:")
print(f"  Epoch actuelle: {checkpoint.get('epoch', 'N/A')}")
print(f"  Meilleure loss: {checkpoint.get('best_loss', 'N/A')}")

if 'losses' in checkpoint:
    print("\n  Historique des pertes:")
    losses = checkpoint['losses']
    for key, value in losses.items():
        if isinstance(value, list):
            print(f"    {key}: {len(value)} valeurs (dernière: {value[-1] if value else 'N/A'})")
        else:
            print(f"    {key}: {value}")

# 5. Analyser l'optimizer
print("\n5. ÉTAT DES OPTIMIZERS:")
for opt_key in ['optimizer_G_state_dict', 'optimizer_D_A_state_dict', 'optimizer_D_B_state_dict']:
    if opt_key in checkpoint:
        opt_state = checkpoint[opt_key]
        print(f"  {opt_key}:")
        if 'param_groups' in opt_state:
            print(f"    Groupes de paramètres: {len(opt_state['param_groups'])}")
            for i, group in enumerate(opt_state['param_groups']):
                print(f"      Groupe {i}: {len(group['params'])} paramètres, lr={group.get('lr', 'N/A')}")
        
        if 'state' in opt_state:
            print(f"    États: {len(opt_state['state'])}")
            # Afficher le premier état
            first_key = next(iter(opt_state['state'].keys())) if opt_state['state'] else None
            if first_key:
                first_state = opt_state['state'][first_key]
                print(f"      Exemple d'état - clés: {list(first_state.keys())}")

# 6. Calculer la taille totale
print("\n6. TAILLE DU MODÈLE:")
total_params = 0
for key, tensor in G_AB_state.items():
    if tensor is not None:
        total_params += tensor.numel()

print(f"  Paramètres G_AB: {total_params:,}")
print(f"  Taille mémoire: {total_params * 4 / 1024**2:.2f} MB (float32)")

# 7. Essayer de deviner l'architecture
print("\n7. DÉDUCTION DE L'ARCHITECTURE:")

# Chercher des motifs dans les noms des couches
all_keys = list(G_AB_state.keys())

# Chercher des blocs résiduels
residual_blocks = []
for key in all_keys:
    if 'block' in key:
        parts = key.split('.')
        for part in parts:
            if 'block' in part and part[5:].isdigit():
                block_num = int(part[5:])
                if block_num not in residual_blocks:
                    residual_blocks.append(block_num)

if residual_blocks:
    print(f"  Blocs résiduels détectés: {sorted(residual_blocks)}")
    print(f"  Nombre de blocs: {len(residual_blocks)}")
else:
    # Compter les couches de convolution pour estimer
    conv_keys = [k for k in all_keys if 'weight' in k and G_AB_state[k].dim() == 4]
    print(f"  Couches de convolution: {len(conv_keys)}")
    
    # Analyser les tailles des filtres
    print("  Tailles des filtres:")
    for i, key in enumerate(conv_keys[:10]):  # Afficher les 10 premiers
        tensor = G_AB_state[key]
        print(f"    {key}: {tensor.shape}")

# 8. Sauvegarder un résumé
print("\n8. SAUVEGARDE DU RÉSUMÉ:")
summary = {
    'epoch': checkpoint.get('epoch', 'N/A'),
    'total_params_G': total_params,
    'conv_layers': conv_layers,
    'norm_layers': norm_layers,
    'residual_blocks': len(residual_blocks) if residual_blocks else 'inconnu',
    'keys_sample': all_keys[:10]  # Premières 10 clés
}

with open('checkpoint_analysis.txt', 'w') as f:
    f.write("ANALYSE DU CHECKPOINT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Epoch: {summary['epoch']}\n")
    f.write(f"Paramètres G_AB: {summary['total_params_G']:,}\n")
    f.write(f"Couches convolution: {summary['conv_layers']}\n")
    f.write(f"Couches normalisation: {summary['norm_layers']}\n")
    f.write(f"Blocs résiduels: {summary['residual_blocks']}\n\n")
    
    f.write("ÉCHANTILLON DES CLÉS:\n")
    for key in summary['keys_sample']:
        f.write(f"  - {key}\n")
    
    f.write("\nCONFIGURATION:\n")
    if 'config' in checkpoint:
        f.write(str(checkpoint['config']))

print("  Résumé sauvegardé dans 'checkpoint_analysis.txt'")
print("\n" + "="*80)
print("ANALYSE TERMINÉE")
print("="*80)