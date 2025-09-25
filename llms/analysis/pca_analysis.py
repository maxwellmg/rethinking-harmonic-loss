import os
import torch
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def load_model_weights(base_dir="out"):
    """Load weights from {model_type}_{distance}_run/ckpt_{iter_num}_{model_type}_{distance}.pt"""
    weights_data = defaultdict(lambda: defaultdict(dict))
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory '{base_dir}' not found!")
        return weights_data
    
    # Find run directories
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith('_run')]
    print(f"Found {len(run_dirs)} run directories")
    
    for run_dir in run_dirs:
        # Parse: {model_type}_{distance}_run
        dir_name = run_dir.name[:-4]  # Remove '_run'
        parts = dir_name.split('_')
        model_type = parts[0]
        distance = '_'.join(parts[1:]) if len(parts) > 1 else "baseline"
        
        print(f"Processing {model_type}_{distance}")
        
        # Find all checkpoints and get the highest iteration number
        checkpoint_files = glob.glob(str(run_dir / "ckpt_*.pt"))
        if not checkpoint_files:
            print(f"  No checkpoints found")
            continue
            
        # Parse iteration numbers and find the highest
        max_iter = -1
        best_ckpt_file = None
        
        for ckpt_file in checkpoint_files:
            filename = Path(ckpt_file).name
            match = re.match(r'ckpt_(\d+)_.*\.pt', filename)
            if match:
                iter_num = int(match.group(1))
                if iter_num > max_iter:
                    max_iter = iter_num
                    best_ckpt_file = ckpt_file
        
        # Load only the highest iteration checkpoint
        if best_ckpt_file:
            try:
                checkpoint = torch.load(best_ckpt_file, map_location='cpu')
                
                # Clean _orig_mod prefix from compiled models
                model_state = checkpoint.get('model', {})
                cleaned_state = {}
                for key, value in model_state.items():
                    clean_key = key.replace('_orig_mod.', '')
                    cleaned_state[clean_key] = value
                
                weights_data[model_type][distance][max_iter] = {
                    'model_state': cleaned_state,
                    'best_val_loss': checkpoint.get('best_val_loss'),
                    'iter_num': max_iter,
                    'file_path': best_ckpt_file
                }
                print(f"  Loaded highest checkpoint: iter {max_iter}")
                
            except Exception as e:
                print(f"  Error loading {best_ckpt_file}: {e}")
    
    return dict(weights_data)

def extract_key_layers(model_state_dict):
    """Extract attention and MLP weight matrices based on your model structure."""
    # Patterns for your specific architectures
    key_patterns = [
        # GPT patterns
        r'transformer\.h\.\d+\.attn\.c_attn\.weight$',
        r'transformer\.h\.\d+\.attn\.c_proj\.weight$', 
        r'transformer\.h\.\d+\.mlp\.c_fc\.weight$',
        r'transformer\.h\.\d+\.mlp\.c_proj\.weight$',
        # BERT patterns  
        r'encoder\.layers\.\d+\.attn\.c_attn\.weight$',
        r'encoder\.layers\.\d+\.attn\.c_proj\.weight$',
        r'encoder\.layers\.\d+\.mlp\.c_fc\.weight$',
        r'encoder\.layers\.\d+\.mlp\.c_proj\.weight$',
        # Qwen patterns
        r'layers\.\d+\.self_attn\.[qkvo]_proj\.weight$',
        r'layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$',
    ]
    
    # Exclude embeddings and norms (too big/less interpretable)
    exclude_patterns = [
        r'.*wte\.weight$', r'.*wpe\.weight$', r'.*embeddings.*', 
        r'.*LayerNorm.*', r'.*ln_\d+.*', r'.*\.bias$'
    ]
    
    extracted = {}
    for name, tensor in model_state_dict.items():
        # Skip if excluded
        if any(re.match(pattern, name) for pattern in exclude_patterns):
            continue
        # Include if matches key patterns
        if any(re.match(pattern, name) for pattern in key_patterns):
            if tensor.dim() >= 2:  # Only matrices
                extracted[name] = tensor.flatten().detach().cpu().numpy()
    
    return extracted

def run_pca_comparison(weights_data, n_components=5):
    """Run PCA to compare models and distance methods."""
    all_samples = []
    metadata = []
    
    print("\nExtracting layer weights for PCA...")
    
    # Collect all layer names for consistency
    all_layer_names = set()
    for model_type, distances in weights_data.items():
        for distance, iterations in distances.items():
            for iter_num, data in iterations.items():
                layer_weights = extract_key_layers(data['model_state'])
                all_layer_names.update(layer_weights.keys())
    
    all_layer_names = sorted(list(all_layer_names))
    print(f"Found {len(all_layer_names)} key layers across models")
    
    # Build feature matrix
    for model_type, distances in weights_data.items():
        for distance, iterations in distances.items():
            for iter_num, data in iterations.items():
                layer_weights = extract_key_layers(data['model_state'])
                
                # Concatenate all layer weights in consistent order
                sample_vector = []
                for layer_name in all_layer_names:
                    if layer_name in layer_weights:
                        sample_vector.extend(layer_weights[layer_name])
                
                if sample_vector:  # Only add if we got weights
                    all_samples.append(sample_vector)
                    metadata.append({
                        'model_type': model_type,
                        'distance': distance,
                        'iteration': iter_num,
                        'val_loss': data.get('best_val_loss'),
                        'label': f"{model_type}_{distance}"
                    })
    
    if not all_samples:
        print("No weight samples found!")
        return
    
    # Handle different sample lengths by truncating to shortest
    min_length = min(len(sample) for sample in all_samples)
    truncated_samples = [sample[:min_length] for sample in all_samples]
    
    # Convert to arrays
    X = np.array(truncated_samples)
    metadata_df = pd.DataFrame(metadata)
    
    print(f"PCA input shape: {X.shape}")
    print(f"Models: {metadata_df['model_type'].unique()}")
    print(f"Distance methods: {metadata_df['distance'].unique()}")
    
    # Standardize and run PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Adjust n_components to not exceed number of samples
    max_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if max_components < 1:
        print("Not enough samples for PCA!")
        return
    
    print(f"Using {max_components} components (limited by {X.shape[0]} samples)")
    
    pca = PCA(n_components=max_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Results
    print(f"\nPCA Results:")
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot by model type
    for i, model in enumerate(metadata_df['model_type'].unique()):
        mask = metadata_df['model_type'] == model
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=model, alpha=0.7, s=60)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('PCA: Models in Weight Space')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot by distance method  
    for i, dist in enumerate(metadata_df['distance'].unique()):
        mask = metadata_df['distance'] == dist
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       label=dist, alpha=0.7, s=60)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('PCA: Distance Methods in Weight Space')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return X_pca, metadata_df, pca

def print_summary(weights_data):
    """Print summary of loaded weights."""
    print("\n" + "="*50)
    print("LOADED WEIGHTS SUMMARY")
    print("="*50)
    
    for model_type, distances in weights_data.items():
        print(f"\nModel: {model_type.upper()}")
        for distance, iterations in distances.items():
            iters = sorted(iterations.keys())
            print(f"  {distance}: iterations {iters}")
            if iters:
                sample = iterations[iters[0]]
                if sample.get('best_val_loss'):
                    print(f"    Best val loss: {sample['best_val_loss']:.4f}")

# Main execution
if __name__ == "__main__":
    print("Loading model weights...")
    weights = load_model_weights("out")
    
    if not weights:
        print("No weights found!")
        exit()
    
    print_summary(weights)
    
    print("\nRunning PCA analysis...")
    try:
        X_pca, metadata_df, pca_model = run_pca_comparison(weights)
        print("Analysis complete!")
    except Exception as e:
        print(f"Error in analysis: {e}")