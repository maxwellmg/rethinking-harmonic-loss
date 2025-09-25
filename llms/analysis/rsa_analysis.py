import os
import torch
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import glob
from tqdm import tqdm

def load_model_weights(base_dir="out"):
    """Load highest iteration checkpoint for each model/distance combination."""
    weights_data = defaultdict(lambda: defaultdict(dict))
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory '{base_dir}' not found!")
        return weights_data
    
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith('_run')]
    print(f"Found {len(run_dirs)} run directories")
    
    for run_dir in run_dirs:
        # Parse: {model_type}_{distance}_run
        dir_name = run_dir.name[:-4]
        parts = dir_name.split('_')
        model_type = parts[0]
        distance = '_'.join(parts[1:]) if len(parts) > 1 else "baseline"
        
        print(f"Processing {model_type}_{distance}")
        
        # Find highest iteration checkpoint
        checkpoint_files = glob.glob(str(run_dir / "ckpt_*.pt"))
        if not checkpoint_files:
            continue
            
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
        
        if best_ckpt_file:
            try:
                checkpoint = torch.load(best_ckpt_file, map_location='cpu')
                model_args = checkpoint.get('model_args', {})
                
                weights_data[model_type][distance] = {
                    'checkpoint': checkpoint,
                    'model_args': model_args,
                    'iter_num': max_iter,
                    'file_path': best_ckpt_file
                }
                print(f"  Loaded checkpoint at iter {max_iter}")
                
            except Exception as e:
                print(f"  Error loading {best_ckpt_file}: {e}")
    
    return dict(weights_data)

def load_models_from_checkpoints(weights_data, device='cpu'):
    """Reconstruct models from checkpoints for activation extraction."""
    models = {}
    
    # Import your model classes (adjust paths as needed)
    try:
        from model.GPT import GPT, GPTConfig
        from model.BERT import BERT, BertConfig  
        from model.QWEN import Qwen2, QwenConfig
    except ImportError as e:
        print(f"Error importing model classes: {e}")
        print("Make sure your model files are in the correct path")
        return models
    
    for model_type, distances in weights_data.items():
        for distance, data in distances.items():
            try:
                checkpoint = data['checkpoint']
                model_args = data['model_args']
                
                # Reconstruct model based on type
                if model_type == 'gpt':
                    config = GPTConfig(**model_args)
                    model = GPT(config)
                elif model_type == 'bert':
                    config = BertConfig(**model_args)
                    model = BERT(config)
                elif model_type == 'qwen':
                    config = QwenConfig(**model_args)
                    model = Qwen2(config)
                else:
                    print(f"Unknown model type: {model_type}")
                    continue
                
                # Load state dict and clean _orig_mod prefix
                state_dict = checkpoint['model']
                cleaned_state = {}
                for key, value in state_dict.items():
                    clean_key = key.replace('_orig_mod.', '')
                    cleaned_state[clean_key] = value
                
                model.load_state_dict(cleaned_state)
                model.to(device)
                model.eval()
                
                models[f"{model_type}_{distance}"] = model
                print(f"Loaded model: {model_type}_{distance}")
                
            except Exception as e:
                print(f"Error loading model {model_type}_{distance}: {e}")
    
    return models

def generate_test_data(vocab_size=50304, seq_length=128, batch_size=100):
    """Generate random text data for activation extraction."""
    # Create diverse test inputs
    inputs = []
    
    # Random sequences
    for _ in range(batch_size // 2):
        seq = torch.randint(1, vocab_size, (seq_length,))
        inputs.append(seq)
    
    # Structured patterns (repeated tokens, gradients, etc.)
    for i in range(batch_size // 2):
        if i % 4 == 0:  # Repeated patterns
            base_seq = torch.randint(1, 100, (seq_length // 4,))
            seq = base_seq.repeat(4)
        elif i % 4 == 1:  # Ascending pattern
            seq = torch.arange(1, seq_length + 1) % vocab_size
        elif i % 4 == 2:  # High frequency tokens
            seq = torch.randint(1, 1000, (seq_length,))
        else:  # Mixed
            seq = torch.randint(1, vocab_size, (seq_length,))
        
        inputs.append(seq)
    
    return torch.stack(inputs)

def extract_activations(model, inputs, model_type):
    """Extract layer-wise activations from model."""
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]  # Take first element if tuple
            activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks for different architectures
    if model_type.startswith('gpt'):
        # GPT layers
        for i, layer in enumerate(model.transformer.h):
            hook = layer.attn.register_forward_hook(make_hook(f'layer_{i}_attn'))
            hooks.append(hook)
            hook = layer.mlp.register_forward_hook(make_hook(f'layer_{i}_mlp'))
            hooks.append(hook)
            
    elif model_type.startswith('bert'):
        # BERT layers
        for i, layer in enumerate(model.encoder.layers):
            hook = layer.attn.register_forward_hook(make_hook(f'layer_{i}_attn'))
            hooks.append(hook)
            hook = layer.mlp.register_forward_hook(make_hook(f'layer_{i}_mlp'))
            hooks.append(hook)
            
    elif model_type.startswith('qwen'):
        # Qwen layers
        for i, layer in enumerate(model.layers):
            hook = layer.self_attn.register_forward_hook(make_hook(f'layer_{i}_attn'))
            hooks.append(hook)
            hook = layer.mlp.register_forward_hook(make_hook(f'layer_{i}_mlp'))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        if model_type.startswith('bert'):
            # BERT needs attention mask
            attention_mask = torch.ones_like(inputs)
            _ = model(inputs, attention_mask=attention_mask)
        else:
            # GPT and Qwen
            _ = model(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations

def compute_cka(X, Y):
    """Compute Centered Kernel Alignment between two activation matrices."""
    # X, Y should be [samples, features]
    X = X.reshape(X.shape[0], -1)  # Flatten spatial dimensions
    Y = Y.reshape(Y.shape[0], -1)
    
    # Center the data
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute kernels
    K = X @ X.T
    L = Y @ Y.T
    
    # Center kernels
    n = K.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n
    K = H @ K @ H
    L = H @ L @ H
    
    # Compute CKA
    numerator = torch.trace(K @ L)
    denominator = torch.sqrt(torch.trace(K @ K) * torch.trace(L @ L))
    
    if denominator == 0:
        return 0.0
    
    return (numerator / denominator).item()

def run_rsa_analysis(models, test_inputs, output_dir="rsa_results"):
    """Run full RSA analysis comparing all model pairs."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting activations from all models...")
    all_activations = {}
    
    for model_name, model in tqdm(models.items()):
        print(f"Processing {model_name}...")
        model_type = model_name.split('_')[0]
        activations = extract_activations(model, test_inputs, model_type)
        all_activations[model_name] = activations
    
    print("Computing CKA similarity matrices...")
    
    # Get all layer names across models
    all_layers = set()
    for activations in all_activations.values():
        all_layers.update(activations.keys())
    all_layers = sorted(list(all_layers))
    
    model_names = list(models.keys())
    results = []
    
    # Compute pairwise CKA for each layer
    for layer_name in tqdm(all_layers, desc="Processing layers"):
        layer_similarities = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if layer_name in all_activations[model1] and layer_name in all_activations[model2]:
                    X = all_activations[model1][layer_name]
                    Y = all_activations[model2][layer_name]
                    
                    try:
                        cka_score = compute_cka(X, Y)
                        layer_similarities[i, j] = cka_score
                    except Exception as e:
                        print(f"Error computing CKA for {layer_name} between {model1} and {model2}: {e}")
                        layer_similarities[i, j] = 0.0
                else:
                    layer_similarities[i, j] = 0.0  # Layer not present in one model
        
        # Store results
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                results.append({
                    'model1': model1,
                    'model2': model2,
                    'layer': layer_name,
                    'cka_similarity': layer_similarities[i, j]
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/cka_results.csv", index=False)
    
    # Create summary statistics
    summary_stats = compute_interpretability_scores(results_df)
    summary_stats.to_csv(f"{output_dir}/interpretability_scores.csv", index=False)
    
    # Create visualizations
    create_rsa_plots(results_df, model_names, all_layers, output_dir)
    
    return results_df, summary_stats

def compute_interpretability_scores(results_df):
    """Compute interpretability scores from CKA results."""
    scores = []
    
    # Group by model pairs
    model_pairs = results_df.groupby(['model1', 'model2'])
    
    for (model1, model2), group in model_pairs:
        if model1 != model2:  # Skip self-comparisons
            # Average CKA across layers
            avg_cka = group['cka_similarity'].mean()
            
            # Parse model info
            m1_type, m1_dist = model1.split('_', 1)
            m2_type, m2_dist = model2.split('_', 1)
            
            scores.append({
                'model1': model1,
                'model2': model2,
                'model1_type': m1_type,
                'model1_distance': m1_dist,
                'model2_type': m2_type, 
                'model2_distance': m2_dist,
                'avg_cka_similarity': avg_cka,
                'same_architecture': m1_type == m2_type,
                'same_distance': m1_dist == m2_dist
            })
    
    scores_df = pd.DataFrame(scores)
    
    # Compute summary metrics
    print("\nInterpretability Summary:")
    print("=" * 50)
    
    # Within architecture similarities (same model type, different distances)
    within_arch = scores_df[scores_df['same_architecture'] & ~scores_df['same_distance']]
    if len(within_arch) > 0:
        print(f"Average within-architecture similarity: {within_arch['avg_cka_similarity'].mean():.3f}")
        
    # Cross architecture similarities
    cross_arch = scores_df[~scores_df['same_architecture']]
    if len(cross_arch) > 0:
        print(f"Average cross-architecture similarity: {cross_arch['avg_cka_similarity'].mean():.3f}")
    
    # Distance method effects
    same_distance = scores_df[scores_df['same_distance'] & ~scores_df['same_architecture']]
    if len(same_distance) > 0:
        print(f"Same distance method (diff arch) similarity: {same_distance['avg_cka_similarity'].mean():.3f}")
    
    return scores_df

def create_rsa_plots(results_df, model_names, layer_names, output_dir):
    """Create visualization plots for RSA results."""
    
    # Average similarity matrix across all layers
    avg_sim_matrix = np.zeros((len(model_names), len(model_names)))
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            subset = results_df[(results_df['model1'] == model1) & (results_df['model2'] == model2)]
            avg_sim_matrix[i, j] = subset['cka_similarity'].mean()
    
    # Plot similarity heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_sim_matrix, 
                xticklabels=model_names, 
                yticklabels=model_names,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                square=True)
    plt.title('Average CKA Similarity Between Models')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cka_similarity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {output_dir}/")
    print("- cka_results.csv: Detailed CKA scores")
    print("- interpretability_scores.csv: Summary statistics")
    print("- cka_similarity_heatmap.png: Visualization")

# Main execution
if __name__ == "__main__":
    print("Starting RSA Analysis for LLM Interpretability...")
    
    # Load checkpoints
    weights_data = load_model_weights("out")
    if not weights_data:
        print("No model weights found!")
        exit()
    
    print(f"\nFound models:")
    for model_type, distances in weights_data.items():
        for distance in distances.keys():
            print(f"  {model_type}_{distance}")
    
    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models = load_models_from_checkpoints(weights_data, device)
    if not models:
        print("Failed to load models!")
        exit()
    
    # Generate test data
    print("Generating test data...")
    test_inputs = generate_test_data(batch_size=50, seq_length=128)
    test_inputs = test_inputs.to(device)
    
    # Run RSA analysis
    print("Running RSA analysis...")
    results_df, summary_stats = run_rsa_analysis(models, test_inputs)
    
    print("\nRSA Analysis Complete!")
    print("Check the rsa_results/ directory for detailed outputs.")