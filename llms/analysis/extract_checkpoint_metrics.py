"""
Single-step metric extraction from checkpoints.
This script loads a checkpoint, runs exactly one evaluation, and saves the metrics.
"""
import os
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import sys
import pandas as pd
from datetime import datetime
import math
import json

from config.config import *
from model.distance_layers import *
from model.model_setup import *
from model.GPT import GPT, GPTConfig
from model.BERT import BERT, BertConfig
from model.QWEN import Qwen2, QwenConfig

import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file')
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
output_csv = args.output_csv

print(f"Processing checkpoint: {checkpoint_path}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# Load checkpoint
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("✓ Checkpoint loaded successfully")
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")
    sys.exit(1)

# Extract metadata from checkpoint
model_args = checkpoint['model_args']
model_type = checkpoint.get('model_type', 'gpt')  # Default to gpt if not specified
iter_num = checkpoint.get('iter_num', 0)
config_dict = checkpoint.get('config', {})

print(f"Model type: {model_type}")
print(f"Iteration: {iter_num}")

# Determine distance type from checkpoint or filename
distance = model_args.get('distance', 'unknown')
if distance == 'unknown':
    # Try to extract from filename
    filename = os.path.basename(checkpoint_path)
    if 'baseline' in filename.lower():
        distance = 'baseline'
    elif 'euclidean' in filename.lower():
        distance = 'euclidean'
    elif 'manhattan' in filename.lower():
        distance = 'manhattan'
    elif 'cosine' in filename.lower():
        distance = 'cosine'
    # Add more as needed

print(f"Distance type: {distance}")

# Data loading
data_dir = os.path.join('./data', dataset)
if not os.path.exists(data_dir):
    print(f"✗ Data directory not found: {data_dir}")
    sys.exit(1)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Get vocab size
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Vocab size: {meta_vocab_size}")

# Batch generation functions
def get_batch_gpt(split):
    """GPT batch generation"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def get_batch_bert_mlm(split, vocab_size, block_size, batch_size, device_type, device):
    """BERT MLM batch generation"""
    import random
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    sequences = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    input_ids = sequences.clone()
    labels = sequences.clone()
    attention_mask = torch.ones_like(input_ids)
    
    for i in range(batch_size):
        valid_positions = torch.arange(block_size)
        num_to_mask = max(1, int(0.15 * len(valid_positions)))
        mask_positions = valid_positions[torch.randperm(len(valid_positions))[:num_to_mask]]
        
        for pos in mask_positions:
            original_token = input_ids[i, pos].item()
            rand_prob = random.random()
            
            if rand_prob < 0.8:
                input_ids[i, pos] = min(103, vocab_size - 1)
            elif rand_prob < 0.9:
                input_ids[i, pos] = torch.randint(0, vocab_size, (1,)).item()
            
            labels[i, pos] = original_token
        
        non_masked = torch.ones(block_size, dtype=torch.bool)
        non_masked[mask_positions] = False
        labels[i][non_masked] = -100
    
    if device_type == 'cuda':
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
    else:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
    
    return input_ids, attention_mask, labels

def get_batch(split, model_type):
    """Unified batch generation"""
    if model_type == 'gpt' or model_type == 'qwen':
        return get_batch_gpt(split)
    elif model_type == 'bert':
        return get_batch_bert_mlm(split, meta_vocab_size or 50304, block_size, batch_size, device_type, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Initialize model based on type
try:
    if model_type == 'gpt':
        gpt_config = GPTConfig(**model_args)
        model = GPT(gpt_config)
    elif model_type == 'bert':
        bert_config = BertConfig(**model_args)
        model = BERT(bert_config)
    elif model_type == 'qwen':
        qwen_config = QwenConfig(**model_args)
        model = Qwen2(qwen_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"✓ Model initialized: {model_type}")
except Exception as e:
    print(f"✗ Failed to initialize model: {e}")
    sys.exit(1)

# Load model state
try:
    state_dict = checkpoint['model']
    
    # Remove DDP prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✓ Model weights loaded")
except Exception as e:
    print(f"✗ Failed to load model weights: {e}")
    sys.exit(1)

def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    if isinstance(loss, torch.Tensor):
        loss_clamped = torch.clamp(loss, max=20.0)
        return torch.exp(loss_clamped).item()
    else:
        loss_clamped = min(loss, 20.0)
        return math.exp(loss_clamped)

def effective_rank(matrix):
    """
    Compute effective rank via singular value entropy (Roy & Vetterli, 2007).
    Works on 2D weight matrices or activation matrices.
    """
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().float().cpu().numpy()
    
    # SVD - only need singular values
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    
    # Filter near-zero singular values for numerical stability
    singular_values = singular_values[singular_values > 1e-10]
    
    if len(singular_values) == 0:
        return 0.0
    
    # Normalize to probability distribution
    p = singular_values / singular_values.sum()
    
    # Effective rank = exp(Shannon entropy)
    entropy = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(entropy))


def compute_model_effective_rank(model, model_type):
    """
    Compute effective rank across key weight matrices.
    Returns per-layer ranks and an aggregate (mean) rank.
    """
    erank_scores = {}
    
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # skip biases and 1D params
        
        # Focus on projection/attention/MLP layers — skip embeddings
        # (embeddings are high-rank by design and not shaped by distance loss)
        skip_keywords = ['embed', 'wpe', 'wte', 'pos_embed', 'tok_emb']
        if any(kw in name.lower() for kw in skip_keywords):
            continue
        
        target_keywords = ['attn', 'mlp', 'proj', 'fc', 'dense', 'query', 'key', 'value', 'out']
        if not any(kw in name.lower() for kw in target_keywords):
            continue
        
        matrix = param.data
        
        # For 3D+ tensors (e.g. Conv), reshape to 2D
        if matrix.dim() > 2:
            matrix = matrix.view(matrix.size(0), -1)
        
        erank_scores[name] = effective_rank(matrix)
    
    if not erank_scores:
        return {}, None
    
    aggregate = float(np.mean(list(erank_scores.values())))
    return erank_scores, aggregate

# Run evaluation
@torch.no_grad()
def evaluate():
    """Run single evaluation and return metrics"""
    model.eval()
    
    metrics = {}
    hidden_states_collected = []
    
    for split in ['train', 'val']:
        losses = []
        accuracies = []
        
        # Run eval_iters evaluations
        for k in range(eval_iters):
            if model_type == 'gpt' or model_type == 'qwen':
                X, Y = get_batch(split, model_type)
                with ctx:
                    if model_type == 'qwen':
                        logits, loss, acc = model(input_ids=X, targets=Y)
                    else:
                        logits, loss, acc = model(X, Y)
                accuracies.append(acc.item() if acc is not None else None)
            else:  # BERT
                input_ids, attention_mask, labels = get_batch(split, model_type)
                with ctx:
                    logits, loss, acc = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            losses.append(loss.item())

            # Collect hidden states from val split on first iter only (memory efficient)
            if split == 'val' and k == 0:
                hidden_states_collected.append(logits.detach().float().cpu())
        
        # Calculate metrics
        avg_loss = np.mean(losses)
        perplexity = calculate_perplexity(avg_loss)
        avg_acc = np.mean([a for a in accuracies if a is not None]) if accuracies else None
        
        metrics[f'{split}_loss'] = avg_loss
        metrics[f'{split}_perplexity'] = perplexity
        if avg_acc is not None:
            metrics[f'{split}_accuracy'] = avg_acc
    
    # Compute activation-based effective rank from collected logits
    # Shape: (batch, seq_len, vocab) -> reshape to (batch*seq_len, vocab)
    if hidden_states_collected:
        stacked = hidden_states_collected[0]  # (batch, seq, vocab)
        activation_matrix = stacked.reshape(-1, stacked.shape[-1])
        
        # INSERT HERE - between reshape and effective_rank call:
        max_rows = 2048
        if activation_matrix.shape[0] > max_rows:
            idx = np.random.choice(activation_matrix.shape[0], max_rows, replace=False)
            activation_matrix = activation_matrix[idx]
        
        metrics['erank_activations'] = effective_rank(activation_matrix)
    
    return metrics

# Run evaluation
print("Running evaluation...")
try:
    metrics = evaluate()
    print("✓ Evaluation complete")
    
    print(f"\nResults:")
    print(f"  Train Loss: {metrics['train_loss']:.4f}, Perplexity: {metrics['train_perplexity']:.2f}")
    print(f"  Val Loss:   {metrics['val_loss']:.4f}, Perplexity: {metrics['val_perplexity']:.2f}")
    if 'train_accuracy' in metrics:
        print(f"  Train Acc:  {metrics['train_accuracy']:.4f}")
    if 'val_accuracy' in metrics:
        print(f"  Val Acc:    {metrics['val_accuracy']:.4f}")
        
except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compute weight-based effective rank
print("Computing effective rank...")
try:
    erank_by_layer, erank_weights_mean = compute_model_effective_rank(model, model_type)
    print(f"✓ Effective rank computed")
    print(f"  Mean weight effective rank: {erank_weights_mean:.4f}")
    if erank_by_layer:
        # Show top 5 layers by rank for diagnostics
        top_layers = sorted(erank_by_layer.items(), key=lambda x: x[1], reverse=True)[:5]
        for layer_name, rank in top_layers:
            print(f"  {layer_name}: {rank:.4f}")
except Exception as e:
    print(f"⚠ Effective rank computation failed: {e}")
    erank_weights_mean = None
    erank_by_layer = {}


# Prepare output data
output_data = {
    'checkpoint_path': checkpoint_path,
    'checkpoint_filename': os.path.basename(checkpoint_path),
    'model_type': model_type,
    'distance_type': distance,
    'iteration': iter_num,
    'timestamp': datetime.now().isoformat(),
    'train_loss': metrics['train_loss'],
    'val_loss': metrics['val_loss'],
    'train_perplexity': metrics['train_perplexity'],
    'val_perplexity': metrics['val_perplexity'],
    'erank_weights_mean': erank_weights_mean,
    'erank_activations': metrics.get('erank_activations', None),
    'erank_by_layer_json': json.dumps(
        {k: round(v, 4) for k, v in erank_by_layer.items()}
    ) if erank_by_layer else None
}

if 'train_accuracy' in metrics:
    output_data['train_accuracy'] = metrics['train_accuracy']
if 'val_accuracy' in metrics:
    output_data['val_accuracy'] = metrics['val_accuracy']

# Add best values from checkpoint if available
if 'best_val_loss' in checkpoint:
    output_data['checkpoint_best_val_loss'] = checkpoint['best_val_loss']
if 'best_val_perplexity' in checkpoint:
    output_data['checkpoint_best_val_perplexity'] = checkpoint['best_val_perplexity']

# Add model configuration
output_data['n_layer'] = model_args.get('n_layer', None)
output_data['n_head'] = model_args.get('n_head', None)
output_data['n_embd'] = model_args.get('n_embd', None)
output_data['vocab_size'] = model_args.get('vocab_size', None)

# Save to CSV
try:
    df = pd.DataFrame([output_data])
    
    # Append to existing CSV or create new one
    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(output_csv, mode='w', header=True, index=False)
    
    print(f"✓ Metrics saved to: {output_csv}")
    
except Exception as e:
    print(f"✗ Failed to save metrics: {e}")
    sys.exit(1)

print("Success! Checkpoint processed.")

