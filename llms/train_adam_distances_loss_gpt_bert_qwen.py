import os
import time
import math
import pickle
from contextlib import nullcontext
import socket
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from codecarbon import EmissionsTracker
from config.config import *
from utils.gradient_check import check_gradients
import importlib
import sys
import pandas as pd
from datetime import datetime
import random

from model.distance_layers import *  # All your distance layers
from model.model_setup import *  # Other model functions
from model.GPT import GPT, GPTConfig  # Specific GPT imports
from model.BERT import BERT, BertConfig  # Specific BERT imports
from model.QWEN import Qwen2, QwenConfig, create_qwen_model

import warnings
from typing import Union, Iterable, List, Dict, Tuple, Optional

import torch
from torch import Tensor, inf

hostname = socket.gethostname()

# Current Run Configuration
wandb_project = 'qwen_full_runs'
#wandb_project = 'bert_full_runs'
#wandb_project = 'gpt_bert_testing'

import argparse

# Enhanced parser for model architecture selection - ADD QWEN
parser = argparse.ArgumentParser()
parser.add_argument('--distance', type=str, default='baseline')
parser.add_argument('--model_type', type=str, default='gpt', choices=['gpt', 'bert', 'qwen'], 
                   help='Choose between GPT, BERT, or Qwen architecture')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--gradient_accumulation_steps', type=int) 
parser.add_argument('--max_iters', type=int)
parser.add_argument('--wandb_project', type=str)
parser.add_argument('--wandb_run_name', type=str)

args, remaining_args = parser.parse_known_args()
distance = args.distance

# Pass ALL remaining args to configurator (including --model_type)
sys.argv = [sys.argv[0]] + remaining_args
exec(open('utils/configurator.py').read())

# Now model_type is available from the configurator
print(f"Using model_type='{model_type}' with distance='{distance}'")

wandb_run_name = f'{model_type}_{distance}_run'
init_from = 'scratch'

run_out_dir = os.path.join(out_dir, wandb_run_name)
os.makedirs(run_out_dir, exist_ok=True)

# Configuration setup
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
config['model_type'] = model_type

# Import architecture configs from config file
exec("from config.config import architecture_configs")

# Apply architecture-specific overrides
if model_type in architecture_configs:
    arch_config = architecture_configs[model_type]
    
    for key, value in arch_config.items():
        if key == 'default_vocab_size':
            pass
        else:
            globals()[key] = value
            config[key] = value
    
    print(f"Applied {model_type.upper()} specific configuration overrides")

# DDP and device setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(ddp_local_rank)
    print(f"DDP: Rank {ddp_rank}, world_size={world_size}, local_rank={ddp_local_rank}")
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    device = torch.device("cuda", ddp_local_rank)
else:
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(5000 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Initialize variables for emissions tracking
cumulative_emissions = 0.0
emissions_data_list = []

# Initialize CodeCarbon tracker for first interval
if master_process:
    emissions_output_file = f"emissions_gpt_bert_test_h100_{wandb_run_name}.csv"
    tracker = EmissionsTracker(
        output_file=f"temp_emissions_{wandb_run_name}_interval.csv",
        project_name=wandb_project,
        experiment_id=f"{wandb_run_name}_interval",
        output_dir=run_out_dir,
        log_level="INFO",
        measure_power_secs=15,
        tracking_mode="process",
        save_to_file=True,
    )
    print(f"CodeCarbon tracking initialized. Final file: {os.path.join(run_out_dir, emissions_output_file)}")
else:
    tracker = None

device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# Data loading
data_dir = os.path.join('./data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Updated batch generation functions
def get_batch_gpt(split):
    """Original GPT batch generation (causal language modeling)"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def get_batch_qwen(split):
    """Qwen batch generation (same as GPT - causal language modeling)"""
    return get_batch_gpt(split)  # Qwen uses same data format as GPT

def get_batch_bert_mlm(split, vocab_size, block_size, batch_size, device_type, device):
    """BERT MLM batch generation - creates meaningful architectural difference"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Get original sequences
    sequences = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    input_ids = sequences.clone()
    labels = sequences.clone()
    attention_mask = torch.ones_like(input_ids)
    
    # MLM masking - this is what makes BERT different
    for i in range(batch_size):
        # Get positions to mask
        valid_positions = torch.arange(block_size)
        
        # Mask 15% of tokens
        num_to_mask = max(1, int(0.15 * len(valid_positions)))
        mask_positions = valid_positions[torch.randperm(len(valid_positions))[:num_to_mask]]
        
        for pos in mask_positions:
            original_token = input_ids[i, pos].item()
            rand_prob = random.random()
            
            if rand_prob < 0.8:
                # 80% of time: replace with [MASK] token
                input_ids[i, pos] = min(103, vocab_size - 1)
            elif rand_prob < 0.9:
                # 10% of time: replace with random token
                input_ids[i, pos] = torch.randint(0, vocab_size, (1,)).item()
            # 10% of time: keep original token
            
            # Label is the original token
            labels[i, pos] = original_token
        
        # Set non-masked positions to -100 (ignored in loss)
        non_masked = torch.ones(block_size, dtype=torch.bool)
        non_masked[mask_positions] = False
        labels[i][non_masked] = -100
    
    # Move to device
    if device_type == 'cuda':
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
    else:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
    
    return input_ids, attention_mask, labels

def get_batch_bert(split):
    """BERT MLM batch - this is the key architectural difference"""
    return get_batch_bert_mlm(split, meta_vocab_size or 50304, block_size, batch_size, device_type, device)

def get_batch_unified(split, model_type):
    if model_type == 'gpt':
        return get_batch_gpt(split)
    elif model_type == 'bert':
        return get_batch_bert(split)
    elif model_type == 'qwen':
        return get_batch_qwen(split)
    else:
        raise ValueError("Invalid model type")

get_batch = lambda split: get_batch_unified(split, model_type)

# Initialize training variables
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size}")

# Updated model initialization based on architecture type
if model_type == 'gpt':
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=None, dropout=dropout, 
                      scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx, 
                      distance=distance)
elif model_type == 'bert':
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                      max_position_embeddings=block_size,
                      bias=bias, vocab_size=None, dropout=dropout,
                      scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx, 
                      distance=distance,
                      type_vocab_size=2,
                      pad_token_id=0)
elif model_type == 'qwen':
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                      max_position_embeddings=block_size,
                      bias=bias, vocab_size=None, dropout=dropout,
                      distance=distance,
                      # Qwen-specific parameters
                      intermediate_size=getattr(config, 'intermediate_size', 4864),
                      num_key_value_heads=getattr(config, 'num_key_value_heads', 2),
                      rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
                      rope_theta=getattr(config, 'rope_theta', 1000000.0))

if init_from == 'scratch':
    print(f"Initializing new {model_type.upper()} model from scratch")
    
    if model_type == 'gpt':
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gpt_config = GPTConfig(**model_args)
        model = GPT(gpt_config)
    elif model_type == 'bert':
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        bert_config = BertConfig(**model_args)
        model = BERT(bert_config)
    elif model_type == 'qwen':
        # Qwen can use either its native vocab or adapt to dataset vocab
        # Option A: Use dataset vocab (requires retokenizing data)
        #model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 151936

        # Option B: Use Qwen's native vocab (recommended)
        model_args['vocab_size'] = 151936  # Always use Qwen's vocab
        qwen_config = QwenConfig(**model_args)
        model = Qwen2(qwen_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    print(f"{model_type.upper()} model initialized with distance: {distance}")

elif init_from == 'resume':
    print(f"Resuming {model_type.upper()} training from {run_out_dir}")
    ckpt_path = os.path.join(out_dir, config['wandb_run_name'], 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    for k in ['n_layer', 'n_head', 'n_embd', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    
    if model_type == 'gpt':
        model_args['block_size'] = checkpoint_model_args['block_size']
        gpt_config = GPTConfig(**model_args)
        model = GPT(gpt_config)
    elif model_type == 'bert':
        model_args['max_position_embeddings'] = checkpoint_model_args.get('max_position_embeddings', 512)
        bert_config = BertConfig(**model_args)
        model = BERT(bert_config)
    elif model_type == 'qwen':
        model_args['max_position_embeddings'] = checkpoint_model_args.get('max_position_embeddings', 32768)
        # Add other Qwen-specific args from checkpoint
        for k in ['intermediate_size', 'num_key_value_heads', 'rms_norm_eps', 'rope_theta']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        qwen_config = QwenConfig(**model_args)
        model = Qwen2(qwen_config)
    
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2') and model_type == 'gpt':
    print(f"Initializing GPT from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

elif init_from.startswith('bert') and model_type == 'bert':
    print(f"Initializing BERT from pretrained weights: {init_from}")
    override_args = dict(dropout=dropout, distance=distance)
    model = BERT.from_pretrained(init_from, override_args)
    
    for k in ['n_layer', 'n_head', 'n_embd', 'bias', 'vocab_size']:
        if hasattr(model.config, k):
            model_args[k] = getattr(model.config, k)

elif init_from.startswith('qwen') and model_type == 'qwen':
    print(f"Initializing Qwen from pretrained weights: {init_from}")
    # Note: This would require implementing from_pretrained for Qwen
    # For now, we'll fall back to scratch initialization
    print("Note: Qwen from_pretrained not implemented, using scratch initialization")
    model_args['vocab_size'] = 151936  # Qwen's native vocab
    qwen_config = QwenConfig(**model_args)
    model = Qwen2(qwen_config)

else:
    raise ValueError(f"Cannot initialize {model_type} model with init_from='{init_from}'")

# Crop model block size if needed
if model_type == 'gpt' and block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
elif model_type == 'bert' and block_size < model.config.max_position_embeddings:
    print(f"Warning: Requested block_size {block_size} < model max_position_embeddings {model.config.max_position_embeddings}")
elif model_type == 'qwen' and block_size < model.config.max_position_embeddings:
    print(f"Warning: Requested block_size {block_size} < model max_position_embeddings {model.config.max_position_embeddings}")

model.to(device)

# Initialize optimizer and scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(optimizer_name, weight_decay, learning_rate, 
                                     (beta1, beta2), rho, gamma, lr_max, device_type)

if init_from == 'resume': 
    optimizer.load_state_dict(checkpoint['optimizer'])
    del state_dict
    del checkpoint

# Compile model if requested
if compile:
    print("Compiling model...")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap in DDP if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Updated loss estimation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if model_type == 'gpt' or model_type == 'qwen':
                # Both GPT and Qwen use causal language modeling
                X, Y = get_batch(split)
                with ctx:
                    if model_type == 'qwen':
                        logits, loss, acc = model(input_ids=X, targets=Y)
                    else: #GPT
                        logits, loss, acc = model(X, Y)
            else:  # BERT with MLM
                input_ids, attention_mask, labels = get_batch(split)
                with ctx:
                    logits, loss, acc = model(input_ids, attention_mask=attention_mask, labels=labels)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Set up fillnan function
def fillnan(x, nan_value=0.):
    """Replace NaN/Inf values with specified value (default: 0)"""
    return torch.nan_to_num(x, nan=nan_value, posinf=nan_value, neginf=nan_value)

# Function to save accumulated emissions data
def save_accumulated_emissions():
    """Save all accumulated emissions data to a single CSV file"""
    if master_process and emissions_data_list:
        try:
            final_emissions_file = os.path.join(run_out_dir, emissions_output_file)
            df = pd.DataFrame(emissions_data_list)
            df.to_csv(final_emissions_file, index=False)
            print(f"Saved accumulated emissions data to: {final_emissions_file}")
        except Exception as e:
            print(f"Error saving accumulated emissions: {e}")

# Initialize WandB logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Start emissions tracking for first interval
if master_process and tracker:
    tracker.start()

# Get initial batch based on model type - UPDATED FOR QWEN
print(f"Starting {model_type.upper()} training...")
if model_type == 'gpt' or model_type == 'qwen':
    X, Y = get_batch('train')
    print(f"{model_type.upper()} batch shapes: X={X.shape}, Y={Y.shape}")
    # Initialize BERT variables to None for consistency
    input_ids, attention_mask, labels = None, None, None
else:  # BERT
    input_ids, attention_mask, labels = get_batch('train')
    print(f"BERT batch shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, labels={labels.shape}")
    # Show masking statistics
    num_masked = (labels != -100).sum().item()
    total_tokens = labels.numel()
    print(f"BERT MLM: {num_masked}/{total_tokens} tokens masked ({num_masked/total_tokens*100:.1f}%)")
    # Initialize GPT/Qwen variables to None for consistency
    X, Y = None, None

t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
clip_time = 0

print("Beginning main training loop...")

try:
    while True:
        # Set learning rate
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and checkpointing with emissions tracking
        if iter_num % eval_interval == 0 and master_process:
            # Stop current emissions tracking and collect data
            interval_emissions_data = None
            interval_emissions = 0.0
            if tracker:
                try:
                    interval_emissions_data = tracker.stop()
                    if interval_emissions_data:
                        if isinstance(interval_emissions_data, (int, float)):
                            interval_emissions = interval_emissions_data
                        else:
                            interval_emissions = getattr(interval_emissions_data, 'emissions', 0.0)
                        
                        cumulative_emissions += interval_emissions
                        print(f"Interval emissions (iter {iter_num}): {interval_emissions:.6f} kg CO2")
                        print(f"Cumulative emissions: {cumulative_emissions:.6f} kg CO2")
                except Exception as e:
                    print(f"Error stopping tracker at iter {iter_num}: {e}")
            
            # Run evaluation
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Store comprehensive emissions data for this interval
            emissions_record = {
                'timestamp': datetime.now().isoformat(),
                'iteration': iter_num,
                'model_type': model_type,
                'distance_type': distance,
                'wandb_run_name': wandb_run_name,
                'interval_emissions_kg_co2': interval_emissions,
                'cumulative_emissions_kg_co2': cumulative_emissions,
                'train_loss': losses['train'].item(),
                'val_loss': losses['val'].item(),
                'learning_rate': lr,
                'best_val_loss': best_val_loss,
            }
            
            # Add comprehensive CodeCarbon data if available
            if interval_emissions_data and not isinstance(interval_emissions_data, (int, float)):
                emissions_record.update({
                    'project_name': getattr(interval_emissions_data, 'project_name', wandb_project),
                    'run_id': getattr(interval_emissions_data, 'run_id', ''),
                    'experiment_id': getattr(interval_emissions_data, 'experiment_id', f"{wandb_run_name}_interval"),
                    'duration': getattr(interval_emissions_data, 'duration', 0.0),
                    'emissions': getattr(interval_emissions_data, 'emissions', interval_emissions),
                    'emissions_rate': getattr(interval_emissions_data, 'emissions_rate', 0.0),
                    'cpu_power': getattr(interval_emissions_data, 'cpu_power', 0.0),
                    'gpu_power': getattr(interval_emissions_data, 'gpu_power', 0.0),
                    'ram_power': getattr(interval_emissions_data, 'ram_power', 0.0),
                    'cpu_energy': getattr(interval_emissions_data, 'cpu_energy', 0.0),
                    'gpu_energy': getattr(interval_emissions_data, 'gpu_energy', 0.0),
                    'ram_energy': getattr(interval_emissions_data, 'ram_energy', 0.0),
                    'energy_consumed': getattr(interval_emissions_data, 'energy_consumed', 0.0),
                    'country_name': getattr(interval_emissions_data, 'country_name', ''),
                    'country_iso_code': getattr(interval_emissions_data, 'country_iso_code', ''),
                    'region': getattr(interval_emissions_data, 'region', ''),
                    'cloud_provider': getattr(interval_emissions_data, 'cloud_provider', ''),
                    'cloud_region': getattr(interval_emissions_data, 'cloud_region', ''),
                    'os': getattr(interval_emissions_data, 'os', ''),
                    'python_version': getattr(interval_emissions_data, 'python_version', ''),
                    'codecarbon_version': getattr(interval_emissions_data, 'codecarbon_version', ''),
                    'cpu_count': getattr(interval_emissions_data, 'cpu_count', 0),
                    'cpu_model': getattr(interval_emissions_data, 'cpu_model', ''),
                    'gpu_count': getattr(interval_emissions_data, 'gpu_count', 0),
                    'gpu_model': getattr(interval_emissions_data, 'gpu_model', ''),
                    'longitude': getattr(interval_emissions_data, 'longitude', 0.0),
                    'latitude': getattr(interval_emissions_data, 'latitude', 0.0),
                    'ram_total_size': getattr(interval_emissions_data, 'ram_total_size', 0.0),
                    'tracking_mode': getattr(interval_emissions_data, 'tracking_mode', 'process'),
                    'on_cloud': getattr(interval_emissions_data, 'on_cloud', False),
                    'pue': getattr(interval_emissions_data, 'pue', 1.0),
                })
            else:
                # Fill with default values if no detailed data available
                emissions_record.update({
                    'project_name': wandb_project,
                    'run_id': '',
                    'experiment_id': f"{wandb_run_name}_interval",
                    'duration': 0.0,
                    'emissions': interval_emissions,
                    'emissions_rate': 0.0,
                    'cpu_power': 0.0,
                    'gpu_power': 0.0,
                    'ram_power': 0.0,
                    'cpu_energy': 0.0,
                    'gpu_energy': 0.0,
                    'ram_energy': 0.0,
                    'energy_consumed': 0.0,
                    'country_name': '',
                    'country_iso_code': '',
                    'region': '',
                    'cloud_provider': '',
                    'cloud_region': '',
                    'os': '',
                    'python_version': '',
                    'codecarbon_version': '',
                    'cpu_count': 0,
                    'cpu_model': '',
                    'gpu_count': 0,
                    'gpu_model': '',
                    'longitude': 0.0,
                    'latitude': 0.0,
                    'ram_total_size': 0.0,
                    'tracking_mode': 'process',
                    'on_cloud': False,
                    'pue': 1.0,
                })
            
            emissions_data_list.append(emissions_record)
            
            # Log to wandb
            if wandb_log:
                log_data = {
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                    "interval_emissions_kg_co2": interval_emissions,
                    "cumulative_emissions_kg_co2": cumulative_emissions,
                    "model_type": model_type,
                }
                wandb.log(log_data, step=iter_num)
                
            # Save checkpoint
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                        'cumulative_emissions_kg_co2': cumulative_emissions,
                        'model_type': model_type,
                    }
                    
                    torch.save(checkpoint, os.path.join(run_out_dir, f'ckpt_{iter_num}_{model_type}_{distance}.pt'))
                    print(f"Saved checkpoint at iteration {iter_num}")
            
            # Restart emissions tracking for next interval (skip for last iteration)
            if tracker and iter_num < max_iters:
                try:
                    tracker = EmissionsTracker(
                        output_file=f"temp_emissions_{wandb_run_name}_interval.csv",
                        project_name=wandb_project,
                        experiment_id=f"{wandb_run_name}_interval",
                        output_dir=run_out_dir,
                        log_level="INFO",
                        measure_power_secs=15,
                        tracking_mode="process",
                        save_to_file=True,
                    )
                    tracker.start()
                    print(f"Restarted emissions tracking for next interval")
                except Exception as e:
                    print(f"Error restarting tracker at iter {iter_num}: {e}")
                    tracker = None

        if iter_num == 0 and eval_only:
            break


        # Updated forward and backward pass in training loop
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
            with ctx:
                if model_type == 'gpt':
                    # GPT and Qwen: Causal language modeling
                    logits, loss, acc = model(X, Y)
                elif model_type == 'qwen':
                    logits, loss, acc = model(input_ids=X, targets=Y)
                else:  # BERT
                    # BERT: Masked language modeling
                    logits, loss, acc = model(input_ids, attention_mask=attention_mask, labels=labels)
                
                if loss.item() > 50.0:
                    print(f"High loss detected: {loss.item():.4f} at iteration {iter_num}")
                    
                loss = loss / gradient_accumulation_steps
                
            # Prefetch next batch - UPDATED FOR QWEN
            if model_type == 'gpt' or model_type == 'qwen':
                X, Y = get_batch('train')
            else:  # BERT
                input_ids, attention_mask, labels = get_batch('train')
                
            scaler.scale(loss).backward()

        # Gradient stability check
        if iter_num % 10 == 0:
            is_stable = check_gradients(raw_model, iter_num)
            if not is_stable:
                print("Stopping training due to gradient instability")
                break

        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if total_norm.item() > grad_clip:
                clip_time += 1

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            accf = acc.item() if acc is not None else 0.0
            
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                
            print(f"iter {iter_num}: loss {lossf:.4f}, acc {accf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            
            # Calculate norms for logging
            params = list(model.parameters())
            total_param_norm = torch.norm(torch.stack([torch.norm(p.data.detach()) for p in params]))
            
            momentum_norm = 0.
            v_norm = 0.
            move_norm = 0.
            
            for state in optimizer.state.values():
                if 'exp_avg' in state:
                    momentum_step = state['exp_avg']
                    v_step = state['exp_avg_sq']
                    move = momentum_step / (torch.sqrt(v_step) + 1e-8)
                    
                    momentum_norm += momentum_step.detach().norm(2) ** 2
                    v_norm += v_step.detach().norm(2) ** 2
                    move_norm += move.detach().norm(2) ** 2
                    
            momentum_norm = torch.sqrt(momentum_norm).item()
            v_norm = torch.sqrt(v_norm).item()
            move_norm = torch.sqrt(move_norm).item()
            
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": lossf,
                    "train/acc": accf,
                    "lr": lr,
                    "param_norm": total_param_norm.item(),
                    "momentum_norm": momentum_norm,
                    "v_norm": v_norm,
                    "move_norm": move_norm,
                    "train/clip_rate": clip_time / (iter_num + 1),
                    "model_type": model_type
                }, step=iter_num)

        iter_num += 1
        local_iter_num += 1

        # Termination condition
        if iter_num > max_iters:
            break

finally:
    # Stop final emissions tracking and save all accumulated data
    if master_process and tracker:
        try:
            final_interval_emissions_data = tracker.stop()
            final_interval_emissions = 0.0
            
            if final_interval_emissions_data:
                # Extract emissions value (could be float or object)
                if isinstance(final_interval_emissions_data, (int, float)):
                    final_interval_emissions = final_interval_emissions_data
                else:
                    final_interval_emissions = getattr(final_interval_emissions_data, 'emissions', 0.0)
                
                cumulative_emissions += final_interval_emissions
                print(f"Final interval emissions: {final_interval_emissions:.6f} kg CO2")
                
                # Add final comprehensive emissions record
                final_emissions_record = {
                    'timestamp': datetime.now().isoformat(),
                    'iteration': iter_num,
                    'model_type': model_type,
                    'distance_type': distance,
                    'wandb_run_name': wandb_run_name,
                    'interval_emissions_kg_co2': final_interval_emissions,
                    'cumulative_emissions_kg_co2': cumulative_emissions,
                    'train_loss': None,  # No evaluation at final step
                    'val_loss': None,
                    'learning_rate': lr,
                    'best_val_loss': best_val_loss,
                }
                
                # Add comprehensive CodeCarbon data if available
                if not isinstance(final_interval_emissions_data, (int, float)):
                    final_emissions_record.update({
                        'project_name': getattr(final_interval_emissions_data, 'project_name', wandb_project),
                        'run_id': getattr(final_interval_emissions_data, 'run_id', ''),
                        'experiment_id': getattr(final_interval_emissions_data, 'experiment_id', f"{wandb_run_name}_interval"),
                        'duration': getattr(final_interval_emissions_data, 'duration', 0.0),
                        'emissions': getattr(final_interval_emissions_data, 'emissions', final_interval_emissions),
                        'emissions_rate': getattr(final_interval_emissions_data, 'emissions_rate', 0.0),
                        'cpu_power': getattr(final_interval_emissions_data, 'cpu_power', 0.0),
                        'gpu_power': getattr(final_interval_emissions_data, 'gpu_power', 0.0),
                        'ram_power': getattr(final_interval_emissions_data, 'ram_power', 0.0),
                        'cpu_energy': getattr(final_interval_emissions_data, 'cpu_energy', 0.0),
                        'gpu_energy': getattr(final_interval_emissions_data, 'gpu_energy', 0.0),
                        'ram_energy': getattr(final_interval_emissions_data, 'ram_energy', 0.0),
                        'energy_consumed': getattr(final_interval_emissions_data, 'energy_consumed', 0.0),
                        'country_name': getattr(final_interval_emissions_data, 'country_name', ''),
                        'country_iso_code': getattr(final_interval_emissions_data, 'country_iso_code', ''),
                        'region': getattr(final_interval_emissions_data, 'region', ''),
                        'cloud_provider': getattr(final_interval_emissions_data, 'cloud_provider', ''),
                        'cloud_region': getattr(final_interval_emissions_data, 'cloud_region', ''),
                        'os': getattr(final_interval_emissions_data, 'os', ''),
                        'python_version': getattr(final_interval_emissions_data, 'python_version', ''),
                        'codecarbon_version': getattr(final_interval_emissions_data, 'codecarbon_version', ''),
                        'cpu_count': getattr(final_interval_emissions_data, 'cpu_count', 0),
                        'cpu_model': getattr(final_interval_emissions_data, 'cpu_model', ''),
                        'gpu_count': getattr(final_interval_emissions_data, 'gpu_count', 0),
                        'gpu_model': getattr(final_interval_emissions_data, 'gpu_model', ''),
                        'longitude': getattr(final_interval_emissions_data, 'longitude', 0.0),
                        'latitude': getattr(final_interval_emissions_data, 'latitude', 0.0),
                        'ram_total_size': getattr(final_interval_emissions_data, 'ram_total_size', 0.0),
                        'tracking_mode': getattr(final_interval_emissions_data, 'tracking_mode', 'process'),
                        'on_cloud': getattr(final_interval_emissions_data, 'on_cloud', False),
                        'pue': getattr(final_interval_emissions_data, 'pue', 1.0),
                    })
                else:
                    # Fill with default values if no detailed data available
                    final_emissions_record.update({
                        'project_name': wandb_project,
                        'run_id': '',
                        'experiment_id': f"{wandb_run_name}_interval",
                        'duration': 0.0,
                        'emissions': final_interval_emissions,
                        'emissions_rate': 0.0,
                        'cpu_power': 0.0,
                        'gpu_power': 0.0,
                        'ram_power': 0.0,
                        'cpu_energy': 0.0,
                        'gpu_energy': 0.0,
                        'ram_energy': 0.0,
                        'energy_consumed': 0.0,
                        'country_name': '',
                        'country_iso_code': '',
                        'region': '',
                        'cloud_provider': '',
                        'cloud_region': '',
                        'os': '',
                        'python_version': '',
                        'codecarbon_version': '',
                        'cpu_count': 0,
                        'cpu_model': '',
                        'gpu_count': 0,
                        'gpu_model': '',
                        'longitude': 0.0,
                        'latitude': 0.0,
                        'ram_total_size': 0.0,
                        'tracking_mode': 'process',
                        'on_cloud': False,
                        'pue': 1.0,
                    })
                
                emissions_data_list.append(final_emissions_record)
            
            print(f"Total training emissions: {cumulative_emissions:.6f} kg CO2")
            
            if wandb_log:
                wandb.log({
                    "final_interval_emissions_kg_co2": final_interval_emissions,
                    "total_emissions_kg_co2": cumulative_emissions,
                    "model_type": model_type
                })
                
        except Exception as e:
            print(f"Error stopping final tracker: {e}")
    
    # Save all accumulated emissions data to single CSV
    save_accumulated_emissions()

if ddp:
    destroy_process_group()

print(f"{model_type.upper()} training finished successfully!")