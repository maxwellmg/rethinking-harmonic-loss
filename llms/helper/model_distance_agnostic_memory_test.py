#!/usr/bin/env python3
"""
Training memory test for GPT, BERT, and Qwen architectures.
Tests batch sizes and gradient accumulation for optimal configuration.
"""

import torch
import torch.cuda
import time
import os
import random
from contextlib import nullcontext
from model.GPT import GPTConfig, GPT
from model.BERT import BertConfig, BERT
from model.QWEN import QwenConfig, Qwen2


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return memory_allocated, memory_reserved, total_memory
    return 0, 0, 0

def create_bert_mlm_batch(batch_size, block_size, vocab_size, device, mask_token_id=103):
    """Create BERT MLM batch with masked tokens"""
    # Generate sequences
    input_ids = torch.randint(1, vocab_size-100, (batch_size, block_size), device=device)  # Avoid special tokens
    targets = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    
    # Apply MLM masking strategy (15% of tokens)
    for i in range(batch_size):
        # Select 15% of positions for masking
        num_mask = max(1, int(0.15 * block_size))
        mask_indices = torch.randperm(block_size)[:num_mask]
        
        for idx in mask_indices:
            prob = random.random()
            if prob < 0.8:
                # 80% of time: replace with [MASK]
                input_ids[i, idx] = mask_token_id
            elif prob < 0.9:
                # 10% of time: replace with random token
                input_ids[i, idx] = torch.randint(1, vocab_size-100, (1,)).item()
            # 10% of time: keep original token
        
        # Set non-masked tokens to -100 (ignored in loss)
        non_masked = torch.ones_like(targets[i], dtype=torch.bool)
        non_masked[mask_indices] = False
        targets[i][non_masked] = -100
    
    return input_ids, attention_mask, targets

def test_gpt_training_batch_size(batch_size, block_size=1024, num_iterations=5, distance="baseline"):
    """Test GPT training with specific batch size"""
    print(f"\n=== Testing GPT Training with batch_size={batch_size}, distance={distance} ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GPT model configuration
    model_args = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768, 
        block_size=block_size,
        bias=False, 
        vocab_size=50304, 
        dropout=0.0, 
        scale_attn_by_inverse_layer_idx=True,
        distance=distance
    )
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model = model.to(device)
    
    # Enable model compilation
    print("Compiling GPT model...")
    model = torch.compile(model)
    
    # Setup optimizer
    optimizer = model.configure_optimizers('adamw', 1e-1, 6e-4, (0.9, 0.95), None, None, None, 'cuda')
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-1)


    # Setup gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Autocast context
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    
    vocab_size = model.config.vocab_size
    
    try:
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Warmup iteration
        x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        
        optimizer.zero_grad()

        with ctx:
            logits, loss, acc = model(x, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Check memory after warmup
        mem_alloc, mem_reserved, total_mem = get_gpu_memory_info()
        print(f"GPU Memory - Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB")
        print(f"Memory Utilization: {(mem_reserved/total_mem)*100:.1f}%")
        
        # Time actual training iterations
        start_time = time.time()
        
        for i in range(num_iterations):
            x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            
            optimizer.zero_grad()
            with ctx:
                logits, loss, acc = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        tokens_per_sec = batch_size * block_size / avg_time
        
        print(f"SUCCESS - Average time: {avg_time:.4f}s, Tokens/sec: {tokens_per_sec:.0f}")
        
        return True, avg_time, mem_reserved
        
    except torch.cuda.OutOfMemoryError:
        print(f"OUT OF MEMORY with batch_size={batch_size}")
        torch.cuda.empty_cache()
        return False, None, None
    except Exception as e:
        print(f"Error with batch_size={batch_size}: {e}")
        torch.cuda.empty_cache()
        return False, None, None

def test_bert_training_batch_size(batch_size, block_size=512, num_iterations=5, distance="baseline"):
    """Test BERT MLM training with specific batch size"""
    print(f"\n=== Testing BERT Training with batch_size={batch_size}, distance={distance} ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # BERT model configuration
    model_args = dict(
        vocab_size=30522,
        #vocab_size=50304,  # Use dataset vocab size for consistency
        max_position_embeddings=block_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=True,
        distance=distance,
        type_vocab_size=2,
        pad_token_id=0
    )
    
    bertconf = BertConfig(**model_args)
    model = BERT(bertconf)
    model = model.to(device)
    
    # Enable model compilation
    # In your BERT test function:
    if distance == "baseline":
        print("Compiling BERT model...")
        model = torch.compile(model)
    else:
        print("Skipping BERT compilation for complex distance layers...")
        # Skip compilation for distance layers that cause XBLOCK issues
    
    # Setup optimizer
    optimizer = model.configure_optimizers('adamw', 1e-1, 1e-4, (0.9, 0.999), None, None, None, 'cuda')
    
    # Setup gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Autocast context
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    
    vocab_size = model.config.vocab_size
    mask_token_id = 103  # [MASK] token ID
    
    try:
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Warmup iteration with MLM data
        input_ids, attention_mask, targets = create_bert_mlm_batch(
            batch_size, block_size, vocab_size, device, mask_token_id
        )
        
        optimizer.zero_grad()
        with ctx:
            # Updated to use 'labels' instead of 'targets'
            logits, loss, acc = model(input_ids, attention_mask=attention_mask, labels=targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Check memory after warmup
        mem_alloc, mem_reserved, total_mem = get_gpu_memory_info()
        print(f"GPU Memory - Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB")
        print(f"Memory Utilization: {(mem_reserved/total_mem)*100:.1f}%")
        
        # Time actual training iterations
        start_time = time.time()
        
        for i in range(num_iterations):
            input_ids, attention_mask, targets = create_bert_mlm_batch(
                batch_size, block_size, vocab_size, device, mask_token_id
            )
            
            optimizer.zero_grad()
            with ctx:
                logits, loss, acc = model(input_ids, attention_mask=attention_mask, labels=targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        tokens_per_sec = batch_size * block_size / avg_time
        
        print(f"SUCCESS - Average time: {avg_time:.4f}s, Tokens/sec: {tokens_per_sec:.0f}")
        
        return True, avg_time, mem_reserved
        
    except torch.cuda.OutOfMemoryError:
        print(f"OUT OF MEMORY with batch_size={batch_size}")
        torch.cuda.empty_cache()
        return False, None, None
    except Exception as e:
        print(f"Error with batch_size={batch_size}: {e}")
        torch.cuda.empty_cache()
        return False, None, None

def test_qwen_training_batch_size(batch_size, block_size=1024, num_iterations=5, distance="baseline"):
    """Test Qwen training with specific batch size"""
    print(f"\n=== Testing Qwen Training with batch_size={batch_size}, distance={distance} ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Qwen model configuration (0.5B parameters)
    model_args = dict(
        vocab_size=151936,  # Qwen's native vocab size
        max_position_embeddings=block_size,
        n_embd=896,  # 0.5B model size
        intermediate_size=4864,
        n_layer=24,
        n_head=14,
        num_key_value_heads=2,  # Grouped query attention
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        dropout=0.0,
        bias=True,
        distance=distance,
        block_size=block_size  # For compatibility
    )
    
    qwenconf = QwenConfig(**model_args)
    model = Qwen2(qwenconf)
    model = model.to(device)
    
    # Enable model compilation
    print("Compiling Qwen model...")
    model = torch.compile(model)
    
    # Setup optimizer
    optimizer = model.configure_optimizers('adamw', 1e-1, 1e-4, (0.9, 0.999), None, None, None, 'cuda')
    
    # Setup gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Autocast context
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    
    vocab_size = model.config.vocab_size
    
    try:
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Warmup iteration
        x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        
        # Warmup iteration
        optimizer.zero_grad()
        with ctx:
            outputs = model(input_ids=x, attention_mask=None, targets=y)
            loss = outputs[1]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Check memory after warmup
        mem_alloc, mem_reserved, total_mem = get_gpu_memory_info()
        print(f"GPU Memory - Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB")
        print(f"Memory Utilization: {(mem_reserved/total_mem)*100:.1f}%")
        
        # Time actual training iterations
        start_time = time.time()
        
        for i in range(num_iterations):
            x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            
            optimizer.zero_grad()
            with ctx:
                outputs = model(input_ids=x, attention_mask=None, targets=y)
                loss = outputs[1]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        tokens_per_sec = batch_size * block_size / avg_time
        
        print(f"SUCCESS - Average time: {avg_time:.4f}s, Tokens/sec: {tokens_per_sec:.0f}")
        
        return True, avg_time, mem_reserved
        
    except torch.cuda.OutOfMemoryError:
        print(f"OUT OF MEMORY with batch_size={batch_size}")
        torch.cuda.empty_cache()
        return False, None, None
    except Exception as e:
        print(f"Error with batch_size={batch_size}: {e}")
        torch.cuda.empty_cache()
        return False, None, None

def find_safe_batch_sizes(model_type="all", distance="baseline"):
    """Find safe batch sizes for all architectures"""
    print(f"Finding safe batch sizes for {model_type.upper()} with distance={distance}...")
    
    # Different batch size ranges based on model size
    batch_sizes_small = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]  # GPT, BERT
    batch_sizes_medium = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]   # Qwen (larger model)
    
    results = {}
    
    if model_type in ["gpt", "all"]:
        print("\n" + "="*60)
        print("TESTING GPT ARCHITECTURE")
        print("="*60)
        
        gpt_results = []
        for batch_size in batch_sizes_small:
            success, time_per_batch, memory_used = test_gpt_training_batch_size(
                batch_size, block_size=1024, distance=distance
            )
            if success:
                gpt_results.append((batch_size, time_per_batch, memory_used))
            else:
                break
        results['gpt'] = gpt_results
    
    if model_type in ["bert", "all"]:
        print("\n" + "="*60)
        print("TESTING BERT ARCHITECTURE")
        print("="*60)
        
        bert_results = []
        for batch_size in batch_sizes_small:
            success, time_per_batch, memory_used = test_bert_training_batch_size(
                batch_size, block_size=512, distance=distance
            )
            if success:
                bert_results.append((batch_size, time_per_batch, memory_used))
            else:
                break
        results['bert'] = bert_results
    
    if model_type in ["qwen", "all"]:
        print("\n" + "="*60)
        print("TESTING QWEN ARCHITECTURE")
        print("="*60)
        
        qwen_results = []
        for batch_size in batch_sizes_medium:
            success, time_per_batch, memory_used = test_qwen_training_batch_size(
                batch_size, block_size=1024, distance=distance
            )
            if success:
                qwen_results.append((batch_size, time_per_batch, memory_used))
            else:
                break
        results['qwen'] = qwen_results
    
    return results

def print_recommendations(results):
    """Print training recommendations for all architectures"""
    print("\n" + "="*60)
    print("TRAINING RECOMMENDATIONS")
    print("="*60)
    
    # Model specifications for reference
    model_specs = {
        'gpt': {'params': '124M', 'block_size': 1024},
        'bert': {'params': '110M', 'block_size': 512},
        'qwen': {'params': '500M', 'block_size': 1024}
    }
    
    for arch_name, arch_results in results.items():
        if not arch_results:
            continue
            
        max_batch_size = arch_results[-1][0]
        safe_batch_size = int(max_batch_size * 0.8)
        block_size = model_specs[arch_name]['block_size']
        params = model_specs[arch_name]['params']
        
        print(f"\n{arch_name.upper()} ARCHITECTURE ({params} parameters):")
        print(f"Maximum batch size: {max_batch_size}")
        print(f"Recommended safe batch size: {safe_batch_size}")
        print(f"Block size: {block_size}")
        
        # Show performance for last few batch sizes
        print(f"\nPerformance comparison:")
        for batch_size, time_per_batch, memory_used in arch_results[-3:]:
            tokens_per_sec = batch_size * block_size / time_per_batch
            print(f"Batch {batch_size:2d}: {tokens_per_sec:8.0f} tokens/sec, {memory_used:5.1f}GB memory")
        
        # Gradient accumulation recommendations
        target_effective_batches = [32, 64, 96, 128] if arch_name == 'qwen' else [64, 96, 128]
        print(f"\nGradient accumulation recommendations for {arch_name.upper()}:")
        for target in target_effective_batches:
            if safe_batch_size <= target:
                accum_steps = target // safe_batch_size
                effective_batch = safe_batch_size * accum_steps
                print(f"  Effective batch {effective_batch}: batch_size={safe_batch_size}, gradient_accumulation_steps={accum_steps}")

def main():
    print("Training Memory Test for GPT, BERT, and Qwen")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    
    mem_alloc, mem_reserved, total_mem = get_gpu_memory_info()
    print(f"Total GPU Memory: {total_mem:.1f}GB")
    
    # Test which architectures to run
    import sys
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type not in ['gpt', 'bert', 'qwen', 'all']:
            print("Usage: python script.py [gpt|bert|qwen|all] [distance]")
            model_type = 'all'
    else:
        model_type = 'all'
    
    # Test which distance to use
    distance = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
    
    print(f"\nTesting model type: {model_type}")
    print(f"Testing distance: {distance}")
    
    results = find_safe_batch_sizes(model_type, distance)
    print_recommendations(results)
    
    print(f"\nIf you still get OOM errors, try:")
    print(f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # Quick comparison summary
    if len(results) > 1:
        print(f"\nQUICK COMPARISON SUMMARY:")
        print(f"=" * 30)
        for arch_name, arch_results in results.items():
            if arch_results:
                safe_batch = int(arch_results[-1][0] * 0.8)
                memory_used = arch_results[-1][2]
                block_size = 1024 if arch_name in ['gpt', 'qwen'] else 512
                tokens_per_sec = safe_batch * block_size / arch_results[-1][1]
                print(f"{arch_name.upper():4s}: batch={safe_batch:2d}, memory={memory_used:4.1f}GB, speed={tokens_per_sec:6.0f} tok/s")

if __name__ == "__main__":
    main()