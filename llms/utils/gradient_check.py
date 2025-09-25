import torch

def check_gradients(model, iter_num):
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Check for problematic gradients
            if torch.isnan(param.grad).any():
                print(f"WARNING: NaN gradient in {name} at iter {iter_num}")
                return False
            if torch.isinf(param.grad).any():
                print(f"WARNING: Inf gradient in {name} at iter {iter_num}")
                return False
            if param_norm > 100.0:  # Much higher threshold
                print(f"WARNING: Very large gradient in {name}: {param_norm:.4f} at iter {iter_num}")
    
    total_norm = total_norm ** 0.5
    print(f"Iter {iter_num}: Total gradient norm: {total_norm:.4f}")
    
    if total_norm > 5000.0:  # Much higher explosion threshold
        print(f"GRADIENT EXPLOSION detected at iter {iter_num}! Norm: {total_norm:.4f}")
        return False
    
    return True

def check_gradients_enhanced(model, iter_num, max_grad_norm=10.0):
    """Enhanced gradient checking with detailed diagnostics"""
    total_norm = 0.0
    param_count = 0
    max_grad = 0.0
    min_grad = float('inf')
    problem_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Track extreme gradients
            grad_max = param.grad.data.abs().max().item()
            grad_min = param.grad.data.abs().min().item()
            
            max_grad = max(max_grad, grad_max)
            min_grad = min(min_grad, grad_min)
            
            # Flag problematic parameters
            if grad_max > 1000:
                problem_params.append((name, grad_max))
            
            # Check for NaN or Inf
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"❌ NaN/Inf gradient in {name}")
                return False
    
    total_norm = total_norm ** 0.5
    
    # Log gradient statistics
    if iter_num % 10 == 0:
        print(f"Gradient stats - Total norm: {total_norm:.4f}, Max: {max_grad:.4f}, Min: {min_grad:.4f}")
    
    # Check for explosion
    if total_norm > max_grad_norm:
        print(f"❌ GRADIENT EXPLOSION at iter {iter_num}! Norm: {total_norm:.4f}")
        if problem_params:
            print("Problematic parameters:")
            for name, grad_val in sorted(problem_params, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {name}: {grad_val:.4f}")
        return False
    
    # Check for vanishing gradients
    if total_norm < 1e-7:
        print(f"⚠️ Vanishing gradients at iter {iter_num}! Norm: {total_norm:.8f}")
    
    return True

'''# Updated training loop with better gradient handling
def train_with_gradient_monitoring():
    """Training loop with enhanced gradient monitoring"""
    
    # Initialize with smaller learning rate
    current_lr = learning_rate * 0.1  # Start with 10% of target LR
    
    for iter_num in range(max_iters):
        # Gradually increase learning rate for first 100 iterations
        if iter_num < 100:
            lr_scale = min(1.0, 0.1 + 0.9 * iter_num / 100)
            current_lr = learning_rate * lr_scale
        else:
            current_lr = get_lr(iter_num) if decay_lr else learning_rate
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Forward pass with error handling
        try:
            with ctx:
                logits, loss, acc = model(X, Y)
                
                # Check for loss explosion
                if loss.item() > 50.0:
                    print(f"❌ Loss explosion: {loss.item():.4f}")
                    break
                
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
        except RuntimeError as e:
            print(f"❌ Runtime error during forward/backward: {e}")
            break
        
        # Enhanced gradient checking
        if not check_gradients_enhanced(model, iter_num):
            print("Stopping due to gradient issues")
            break
        
        # Gradient clipping with monitoring
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            if total_norm > grad_clip:
                print(f"Gradients clipped: {total_norm:.4f} -> {grad_clip}")
        
        # Optimizer step with error handling
        try:
            scaler.step(optimizer)
            scaler.update()
        except RuntimeError as e:
            print(f"❌ Optimizer step failed: {e}")
            break
        
        optimizer.zero_grad(set_to_none=True)
        
        # Get next batch
        X, Y = get_batch('train')
        
        # Print progress
        if iter_num % log_interval == 0:
            print(f"Iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, lr {current_lr:.6f}")
    
    return iter_num'''