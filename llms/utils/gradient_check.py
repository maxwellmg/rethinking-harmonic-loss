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
