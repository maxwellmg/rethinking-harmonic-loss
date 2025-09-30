import torch

# ANALYSIS: Different fillnan strategies and their effects

def fillnan_original(x, nan_value=0.):
    """Original: Replace with 0 or small constants"""
    return torch.nan_to_num(x, nan=nan_value, posinf=nan_value, neginf=nan_value)

def fillnan_extreme_values(x, nan_replacement="large"):
    """Replace NaN/Inf with very large or very small numbers"""
    if nan_replacement == "large":
        # Use large positive numbers
        return torch.nan_to_num(x, nan=1e6, posinf=1e6, neginf=-1e6)
    elif nan_replacement == "small":
        # Use very small numbers
        return torch.nan_to_num(x, nan=1e-8, posinf=1e-8, neginf=-1e-8)
    elif nan_replacement == "adaptive":
        # Use values based on the tensor's existing range
        finite_mask = torch.isfinite(x)
        if finite_mask.any():
            finite_vals = x[finite_mask]
            max_val = finite_vals.max().item()
            min_val = finite_vals.min().item()
            range_val = max_val - min_val
            
            # Replace with values just outside the existing range
            large_replacement = max_val + range_val
            small_replacement = min_val - range_val
            return torch.nan_to_num(x, nan=0.0, posinf=large_replacement, neginf=small_replacement)
        else:
            # Fallback if no finite values
            return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

# ===== BETTER APPROACH: CONTEXT-AWARE REPLACEMENT =====

def smart_fillnan_for_distances(x, context="euclidean_distance"):
    """Context-aware NaN/Inf replacement for distance layers"""
    
    if context == "euclidean_distance":
        # For distances: NaN -> large distance, +inf -> large distance, -inf -> 0
        return torch.nan_to_num(x, nan=1000.0, posinf=1000.0, neginf=0.0)
    
    elif context == "cosine_similarity":
        # For similarities: NaN -> 0 (neutral), +inf -> 1 (max sim), -inf -> -1 (min sim)
        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    elif context == "distance_power":
        # For distance^(-n): NaN -> small similarity, +inf -> 0, -inf -> 0
        return torch.nan_to_num(x, nan=1e-6, posinf=0.0, neginf=0.0)
    
    elif context == "logits":
        # For logits: Keep reasonable range for softmax
        return torch.nan_to_num(x, nan=-10.0, posinf=10.0, neginf=-10.0)
    
    elif context == "probabilities":
        # For probabilities: Must stay in [0,1]
        return torch.nan_to_num(x, nan=1e-8, posinf=1.0, neginf=0.0)
    
    else:
        # Default: conservative replacement
        return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

def improved_fillnan(x, context="general"):
    """Context-aware fillnan replacement"""
    context_mapping = {
        "euclidean_distance": {"nan": 100.0, "posinf": 100.0, "neginf": 0.0},
        "manhattan_distance": {"nan": 100.0, "posinf": 100.0, "neginf": 0.0}, 
        "cosine_similarity": {"nan": 0.0, "posinf": 1.0, "neginf": -1.0},
        "distance_power": {"nan": 1e-6, "posinf": 0.0, "neginf": 0.0},
        "logits": {"nan": -10.0, "posinf": 10.0, "neginf": -10.0},
        "general": {"nan": 0.0, "posinf": 1e3, "neginf": -1e3}
    }
    
    values = context_mapping.get(context, context_mapping["general"])
    return torch.nan_to_num(x, nan=values["nan"], posinf=values["posinf"], neginf=values["neginf"])

def large_number_fillnan(x, nan_value=1e6):
    """Replace with large numbers"""
    return torch.nan_to_num(x, nan=nan_value, posinf=nan_value, neginf=-nan_value)

def small_number_fillnan(x, nan_value=1e-8):
    """Replace with small numbers"""
    return torch.nan_to_num(x, nan=nan_value, posinf=nan_value, neginf=nan_value)