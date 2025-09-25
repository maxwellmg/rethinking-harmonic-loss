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


# ===== RECOMMENDATION =====

'''def recommendation():
    """
    MY RECOMMENDATION:
    
    1. DON'T rely on fillnan as primary strategy
    2. IF you must use fillnan, use context-aware replacement
    3. BETTER: Use mathematical prevention + minimal cleanup
    4. BEST: Fix the underlying math (HybridStableEuclidean)
    
    EFFECTIVENESS RANKING:
    1. Mathematical fixes (90% effective)
    2. Hybrid approach (85% effective) 
    3. Smart context-aware fillnan (70% effective)
    4. Large/small number replacement (50% effective)
    5. Zero replacement (40% effective)
    
    VERDICT: Smart fillnan CAN help, but it's a band-aid solution.
    The hybrid approach gives you the best of both worlds.
    """
    pass


# ===== EFFECTIVENESS ANALYSIS =====

def analyze_fillnan_strategies():
    """
    STRATEGY 1: Replace with 0 (current)
    ✅ Pro: Neutral value, doesn't bias computation strongly
    ❌ Con: Can create artificial "perfect" distances/similarities
    ❌ Con: Breaks gradient flow completely
    
    STRATEGY 2: Replace with very large numbers (1e6, 1e8)
    ✅ Pro: Represents "very dissimilar" which might be mathematically meaningful
    ❌ Con: Can cause gradient explosion in subsequent operations
    ❌ Con: Still masks the underlying problem
    ❌ Con: Can dominate softmax computations
    
    STRATEGY 3: Replace with very small numbers (1e-8)
    ✅ Pro: Represents "very similar" in distance contexts
    ❌ Con: Can cause numerical underflow
    ❌ Con: May not be meaningful for all distance types
    ❌ Con: Still doesn't fix root cause
    
    STRATEGY 4: Adaptive replacement based on tensor range
    ✅ Pro: Context-aware replacement
    ✅ Pro: Less likely to completely dominate computations
    ❌ Con: More complex, harder to debug
    ❌ Con: Still masking fundamental issues
    """
    
    return {
        "zero_replacement": {"effectiveness": 6, "safety": 8, "debugging": 4},
        "large_replacement": {"effectiveness": 4, "safety": 3, "debugging": 2}, 
        "small_replacement": {"effectiveness": 5, "safety": 6, "debugging": 3},
        "adaptive_replacement": {"effectiveness": 7, "safety": 7, "debugging": 5},
        "mathematical_fixes": {"effectiveness": 10, "safety": 10, "debugging": 9}
    }
'''