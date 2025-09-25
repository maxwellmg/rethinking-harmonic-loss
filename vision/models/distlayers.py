import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import math

import sys
# import keyboard

from tqdm import tqdm

import torch.nn.functional as F

use_custom_loss = False
custom_loss = None  

class EuclideanDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(EuclideanDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist_sq: (B, V)
        n_embd = x.size(-1,)
        w = self.weight
        wx = torch.einsum('bn,vn->bv', x, w) # (B, V)
        ww = torch.norm(w, dim=-1)**2 # (V,)
        xx = torch.norm(x, dim=-1)**2 # (B,)
        
        dist_sq = ww[None,:] + xx[:,None] - 2 * wx + self.eps
        dist_sq = dist_sq / torch.min(dist_sq, dim=-1, keepdim = True)[0]
        return (dist_sq)**(-self.n)

class ManhattanDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(ManhattanDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        batch_size = x.size(0)
        vocab_size = self.weight.size(0)
        embedding_dim = x.size(1)
        
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = self.weight.unsqueeze(0)  # (1, V, N)
        
        manhattan_dist = torch.sum(torch.abs(x_expanded - w_expanded), dim=2) + self.eps
        manhattan_dist = manhattan_dist / torch.min(manhattan_dist, dim=-1, keepdim=True)[0]
        
        return (manhattan_dist) ** (-self.n)

class CosineDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, stable=True):
        super(CosineDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        self.stable = stable
        
    def forward(self, x, scale=False):
        if self.stable:
            # Stable version using F.normalize
            x_normalized = F.normalize(x, p=2, dim=-1)
            w_normalized = F.normalize(self.weight, p=2, dim=-1)
            cosine_similarity = torch.einsum('bn,vn->bv', x_normalized, w_normalized)
        else:
            # Original version
            wx = torch.einsum('bn,vn->bv', x, self.weight)
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            w_norm = torch.norm(self.weight, dim=-1)
            cosine_similarity = wx / ((x_norm * w_norm[None, :]) + self.eps)
        
        cosine_distance = 1 - cosine_similarity + self.eps
        cosine_distance = cosine_distance / torch.min(cosine_distance, dim=-1, keepdim=True)[0]
        
        return (cosine_distance) ** (-self.n)

class MinkowskiDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, p=1.5, n=1., eps=1e-4, bias=False):
        super(MinkowskiDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.p = p
        self.n = n
        self.eps = max(eps, 1e-6)
        
    def forward(self, x, scale=False):
        w = self.weight
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        diff = torch.abs(x_expanded - w_expanded)
        diff = torch.clamp(diff + self.eps, min=self.eps, max=1e6)
        diff_pow = torch.pow(diff, self.p)
        
        dist_pow_p = torch.sum(diff_pow, dim=-1)
        dist_pow_p = torch.clamp(dist_pow_p + self.eps, min=self.eps)
        dist = torch.pow(dist_pow_p, 1.0/self.p)
        
        if scale:
            min_dist = torch.min(dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            dist = dist / min_dist
        
        dist_clamped = torch.clamp(dist + self.eps, min=self.eps)
        return torch.pow(dist_clamped, -self.n)

'''class HammingDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, 
                 threshold=0.5, temperature=1.0, variant="soft"):
        super(HammingDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        self.threshold = threshold
        self.temperature = temperature
        self.variant = variant
        
    def forward(self, x, scale=False):
        w = self.weight
        
        if self.variant == "soft":
            x_soft = torch.sigmoid(x / self.temperature)
            w_soft = torch.sigmoid(w / self.temperature)
            x_expanded = x_soft.unsqueeze(1)
            w_expanded = w_soft.unsqueeze(0)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)
        elif self.variant == "gumbel":
            def gumbel_sigmoid(logits, temperature):
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + self.eps) + self.eps)
                return torch.sigmoid((logits + gumbel_noise) / temperature)
            
            if self.training:
                x_binary = gumbel_sigmoid(x, self.temperature)
                w_binary = gumbel_sigmoid(w, self.temperature)
            else:
                x_binary = torch.sigmoid(x / self.temperature)
                w_binary = torch.sigmoid(w / self.temperature)
            
            x_expanded = x_binary.unsqueeze(1)
            w_expanded = w_binary.unsqueeze(0)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)
        else:  # "hard" - with gradient issues
            temperature = 0.1
            x_binary = torch.sigmoid((x - self.threshold) / temperature)
            w_binary = torch.sigmoid((w - self.threshold) / temperature)
            x_expanded = x_binary.unsqueeze(1)
            w_expanded = w_binary.unsqueeze(0)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)
        
        if scale:
            if self.variant == "soft":
                max_dist = torch.max(hamming_dist, dim=-1, keepdim=True)[0]
                hamming_dist = hamming_dist / (max_dist + self.eps)
            else:
                min_dist = torch.min(hamming_dist, dim=-1, keepdim=True)[0]
                hamming_dist = hamming_dist / (min_dist + self.eps)
        
        result = torch.pow(hamming_dist + self.eps, -self.n)
        return result'''

class HammingDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-8, bias=False, 
                 threshold=0.5, temperature=1.0, variant="soft"):
        super(HammingDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-8)
        self.threshold = threshold
        self.temperature = max(temperature, 0.1)  # Prevent too small temperature
        self.variant = variant
        
    def forward(self, x, scale=False):
        w = self.weight
        
        if self.variant == "soft":
            # Soft variant - use sigmoid approximation
            x_soft = torch.sigmoid(x / self.temperature)
            w_soft = torch.sigmoid(w / self.temperature)
            x_expanded = x_soft.unsqueeze(1)
            w_expanded = w_soft.unsqueeze(0)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)
            
        elif self.variant == "gumbel":
            # FIXED: More stable Gumbel implementation
            def stable_gumbel_sigmoid(logits, temperature):
                # More stable Gumbel noise generation
                uniform = torch.rand_like(logits)
                # Clamp to prevent log(0) issues
                uniform = torch.clamp(uniform, min=1e-7, max=1.0 - 1e-7)
                
                # Stable Gumbel noise: avoid double log issues
                gumbel_noise = -torch.log(-torch.log(uniform))
                # Clamp Gumbel noise to prevent extreme values
                gumbel_noise = torch.clamp(gumbel_noise, min=-10.0, max=10.0)
                
                # Apply temperature scaling and sigmoid
                return torch.sigmoid((logits + gumbel_noise) / temperature)
            
            if self.training:
                # Use Gumbel-softmax trick during training
                x_binary = stable_gumbel_sigmoid(x, self.temperature)
                w_binary = stable_gumbel_sigmoid(w, self.temperature)
            else:
                # Use deterministic sigmoid during evaluation
                x_binary = torch.sigmoid(x / self.temperature)
                w_binary = torch.sigmoid(w / self.temperature)
            
            x_expanded = x_binary.unsqueeze(1)
            w_expanded = w_binary.unsqueeze(0)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)
            
        else:  # "hard" variant
            # Hard variant with straight-through estimator
            temperature = max(self.temperature, 0.1)  # Prevent too small temp
            
            # Forward pass: hard thresholding
            x_hard = (x > self.threshold).float()
            w_hard = (w > self.threshold).float()
            
            # Backward pass: use sigmoid gradients (straight-through estimator)
            x_binary = x_hard + torch.sigmoid(x / temperature) - torch.sigmoid(x / temperature).detach()
            w_binary = w_hard + torch.sigmoid(w / temperature) - torch.sigmoid(w / temperature).detach()
            
            x_expanded = x_binary.unsqueeze(1)
            w_expanded = w_binary.unsqueeze(0)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)
        
        # Debug: Check for problematic values
        if torch.isnan(hamming_dist).any() or torch.isinf(hamming_dist).any():
            print(f"WARNING: Hamming distance has NaN/Inf values!")
            print(f"Variant: {self.variant}, Temperature: {self.temperature}")
            print(f"Distance range: {hamming_dist.min():.6f} to {hamming_dist.max():.6f}")
        
        # Optional scaling
        if scale:
            if self.variant == "soft":
                # For soft variant, normalize by maximum possible distance
                max_dist = torch.max(hamming_dist, dim=-1, keepdim=True)[0]
                max_dist = torch.clamp(max_dist, min=self.eps)
                hamming_dist = hamming_dist / max_dist
            else:
                # For binary variants, normalize by minimum distance
                min_dist = torch.min(hamming_dist, dim=-1, keepdim=True)[0]
                min_dist = torch.clamp(min_dist, min=self.eps)
                hamming_dist = hamming_dist / min_dist
        
        # Convert to similarity
        hamming_dist = torch.clamp(hamming_dist + self.eps, min=self.eps, max=1e6)
        similarity = torch.pow(hamming_dist, -self.n)
        
        # Final safety check
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            print(f"WARNING: Hamming similarity has NaN/Inf values!")
            similarity = torch.ones_like(similarity) / similarity.size(-1)
        
        return similarity

class ChebyshevDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-8, bias=False, 
                 smooth=False, alpha=10.0):
        super(ChebyshevDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-8)
        self.smooth = smooth
        # FIXED: Clamp alpha to reasonable range to prevent overflow
        self.alpha = max(min(alpha, 20.0), 1.0)  # Between 1.0 and 20.0
        
    def forward(self, x, scale=False):
        w = self.weight
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        diff = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        
        # FIXED: Normalize features to prevent scale-dependent issues
        # Chebyshev distance is very sensitive to feature magnitudes
        diff_normalized = diff / (torch.std(diff, dim=-1, keepdim=True) + self.eps)
        
        if self.smooth:
            # FIXED: More stable smooth approximation
            # Clamp inputs to logsumexp to prevent overflow
            scaled_diff = torch.clamp(self.alpha * diff_normalized, min=-50, max=50)
            
            try:
                chebyshev_dist = torch.logsumexp(scaled_diff, dim=-1) / self.alpha
            except Exception as e:
                print(f"WARNING: Chebyshev logsumexp failed ({e}), falling back to max")
                chebyshev_dist = torch.max(diff_normalized, dim=-1)[0]
        else:
            # FIXED: Use softmax for better gradients than hard max
            # This provides a differentiable approximation to max
            temp = 1.0  # Temperature for softmax
            weights = torch.softmax(diff_normalized / temp, dim=-1)
            chebyshev_dist = torch.sum(weights * diff_normalized, dim=-1)
        
        # Debug: Check for problematic values
        if torch.isnan(chebyshev_dist).any() or torch.isinf(chebyshev_dist).any():
            print(f"WARNING: Chebyshev distance has NaN/Inf values!")
            print(f"Smooth: {self.smooth}, Alpha: {self.alpha}")
            print(f"Distance range: {chebyshev_dist.min():.6f} to {chebyshev_dist.max():.6f}")
            # Fallback to simple max
            chebyshev_dist = torch.max(diff_normalized, dim=-1)[0]
        
        # Optional scaling
        if scale:
            min_dist = torch.min(chebyshev_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            chebyshev_dist = chebyshev_dist / min_dist
        
        # Convert to similarity
        dist_clamped = torch.clamp(chebyshev_dist + self.eps, min=self.eps, max=1e6)
        similarity = torch.pow(dist_clamped, -self.n)
        
        # Final safety check
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            print(f"WARNING: Chebyshev similarity has NaN/Inf values!")
            similarity = torch.ones_like(similarity) / similarity.size(-1)
        
        return similarity
        
'''class ChebyshevDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, 
                 smooth=False, alpha=10.0):
        super(ChebyshevDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        self.smooth = smooth
        self.alpha = alpha
        
    def forward(self, x, scale=False):
        w = self.weight
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        diff = torch.abs(x_expanded - w_expanded)
        
        if self.smooth:
            chebyshev_dist = torch.logsumexp(self.alpha * diff, dim=-1) / self.alpha
        else:
            chebyshev_dist = torch.max(diff, dim=-1)[0]
        
        if scale:
            chebyshev_dist = chebyshev_dist / torch.min(chebyshev_dist, dim=-1, keepdim=True)[0]
        
        return torch.pow(chebyshev_dist + self.eps, -self.n)'''



'''class CanberraDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False,
                 variant="standard", min_denom=1e-3, weight_power=1.0, normalize_weights=True):
        super(CanberraDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        self.variant = variant
        self.min_denom = min_denom
        self.weight_power = weight_power
        self.normalize_weights = normalize_weights
        
    def forward(self, x, scale=False):
        w = self.weight
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        numerator = torch.abs(x_expanded - w_expanded)
        
        if self.variant == "robust":
            raw_denominator = torch.abs(x_expanded) + torch.abs(w_expanded)
            denominator = torch.clamp(raw_denominator, min=self.min_denom)
        elif self.variant == "weighted":
            denominator = torch.abs(x_expanded) + torch.abs(w_expanded) + self.eps
            feature_weights = torch.pow(denominator, self.weight_power)
            if self.normalize_weights:
                feature_weights = feature_weights / torch.sum(feature_weights, dim=-1, keepdim=True)
            weighted_terms = feature_weights * (numerator / denominator)
            canberra_dist = torch.sum(weighted_terms, dim=-1)
            if scale:
                canberra_dist = canberra_dist / torch.min(canberra_dist, dim=-1, keepdim=True)[0]
            return torch.pow(canberra_dist + self.eps, -self.n)
        else:  # standard
            denominator = torch.abs(x_expanded) + torch.abs(w_expanded) + self.eps
        
        canberra_dist = torch.sum(numerator / denominator, dim=-1)
        
        if scale:
            canberra_dist = canberra_dist / torch.min(canberra_dist, dim=-1, keepdim=True)[0]
        
        return torch.pow(canberra_dist + self.eps, -self.n)'''

class CanberraDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-8, bias=False,
                 variant="standard", min_denom=1e-8, weight_power=1.0, normalize_weights=True):
        super(CanberraDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-8)  # Ensure reasonable epsilon
        self.variant = variant
        self.min_denom = max(min_denom, 1e-8)  # Much smaller minimum
        self.weight_power = weight_power
        self.normalize_weights = normalize_weights
        
    def forward(self, x, scale=False):
        w = self.weight
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        numerator = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        
        if self.variant == "robust":
            # More conservative robust variant
            denominator = torch.abs(x_expanded) + torch.abs(w_expanded)
            # Only clamp when denominator is actually problematic
            denominator = torch.clamp(denominator, min=self.min_denom)
            
        elif self.variant == "weighted":
            # Keep existing weighted logic
            denominator = torch.abs(x_expanded) + torch.abs(w_expanded) + self.eps
            feature_weights = torch.pow(denominator, self.weight_power)
            if self.normalize_weights:
                feature_weights = feature_weights / torch.sum(feature_weights, dim=-1, keepdim=True)
            weighted_terms = feature_weights * (numerator / denominator)
            canberra_dist = torch.sum(weighted_terms, dim=-1)
            if scale:
                min_dist = torch.min(canberra_dist, dim=-1, keepdim=True)[0]
                min_dist = torch.clamp(min_dist, min=self.eps)
                canberra_dist = canberra_dist / min_dist
            return torch.pow(canberra_dist + self.eps, -self.n)
            
        else:  # "standard"
            # FIXED: Proper Canberra distance - only add epsilon when needed
            denominator = torch.abs(x_expanded) + torch.abs(w_expanded)
            # Add epsilon only for numerical stability when denominator is near zero
            denominator = torch.where(denominator < self.eps, 
                                    torch.full_like(denominator, self.eps), 
                                    denominator)
        
        # Calculate Canberra terms
        canberra_terms = numerator / denominator  # (B, V, N)
        canberra_dist = torch.sum(canberra_terms, dim=-1)  # (B, V)
        
        # Debug: Check for problematic values
        if torch.isnan(canberra_dist).any() or torch.isinf(canberra_dist).any():
            print(f"WARNING: Canberra distance has NaN/Inf values!")
            print(f"Numerator range: {numerator.min():.6f} to {numerator.max():.6f}")
            print(f"Denominator range: {denominator.min():.6f} to {denominator.max():.6f}")
            print(f"Distance range: {canberra_dist.min():.6f} to {canberra_dist.max():.6f}")
        
        # Optional scaling (often problematic)
        if scale:
            min_dist = torch.min(canberra_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            canberra_dist = canberra_dist / min_dist
        
        # Convert to similarity: distance^(-n)
        # Clamp distance to prevent overflow in the power operation
        canberra_dist = torch.clamp(canberra_dist + self.eps, min=self.eps, max=1e6)
        similarity = torch.pow(canberra_dist, -self.n)
        
        # Final check for problematic values
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            print(f"WARNING: Canberra similarity has NaN/Inf values!")
            # Fallback to uniform distribution
            similarity = torch.ones_like(similarity) / similarity.size(-1)
        
        return similarity

class BrayCurtisDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-3, bias=False,
                 variant="standard", normalize_inputs=True, min_sum=1e-3):
        super(BrayCurtisDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.variant = variant
        self.normalize_inputs = normalize_inputs
        self.min_sum = max(min_sum, self.eps * 10)
        
    def forward(self, x, scale=False):
        w = self.weight
        
        if self.variant == "normalized" and self.normalize_inputs:
            x_pos = torch.abs(x) + self.eps
            w_pos = torch.abs(w) + self.eps
            x_sum = torch.clamp(torch.sum(x_pos, dim=-1, keepdim=True), min=self.min_sum)
            w_sum = torch.clamp(torch.sum(w_pos, dim=-1, keepdim=True), min=self.min_sum)
            x_norm = x_pos / x_sum
            w_norm = w_pos / w_sum
            x_expanded = x_norm.unsqueeze(1)
            w_expanded = w_norm.unsqueeze(0)
        else:
            x_expanded = x.unsqueeze(1)
            w_expanded = w.unsqueeze(0)
        
        numerator = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1)
        
        if self.variant == "abs":
            denominator = torch.sum(torch.abs(x_expanded) + torch.abs(w_expanded), dim=-1)
        else:  # standard or normalized
            denominator = torch.sum(x_expanded + w_expanded, dim=-1)
        
        denominator = torch.clamp(torch.abs(denominator) + self.eps, min=self.eps * 10)
        bray_curtis_dist = numerator / denominator
        
        if scale:
            min_dist = torch.min(bray_curtis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            bray_curtis_dist = bray_curtis_dist / min_dist
        
        dist_clamped = torch.clamp(bray_curtis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)

'''class MahalanobisDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False,
                 variant="standard", learn_cov=True, init_identity=True, 
                 regularize_cov=True, reg_lambda=1e-2):
        super(MahalanobisDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.variant = variant
        self.learn_cov = learn_cov
        self.regularize_cov = regularize_cov
        self.reg_lambda = max(reg_lambda, 1e-4)
        
        if variant == "diagonal":
            self.diag_cov_inv = nn.Parameter(torch.zeros(in_features))
        elif variant == "cholesky":
            self.chol_factor = nn.Parameter(torch.eye(in_features) * 0.1 + torch.randn(in_features, in_features) * 0.01)
        elif learn_cov:
            if init_identity:
                self.cov_inv = nn.Parameter(torch.eye(in_features) * (1.0 + self.reg_lambda))
            else:
                L = torch.randn(in_features, in_features) * 0.01
                self.cov_inv = nn.Parameter(L @ L.T + torch.eye(in_features) * self.reg_lambda)
        else:
            self.register_buffer('cov_inv', torch.eye(in_features))
        
    def forward(self, x, scale=False):
        w = self.weight
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        diff = x_expanded - w_expanded
        
        if self.variant == "diagonal":
            diag_pos = torch.exp(torch.clamp(self.diag_cov_inv, min=-10, max=10)) + self.eps
            diff_sq = diff * diff
            weighted_diff_sq = diff_sq * diag_pos.unsqueeze(0).unsqueeze(0)
            mahalanobis_sq = torch.sum(weighted_diff_sq, dim=-1)
        elif self.variant == "cholesky":
            L = torch.tril(self.chol_factor)
            diag_indices = torch.arange(L.size(0), device=L.device)
            L[diag_indices, diag_indices] = torch.clamp(L[diag_indices, diag_indices], min=self.eps)
            cov_inv = L @ L.T + torch.eye(L.size(0), device=L.device) * self.eps * 10
            try:
                diff_transformed = torch.einsum('bvn,nm->bvm', diff, cov_inv)
                mahalanobis_sq = torch.sum(diff_transformed * diff, dim=-1)
            except:
                mahalanobis_sq = torch.sum(diff * diff, dim=-1)
        else:  # standard
            if self.learn_cov:
                if self.regularize_cov:
                    cov_inv_reg = self.cov_inv + torch.eye(self.cov_inv.size(0), 
                                                         device=self.cov_inv.device) * self.reg_lambda
                else:
                    try:
                        U, S, V = torch.svd(self.cov_inv)
                        S_pos = torch.clamp(S, min=self.eps * 10)
                        cov_inv_reg = U @ torch.diag(S_pos) @ V.T
                    except:
                        cov_inv_reg = torch.eye(self.cov_inv.size(0), device=self.cov_inv.device) * self.reg_lambda
            else:
                cov_inv_reg = self.cov_inv
            
            try:
                diff_transformed = torch.einsum('bvn,nm->bvm', diff, cov_inv_reg)
                mahalanobis_sq = torch.sum(diff_transformed * diff, dim=-1)
            except:
                mahalanobis_sq = torch.sum(diff * diff, dim=-1)
        
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        
        if scale:
            min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            mahalanobis_dist = mahalanobis_dist / min_dist
        
        dist_clamped = torch.clamp(mahalanobis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)'''

class MahalanobisDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-6, bias=False,
                 variant="standard", learn_cov=True, init_identity=True, 
                 regularize_cov=True, reg_lambda=1e-2):
        super(MahalanobisDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-8)
        self.variant = variant
        self.learn_cov = learn_cov
        self.regularize_cov = regularize_cov
        self.reg_lambda = max(reg_lambda, 1e-4)  # Ensure minimum regularization
        
        if variant == "diagonal":
            # Learn diagonal covariance inverse (log scale for positivity)
            self.log_diag_cov_inv = nn.Parameter(torch.zeros(in_features))
        elif variant == "cholesky":
            # Learn Cholesky factor of precision matrix
            self.chol_factor = nn.Parameter(torch.eye(in_features) * 0.1)
        elif learn_cov:
            if init_identity:
                # Initialize as regularized identity
                self.precision_matrix = nn.Parameter(torch.eye(in_features) * (1.0 + self.reg_lambda))
            else:
                # Initialize with small random perturbation around identity
                L = torch.randn(in_features, in_features) * 0.01
                init_matrix = L @ L.T + torch.eye(in_features) * self.reg_lambda
                self.precision_matrix = nn.Parameter(init_matrix)
        else:
            # Fixed identity precision matrix
            self.register_buffer('precision_matrix', torch.eye(in_features))
        
    def forward(self, x, scale=False):
        w = self.weight
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        diff = x_expanded - w_expanded  # (B, V, N)
        
        try:
            if self.variant == "diagonal":
                # Diagonal covariance: much more stable
                diag_precision = torch.exp(torch.clamp(self.log_diag_cov_inv, min=-10, max=10))
                diag_precision = diag_precision + self.eps  # Ensure positivity
                
                # Weighted squared differences
                weighted_diff_sq = (diff * diff) * diag_precision.unsqueeze(0).unsqueeze(0)
                mahalanobis_sq = torch.sum(weighted_diff_sq, dim=-1)
                
            elif self.variant == "cholesky":
                # Use Cholesky decomposition for stability
                L = torch.tril(self.chol_factor)  # Ensure lower triangular
                
                # Ensure positive diagonal elements
                diag_indices = torch.arange(L.size(0), device=L.device)
                L[diag_indices, diag_indices] = torch.clamp(
                    L[diag_indices, diag_indices], min=self.eps
                )
                
                # Precision matrix from Cholesky: P = L @ L.T
                precision = L @ L.T + torch.eye(L.size(0), device=L.device) * self.eps
                
                # Apply precision matrix: diff.T @ P @ diff
                diff_transformed = torch.einsum('bvn,nm->bvm', diff, precision)
                mahalanobis_sq = torch.sum(diff_transformed * diff, dim=-1)
                
            else:  # "standard"
                if self.learn_cov:
                    # FIXED: Always use regularization for numerical stability
                    precision = self.precision_matrix
                    
                    if self.regularize_cov:
                        # Add regularization to diagonal
                        precision = precision + torch.eye(
                            precision.size(0), device=precision.device
                        ) * self.reg_lambda
                    
                    # Ensure the matrix is symmetric and positive definite
                    precision = (precision + precision.T) / 2  # Symmetrize
                    
                    # Add small diagonal regularization for numerical stability
                    precision = precision + torch.eye(
                        precision.size(0), device=precision.device
                    ) * self.eps * 10
                    
                else:
                    # Use fixed identity precision
                    precision = self.precision_matrix
                
                # Apply precision matrix
                diff_transformed = torch.einsum('bvn,nm->bvm', diff, precision)
                mahalanobis_sq = torch.sum(diff_transformed * diff, dim=-1)
        
        except Exception as e:
            # Fallback to Euclidean distance if Mahalanobis computation fails
            print(f"WARNING: Mahalanobis computation failed ({e}), falling back to Euclidean")
            mahalanobis_sq = torch.sum(diff * diff, dim=-1)
        
        # Ensure non-negative distances
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        
        # Debug: Check for problematic values
        if torch.isnan(mahalanobis_dist).any() or torch.isinf(mahalanobis_dist).any():
            print(f"WARNING: Mahalanobis distance has NaN/Inf values!")
            print(f"Variant: {self.variant}, Distance range: {mahalanobis_dist.min():.6f} to {mahalanobis_dist.max():.6f}")
            # Fallback to uniform distances
            mahalanobis_dist = torch.ones_like(mahalanobis_dist)
        
        # Optional scaling
        if scale:
            min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            mahalanobis_dist = mahalanobis_dist / min_dist
        
        # Convert to similarity
        dist_clamped = torch.clamp(mahalanobis_dist + self.eps, min=self.eps, max=1e6)
        similarity = torch.pow(dist_clamped, -self.n)
        
        # Final safety check
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            print(f"WARNING: Mahalanobis similarity has NaN/Inf values!")
            similarity = torch.ones_like(similarity) / similarity.size(-1)
        
        return similarity

# Unified Image MLP with all distance types
class SimpleImageMLP_WithDistLayer(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[256, 128], 
                 distance_type="euclidean", n=1., eps=1e-4,
                 # Distance-specific parameters
                 minkowski_p=1.5,                    # For Minkowski
                 hamming_threshold=0.5,              # For Hamming
                 hamming_temperature=1.0,            # For Hamming
                 hamming_variant="soft",             # For Hamming: "soft", "gumbel", "hard"
                 chebyshev_smooth=False,             # For Chebyshev
                 chebyshev_alpha=10.0,               # For Chebyshev smooth
                 canberra_variant="standard",        # For Canberra: "standard", "robust", "weighted"
                 canberra_min_denom=1e-3,           # For Canberra robust
                 canberra_weight_power=1.0,         # For Canberra weighted
                 canberra_normalize_weights=True,    # For Canberra weighted
                 bray_curtis_variant="standard",     # For BrayCurtis: "standard", "abs", "normalized"
                 bray_curtis_normalize_inputs=True,  # For BrayCurtis normalized
                 bray_curtis_min_sum=1e-3,          # For BrayCurtis normalized
                 mahalanobis_variant="standard",     # For Mahalanobis: "standard", "diagonal", "cholesky"
                 mahalanobis_learn_cov=True,        # For Mahalanobis
                 mahalanobis_init_identity=True,    # For Mahalanobis
                 mahalanobis_regularize_cov=True,   # For Mahalanobis
                 mahalanobis_reg_lambda=1e-2,       # For Mahalanobis
                 cosine_stable=True,                 # For Cosine
                 scale_distances=False               # Whether to scale distances by minimum
                 ):
        """
        Unified Image MLP with multiple distance layer options.
        
        Args:
            distance_type: One of ["euclidean", "manhattan", "cosine", "minkowski", 
                          "hamming", "chebyshev", "canberra", "bray_curtis", "mahalanobis"]
            n: Power for final transformation (inverse distance)
            eps: Small epsilon for numerical stability
            scale_distances: Whether to normalize distances by minimum
            
            # Distance-specific parameters (see individual classes for details)
        """
        super().__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Final layer with chosen distance type
        distance_type = distance_type.lower()
        
        if distance_type == "euclidean":
            dist_layer = EuclideanDistLayer(prev_size, num_classes, n=n, eps=eps)
        elif distance_type == "manhattan":
            dist_layer = ManhattanDistLayer(prev_size, num_classes, n=n, eps=eps)
        elif distance_type == "cosine":
            dist_layer = CosineDistLayer(prev_size, num_classes, n=n, eps=eps, stable=cosine_stable)
        elif distance_type == "minkowski":
            dist_layer = MinkowskiDistLayer(prev_size, num_classes, p=minkowski_p, n=n, eps=eps)
        elif distance_type == "hamming":
            dist_layer = HammingDistLayer(prev_size, num_classes, n=n, eps=eps,
                                        threshold=hamming_threshold, temperature=hamming_temperature,
                                        variant=hamming_variant)
        elif distance_type == "chebyshev":
            dist_layer = ChebyshevDistLayer(prev_size, num_classes, n=n, eps=eps,
                                          smooth=chebyshev_smooth, alpha=chebyshev_alpha)
        elif distance_type == "canberra":
            dist_layer = CanberraDistLayer(prev_size, num_classes, n=n, eps=eps,
                                         variant=canberra_variant, min_denom=canberra_min_denom,
                                         weight_power=canberra_weight_power, 
                                         normalize_weights=canberra_normalize_weights)
        elif distance_type == "bray_curtis":
            dist_layer = BrayCurtisDistLayer(prev_size, num_classes, n=n, eps=eps,
                                           variant=bray_curtis_variant,
                                           normalize_inputs=bray_curtis_normalize_inputs,
                                           min_sum=bray_curtis_min_sum)
        elif distance_type == "mahalanobis":
            dist_layer = MahalanobisDistLayer(prev_size, num_classes, n=n, eps=eps,
                                            variant=mahalanobis_variant,
                                            learn_cov=mahalanobis_learn_cov,
                                            init_identity=mahalanobis_init_identity,
                                            regularize_cov=mahalanobis_regularize_cov,
                                            reg_lambda=mahalanobis_reg_lambda)
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}. Supported types: "
                           f"['euclidean', 'manhattan', 'cosine', 'minkowski', 'hamming', "
                           f"'chebyshev', 'canberra', 'bray_curtis', 'mahalanobis']")
        
        layers.append(dist_layer)
        
        self.layers = nn.Sequential(*layers[:-1])  # All but the last layer
        self.dist_layer = layers[-1]  # The distance layer
        self.distance_type = distance_type
        self.scale_distances = scale_distances
        
    def forward(self, x, return_embedding=False):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        embedding = self.layers(x)
        
        # Final distance layer
        distances = self.dist_layer(embedding, scale=self.scale_distances)
        
        # Convert to probabilities and logits
        prob_unnorm = distances
        prob = prob_unnorm / torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob + 1e-8)
        
        if return_embedding:
            return logits, embedding
        return logits
