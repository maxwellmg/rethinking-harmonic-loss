import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import math
from src.utils.dataset import *
from src.utils.visualization import *

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
        # x: (B, N) - batch of input embeddings
        # self.weight: (V, N) - class embeddings (vocab_size x embedding_dim)
        # Returns: (B, V) - inverse Manhattan distances
        
        batch_size = x.size(0)
        vocab_size = self.weight.size(0)
        embedding_dim = x.size(1)
        
        # Expand dimensions for broadcasting
        # x_expanded: (B, 1, N)
        # w_expanded: (1, V, N)
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = self.weight.unsqueeze(0)  # (1, V, N)
        
        # Compute Manhattan distance: sum of absolute differences
        # |x - w| for each dimension, then sum across dimensions
        manhattan_dist = torch.sum(torch.abs(x_expanded - w_expanded), dim=2) + self.eps
        # Shape: (B, V)
        
        # Optional: normalize by minimum distance (as in original)
        manhattan_dist = manhattan_dist / torch.min(manhattan_dist, dim=-1, keepdim=True)[0]
        
        # Return inverse distance to the power of n
        return (manhattan_dist) ** (-self.n)

class CosineDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(CosineDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        # x: (B, N) - batch of input embeddings
        # self.weight: (V, N) - class embeddings (vocab_size x embedding_dim)
        # Returns: (B, V) - inverse cosine distances
        
        # Cosine distance = 1 - cosine_similarity
        # cosine_similarity = (x · w) / (||x|| * ||w||)
        
        # Compute dot products: x · w
        wx = torch.einsum('bn,vn->bv', x, self.weight)  # (B, V)
        
        # Compute norms
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # (B, 1)
        w_norm = torch.norm(self.weight, dim=-1)      # (V,)
        
        # Compute cosine similarity: (x · w) / (||x|| * ||w||)
        # Add eps to prevent division by zero
        cosine_similarity = wx / ((x_norm * w_norm[None, :]) + self.eps)
        
        # Compute cosine distance: 1 - cosine_similarity
        cosine_distance = 1 - cosine_similarity + self.eps
        
        # Optional: normalize by minimum distance (as in original)
        cosine_distance = cosine_distance / torch.min(cosine_distance, dim=-1, keepdim=True)[0]
        
        # Return inverse distance to the power of n
        return (cosine_distance) ** (-self.n)


# Alternative implementation using F.normalize for numerical stability
class CosineDistLayerStable(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(CosineDistLayerStable, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        # Normalize both input and weight vectors to unit length
        x_normalized = F.normalize(x, p=2, dim=-1)           # (B, N)
        w_normalized = F.normalize(self.weight, p=2, dim=-1) # (V, N)
        
        # Compute cosine similarity (dot product of normalized vectors)
        cosine_similarity = torch.einsum('bn,vn->bv', x_normalized, w_normalized)  # (B, V)
        
        # Compute cosine distance: 1 - cosine_similarity
        cosine_distance = 1 - cosine_similarity + self.eps
        
        # Optional: normalize by minimum distance (following original pattern)
        cosine_distance = cosine_distance / torch.min(cosine_distance, dim=-1, keepdim=True)[0]
        
        # Return inverse distance to the power of n
        return (cosine_distance) ** (-self.n)

class MinkowskiDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, p=1.5, n=1., eps=1e-4, bias=False):
        super(MinkowskiDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.p = p  # Minkowski parameter (p=1 for Manhattan, p=2 for Euclidean)
        self.n = n  # Power for final transformation
        self.eps = max(eps, 1e-6)  # Increased minimum eps
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist: (B, V)
        w = self.weight
        
        # Compute Minkowski distance
        # |x - w|_p = (sum(|x_i - w_i|^p))^(1/p)
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        # Compute |x_i - w_i|^p with better numerical stability
        diff = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        
        # FIXED: Clamp to prevent extreme values
        diff = torch.clamp(diff + self.eps, min=self.eps, max=1e6)
        diff_pow = torch.pow(diff, self.p)  # (B, V, N)
        
        # Sum over features and take p-th root
        dist_pow_p = torch.sum(diff_pow, dim=-1)  # (B, V)
        
        # FIXED: More robust p-th root calculation
        dist_pow_p = torch.clamp(dist_pow_p + self.eps, min=self.eps)
        dist = torch.pow(dist_pow_p, 1.0/self.p)  # (B, V)
        
        # Normalize distances
        if scale:
            min_dist = torch.min(dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)  # FIXED: Prevent division by zero
            dist = dist / min_dist
        
        # Apply final transformation with clamping
        dist_clamped = torch.clamp(dist + self.eps, min=self.eps)
        return torch.pow(dist_clamped, -self.n)

class HammingDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, threshold=0.5):
        super(HammingDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n  # Power for final transformation
        self.eps = eps
        self.threshold = threshold  # Threshold for binarization
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist: (B, V)
        w = self.weight
        
        # ISSUE: Hard binarization breaks gradients
        # OLD (BROKEN): x_binary = (x > self.threshold).float()
        # NEW (FIXED): Use differentiable approximation
        
        # Smooth binarization using sigmoid with steep slope
        temperature = 0.1  # Small temperature for sharp transition
        x_binary = torch.sigmoid((x - self.threshold) / temperature)
        w_binary = torch.sigmoid((w - self.threshold) / temperature)
        
        # Compute Hamming distance
        x_expanded = x_binary.unsqueeze(1)  # (B, 1, N)
        w_expanded = w_binary.unsqueeze(0)  # (1, V, N)
        
        # Count mismatches: soft XOR operation
        diff = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        hamming_dist = torch.sum(diff, dim=-1)  # (B, V)
        
        # Normalize distances if requested
        if scale:
            min_dist = torch.min(hamming_dist, dim=-1, keepdim=True)[0]
            hamming_dist = hamming_dist / (min_dist + self.eps)
        
        # Apply final transformation (ensure gradients flow)
        result = torch.pow(hamming_dist + self.eps, -self.n)
        
        # Debug: Check if result requires grad
        if not result.requires_grad and x.requires_grad:
            print("WARNING: Hamming result doesn't require grad!")
            print(f"x.requires_grad: {x.requires_grad}")
            print(f"w.requires_grad: {w.requires_grad}")
            print(f"result.requires_grad: {result.requires_grad}")
        
        return result


class HammingDistLayerSoft(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, temperature=1.0):
        super(HammingDistLayerSoft, self).__init__(in_features, out_features, bias=bias)
        self.n = n  # Power for final transformation
        self.eps = eps
        self.temperature = temperature  # Temperature for soft binarization
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist: (B, V)
        w = self.weight
        
        # Soft binarization using sigmoid (this preserves gradients)
        x_soft = torch.sigmoid(x / self.temperature)
        w_soft = torch.sigmoid(w / self.temperature)
        
        # Compute soft Hamming distance
        x_expanded = x_soft.unsqueeze(1)  # (B, 1, N)
        w_expanded = w_soft.unsqueeze(0)  # (1, V, N)
        
        # Soft XOR: |p - q| where p, q are probabilities
        diff = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        soft_hamming_dist = torch.sum(diff, dim=-1)  # (B, V)
        
        # Normalize distances if requested
        if scale:
            max_dist = torch.max(soft_hamming_dist, dim=-1, keepdim=True)[0]
            soft_hamming_dist = soft_hamming_dist / (max_dist + self.eps)
        
        # Apply final transformation
        result = torch.pow(soft_hamming_dist + self.eps, -self.n)
        
        return result


class HammingDistLayerGumbel(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, temperature=1.0):
        super(HammingDistLayerGumbel, self).__init__(in_features, out_features, bias=bias)
        self.n = n  # Power for final transformation
        self.eps = eps
        self.temperature = temperature
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        w = self.weight
        
        # Gumbel-sigmoid for differentiable discrete sampling
        def gumbel_sigmoid(logits, temperature):
            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + self.eps) + self.eps)
            # Apply Gumbel-sigmoid
            return torch.sigmoid((logits + gumbel_noise) / temperature)
        
        # Convert to logits (assuming input is roughly in [-inf, inf])
        x_logits = x
        w_logits = w
        
        # Apply Gumbel-sigmoid
        if self.training:
            x_binary = gumbel_sigmoid(x_logits, self.temperature)
            w_binary = gumbel_sigmoid(w_logits, self.temperature)
        else:
            # Use deterministic sigmoid during inference
            x_binary = torch.sigmoid(x_logits / self.temperature)
            w_binary = torch.sigmoid(w_logits / self.temperature)
        
        # Compute soft Hamming distance
        x_expanded = x_binary.unsqueeze(1)  # (B, 1, N)
        w_expanded = w_binary.unsqueeze(0)  # (1, V, N)
        
        diff = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        hamming_dist = torch.sum(diff, dim=-1)  # (B, V)
        
        if scale:
            max_dist = torch.max(hamming_dist, dim=-1, keepdim=True)[0]
            hamming_dist = hamming_dist / (max_dist + self.eps)
        
        result = torch.pow(hamming_dist + self.eps, -self.n)
        return result

class ChebyshevDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(ChebyshevDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n  # Power for final transformation
        self.eps = eps
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist: (B, V)
        w = self.weight
        
        # Compute Chebyshev distance (L∞ norm)
        # ||x - w||_∞ = max_i |x_i - w_i|
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        # Compute absolute differences
        diff = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        
        # Take maximum across features (Chebyshev distance)
        chebyshev_dist = torch.max(diff, dim=-1)[0]  # (B, V)
        
        # Normalize distances if requested
        if scale:
            chebyshev_dist = chebyshev_dist / torch.min(chebyshev_dist, dim=-1, keepdim=True)[0]
        
        # Apply final transformation (add eps to avoid division by zero)
        return torch.pow(chebyshev_dist + self.eps, -self.n)


class ChebyshevDistLayerSmooth(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, alpha=10.0):
        super(ChebyshevDistLayerSmooth, self).__init__(in_features, out_features, bias=bias)
        self.n = n  # Power for final transformation
        self.eps = eps
        self.alpha = alpha  # Smoothing parameter for soft max
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist: (B, V)
        w = self.weight
        
        # Compute smooth Chebyshev distance using log-sum-exp trick
        # Smooth max approximation: (1/α) * log(Σ exp(α * |x_i - w_i|))
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        # Compute absolute differences
        diff = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        
        # Smooth maximum using log-sum-exp
        # This approximates max(diff) but remains differentiable
        smooth_chebyshev_dist = torch.logsumexp(self.alpha * diff, dim=-1) / self.alpha  # (B, V)
        
        # Normalize distances if requested
        if scale:
            smooth_chebyshev_dist = smooth_chebyshev_dist / torch.min(smooth_chebyshev_dist, dim=-1, keepdim=True)[0]
        
        # Apply final transformation
        return torch.pow(smooth_chebyshev_dist + self.eps, -self.n)

class CanberraDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(CanberraDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n  # Power for final transformation
        self.eps = eps
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist: (B, V)
        w = self.weight
        
        # Compute Canberra distance
        # d(x, w) = Σ |x_i - w_i| / (|x_i| + |w_i|)
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        # Compute numerator: |x_i - w_i|
        numerator = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        
        # Compute denominator: |x_i| + |w_i|
        denominator = torch.abs(x_expanded) + torch.abs(w_expanded)  # (B, V, N)
        
        # Add eps to avoid division by zero
        denominator = denominator + self.eps
        
        # Compute Canberra distance
        canberra_dist = torch.sum(numerator / denominator, dim=-1)  # (B, V)
        
        # Normalize distances if requested
        if scale:
            canberra_dist = canberra_dist / torch.min(canberra_dist, dim=-1, keepdim=True)[0]
        
        # Apply final transformation (add eps to avoid division by zero)
        return torch.pow(canberra_dist + self.eps, -self.n)


class CanberraDistLayerRobust(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, min_denom=1e-3):
        super(CanberraDistLayerRobust, self).__init__(in_features, out_features, bias=bias)
        self.n = n  # Power for final transformation
        self.eps = eps
        self.min_denom = min_denom  # Minimum denominator to prevent instability
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist: (B, V)
        w = self.weight
        
        # Compute robust Canberra distance
        # d(x, w) = Σ |x_i - w_i| / max(|x_i| + |w_i|, min_denom)
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        # Compute numerator: |x_i - w_i|
        numerator = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        
        # Compute denominator: max(|x_i| + |w_i|, min_denom)
        raw_denominator = torch.abs(x_expanded) + torch.abs(w_expanded)  # (B, V, N)
        denominator = torch.clamp(raw_denominator, min=self.min_denom)  # (B, V, N)
        
        # Compute robust Canberra distance
        canberra_dist = torch.sum(numerator / denominator, dim=-1)  # (B, V)
        
        # Normalize distances if requested
        if scale:
            canberra_dist = canberra_dist / torch.min(canberra_dist, dim=-1, keepdim=True)[0]
        
        # Apply final transformation
        return torch.pow(canberra_dist + self.eps, -self.n)


class CanberraDistLayerWeighted(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, 
                 weight_power=1.0, normalize_weights=True):
        super(CanberraDistLayerWeighted, self).__init__(in_features, out_features, bias=bias)
        self.n = n  # Power for final transformation
        self.eps = eps
        self.weight_power = weight_power  # Power for weighting scheme
        self.normalize_weights = normalize_weights  # Whether to normalize weights
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist: (B, V)
        w = self.weight
        
        # Compute weighted Canberra distance
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        # Compute numerator: |x_i - w_i|
        numerator = torch.abs(x_expanded - w_expanded)  # (B, V, N)
        
        # Compute denominator: |x_i| + |w_i|
        denominator = torch.abs(x_expanded) + torch.abs(w_expanded) + self.eps  # (B, V, N)
        
        # Compute feature weights based on magnitude
        feature_weights = torch.pow(denominator, self.weight_power)  # (B, V, N)
        
        # Normalize weights if requested
        if self.normalize_weights:
            feature_weights = feature_weights / torch.sum(feature_weights, dim=-1, keepdim=True)
        
        # Compute weighted Canberra distance
        weighted_terms = feature_weights * (numerator / denominator)  # (B, V, N)
        canberra_dist = torch.sum(weighted_terms, dim=-1)  # (B, V)
        
        # Normalize distances if requested
        if scale:
            canberra_dist = canberra_dist / torch.min(canberra_dist, dim=-1, keepdim=True)[0]
        
        # Apply final transformation
        return torch.pow(canberra_dist + self.eps, -self.n)


# FIXED BRAY-CURTIS DISTANCE LAYER
class BrayCurtisDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-3, bias=False):  # INCREASED eps
        super(BrayCurtisDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)  # Ensure minimum eps
        
    def forward(self, x, scale=False):
        w = self.weight
        
        # Compute Bray-Curtis distance
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        # Compute numerator: Σ |x_i - w_i|
        numerator = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1)  # (B, V)
        
        # FIXED: More robust denominator calculation
        denominator = torch.sum(x_expanded + w_expanded, dim=-1)  # (B, V)
        
        # CRITICAL FIX: Ensure denominator is always positive and significant
        denominator = torch.clamp(torch.abs(denominator) + self.eps, min=self.eps * 10)
        
        # Compute Bray-Curtis distance
        bray_curtis_dist = numerator / denominator  # (B, V)
        
        # FIXED: Better normalization
        if scale:
            min_dist = torch.min(bray_curtis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            bray_curtis_dist = bray_curtis_dist / min_dist
        
        # FIXED: Clamp before power operation
        dist_clamped = torch.clamp(bray_curtis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


# FIXED BRAY-CURTIS ABSOLUTE LAYER
class BrayCurtisDistLayerAbs(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-3, bias=False):
        super(BrayCurtisDistLayerAbs, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        
    def forward(self, x, scale=False):
        w = self.weight
        
        x_expanded = x.unsqueeze(1)  # (B, 1, N)
        w_expanded = w.unsqueeze(0)  # (1, V, N)
        
        # Compute numerator: Σ |x_i - w_i|
        numerator = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1)  # (B, V)
        
        # FIXED: More robust absolute denominator
        denominator = torch.sum(torch.abs(x_expanded) + torch.abs(w_expanded), dim=-1)  # (B, V)
        denominator = torch.clamp(denominator + self.eps, min=self.eps * 10)
        
        # Compute distance
        bray_curtis_dist = numerator / denominator
        
        if scale:
            min_dist = torch.min(bray_curtis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            bray_curtis_dist = bray_curtis_dist / min_dist
        
        dist_clamped = torch.clamp(bray_curtis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


# FIXED BRAY-CURTIS NORMALIZED LAYER  
class BrayCurtisDistLayerNormalized(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-3, bias=False, 
                 normalize_inputs=True, min_sum=1e-3):  # INCREASED min_sum
        super(BrayCurtisDistLayerNormalized, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.normalize_inputs = normalize_inputs
        self.min_sum = max(min_sum, self.eps * 10)  # Ensure min_sum > eps
        
    def forward(self, x, scale=False):
        w = self.weight
        
        # FIXED: More robust normalization
        if self.normalize_inputs:
            # Make inputs non-negative with better stability
            x_pos = torch.abs(x) + self.eps
            w_pos = torch.abs(w) + self.eps
            
            # FIXED: More robust sum calculation
            x_sum = torch.sum(x_pos, dim=-1, keepdim=True)
            w_sum = torch.sum(w_pos, dim=-1, keepdim=True)
            
            # CRITICAL FIX: Better clamping strategy
            x_sum_clamped = torch.clamp(x_sum, min=self.min_sum)
            w_sum_clamped = torch.clamp(w_sum, min=self.min_sum)
            
            x_norm = x_pos / x_sum_clamped
            w_norm = w_pos / w_sum_clamped
        else:
            x_norm = x
            w_norm = w
        
        # Compute distance on normalized inputs
        x_expanded = x_norm.unsqueeze(1)
        w_expanded = w_norm.unsqueeze(0)
        
        numerator = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1)
        denominator = torch.sum(x_expanded + w_expanded, dim=-1)
        
        # FIXED: Robust denominator handling
        denominator = torch.clamp(denominator + self.eps, min=self.eps * 10)
        
        bray_curtis_dist = numerator / denominator
        
        if scale:
            min_dist = torch.min(bray_curtis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            bray_curtis_dist = bray_curtis_dist / min_dist
        
        dist_clamped = torch.clamp(bray_curtis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


# FIXED MAHALANOBIS STANDARD LAYER
class MahalanobisDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, 
                 learn_cov=True, init_identity=True, regularize_cov=True, reg_lambda=1e-2):  # INCREASED reg_lambda
        super(MahalanobisDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.learn_cov = learn_cov
        self.regularize_cov = regularize_cov
        self.reg_lambda = max(reg_lambda, 1e-4)  # Ensure minimum regularization
        
        if learn_cov:
            if init_identity:
                # FIXED: Better initialization
                self.cov_inv = nn.Parameter(torch.eye(in_features) * (1.0 + self.reg_lambda))
            else:
                # FIXED: More stable random initialization
                L = torch.randn(in_features, in_features) * 0.01  # Smaller values
                self.cov_inv = nn.Parameter(L @ L.T + torch.eye(in_features) * self.reg_lambda)
        else:
            self.register_buffer('cov_inv', torch.eye(in_features))
        
    def forward(self, x, scale=False):
        w = self.weight
        
        # FIXED: More robust covariance handling
        if self.learn_cov:
            if self.regularize_cov:
                # FIXED: Stronger regularization
                cov_inv_reg = self.cov_inv + torch.eye(self.cov_inv.size(0), 
                                                     device=self.cov_inv.device) * self.reg_lambda
            else:
                # FIXED: More stable SVD approach
                try:
                    U, S, V = torch.svd(self.cov_inv)
                    S_pos = torch.clamp(S, min=self.eps * 10)  # More conservative clamping
                    cov_inv_reg = U @ torch.diag(S_pos) @ V.T
                except:
                    # Fallback to identity if SVD fails
                    cov_inv_reg = torch.eye(self.cov_inv.size(0), device=self.cov_inv.device) * self.reg_lambda
        else:
            cov_inv_reg = self.cov_inv
        
        # Compute Mahalanobis distance
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        diff = x_expanded - w_expanded
        
        # FIXED: More stable matrix multiplication
        try:
            diff_transformed = torch.einsum('bvn,nm->bvm', diff, cov_inv_reg)
            mahalanobis_sq = torch.sum(diff_transformed * diff, dim=-1)
        except:
            # Fallback to Euclidean distance if einsum fails
            mahalanobis_sq = torch.sum(diff * diff, dim=-1)
        
        # FIXED: More robust square root
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)  # Ensure non-negative
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        
        if scale:
            min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            mahalanobis_dist = mahalanobis_dist / min_dist
        
        dist_clamped = torch.clamp(mahalanobis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


# FIXED MAHALANOBIS CHOLESKY LAYER
class MahalanobisDistLayerCholesky(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(MahalanobisDistLayerCholesky, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        
        # FIXED: Better Cholesky initialization
        self.chol_factor = nn.Parameter(torch.eye(in_features) * 0.1 + torch.randn(in_features, in_features) * 0.01)
        
    def forward(self, x, scale=False):
        w = self.weight
        
        # FIXED: More robust Cholesky construction
        L = torch.tril(self.chol_factor)
        
        diag_indices = torch.arange(L.size(0), device=L.device)
        L[diag_indices, diag_indices] = torch.clamp(L[diag_indices, diag_indices], min=self.eps)
        
        cov_inv = L @ L.T + torch.eye(L.size(0), device=L.device) * self.eps * 10
        
        # Compute distance with fallback
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        diff = x_expanded - w_expanded
        
        try:
            diff_transformed = torch.einsum('bvn,nm->bvm', diff, cov_inv)
            mahalanobis_sq = torch.sum(diff_transformed * diff, dim=-1)
        except:
            # Fallback to Euclidean
            mahalanobis_sq = torch.sum(diff * diff, dim=-1)
        
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        
        if scale:
            min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            mahalanobis_dist = mahalanobis_dist / min_dist
        
        dist_clamped = torch.clamp(mahalanobis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


# FIXED MAHALANOBIS DIAGONAL LAYER
class MahalanobisDistLayerDiagonal(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(MahalanobisDistLayerDiagonal, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        
        # FIXED: Better initialization for diagonal elements
        self.diag_cov_inv = nn.Parameter(torch.zeros(in_features))  # Will be exponentiated
        
    def forward(self, x, scale=False):
        w = self.weight
        
        # FIXED: More robust positive diagonal elements
        diag_pos = torch.exp(torch.clamp(self.diag_cov_inv, min=-10, max=10)) + self.eps
        
        # Compute diagonal Mahalanobis distance
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        diff = x_expanded - w_expanded
        diff_sq = diff * diff
        
        # Apply diagonal covariance inverse
        weighted_diff_sq = diff_sq * diag_pos.unsqueeze(0).unsqueeze(0)
        mahalanobis_sq = torch.sum(weighted_diff_sq, dim=-1)
        
        # FIXED: Ensure non-negative before sqrt
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        
        if scale:
            min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            mahalanobis_dist = mahalanobis_dist / min_dist
        
        dist_clamped = torch.clamp(mahalanobis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)