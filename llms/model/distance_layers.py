import torch
import torch.nn as nn
from torch.nn import functional as F
from .model_setup import *

#### Distance Layer Implementations ####

class EuclideanDistLayer(torch.nn.Linear):
    """Euclidean distance-based layer (harmonic similarity)"""
    def __init__(self, in_features, out_features, bias=False):
        super(EuclideanDistLayer, self).__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        # x: (B, T, N)
        # w: (V, N)
        # dist_sq: (B, T, V)
        n_embd = x.size(-1)
        w = self.weight
        
        wx = torch.einsum('btn,vn->btv', x, w) # (B, T, V)
        wx = fillnan(wx, nan_value=0.)
        
        ww = torch.norm(w, dim=-1)**2 # (V,)
        ww = fillnan(ww, nan_value=1.)
        
        xx = torch.norm(x, dim=-1)**2 # (B,T)
        xx = fillnan(xx, nan_value=1.)
        
        dist_sq = ww[...,:] + xx[:,:,None] - 2 * wx
        dist_sq = fillnan(dist_sq, nan_value=768.)

        dist_sq = dist_sq / torch.tensor(n_embd)
        dist_sq = fillnan(dist_sq, nan_value=1.)
        
        dist_sq = dist_sq / torch.min(dist_sq, dim=-1, keepdim=True)[0]
        dist_sq = fillnan(dist_sq, nan_value=1.)
        
        # pow_n is an important hyperparameter in harmonic loss
        # suggest trying pow_n = 1, sqrt(n_embd), n_embd
        pow_n = torch.tensor(n_embd)
        dist_sq = dist_sq ** (-pow_n)
        dist_sq = fillnan(dist_sq, nan_value=1.)
        
        return dist_sq

class ManhattanDistLayerLong(torch.nn.Linear):
    """Manhattan distance-based layer - ORIGINAL (SLOW but most accurate)"""
    def __init__(self, in_features, out_features, bias=False):
        super(ManhattanDistLayerLong, self).__init__(in_features, out_features, bias=bias)
    
    def forward(self, x):
        n_embd = x.size(-1)
        w = self.weight
        
        # Compute Manhattan distance: sum of absolute differences
        x_expanded = x.unsqueeze(-2)  # (B, T, 1, N)
        w_expanded = w.unsqueeze(0).unsqueeze(0)  # (1, 1, V, N)
        
        manhattan_dist = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1)  # (B, T, V)
        manhattan_dist = fillnan(manhattan_dist, nan_value=float(n_embd))
        
        # Normalize by embedding dimension
        manhattan_dist = manhattan_dist / torch.tensor(n_embd, dtype=torch.float32)
        manhattan_dist = fillnan(manhattan_dist, nan_value=1.)
        
        # Normalize by minimum distance
        manhattan_dist = manhattan_dist / torch.min(manhattan_dist, dim=-1, keepdim=True)[0]
        manhattan_dist = fillnan(manhattan_dist, nan_value=1.)
        
        # Convert distance to similarity
        pow_n = torch.tensor(n_embd, dtype=torch.float32)
        similarity = manhattan_dist ** (-pow_n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity

class ManhattanDistLayerIntermediate(torch.nn.Linear):
    """Manhattan distance-based layer - CHUNKED (Balanced speed/accuracy)"""
    def __init__(self, in_features, out_features, bias=False):
        super(ManhattanDistLayerIntermediate, self).__init__(in_features, out_features, bias=bias)
    
    def forward(self, x):
        # Use efficient chunked computation to avoid memory explosion
        B, T, N = x.shape
        V = self.weight.shape[0]
        
        # Flatten x for efficient computation
        x_flat = x.view(-1, N)  # (B*T, N)
        
        # Chunked computation to avoid memory explosion
        chunk_size = 1024  # Adjust based on memory - smaller = less memory, more chunks
        distances = []
        
        for i in range(0, V, chunk_size):
            end_idx = min(i + chunk_size, V)
            w_chunk = self.weight[i:end_idx]  # (chunk_size, N)
            
            # Compute Manhattan distance for this chunk
            x_expanded = x_flat.unsqueeze(1)  # (B*T, 1, N)
            w_expanded = w_chunk.unsqueeze(0)  # (1, chunk_size, N)
            
            chunk_dist = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1)  # (B*T, chunk_size)
            distances.append(chunk_dist)
        
        manhattan_dist = torch.cat(distances, dim=1)  # (B*T, V)
        manhattan_dist = manhattan_dist.view(B, T, V)  # (B, T, V)
        manhattan_dist = fillnan(manhattan_dist, nan_value=float(N))
        
        # Normalize by embedding dimension
        manhattan_dist = manhattan_dist / N
        manhattan_dist = fillnan(manhattan_dist, nan_value=1.)
        
        # Normalize by minimum distance
        min_dist = torch.min(manhattan_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=1e-8)
        manhattan_dist = manhattan_dist / min_dist
        manhattan_dist = fillnan(manhattan_dist, nan_value=1.)
        
        # Convert to similarity (avoid expensive exponentiation)
        # Use simple inverse instead of power
        similarity = 1.0 / (1.0 + manhattan_dist)
        similarity = fillnan(similarity, nan_value=0.0)
        
        return similarity

class ManhattanDistLayerFast(torch.nn.Linear):
    """Manhattan distance-based layer - FAST (Manhattan-inspired approximation)"""
    def __init__(self, in_features, out_features, bias=False):
        super(ManhattanDistLayerFast, self).__init__(in_features, out_features, bias=bias)
    
    def forward(self, x):
        # Much simpler and faster Manhattan-inspired computation
        B, T, N = x.shape
        
        # Compute L1 norm-based similarity using matrix operations
        x_norm = torch.norm(x, p=1, dim=-1, keepdim=True)  # (B, T, 1)
        w_norm = torch.norm(self.weight, p=1, dim=-1)  # (V,)
        
        # Simple dot product scaled by norms (Manhattan-inspired)
        logits = torch.matmul(x, self.weight.t())  # (B, T, V)
        
        # Scale by L1 norms to approximate Manhattan distance behavior
        scale = x_norm * w_norm.unsqueeze(0).unsqueeze(0)  # (B, T, V)
        similarity = logits / (scale + 1e-8)
        similarity = fillnan(similarity, nan_value=0.0)
        
        # Apply sigmoid to ensure positive values
        similarity = torch.sigmoid(similarity)
        similarity = fillnan(similarity, nan_value=0.5)
        
        return similarity

class OptimizedMinkowskiDistLayer(torch.nn.Linear):
    """
    Optimized Minkowski distance layer using efficient computation patterns
    
    For L1 and L2, uses algebraic shortcuts. For other norms, uses optimized
    broadcasting with minimal memory overhead.
    """
    
    def __init__(self, in_features, out_features, temperature=2.0, bias=False, 
                 chunk_size=2048, use_fast_l2=True):
        super(OptimizedMinkowskiDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.use_fast_l2 = use_fast_l2
        
        # Precompute properties for efficiency
        self.is_l1 = abs(temperature - 1.0) < 1e-6
        self.is_l2 = abs(temperature - 2.0) < 1e-6
        self.is_integer = abs(temperature - round(temperature)) < 1e-6
        self.int_temp = round(temperature) if self.is_integer else None
        
        print(f"OptimizedMinkowskiDistLayer: L{temperature} norm, fast_l2={use_fast_l2}")
    
    def forward(self, x):
        # x: (B, T, N), w: (V, N) -> output: (B, T, V)
        B, T, N = x.shape
        V = self.weight.shape[0]
        w = self.weight
        
        # Route to most efficient implementation based on norm type
        if self.is_l2 and self.use_fast_l2:
            return self._forward_l2_fast(x, w, N)
        elif self.is_l1:
            return self._forward_l1_optimized(x, w, B, T, N, V)
        elif V > self.chunk_size:
            return self._forward_chunked_optimized(x, w, B, T, N, V)
        else:
            return self._forward_direct_optimized(x, w, B, T, N, V)
    
    def _forward_l2_fast(self, x, w, n_embd):
        """Ultra-fast L2 using algebraic identity (same as EuclideanDistLayer)"""
        # ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩
        wx = torch.einsum('btn,vn->btv', x, w)
        wx = fillnan(wx, nan_value=0.)
        
        ww = torch.sum(w**2, dim=-1)  # (V,)
        ww = fillnan(ww, nan_value=1.)
        
        xx = torch.sum(x**2, dim=-1)  # (B,T)
        xx = fillnan(xx, nan_value=1.)
        
        dist_sq = ww + xx.unsqueeze(-1) - 2 * wx
        dist_sq = torch.clamp(dist_sq, min=0)
        dist_sq = fillnan(dist_sq, nan_value=float(n_embd))
        
        # Convert to distance, normalize, and return similarity
        minkowski_dist = torch.sqrt(dist_sq + 1e-8)
        return self._distance_to_similarity(minkowski_dist, n_embd)
    
    def _forward_l1_optimized(self, x, w, B, T, N, V):
        """Optimized L1 computation using efficient broadcasting"""
        if V > self.chunk_size:
            return self._forward_l1_chunked(x, w, B, T, N, V)
        
        # For smaller vocabularies, use direct computation but with better memory management
        # Use torch.cdist with p=1 for L1 distance
        x_flat = x.view(-1, N)  # (B*T, N)
        
        # Compute L1 distances efficiently
        distances = torch.cdist(x_flat, w, p=1)  # (B*T, V)
        minkowski_dist = distances.view(B, T, V)
        minkowski_dist = fillnan(minkowski_dist, nan_value=float(N))
        
        return self._distance_to_similarity(minkowski_dist, N)
    
    def _forward_l1_chunked(self, x, w, B, T, N, V):
        """Chunked L1 computation for large vocabularies"""
        x_flat = x.view(-1, N)  # (B*T, N)
        distances = []
        
        for i in range(0, V, self.chunk_size):
            end_idx = min(i + self.chunk_size, V)
            w_chunk = w[i:end_idx]
            
            # Use cdist for this chunk
            chunk_dist = torch.cdist(x_flat, w_chunk, p=1)  # (B*T, chunk_size)
            distances.append(chunk_dist)
        
        minkowski_dist = torch.cat(distances, dim=1)  # (B*T, V)
        minkowski_dist = minkowski_dist.view(B, T, V)
        minkowski_dist = fillnan(minkowski_dist, nan_value=float(N))
        
        return self._distance_to_similarity(minkowski_dist, N)
    
    def _forward_direct_optimized(self, x, w, B, T, N, V):
        """Optimized direct computation for general Lp norms"""
        # Use more memory-efficient broadcasting
        x_flat = x.view(-1, N)  # (B*T, N)
        
        if self.is_integer and self.int_temp > 0 and self.int_temp <= 10:
            # Use cdist for integer norms up to 10
            try:
                distances = torch.cdist(x_flat, w, p=self.int_temp)  # (B*T, V)
                minkowski_dist = distances.view(B, T, V)
            except:
                # Fallback to manual computation
                minkowski_dist = self._compute_general_norm(x, w, B, T, N, V)
        else:
            # Manual computation for fractional or very high norms
            minkowski_dist = self._compute_general_norm(x, w, B, T, N, V)
        
        minkowski_dist = fillnan(minkowski_dist, nan_value=float(N))
        return self._distance_to_similarity(minkowski_dist, N)
    
    def _forward_chunked_optimized(self, x, w, B, T, N, V):
        """Optimized chunked computation for large vocabularies"""
        x_flat = x.view(-1, N)  # (B*T, N)
        distances = []
        
        for i in range(0, V, self.chunk_size):
            end_idx = min(i + self.chunk_size, V)
            w_chunk = w[i:end_idx]
            
            if self.is_integer and self.int_temp > 0 and self.int_temp <= 10:
                try:
                    chunk_dist = torch.cdist(x_flat, w_chunk, p=self.int_temp)
                except:
                    chunk_dist = self._compute_chunk_norm(x_flat, w_chunk)
            else:
                chunk_dist = self._compute_chunk_norm(x_flat, w_chunk)
            
            distances.append(chunk_dist)
        
        minkowski_dist = torch.cat(distances, dim=1)  # (B*T, V)
        minkowski_dist = minkowski_dist.view(B, T, V)
        minkowski_dist = fillnan(minkowski_dist, nan_value=float(N))
        
        return self._distance_to_similarity(minkowski_dist, N)
    
    def _compute_general_norm(self, x, w, B, T, N, V):
        """Compute general Lp norm with minimal memory usage"""
        # Use sequential computation to minimize peak memory
        x_flat = x.view(-1, N)  # (B*T, N)
        distances = torch.zeros(B*T, V, device=x.device, dtype=x.dtype)
        
        # Process in smaller batches to control memory
        batch_size = min(1024, B*T)
        
        for i in range(0, B*T, batch_size):
            end_i = min(i + batch_size, B*T)
            x_batch = x_flat[i:end_i]  # (batch_size, N)
            
            # Compute distances for this batch
            x_expanded = x_batch.unsqueeze(1)  # (batch_size, 1, N)
            w_expanded = w.unsqueeze(0)  # (1, V, N)
            abs_diff = torch.abs(x_expanded - w_expanded)  # (batch_size, V, N)
            
            if self.is_integer and self.int_temp > 0:
                batch_dist = torch.sum(abs_diff**self.int_temp, dim=-1)**(1.0/self.int_temp)
            else:
                powered = torch.pow(abs_diff + 1e-8, self.temperature)
                batch_dist = torch.pow(torch.sum(powered, dim=-1) + 1e-8, 1.0/self.temperature)
            
            distances[i:end_i] = batch_dist
        
        return distances.view(B, T, V)
    
    def _compute_chunk_norm(self, x_flat, w_chunk):
        """Compute norm for a chunk of weights"""
        x_expanded = x_flat.unsqueeze(1)  # (B*T, 1, N)
        w_expanded = w_chunk.unsqueeze(0)  # (1, chunk_size, N)
        abs_diff = torch.abs(x_expanded - w_expanded)  # (B*T, chunk_size, N)
        
        if self.is_integer and self.int_temp > 0:
            return torch.sum(abs_diff**self.int_temp, dim=-1)**(1.0/self.int_temp)
        else:
            powered = torch.pow(abs_diff + 1e-8, self.temperature)
            return torch.pow(torch.sum(powered, dim=-1) + 1e-8, 1.0/self.temperature)
    
    def _distance_to_similarity(self, minkowski_dist, n_embd):
        """Convert distance to similarity (same as original)"""
        # Normalize by embedding dimension
        minkowski_dist = minkowski_dist / n_embd
        minkowski_dist = fillnan(minkowski_dist, nan_value=1.)
        
        # Normalize by minimum distance for relative scaling
        min_dist = torch.min(minkowski_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=1e-8)
        minkowski_dist = minkowski_dist / min_dist
        minkowski_dist = fillnan(minkowski_dist, nan_value=1.)
        
        # Convert distance to similarity using power law
        pow_n = torch.tensor(n_embd, dtype=torch.float32)
        similarity = minkowski_dist ** (-pow_n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


class UltraFastMinkowskiL1(torch.nn.Linear):
    """
    Ultra-fast L1 (Manhattan) distance using coordinate-wise optimization
    """
    def __init__(self, in_features, out_features, bias=False):
        super(UltraFastMinkowskiL1, self).__init__(in_features, out_features, bias=bias)
    
    def forward(self, x):
        # Use torch.cdist with p=1 for maximum efficiency
        n_embd = x.size(-1)
        B, T, N = x.shape
        x_flat = x.view(-1, N)  # (B*T, N)
        
        distances = torch.cdist(x_flat, self.weight, p=1)  # (B*T, V)
        minkowski_dist = distances.view(B, T, -1)
        minkowski_dist = fillnan(minkowski_dist, nan_value=float(n_embd))
        
        return self._distance_to_similarity(minkowski_dist, n_embd)
    
    def _distance_to_similarity(self, minkowski_dist, n_embd):
        """Convert distance to similarity"""
        minkowski_dist = minkowski_dist / n_embd
        minkowski_dist = fillnan(minkowski_dist, nan_value=1.)
        
        min_dist = torch.min(minkowski_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=1e-8)
        minkowski_dist = minkowski_dist / min_dist
        minkowski_dist = fillnan(minkowski_dist, nan_value=1.)
        
        pow_n = torch.tensor(n_embd, dtype=torch.float32)
        similarity = minkowski_dist ** (-pow_n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


# Factory function for optimized Minkowski layers
def create_optimized_minkowski_layer(in_features, out_features, temperature, bias=False, **kwargs):
    """
    Factory function to create optimized Minkowski layers
    
    Automatically chooses the most efficient implementation based on temperature
    """
    if abs(temperature - 1.0) < 1e-6:
        # Use ultra-fast L1 implementation
        layer = UltraFastMinkowskiL1(in_features, out_features, bias=bias)
        layer.description = "Ultra-fast Manhattan (L1)"
        layer.temperature = 1.0
    else:
        # Use general optimized implementation
        layer = OptimizedMinkowskiDistLayer(in_features, out_features, temperature, bias=bias, **kwargs)
        if abs(temperature - 2.0) < 1e-6:
            layer.description = "Optimized Euclidean (L2)"
        elif temperature < 1.0:
            layer.description = f"Optimized Sub-linear (L{temperature})"
        else:
            layer.description = f"Optimized L{temperature} norm"
    
    return layer

class MahalanobisDistLayerStandard(torch.nn.Linear):
    """Optimized Standard Mahalanobis distance layer using algebraic identity"""
    
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False,
                 learn_cov=True, init_identity=True, regularize_cov=True, reg_lambda=1e-2):
        super(MahalanobisDistLayerStandard, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.learn_cov = learn_cov
        self.regularize_cov = regularize_cov
        self.reg_lambda = max(reg_lambda, 1e-4)
        
        if learn_cov:
            if init_identity:
                self.cov_inv = nn.Parameter(torch.eye(in_features) * (1.0 + self.reg_lambda))
            else:
                L = torch.randn(in_features, in_features) * 0.01
                self.cov_inv = nn.Parameter(L @ L.T + torch.eye(in_features) * self.reg_lambda)
        else:
            self.register_buffer('cov_inv', torch.eye(in_features))
    
    def forward(self, x):
        # x: (B, T, N), w: (V, N) -> output: (B, T, V)
        n_embd = x.size(-1)
        w = self.weight
        
        # Get regularized covariance inverse
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
        
        # Transform inputs and weights by covariance
        # Use Cholesky decomposition for numerical stability
        try:
            L = torch.linalg.cholesky(cov_inv_reg + torch.eye(cov_inv_reg.size(0), device=cov_inv_reg.device) * self.eps)
            x_transformed = torch.einsum('btn,nm->btm', x, L)  # (B, T, N)
            w_transformed = torch.einsum('vn,nm->vm', w, L)    # (V, N)
        except:
            # Fallback to direct multiplication if Cholesky fails
            x_transformed = torch.einsum('btn,nm->btm', x, cov_inv_reg)
            w_transformed = torch.einsum('vn,nm->vm', w, cov_inv_reg)
        
        # Now use Euclidean distance formula on transformed space
        # ||x_t - w_t||² = ||x_t||² + ||w_t||² - 2⟨x_t, w_t⟩
        xw = torch.einsum('btn,vn->btv', x_transformed, w_transformed)  # (B, T, V)
        xw = fillnan(xw, nan_value=0.)
        
        ww = torch.sum(w_transformed**2, dim=-1)  # (V,)
        ww = fillnan(ww, nan_value=1.)
        
        xx = torch.sum(x_transformed**2, dim=-1)  # (B, T)
        xx = fillnan(xx, nan_value=1.)
        
        # Mahalanobis squared distance
        mahalanobis_sq = ww + xx.unsqueeze(-1) - 2 * xw
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)
        mahalanobis_sq = fillnan(mahalanobis_sq, nan_value=float(n_embd))
        
        # Convert to distance and then similarity
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        mahalanobis_dist = fillnan(mahalanobis_dist, nan_value=1.)
        
        # Normalize by minimum distance for relative scaling
        min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=self.eps)
        mahalanobis_dist = mahalanobis_dist / min_dist
        mahalanobis_dist = fillnan(mahalanobis_dist, nan_value=1.)
        
        # Convert to similarity using power law
        similarity = torch.pow(mahalanobis_dist + self.eps, -self.n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


class MahalanobisDistLayerDiagonal(torch.nn.Linear):
    """Optimized Diagonal Mahalanobis distance layer"""
    
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(MahalanobisDistLayerDiagonal, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        
        # Diagonal covariance inverse (learnable precision parameters)
        self.log_diag_cov_inv = nn.Parameter(torch.zeros(in_features))
    
    def forward(self, x):
        # x: (B, T, N), w: (V, N) -> output: (B, T, V)
        n_embd = x.size(-1)
        w = self.weight
        
        # Get diagonal precision matrix (ensure positive)
        diag_cov_inv = torch.exp(torch.clamp(self.log_diag_cov_inv, min=-10, max=10)) + self.eps
        
        # Scale inputs and weights by sqrt of diagonal precision
        sqrt_diag = torch.sqrt(diag_cov_inv)
        x_scaled = x * sqrt_diag.unsqueeze(0).unsqueeze(0)  # (B, T, N)
        w_scaled = w * sqrt_diag.unsqueeze(0)               # (V, N)
        
        # Use Euclidean distance formula on scaled space
        xw = torch.einsum('btn,vn->btv', x_scaled, w_scaled)  # (B, T, V)
        xw = fillnan(xw, nan_value=0.)
        
        ww = torch.sum(w_scaled**2, dim=-1)  # (V,)
        ww = fillnan(ww, nan_value=1.)
        
        xx = torch.sum(x_scaled**2, dim=-1)  # (B, T)
        xx = fillnan(xx, nan_value=1.)
        
        # Mahalanobis squared distance
        mahalanobis_sq = ww + xx.unsqueeze(-1) - 2 * xw
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)
        mahalanobis_sq = fillnan(mahalanobis_sq, nan_value=float(n_embd))
        
        # Convert to distance and then similarity
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        mahalanobis_dist = fillnan(mahalanobis_dist, nan_value=1.)
        
        # Normalize by minimum distance
        min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=self.eps)
        mahalanobis_dist = mahalanobis_dist / min_dist
        mahalanobis_dist = fillnan(mahalanobis_dist, nan_value=1.)
        
        # Convert to similarity
        similarity = torch.pow(mahalanobis_dist + self.eps, -self.n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


class MahalanobisDistLayerCholesky(torch.nn.Linear):
    """Optimized Cholesky Mahalanobis distance layer"""
    
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(MahalanobisDistLayerCholesky, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        
        # Cholesky factor (lower triangular)
        self.chol_factor = nn.Parameter(
            torch.eye(in_features) * 0.1 + torch.randn(in_features, in_features) * 0.01
        )
    
    def forward(self, x):
        # x: (B, T, N), w: (V, N) -> output: (B, T, V)
        n_embd = x.size(-1)
        w = self.weight
        
        # Get lower triangular Cholesky factor
        L = torch.tril(self.chol_factor)
        # Ensure positive diagonal elements
        diag_indices = torch.arange(L.size(0), device=L.device)
        L = L.clone()
        L[diag_indices, diag_indices] = torch.clamp(L[diag_indices, diag_indices], min=self.eps)
        
        # Transform inputs and weights using Cholesky factor
        # Since we want x^T Σ^{-1} x and Σ^{-1} = L L^T, we multiply by L^T
        try:
            x_transformed = torch.einsum('btn,mn->btm', x, L.T)  # (B, T, N)
            w_transformed = torch.einsum('vn,mn->vm', w, L.T)    # (V, N)
        except:
            # Fallback to identity transformation
            x_transformed = x
            w_transformed = w
        
        # Use Euclidean distance formula on transformed space
        xw = torch.einsum('btn,vn->btv', x_transformed, w_transformed)  # (B, T, V)
        xw = fillnan(xw, nan_value=0.)
        
        ww = torch.sum(w_transformed**2, dim=-1)  # (V,)
        ww = fillnan(ww, nan_value=1.)
        
        xx = torch.sum(x_transformed**2, dim=-1)  # (B, T)
        xx = fillnan(xx, nan_value=1.)
        
        # Mahalanobis squared distance
        mahalanobis_sq = ww + xx.unsqueeze(-1) - 2 * xw
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)
        mahalanobis_sq = fillnan(mahalanobis_sq, nan_value=float(n_embd))
        
        # Convert to distance and then similarity
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        mahalanobis_dist = fillnan(mahalanobis_dist, nan_value=1.)
        
        # Normalize by minimum distance
        min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=self.eps)
        mahalanobis_dist = mahalanobis_dist / min_dist
        mahalanobis_dist = fillnan(mahalanobis_dist, nan_value=1.)
        
        # Convert to similarity
        similarity = torch.pow(mahalanobis_dist + self.eps, -self.n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


# Factory function for easy creation
def create_mahalanobis_layer(in_features, out_features, variant="diagonal", **kwargs):
    """
    Factory function to create Mahalanobis layers
    
    Args:
        variant: "standard", "diagonal", or "cholesky"
        **kwargs: Additional parameters for the specific variant
    """
    if variant == "standard":
        return MahalanobisDistLayerStandard(in_features, out_features, **kwargs)
    elif variant == "diagonal":
        return MahalanobisDistLayerDiagonal(in_features, out_features, **kwargs)
    elif variant == "cholesky":
        return MahalanobisDistLayerCholesky(in_features, out_features, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from 'standard', 'diagonal', 'cholesky'")

class CosineDistLayer(torch.nn.Linear):
    """Cosine distance-based layer"""
    def __init__(self, in_features, out_features, bias=False):
        super(CosineDistLayer, self).__init__(in_features, out_features, bias=bias)
    
    def forward(self, x):
        n_embd = x.size(-1)
        w = self.weight
        
        # Normalize vectors for cosine similarity
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)  # (B, T, N)
        w_norm = torch.nn.functional.normalize(w, p=2, dim=-1)  # (V, N)
        
        x_norm = fillnan(x_norm, nan_value=0.)
        w_norm = fillnan(w_norm, nan_value=0.)
        
        # Compute cosine similarity
        cosine_sim = torch.einsum('btn,vn->btv', x_norm, w_norm)  # (B, T, V)
        cosine_sim = fillnan(cosine_sim, nan_value=0.)
        
        # Convert to distance and then similarity
        cosine_dist = 1.0 - cosine_sim
        cosine_dist = fillnan(cosine_dist, nan_value=1.)
        
        # Normalize by minimum distance
        min_dist = torch.min(cosine_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=1e-8)
        cosine_dist = cosine_dist / min_dist
        cosine_dist = fillnan(cosine_dist, nan_value=1.)
        
        # Convert to similarity
        pow_n = torch.tensor(n_embd, dtype=torch.float32)
        similarity = cosine_dist ** (-pow_n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity

class CosineSimpleDistLayer(torch.nn.Linear):
    """Cosine distance-based layer"""
    def __init__(self, in_features, out_features, bias=False):
        super(CosineSimpleDistLayer, self).__init__(in_features, out_features, bias=bias)
    def forward(self, x):
        # Much simpler cosine similarity
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)  # (B, T, N)
        w_norm = torch.nn.functional.normalize(self.weight, p=2, dim=-1)  # (V, N)
        
        # Direct cosine similarity
        cosine_sim = torch.einsum('btn,vn->btv', x_norm, w_norm)  # (B, T, V)
        cosine_sim = fillnan(cosine_sim, nan_value=0.)
        
        # Convert to positive similarity (shift and scale)
        similarity = (cosine_sim + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        similarity = fillnan(similarity, nan_value=0.5)
        
        return similarity

class CosineTempScaleDistLayer(torch.nn.Linear):
    """Cosine distance-based layer with temperature scaling"""
    def __init__(self, in_features, out_features, temperature=1.0, bias=False):
        super(CosineTempScaleDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.temperature = temperature
    
    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        w_norm = torch.nn.functional.normalize(self.weight, p=2, dim=-1)
        
        cosine_sim = torch.einsum('btn,vn->btv', x_norm, w_norm)
        cosine_sim = fillnan(cosine_sim, nan_value=0.)
        
        # Apply temperature scaling
        scaled_sim = cosine_sim / self.temperature
        
        return torch.softmax(scaled_sim, dim=-1)

# Add these new distance layers to your existing distance_layers.py file

class HammingDistLayer(torch.nn.Linear):
    """Hamming distance-based layer with multiple variants"""
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, 
                 threshold=0.5, temperature=1.0, variant="soft"):
        super(HammingDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.threshold = threshold
        self.temperature = temperature
        self.variant = variant
        
    def forward(self, x):
        # x: (B, T, N), w: (V, N) -> output: (B, T, V)
        n_embd = x.size(-1)
        w = self.weight
        
        if self.variant == "soft":
            # Soft binarization using sigmoid
            x_soft = torch.sigmoid(x / self.temperature)
            w_soft = torch.sigmoid(w / self.temperature)
            x_expanded = x_soft.unsqueeze(-2)  # (B, T, 1, N)
            w_expanded = w_soft.unsqueeze(0).unsqueeze(0)  # (1, 1, V, N)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)  # (B, T, V)
            
        elif self.variant == "gumbel":
            # Gumbel-softmax for differentiable sampling
            def gumbel_sigmoid(logits, temperature):
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + self.eps) + self.eps)
                return torch.sigmoid((logits + gumbel_noise) / temperature)
            
            if self.training:
                x_binary = gumbel_sigmoid(x, self.temperature)
                w_binary = gumbel_sigmoid(w, self.temperature)
            else:
                x_binary = torch.sigmoid(x / self.temperature)
                w_binary = torch.sigmoid(w / self.temperature)
            
            x_expanded = x_binary.unsqueeze(-2)  # (B, T, 1, N)
            w_expanded = w_binary.unsqueeze(0).unsqueeze(0)  # (1, 1, V, N)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)  # (B, T, V)
            
        else:  # "hard" variant
            # Hard thresholding with small temperature for gradient flow
            temp = 0.1
            x_binary = torch.sigmoid((x - self.threshold) / temp)
            w_binary = torch.sigmoid((w - self.threshold) / temp)
            x_expanded = x_binary.unsqueeze(-2)  # (B, T, 1, N)
            w_expanded = w_binary.unsqueeze(0).unsqueeze(0)  # (1, 1, V, N)
            diff = torch.abs(x_expanded - w_expanded)
            hamming_dist = torch.sum(diff, dim=-1)  # (B, T, V)
        
        hamming_dist = fillnan(hamming_dist, nan_value=float(n_embd))
        
        # Normalize by embedding dimension
        hamming_dist = hamming_dist / n_embd
        hamming_dist = fillnan(hamming_dist, nan_value=1.)
        
        # Normalize by minimum distance
        min_dist = torch.min(hamming_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=self.eps)
        hamming_dist = hamming_dist / min_dist
        hamming_dist = fillnan(hamming_dist, nan_value=1.)
        
        # Convert to similarity
        pow_n = torch.tensor(n_embd, dtype=torch.float32)
        similarity = torch.pow(hamming_dist + self.eps, -pow_n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


class ChebyshevDistLayer(torch.nn.Linear):
    """Chebyshev distance-based layer (L∞ norm)"""
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, 
                 smooth=False, alpha=10.0):
        super(ChebyshevDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.smooth = smooth
        self.alpha = alpha
        
    def forward(self, x):
        # x: (B, T, N), w: (V, N) -> output: (B, T, V)
        n_embd = x.size(-1)
        w = self.weight
        
        x_expanded = x.unsqueeze(-2)  # (B, T, 1, N)
        w_expanded = w.unsqueeze(0).unsqueeze(0)  # (1, 1, V, N)
        diff = torch.abs(x_expanded - w_expanded)  # (B, T, V, N)
        
        if self.smooth:
            # Smooth approximation using logsumexp
            chebyshev_dist = torch.logsumexp(self.alpha * diff, dim=-1) / self.alpha
        else:
            # Standard max operation
            chebyshev_dist = torch.max(diff, dim=-1)[0]  # (B, T, V)
        
        chebyshev_dist = fillnan(chebyshev_dist, nan_value=float(n_embd))
        
        # Normalize by embedding dimension
        chebyshev_dist = chebyshev_dist / n_embd
        chebyshev_dist = fillnan(chebyshev_dist, nan_value=1.)
        
        # Normalize by minimum distance
        min_dist = torch.min(chebyshev_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=self.eps)
        chebyshev_dist = chebyshev_dist / min_dist
        chebyshev_dist = fillnan(chebyshev_dist, nan_value=1.)
        
        # Convert to similarity
        pow_n = torch.tensor(n_embd, dtype=torch.float32)
        similarity = torch.pow(chebyshev_dist + self.eps, -pow_n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


class CanberraDistLayer(torch.nn.Linear):
    """Canberra distance-based layer with multiple variants"""
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False,
                 variant="standard", min_denom=1e-3, weight_power=1.0, normalize_weights=True):
        super(CanberraDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.variant = variant
        self.min_denom = max(min_denom, self.eps)
        self.weight_power = weight_power
        self.normalize_weights = normalize_weights
        
    def forward(self, x):
        # x: (B, T, N), w: (V, N) -> output: (B, T, V)
        n_embd = x.size(-1)
        w = self.weight
        
        x_expanded = x.unsqueeze(-2)  # (B, T, 1, N)
        w_expanded = w.unsqueeze(0).unsqueeze(0)  # (1, 1, V, N)
        numerator = torch.abs(x_expanded - w_expanded)  # (B, T, V, N)
        
        if self.variant == "robust":
            # Robust variant with minimum denominator
            raw_denominator = torch.abs(x_expanded) + torch.abs(w_expanded)
            denominator = torch.clamp(raw_denominator, min=self.min_denom)
            canberra_dist = torch.sum(numerator / denominator, dim=-1)  # (B, T, V)
            
        elif self.variant == "weighted":
            # Weighted variant with feature importance
            denominator = torch.abs(x_expanded) + torch.abs(w_expanded) + self.eps
            feature_weights = torch.pow(denominator, self.weight_power)
            
            if self.normalize_weights:
                feature_weights = feature_weights / torch.sum(feature_weights, dim=-1, keepdim=True)
            
            weighted_terms = feature_weights * (numerator / denominator)
            canberra_dist = torch.sum(weighted_terms, dim=-1)  # (B, T, V)
            
        else:  # "standard"
            # Standard Canberra distance
            denominator = torch.abs(x_expanded) + torch.abs(w_expanded) + self.eps
            canberra_dist = torch.sum(numerator / denominator, dim=-1)  # (B, T, V)
        
        canberra_dist = fillnan(canberra_dist, nan_value=float(n_embd))
        
        # Normalize by embedding dimension
        canberra_dist = canberra_dist / n_embd
        canberra_dist = fillnan(canberra_dist, nan_value=1.)
        
        # Normalize by minimum distance
        min_dist = torch.min(canberra_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=self.eps)
        canberra_dist = canberra_dist / min_dist
        canberra_dist = fillnan(canberra_dist, nan_value=1.)
        
        # Convert to similarity
        pow_n = torch.tensor(n_embd, dtype=torch.float32)
        similarity = torch.pow(canberra_dist + self.eps, -pow_n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


class BrayCurtisDistLayer(torch.nn.Linear):
    """Bray-Curtis distance-based layer with multiple variants"""
    def __init__(self, in_features, out_features, n=1., eps=1e-3, bias=False,
                 variant="standard", normalize_inputs=True, min_sum=1e-3):
        super(BrayCurtisDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.variant = variant
        self.normalize_inputs = normalize_inputs
        self.min_sum = max(min_sum, self.eps * 10)
        
    def forward(self, x):
        # x: (B, T, N), w: (V, N) -> output: (B, T, V)
        n_embd = x.size(-1)
        w = self.weight
        
        if self.variant == "normalized" and self.normalize_inputs:
            # Normalize inputs to positive values and unit sum
            x_pos = torch.abs(x) + self.eps
            w_pos = torch.abs(w) + self.eps
            x_sum = torch.clamp(torch.sum(x_pos, dim=-1, keepdim=True), min=self.min_sum)
            w_sum = torch.clamp(torch.sum(w_pos, dim=-1, keepdim=True), min=self.min_sum)
            x_norm = x_pos / x_sum
            w_norm = w_pos / w_sum
            x_expanded = x_norm.unsqueeze(-2)  # (B, T, 1, N)
            w_expanded = w_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, V, N)
        else:
            x_expanded = x.unsqueeze(-2)  # (B, T, 1, N)
            w_expanded = w.unsqueeze(0).unsqueeze(0)  # (1, 1, V, N)
        
        # Compute numerator (absolute differences)
        numerator = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1)  # (B, T, V)
        
        # Compute denominator based on variant
        if self.variant == "abs":
            # Use absolute values in denominator
            denominator = torch.sum(torch.abs(x_expanded) + torch.abs(w_expanded), dim=-1)
        else:  # "standard" or "normalized"
            # Standard Bray-Curtis denominator
            denominator = torch.sum(x_expanded + w_expanded, dim=-1)
        
        # Ensure positive denominator
        denominator = torch.clamp(torch.abs(denominator) + self.eps, min=self.eps * 10)
        bray_curtis_dist = numerator / denominator
        
        bray_curtis_dist = fillnan(bray_curtis_dist, nan_value=1.)
        
        # Normalize by minimum distance
        min_dist = torch.min(bray_curtis_dist, dim=-1, keepdim=True)[0]
        min_dist = torch.clamp(min_dist, min=self.eps)
        bray_curtis_dist = bray_curtis_dist / min_dist
        bray_curtis_dist = fillnan(bray_curtis_dist, nan_value=1.)
        
        # Convert to similarity
        pow_n = torch.tensor(n_embd, dtype=torch.float32)
        dist_clamped = torch.clamp(bray_curtis_dist + self.eps, min=self.eps, max=1e6)
        similarity = torch.pow(dist_clamped, -pow_n)
        similarity = fillnan(similarity, nan_value=1.)
        
        return similarity


# Factory functions for easy creation with variants
def create_hamming_layer(in_features, out_features, variant="soft", **kwargs):
    """Factory function to create Hamming layers with different variants"""
    return HammingDistLayer(in_features, out_features, variant=variant, **kwargs)

def create_chebyshev_layer(in_features, out_features, smooth=False, **kwargs):
    """Factory function to create Chebyshev layers"""
    return ChebyshevDistLayer(in_features, out_features, smooth=smooth, **kwargs)

def create_canberra_layer(in_features, out_features, variant="standard", **kwargs):
    """Factory function to create Canberra layers with different variants"""
    return CanberraDistLayer(in_features, out_features, variant=variant, **kwargs)

def create_bray_curtis_layer(in_features, out_features, variant="standard", **kwargs):
    """Factory function to create Bray-Curtis layers with different variants"""
    return BrayCurtisDistLayer(in_features, out_features, variant=variant, **kwargs)