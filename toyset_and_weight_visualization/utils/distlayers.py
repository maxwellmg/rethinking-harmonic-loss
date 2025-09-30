import torch.nn as nn
import torch
import torch.optim as optim
from utils.dataset import *
from utils.visualization import *
import torch.nn.functional as F

use_custom_loss = False
custom_loss = None  

class EuclideanDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(EuclideanDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        n_embd = x.size(-1,)
        w = self.weight
        wx = torch.einsum('bn,vn->bv', x, w)
        ww = torch.norm(w, dim=-1)**2
        xx = torch.norm(x, dim=-1)**2

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
        
        x_expanded = x.unsqueeze(1)
        w_expanded = self.weight.unsqueeze(0)
        
        # |x - w| for each dimension, then sum across dimensions
        manhattan_dist = torch.sum(torch.abs(x_expanded - w_expanded), dim=2) + self.eps

        manhattan_dist = manhattan_dist / torch.min(manhattan_dist, dim=-1, keepdim=True)[0]
        
        return (manhattan_dist) ** (-self.n)

class CosineDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(CosineDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        # Cosine distance = 1 - cosine_similarity
        # cosine_similarity = (x · w) / (||x|| * ||w||)
        
        # Compute dot products: x · w
        wx = torch.einsum('bn,vn->bv', x, self.weight) 
        
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # (B, 1)
        w_norm = torch.norm(self.weight, dim=-1)     
        
        cosine_similarity = wx / ((x_norm * w_norm[None, :]) + self.eps)
        
        cosine_distance = 1 - cosine_similarity + self.eps
        
        cosine_distance = cosine_distance / torch.min(cosine_distance, dim=-1, keepdim=True)[0]
        
        return (cosine_distance) ** (-self.n)

class CosineDistLayerStable(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(CosineDistLayerStable, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        x_normalized = F.normalize(x, p=2, dim=-1)           # (B, N)
        w_normalized = F.normalize(self.weight, p=2, dim=-1) # (V, N)
        
        cosine_similarity = torch.einsum('bn,vn->bv', x_normalized, w_normalized) 
        
        cosine_distance = 1 - cosine_similarity + self.eps
        
        cosine_distance = cosine_distance / torch.min(cosine_distance, dim=-1, keepdim=True)[0]
        
        return (cosine_distance) ** (-self.n)

class MinkowskiDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, p=1.5, n=1., eps=1e-4, bias=False):
        super(MinkowskiDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.p = p  # Minkowski parameter (p=1 for Manhattan, p=2 for Euclidean)
        self.n = n  
        self.eps = max(eps, 1e-6)  # Increased minimum eps
        
    def forward(self, x, scale=False):

        w = self.weight
        
        # Compute Minkowski distance
        # |x - w|_p = (sum(|x_i - w_i|^p))^(1/p)
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        # Compute |x_i - w_i|^p with better numerical stability
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

class HammingDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, threshold=0.5):
        super(HammingDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n  
        self.eps = eps
        self.threshold = threshold  # Threshold for binarization
        
    def forward(self, x, scale=False):

        w = self.weight
        
        temperature = 0.1 
        x_binary = torch.sigmoid((x - self.threshold) / temperature)
        w_binary = torch.sigmoid((w - self.threshold) / temperature)
        
        x_expanded = x_binary.unsqueeze(1)
        w_expanded = w_binary.unsqueeze(0)

        diff = torch.abs(x_expanded - w_expanded)
        hamming_dist = torch.sum(diff, dim=-1) 

        if scale:
            min_dist = torch.min(hamming_dist, dim=-1, keepdim=True)[0]
            hamming_dist = hamming_dist / (min_dist + self.eps)

        result = torch.pow(hamming_dist + self.eps, -self.n)

        if not result.requires_grad and x.requires_grad:
            print("WARNING: Hamming result doesn't require grad!")
            print(f"x.requires_grad: {x.requires_grad}")
            print(f"w.requires_grad: {w.requires_grad}")
            print(f"result.requires_grad: {result.requires_grad}")
        
        return result


class HammingDistLayerSoft(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, temperature=1.0):
        super(HammingDistLayerSoft, self).__init__(in_features, out_features, bias=bias)
        self.n = n  
        self.eps = eps
        self.temperature = temperature  # Temperature for soft binarization
        
    def forward(self, x, scale=False):

        w = self.weight
        
        # Soft binarization using sigmoid (this preserves gradients)
        x_soft = torch.sigmoid(x / self.temperature)
        w_soft = torch.sigmoid(w / self.temperature)
        
        # Compute soft Hamming distance
        x_expanded = x_soft.unsqueeze(1)
        w_expanded = w_soft.unsqueeze(0)
        
        diff = torch.abs(x_expanded - w_expanded)
        soft_hamming_dist = torch.sum(diff, dim=-1) 
        
        if scale:
            max_dist = torch.max(soft_hamming_dist, dim=-1, keepdim=True)[0]
            soft_hamming_dist = soft_hamming_dist / (max_dist + self.eps)
        
        result = torch.pow(soft_hamming_dist + self.eps, -self.n)
        
        return result


class HammingDistLayerGumbel(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, temperature=1.0):
        super(HammingDistLayerGumbel, self).__init__(in_features, out_features, bias=bias)
        self.n = n  
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
        
        x_logits = x
        w_logits = w
        
        if self.training:
            x_binary = gumbel_sigmoid(x_logits, self.temperature)
            w_binary = gumbel_sigmoid(w_logits, self.temperature)
        else:
            x_binary = torch.sigmoid(x_logits / self.temperature)
            w_binary = torch.sigmoid(w_logits / self.temperature)
        
        x_expanded = x_binary.unsqueeze(1)
        w_expanded = w_binary.unsqueeze(0)
        
        diff = torch.abs(x_expanded - w_expanded)
        hamming_dist = torch.sum(diff, dim=-1) 
        
        if scale:
            max_dist = torch.max(hamming_dist, dim=-1, keepdim=True)[0]
            hamming_dist = hamming_dist / (max_dist + self.eps)
        
        result = torch.pow(hamming_dist + self.eps, -self.n)
        return result

class ChebyshevDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(ChebyshevDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n  
        self.eps = eps
        
    def forward(self, x, scale=False):

        w = self.weight
        
        # Compute Chebyshev distance (L∞ norm)
        # ||x - w||_∞ = max_i |x_i - w_i|
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        diff = torch.abs(x_expanded - w_expanded)
        
        chebyshev_dist = torch.max(diff, dim=-1)[0] 
        
        if scale:
            chebyshev_dist = chebyshev_dist / torch.min(chebyshev_dist, dim=-1, keepdim=True)[0]
        
        return torch.pow(chebyshev_dist + self.eps, -self.n)


class ChebyshevDistLayerSmooth(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, alpha=10.0):
        super(ChebyshevDistLayerSmooth, self).__init__(in_features, out_features, bias=bias)
        self.n = n  
        self.eps = eps
        self.alpha = alpha  # Smoothing parameter for soft max
        
    def forward(self, x, scale=False):

        w = self.weight
        
        # Compute smooth Chebyshev distance using log-sum-exp trick
        # Smooth max approximation: (1/α) * log(Σ exp(α * |x_i - w_i|))
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        diff = torch.abs(x_expanded - w_expanded)
        
        smooth_chebyshev_dist = torch.logsumexp(self.alpha * diff, dim=-1) / self.alpha 
        
        if scale:
            smooth_chebyshev_dist = smooth_chebyshev_dist / torch.min(smooth_chebyshev_dist, dim=-1, keepdim=True)[0]
        
        return torch.pow(smooth_chebyshev_dist + self.eps, -self.n)

class CanberraDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(CanberraDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n  
        self.eps = eps
        
    def forward(self, x, scale=False):
        w = self.weight
        
        # Compute Canberra distance
        # d(x, w) = Σ |x_i - w_i| / (|x_i| + |w_i|)
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        # Compute numerator: |x_i - w_i|
        numerator = torch.abs(x_expanded - w_expanded)
        
        # Compute denominator: |x_i| + |w_i|
        denominator = torch.abs(x_expanded) + torch.abs(w_expanded)
        
        denominator = denominator + self.eps
        
        canberra_dist = torch.sum(numerator / denominator, dim=-1) 
        
        if scale:
            canberra_dist = canberra_dist / torch.min(canberra_dist, dim=-1, keepdim=True)[0]
        
        return torch.pow(canberra_dist + self.eps, -self.n)


class CanberraDistLayerRobust(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, min_denom=1e-3):
        super(CanberraDistLayerRobust, self).__init__(in_features, out_features, bias=bias)
        self.n = n  
        self.eps = eps
        self.min_denom = min_denom  # Minimum denominator to prevent instability
        
    def forward(self, x, scale=False):
        w = self.weight
        
        # Compute robust Canberra distance
        # d(x, w) = Σ |x_i - w_i| / max(|x_i| + |w_i|, min_denom)
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        # Compute numerator: |x_i - w_i|
        numerator = torch.abs(x_expanded - w_expanded)
        
        # Compute denominator: max(|x_i| + |w_i|, min_denom)
        raw_denominator = torch.abs(x_expanded) + torch.abs(w_expanded)
        denominator = torch.clamp(raw_denominator, min=self.min_denom)
        
        canberra_dist = torch.sum(numerator / denominator, dim=-1) 
        
        if scale:
            canberra_dist = canberra_dist / torch.min(canberra_dist, dim=-1, keepdim=True)[0]
        
        return torch.pow(canberra_dist + self.eps, -self.n)


class CanberraDistLayerWeighted(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, 
                 weight_power=1.0, normalize_weights=True):
        super(CanberraDistLayerWeighted, self).__init__(in_features, out_features, bias=bias)
        self.n = n  
        self.eps = eps
        self.weight_power = weight_power  # Power for weighting scheme
        self.normalize_weights = normalize_weights  # Whether to normalize weights
        
    def forward(self, x, scale=False):
        w = self.weight

        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        # Compute numerator: |x_i - w_i|
        numerator = torch.abs(x_expanded - w_expanded)
        
        # Compute denominator: |x_i| + |w_i|
        denominator = torch.abs(x_expanded) + torch.abs(w_expanded) + self.eps
        
        feature_weights = torch.pow(denominator, self.weight_power)
        
        if self.normalize_weights:
            feature_weights = feature_weights / torch.sum(feature_weights, dim=-1, keepdim=True)
        
        weighted_terms = feature_weights * (numerator / denominator)
        canberra_dist = torch.sum(weighted_terms, dim=-1) 
        
        if scale:
            canberra_dist = canberra_dist / torch.min(canberra_dist, dim=-1, keepdim=True)[0]
        
        return torch.pow(canberra_dist + self.eps, -self.n)


class BrayCurtisDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-3, bias=False):  # INCREASED eps
        super(BrayCurtisDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)  # Ensure minimum eps
        
    def forward(self, x, scale=False):
        w = self.weight
        
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        # Compute numerator: Σ |x_i - w_i|
        numerator = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1) 
        
        denominator = torch.sum(x_expanded + w_expanded, dim=-1) 
        
        # Ensure denominator is always positive and significant
        denominator = torch.clamp(torch.abs(denominator) + self.eps, min=self.eps * 10)
        
        bray_curtis_dist = numerator / denominator 
        
        if scale:
            min_dist = torch.min(bray_curtis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            bray_curtis_dist = bray_curtis_dist / min_dist

        dist_clamped = torch.clamp(bray_curtis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


class BrayCurtisDistLayerAbs(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-3, bias=False):
        super(BrayCurtisDistLayerAbs, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        
    def forward(self, x, scale=False):
        w = self.weight
        
        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        
        # Compute numerator: Σ |x_i - w_i|
        numerator = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1) 
        
        # Robust absolute denominator
        denominator = torch.sum(torch.abs(x_expanded) + torch.abs(w_expanded), dim=-1) 
        denominator = torch.clamp(denominator + self.eps, min=self.eps * 10)
        
        bray_curtis_dist = numerator / denominator
        
        if scale:
            min_dist = torch.min(bray_curtis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            bray_curtis_dist = bray_curtis_dist / min_dist
        
        dist_clamped = torch.clamp(bray_curtis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


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
        
        # Robust normalization
        if self.normalize_inputs:
            x_pos = torch.abs(x) + self.eps
            w_pos = torch.abs(w) + self.eps
            
            x_sum = torch.sum(x_pos, dim=-1, keepdim=True)
            w_sum = torch.sum(w_pos, dim=-1, keepdim=True)
            
            x_sum_clamped = torch.clamp(x_sum, min=self.min_sum)
            w_sum_clamped = torch.clamp(w_sum, min=self.min_sum)
            
            x_norm = x_pos / x_sum_clamped
            w_norm = w_pos / w_sum_clamped
        else:
            x_norm = x
            w_norm = w
        
        x_expanded = x_norm.unsqueeze(1)
        w_expanded = w_norm.unsqueeze(0)
        
        numerator = torch.sum(torch.abs(x_expanded - w_expanded), dim=-1)
        #denominator = torch.sum(x_expanded + w_expanded, dim=-1)
        denominator = torch.clamp(denominator + self.eps, min=self.eps * 10)
        
        bray_curtis_dist = numerator / denominator
        
        if scale:
            min_dist = torch.min(bray_curtis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            bray_curtis_dist = bray_curtis_dist / min_dist
        
        dist_clamped = torch.clamp(bray_curtis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


class MahalanobisDistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False, 
                 learn_cov=True, init_identity=True, regularize_cov=True, reg_lambda=1e-2):
        super(MahalanobisDistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        self.learn_cov = learn_cov
        self.regularize_cov = regularize_cov
        self.reg_lambda = max(reg_lambda, 1e-4)  # Ensure minimum regularization
        
        if learn_cov:
            if init_identity:
                self.cov_inv = nn.Parameter(torch.eye(in_features) * (1.0 + self.reg_lambda))
            else:
                L = torch.randn(in_features, in_features) * 0.01  # Smaller values
                self.cov_inv = nn.Parameter(L @ L.T + torch.eye(in_features) * self.reg_lambda)
        else:
            self.register_buffer('cov_inv', torch.eye(in_features))
        
    def forward(self, x, scale=False):
        w = self.weight

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

        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        diff = x_expanded - w_expanded
        
        try:
            diff_transformed = torch.einsum('bvn,nm->bvm', diff, cov_inv_reg)
            mahalanobis_sq = torch.sum(diff_transformed * diff, dim=-1)
        except:
            mahalanobis_sq = torch.sum(diff * diff, dim=-1)

        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)  # Ensure non-negative
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        
        if scale:
            min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            mahalanobis_dist = mahalanobis_dist / min_dist
        
        dist_clamped = torch.clamp(mahalanobis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)


class MahalanobisDistLayerCholesky(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(MahalanobisDistLayerCholesky, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)
        
        self.chol_factor = nn.Parameter(torch.eye(in_features) * 0.1 + torch.randn(in_features, in_features) * 0.01)
        
    def forward(self, x, scale=False):
        w = self.weight
        
        L = torch.tril(self.chol_factor)
        
        diag_indices = torch.arange(L.size(0), device=L.device)
        L[diag_indices, diag_indices] = torch.clamp(L[diag_indices, diag_indices], min=self.eps)
        
        cov_inv = L @ L.T + torch.eye(L.size(0), device=L.device) * self.eps * 10

        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        diff = x_expanded - w_expanded
        
        try:
            diff_transformed = torch.einsum('bvn,nm->bvm', diff, cov_inv)
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
        return torch.pow(dist_clamped, -self.n)


class MahalanobisDistLayerDiagonal(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(MahalanobisDistLayerDiagonal, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = max(eps, 1e-6)

        self.diag_cov_inv = nn.Parameter(torch.zeros(in_features))  # Will be exponentiated
        
    def forward(self, x, scale=False):
        w = self.weight

        diag_pos = torch.exp(torch.clamp(self.diag_cov_inv, min=-10, max=10)) + self.eps

        x_expanded = x.unsqueeze(1)
        w_expanded = w.unsqueeze(0)
        diff = x_expanded - w_expanded
        diff_sq = diff * diff

        weighted_diff_sq = diff_sq * diag_pos.unsqueeze(0).unsqueeze(0)
        mahalanobis_sq = torch.sum(weighted_diff_sq, dim=-1)

        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0)
        mahalanobis_dist = torch.sqrt(mahalanobis_sq + self.eps)
        
        if scale:
            min_dist = torch.min(mahalanobis_dist, dim=-1, keepdim=True)[0]
            min_dist = torch.clamp(min_dist, min=self.eps)
            mahalanobis_dist = mahalanobis_dist / min_dist
        
        dist_clamped = torch.clamp(mahalanobis_dist + self.eps, min=self.eps, max=1e6)
        return torch.pow(dist_clamped, -self.n)