import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from .model_setup import *
from .distance_layers import *
import torch.optim as optim

# Add optimizer dictionary
optimizer_dict = {'adamw': torch.optim.AdamW}

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

class QwenMLP(nn.Module):
    """Qwen MLP with SwiGLU activation"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenAttention(nn.Module):
    """Qwen attention with grouped query attention and RoPE"""
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat k/v heads if num_key_value_heads < num_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class QwenDecoderLayer(nn.Module):
    """Qwen decoder layer"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.n_embd
        
        self.self_attn = QwenAttention(config, layer_idx)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

@dataclass
class QwenConfig:
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    n_embd: int = 896  # hidden_size for 0.5B model
    intermediate_size: int = 4864  # MLP intermediate size
    n_layer: int = 24
    n_head: int = 14
    num_key_value_heads: int = 2  # For grouped query attention
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    dropout: float = 0.0
    bias: bool = True
    distance: str = "baseline"
    
    # For compatibility with your existing code
    block_size: int = 32768  # Same as max_position_embeddings
    scale_attn_by_inverse_layer_idx: bool = False

class Qwen2(nn.Module):
    """Qwen2-0.5B model with custom distance layers"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([QwenDecoderLayer(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        
        # Choose the appropriate head layer based on distance type - SAME AS YOUR EXISTING MODELS
        if config.distance == "baseline":
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "euclidean":
            self.lm_head = EuclideanDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_long":
            self.lm_head = ManhattanDistLayerLong(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_intermediate":
            self.lm_head = ManhattanDistLayerIntermediate(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_fast":
            self.lm_head = ManhattanDistLayerFast(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine":
            self.lm_head = CosineDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine_simple":
            self.lm_head = CosineSimpleDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine_temp_scale_0_1":
            self.lm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.1, bias=False)
        elif config.distance == "cosine_temp_scale_0_3":
            self.lm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.3, bias=False)
        elif config.distance == "cosine_temp_scale_0_5":
            self.lm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.5, bias=False)
        elif config.distance == "cosine_temp_scale_1_0":
            self.lm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=1.0, bias=False)
        elif config.distance == "cosine_temp_scale_2_0":
            self.lm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=2.0, bias=False)
        elif config.distance == "minkowski_l1":
            self.lm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=1.0)
        elif config.distance == "minkowski_l2":
            self.lm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=2.0)
        elif config.distance == "minkowski_l1_5":
            self.lm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=1.5)
        elif config.distance == "minkowski_l3":
            self.lm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=3.0)
        elif config.distance == "minkowski_l0_5":
            self.lm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=0.5)
        elif config.distance == "mahalanobis_diagonal":
            self.lm_head = MahalanobisDistLayerDiagonal(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "mahalanobis_cholesky":
            self.lm_head = MahalanobisDistLayerCholesky(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "mahalanobis_standard":
            self.lm_head = MahalanobisDistLayerStandard(config.n_embd, config.vocab_size, bias=False)
        # Add your new distance layers
        elif config.distance == "hamming_soft":
            self.lm_head = create_hamming_layer(config.n_embd, config.vocab_size, variant="soft", bias=False)
        elif config.distance == "hamming_gumbel":
            self.lm_head = create_hamming_layer(config.n_embd, config.vocab_size, variant="gumbel", bias=False)
        elif config.distance == "hamming_hard":
            self.lm_head = create_hamming_layer(config.n_embd, config.vocab_size, variant="hard", bias=False)
        elif config.distance == "chebyshev":
            self.lm_head = create_chebyshev_layer(config.n_embd, config.vocab_size, smooth=False, bias=False)
        elif config.distance == "chebyshev_smooth":
            self.lm_head = create_chebyshev_layer(config.n_embd, config.vocab_size, smooth=True, bias=False)
        elif config.distance == "canberra_standard":
            self.lm_head = create_canberra_layer(config.n_embd, config.vocab_size, variant="standard", bias=False)
        elif config.distance == "canberra_robust":
            self.lm_head = create_canberra_layer(config.n_embd, config.vocab_size, variant="robust", bias=False)
        elif config.distance == "canberra_weighted":
            self.lm_head = create_canberra_layer(config.n_embd, config.vocab_size, variant="weighted", bias=False)
        elif config.distance == "bray_curtis_standard":
            self.lm_head = create_bray_curtis_layer(config.n_embd, config.vocab_size, variant="standard", bias=False)
        elif config.distance == "bray_curtis_abs":
            self.lm_head = create_bray_curtis_layer(config.n_embd, config.vocab_size, variant="abs", bias=False)
        elif config.distance == "bray_curtis_normalized":
            self.lm_head = create_bray_curtis_layer(config.n_embd, config.vocab_size, variant="normalized", bias=False)
        else:
            raise ValueError(f"Unknown distance type: {config.distance}")
        
        # Weight tying
        self.embed_tokens.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def _get_causal_mask(self, seq_len, device):
        """Generate causal attention mask"""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]  # Add batch and head dims
    
    def forward(self, input_ids, attention_mask=None, targets=None):
        print(f"Forward called with input_ids shape: {input_ids.shape}")
        print(f"targets: {targets.shape if targets is not None else None}")
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Causal attention mask
        causal_mask = self._get_causal_mask(seq_len, device)
        if attention_mask is not None:
            # Combine causal mask with padding mask
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            causal_mask = causal_mask + attention_mask
        
        # Forward through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask)
        
        hidden_states = self.norm(hidden_states)
        
        if targets is not None:
            # Training case - same logic as your existing models
            if self.config.distance == "baseline":
                logits = self.lm_head(hidden_states)
                logits = torch.clamp(logits, -30, 30)
            else:
                # Distance-based processing
                dist_output = self.lm_head(hidden_states)
                dist_output = torch.clamp(dist_output, 1e-8, 1e8)
                
                sum_dist = torch.sum(dist_output, dim=-1, keepdim=True)
                sum_dist = torch.clamp(sum_dist, min=1e-8)
                prob = dist_output / sum_dist
                
                # Add smoothing
                alpha = 0.01
                prob = prob + alpha / self.config.vocab_size
                
                prob = torch.clamp(prob, 1e-8, 1.0)
                logits = torch.log(prob)
            
            # Compute loss
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            
            loss = F.cross_entropy(logits, targets, ignore_index=-1, 
                                reduction='none', label_smoothing=0.1)
            loss = torch.clamp(loss, 0, 20)
            loss = torch.mean(loss)
            
            acc = torch.mean((torch.argmax(logits, dim=1) == targets).float())
            
        else:
            # Inference case
            if self.config.distance == "baseline":
                logits = self.lm_head(hidden_states[:, [-1], :])
                logits = torch.clamp(logits, -30, 30)
            else:
                dist_output = self.lm_head(hidden_states[:, [-1], :])
                dist_output = torch.clamp(dist_output, 1e-8, 1e8)
                sum_dist = torch.sum(dist_output, dim=-1, keepdim=True)
                sum_dist = torch.clamp(sum_dist, min=1e-8)
                prob = dist_output / sum_dist
                alpha = 0.01
                prob = prob + alpha / self.config.vocab_size
                prob = torch.clamp(prob, 1e-8, 1.0)
                logits = torch.log(prob)
            
            loss = None
            acc = None
        
        return logits, loss, acc
    
    def configure_optimizers(self, optimizer_name, weight_decay, learning_rate, betas, rho, gamma, lr_max, device_type):
        """Configure optimizers - same pattern as your existing models"""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EuclideanDistLayer, ManhattanDistLayerLong, 
                                  ManhattanDistLayerIntermediate, ManhattanDistLayerFast, 
                                  CosineDistLayer, CosineSimpleDistLayer, CosineTempScaleDistLayer,
                                  OptimizedMinkowskiDistLayer, UltraFastMinkowskiL1,
                                  MahalanobisDistLayerStandard, MahalanobisDistLayerDiagonal, 
                                  MahalanobisDistLayerCholesky, HammingDistLayer, 
                                  ChebyshevDistLayer, CanberraDistLayer, BrayCurtisDistLayer)
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # Remove tied weights from decay set
        if 'lm_head.weight' in decay:
            decay.remove('lm_head.weight')
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        #from .model_setup import optimizer_dict
        opt_func = optimizer_dict[optimizer_name]
        if optimizer_name == 'adamw':
            use_fused = False
            print(f"using fused AdamW: {use_fused}")
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = opt_func(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        else:
            raise ValueError('Invalid optimizer.')
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 125e12  # V100 baseline
        mfu = flops_achieved / flops_promised
        return mfu


# Factory function for creating Qwen models
def create_qwen_model(distance_type="baseline", **kwargs):
    """Factory function to create Qwen2 with custom distance layers"""
    config = QwenConfig(distance=distance_type, **kwargs)
    return Qwen2(config)