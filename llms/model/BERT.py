import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass

# Keep all your existing distance layer implementations
from .model_setup import *
from .distance_layers import *

optimizer_dict = {'adamw': torch.optim.AdamW}

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass
from .distance_layers import *


@dataclass
class BertConfig:
    vocab_size: int = 50304  # Match your dataset
    max_position_embeddings: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    distance: str = "baseline"
    type_vocab_size: int = 2
    pad_token_id: int = 0
    mask_token_id: int = 103
    cls_token_id: int = 101
    sep_token_id: int = 102
    # MLM settings
    mlm_probability: float = 0.15
    # Distance layer settings
    minkowski_temperature: float = 2.0

class BertBlock(nn.Module):
    """BERT transformer block with bidirectional attention"""
    
    def __init__(self, config, idx_layer):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BertSelfAttention(config, idx_layer)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = BertMLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class BertMLP(nn.Module):
    """BERT-style MLP (same as GPT MLP)"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)  # Import this from your model_setup
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class BertSelfAttention(nn.Module):
    """BERT bidirectional attention - key difference from GPT"""
    
    def __init__(self, config, idx_layer):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.idx_layer = idx_layer
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        
        # NO causal mask - this is the key difference from GPT

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.scale_attn_by_inverse_layer_idx:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)) / float(self.idx_layer + 1))
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply padding mask (no causal mask!)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attention_mask = (1.0 - attention_mask) * -10000.0
            att = att + attention_mask
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class BERT(nn.Module):
    """BERT with proper MLM task and bidirectional attention"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_position_embeddings is not None
        self.config = config

        # BERT embeddings
        self.embeddings = nn.ModuleDict(dict(
            word_embeddings = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id),
            position_embeddings = nn.Embedding(config.max_position_embeddings, config.n_embd),
            token_type_embeddings = nn.Embedding(config.type_vocab_size, config.n_embd),
            LayerNorm = LayerNorm(config.n_embd, bias=config.bias),
            dropout = nn.Dropout(config.dropout),
        ))

        # BERT encoder layers (bidirectional attention)
        self.encoder = nn.ModuleDict(dict(
            layers = nn.ModuleList([BertBlock(config, idx_layer) for idx_layer in range(config.n_layer)]),
        ))

        # MLM head with your distance layers
        if config.distance == "baseline":
            self.mlm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "euclidean":
            self.mlm_head = EuclideanDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_long":
            self.mlm_head = ManhattanDistLayerLong(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_intermediate":
            self.mlm_head = ManhattanDistLayerIntermediate(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_fast":
            self.mlm_head = ManhattanDistLayerFast(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine":
            self.mlm_head = CosineDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine_simple":
            self.mlm_head = CosineSimpleDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine_temp_scale_0_1":
            self.mlm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.1, bias=False)
        elif config.distance == "cosine_temp_scale_0_3":
            self.mlm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.3, bias=False)
        elif config.distance == "cosine_temp_scale_0_5":
            self.mlm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.5, bias=False)
        elif config.distance == "cosine_temp_scale_1_0":
            self.mlm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=1.0, bias=False)
        elif config.distance == "cosine_temp_scale_2_0":
            self.mlm_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=2.0, bias=False)
        # Replace your existing Minkowski entries with:
        elif config.distance == "optimized_minkowski_l1":
            self.mlm_head = UltraFastMinkowskiL1(config.n_embd, config.vocab_size)
        elif config.distance == "optimized_minkowski_l2":
            self.mlm_head = OptimizedMinkowskiDistLayer(config.n_embd, config.vocab_size, temperature=2.0)
        elif config.distance == "minkowski_l1":
            self.mlm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=1.0)
        elif config.distance == "minkowski_l2":
            self.mlm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=2.0)
        elif config.distance == "minkowski_l1_5":
            self.mlm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=1.5)
        elif config.distance == "minkowski_l3":
            self.mlm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=3.0)
        elif config.distance == "minkowski_l0_5":
            self.mlm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=0.5)
        elif config.distance == "minkowski_custom":
            self.mlm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, 
                                                          temperature=config.minkowski_temperature)
        elif config.distance == "mahalanobis_diagonal":
            self.mlm_head = MahalanobisDistLayerDiagonal(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "mahalanobis_cholesky":
            self.mlm_head = MahalanobisDistLayerCholesky(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "mahalanobis_standard":
            self.mlm_head = MahalanobisDistLayerStandard(config.n_embd, config.vocab_size, bias=False)
        # New Layers
        elif config.distance == "hamming_soft":
            self.mlm_head = create_hamming_layer(config.n_embd, config.vocab_size, variant="soft", bias=False)
        elif config.distance == "hamming_gumbel":
            self.mlm_head = create_hamming_layer(config.n_embd, config.vocab_size, variant="gumbel", bias=False)
        elif config.distance == "hamming_hard":
            self.mlm_head = create_hamming_layer(config.n_embd, config.vocab_size, variant="hard", bias=False)
        elif config.distance == "chebyshev":
            self.mlm_head = create_chebyshev_layer(config.n_embd, config.vocab_size, smooth=False, bias=False)
        elif config.distance == "chebyshev_smooth":
            self.mlm_head = create_chebyshev_layer(config.n_embd, config.vocab_size, smooth=True, bias=False)
        elif config.distance == "canberra_standard":
            self.mlm_head = create_canberra_layer(config.n_embd, config.vocab_size, variant="standard", bias=False)
        elif config.distance == "canberra_robust":
            self.mlm_head = create_canberra_layer(config.n_embd, config.vocab_size, variant="robust", bias=False)
        elif config.distance == "canberra_weighted":
            self.mlm_head = create_canberra_layer(config.n_embd, config.vocab_size, variant="weighted", bias=False)
        elif config.distance == "bray_curtis_standard":
            self.mlm_head = create_bray_curtis_layer(config.n_embd, config.vocab_size, variant="standard", bias=False)
        elif config.distance == "bray_curtis_abs":
            self.mlm_head = create_bray_curtis_layer(config.n_embd, config.vocab_size, variant="abs", bias=False)
        elif config.distance == "bray_curtis_normalized":
            self.mlm_head = create_bray_curtis_layer(config.n_embd, config.vocab_size, variant="normalized", bias=False)
        else:
            raise ValueError(f"Unknown distance type: {config.distance}")

        # Weight tying
        self.embeddings.word_embeddings.weight = self.mlm_head.weight
        self.n_embd = config.n_embd

        self.apply(self._init_weights)
        
        # Apply residual scaling
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("BERT parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.position_embeddings.weight.numel()
            n_params -= self.embeddings.token_type_embeddings.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Same initialization as your GPT"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    # Add this method to your BERT class in model/BERT.py
    # Add this method to your BERT class in model/BERT.py

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 
        Estimate model flops utilization (MFU) for BERT
        Adapted from GPT implementation but using BERT's config structure
        """
        # First estimate the number of flops we do per iteration
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.max_position_embeddings
        
        # For BERT, we use max_position_embeddings instead of block_size
        # The flops calculation is similar to GPT but accounts for bidirectional attention
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Express our flops throughput as ratio of hardware peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 125e12  # Using same hardware specs as GPT
        mfu = flops_achieved / flops_promised
        return mfu
        
    def configure_optimizers(self, optimizer_name, weight_decay, learning_rate, betas, rho, gamma, lr_max, device_type):
        """
        Configure optimizer for BERT - same logic as GPT
        """
        # Import optimizer dictionary
        optimizer_dict = {'adamw': torch.optim.AdamW}
        
        # Separate parameters for weight decay
        decay = set()
        no_decay = set()
        
        whitelist_weight_modules = (torch.nn.Linear, EuclideanDistLayer, ManhattanDistLayerLong,ManhattanDistLayerIntermediate, ManhattanDistLayerFast, CosineDistLayer, CosineSimpleDistLayer, CosineTempScaleDistLayer, OptimizedMinkowskiDistLayer, UltraFastMinkowskiL1, HammingDistLayer, ChebyshevDistLayer, CanberraDistLayer, BrayCurtisDistLayer)

        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Remove tied weights from decay set (same as GPT logic)
        if 'mlm_head.weight' in decay:  # Note: BERT uses mlm_head instead of lm_head
            decay.remove('mlm_head.weight')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        opt_func = optimizer_dict[optimizer_name]
        if optimizer_name == 'adamw':
            use_fused = False
            print(f"using fused AdamW: {use_fused}")
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = opt_func(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        else:
            raise ValueError('Invalid optimizer.')
        return optimizer

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        MLM forward pass - this is what makes BERT meaningfully different
        """
        device = input_ids.device
        b, t = input_ids.size()
        
        position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Embeddings
        word_emb = self.embeddings.word_embeddings(input_ids)
        pos_emb = self.embeddings.position_embeddings(position_ids)
        type_emb = self.embeddings.token_type_embeddings(token_type_ids)
        
        embeddings = word_emb + pos_emb + type_emb
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        
        # Stability clamps
        embeddings = torch.clamp(embeddings, -10, 10)

        # Forward through encoder layers (bidirectional attention)
        hidden_states = embeddings
        for i, layer in enumerate(self.encoder.layers):
            hidden_states_prev = hidden_states
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
            
            if self.training:
                hidden_states = torch.clamp(hidden_states, -50, 50)
            
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print(f"NaN/Inf detected after layer {i}")
                hidden_states = hidden_states_prev
                break

        hidden_states = torch.clamp(hidden_states, -20, 20)

        # MLM prediction
        if labels is not None:
            # Training case - only predict masked tokens
            if self.config.distance == "baseline":
                prediction_scores = self.mlm_head(hidden_states)
                prediction_scores = torch.clamp(prediction_scores, -30, 30)
            else:
                # Your distance layer processing
                dist_output = self.mlm_head(hidden_states)
                dist_output = torch.clamp(dist_output, 1e-8, 1e8)
                
                sum_dist = torch.sum(dist_output, dim=-1, keepdim=True)
                sum_dist = torch.clamp(sum_dist, min=1e-8)
                prob = dist_output / sum_dist
                
                alpha = 0.01
                prob = prob + alpha / self.config.vocab_size
                prob = torch.clamp(prob, 1e-8, 1.0)
                prediction_scores = torch.log(prob)
            
            # Only compute loss on masked tokens (labels != -100)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), 
                labels.view(-1)
            )
            
            # Calculate accuracy only on masked tokens
            active_predictions = prediction_scores.view(-1, self.config.vocab_size)[labels.view(-1) != -100]
            active_labels = labels.view(-1)[labels.view(-1) != -100]
            
            if len(active_labels) > 0:
                acc = torch.mean((torch.argmax(active_predictions, dim=1) == active_labels).float())
            else:
                acc = torch.tensor(0.0, device=device)
            
            return prediction_scores, masked_lm_loss, acc
        else:
            # Inference case
            if self.config.distance == "baseline":
                prediction_scores = self.mlm_head(hidden_states)
            else:
                dist_output = self.mlm_head(hidden_states)
                dist_output = torch.clamp(dist_output, 1e-8, 1e8)
                sum_dist = torch.sum(dist_output, dim=-1, keepdim=True)
                sum_dist = torch.clamp(sum_dist, min=1e-8)
                prob = dist_output / sum_dist
                alpha = 0.01
                prob = prob + alpha / self.config.vocab_size
                prob = torch.clamp(prob, 1e-8, 1.0)
                prediction_scores = torch.log(prob)
            
            return prediction_scores, None, None

'''
class BertSelfAttention(nn.Module):
    """BERT-style bidirectional self-attention (no causal mask)"""
    
    def __init__(self, config, idx_layer):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Combined query, key, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.idx_layer = idx_layer
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        # Calculate query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Attention scores
        if self.scale_attn_by_inverse_layer_idx:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)) / float(self.idx_layer + 1))
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply attention mask (for padding tokens)
        if attention_mask is not None:
            # Convert attention mask to additive form
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attention_mask = (1.0 - attention_mask) * -10000.0
            att = att + attention_mask
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class BertMLP(nn.Module):
    """BERT-style MLP (same as GPT but included for clarity)"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)  # Using your existing GELU implementation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BertBlock(nn.Module):
    """BERT transformer block"""
    
    def __init__(self, config, idx_layer):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BertSelfAttention(config, idx_layer)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = BertMLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class BertConfig:
    """BERT configuration matching your GPT setup"""
    vocab_size: int = 30522  # BERT vocab size
    max_position_embeddings: int = 512  # BERT's max sequence length
    n_layer: int = 12  # 12 layers for BERT-base
    n_head: int = 12   # 12 attention heads for BERT-base
    n_embd: int = 768  # 768 hidden size for BERT-base
    dropout: float = 0.1  # BERT typically uses 0.1
    bias: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    distance: str = "baseline"  # Keep your distance options
    # Add BERT-specific configs
    type_vocab_size: int = 2  # For segment embeddings
    pad_token_id: int = 0
    
    # For custom Minkowski layers
    minkowski_temperature: float = 2.0  # Default L2


class BERT(nn.Module):
    """BERT model with your custom distance layers"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_position_embeddings is not None
        self.config = config

        # BERT embeddings
        self.embeddings = nn.ModuleDict(dict(
            word_embeddings = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id),
            position_embeddings = nn.Embedding(config.max_position_embeddings, config.n_embd),
            token_type_embeddings = nn.Embedding(config.type_vocab_size, config.n_embd),
            LayerNorm = LayerNorm(config.n_embd, bias=config.bias),
            dropout = nn.Dropout(config.dropout),
        ))

        # BERT encoder layers
        self.encoder = nn.ModuleDict(dict(
            layers = nn.ModuleList([BertBlock(config, idx_layer) for idx_layer in range(config.n_layer)]),
        ))

        # Pooler for classification tasks (optional)
        self.pooler = nn.Linear(config.n_embd, config.n_embd)

        # Choose appropriate head based on distance type - SAME AS YOUR GPT CODE
        if config.distance == "baseline":
            self.cls_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "euclidean":
            self.cls_head = EuclideanDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_long":
            self.cls_head = ManhattanDistLayerLong(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_intermediate":
            self.cls_head = ManhattanDistLayerIntermediate(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "manhattan_fast":
            self.cls_head = ManhattanDistLayerFast(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine":
            self.cls_head = CosineDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine_simple":
            self.cls_head = CosineSimpleDistLayer(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "cosine_temp_scale_0_1":
            self.cls_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.1, bias=False)
        elif config.distance == "cosine_temp_scale_0_3":
            self.cls_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.3, bias=False)
        elif config.distance == "cosine_temp_scale_0_5":
            self.cls_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=0.5, bias=False)
        elif config.distance == "cosine_temp_scale_1_0":
            self.cls_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=1.0, bias=False)
        elif config.distance == "cosine_temp_scale_2_0":
            self.cls_head = CosineTempScaleDistLayer(config.n_embd, config.vocab_size, temperature=2.0, bias=False)
        elif config.distance == "minkowski_l1":
            self.cls_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=1.0)
        elif config.distance == "minkowski_l2":
            self.cls_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=2.0)
        elif config.distance == "minkowski_l1_5":
            self.cls_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=1.5)
        elif config.distance == "minkowski_l3":
            self.cls_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=3.0)
        elif config.distance == "minkowski_l0_5":
            self.cls_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=0.5)
        elif config.distance == "minkowski_custom":
            self.cls_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, 
                                                          temperature=config.minkowski_temperature)
        elif config.distance == "mahalanobis_diagonal":
            self.cls_head = MahalanobisDistLayerDiagonal(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "mahalanobis_cholesky":
            self.cls_head = MahalanobisDistLayerCholesky(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "mahalanobis_standard":
            self.cls_head = MahalanobisDistLayerStandard(config.n_embd, config.vocab_size, bias=False)
        else:
            raise ValueError(f"Unknown distance type: {config.distance}")

        # Weight tying (optional for BERT, but keeping for consistency)
        self.embeddings.word_embeddings.weight = self.cls_head.weight
        self.n_embd = config.n_embd

        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.position_embeddings.weight.numel()
            n_params -= self.embeddings.token_type_embeddings.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights - using your stable initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Distance layers - use smaller initialization
        elif isinstance(module, (EuclideanDistLayer, ManhattanDistLayerLong, 
                            ManhattanDistLayerIntermediate, ManhattanDistLayerFast, 
                            CosineDistLayer, CosineSimpleDistLayer, CosineTempScaleDistLayer)):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, targets=None, task_type="mlm"):
        """
        Forward pass supporting multiple tasks:
        - task_type="mlm": Masked Language Modeling (BERT's pretraining task)
        - task_type="classification": Sequence classification
        """
        device = input_ids.device
        b, t = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Default token type ids to zeros if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Default attention mask to ones if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Embeddings
        word_emb = self.embeddings.word_embeddings(input_ids)
        pos_emb = self.embeddings.position_embeddings(position_ids)
        type_emb = self.embeddings.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = word_emb + pos_emb + type_emb
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        
        # Clamp embeddings for stability
        embeddings = torch.clamp(embeddings, -10, 10)

        # Forward through encoder layers
        hidden_states = embeddings
        for i, layer in enumerate(self.encoder.layers):
            hidden_states_prev = hidden_states
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
            
            # Stability checks (same as your GPT)
            if self.training:
                hidden_states = torch.clamp(hidden_states, -50, 50)
            
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print(f"NaN/Inf detected after layer {i}, using previous values")
                hidden_states = hidden_states_prev
                break

        # Final clamp
        hidden_states = torch.clamp(hidden_states, -20, 20)

        if task_type == "classification":
            # Use pooler for classification (first token representation)
            pooled_output = self.pooler(hidden_states[:, 0])  # [CLS] token
            pooled_output = torch.tanh(pooled_output)
            return pooled_output, None, None

        # For MLM or other token-level tasks
        if targets is not None:
            # Training case - same logic as your GPT
            if self.config.distance == "baseline":
                logits = self.cls_head(hidden_states)
                logits = torch.clamp(logits, -30, 30)
            else:
                # Distance-based processing with stability
                dist_output = self.cls_head(hidden_states)
                dist_output = torch.clamp(dist_output, 1e-8, 1e8)
                
                sum_dist = torch.sum(dist_output, dim=-1, keepdim=True)
                sum_dist = torch.clamp(sum_dist, min=1e-8)
                prob = dist_output / sum_dist
                
                # Add smoothing
                alpha = 0.01
                prob = prob + alpha / self.config.vocab_size
                
                prob = torch.clamp(prob, 1e-8, 1.0)
                logits = torch.log(prob)
            
            # Only compute loss on masked positions (for MLM)
            if targets.dim() == 2:  # Same shape as input
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.size(-1))[active_loss]
                active_labels = targets.view(-1)[active_loss]
            else:
                active_logits = logits.reshape(-1, logits.size(-1))
                active_labels = targets.reshape(-1)

            loss = F.cross_entropy(active_logits, active_labels, ignore_index=-100, 
                                reduction='none', label_smoothing=0.1)
            loss = torch.clamp(loss, 0, 20)
            loss = torch.mean(loss)
            
            acc = torch.mean((torch.argmax(active_logits, dim=1) == active_labels).float())
            
        else:
            # Inference case
            if self.config.distance == "baseline":
                logits = self.cls_head(hidden_states)
                logits = torch.clamp(logits, -30, 30)
            else:
                dist_output = self.cls_head(hidden_states)
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
        """Same optimizer configuration as your GPT"""
        # Separate parameters for weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EuclideanDistLayer, ManhattanDistLayerLong, 
                                  ManhattanDistLayerIntermediate, ManhattanDistLayerFast, 
                                  CosineDistLayer, CosineSimpleDistLayer, CosineTempScaleDistLayer)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        
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
        if 'cls_head.weight' in decay:
            decay.remove('cls_head.weight')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        opt_func = optimizer_dict[optimizer_name]
        if optimizer_name == 'adamw':
            use_fused = False
            print(f"using fused AdamW: {use_fused}")
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = opt_func(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        else:
            raise ValueError('Invalid optimizer.')
        return optimizer

    @classmethod
    def from_pretrained(cls, model_name="bert-base-uncased", override_args=None):
        """Load pretrained BERT weights"""
        override_args = override_args or {}
        
        try:
            from transformers import BertModel, BertConfig as HFBertConfig
        except ImportError:
            raise ImportError("transformers library required for from_pretrained")
        
        print(f"loading weights from pretrained BERT: {model_name}")
        
        # Load HuggingFace BERT
        hf_model = BertModel.from_pretrained(model_name)
        hf_config = hf_model.config
        
        # Create our config
        config_args = {
            'vocab_size': hf_config.vocab_size,
            'max_position_embeddings': hf_config.max_position_embeddings,
            'n_layer': hf_config.num_hidden_layers,
            'n_head': hf_config.num_attention_heads,
            'n_embd': hf_config.hidden_size,
            'dropout': hf_config.hidden_dropout_prob,
            'bias': True,
            'type_vocab_size': hf_config.type_vocab_size,
        }
        
        # Override with user args
        config_args.update(override_args)
        
        config = BertConfig(**config_args)
        model = cls(config)
        
        # Copy weights (this would need detailed mapping - simplified here)
        print("Note: Weight copying from HF BERT needs detailed implementation")
        print("For now, using random initialization with BERT architecture")
        
        return model


# Usage example:
def create_bert_model(distance_type="baseline", **kwargs):
    """Factory function to create BERT with custom distance layers"""
    config = BertConfig(
        distance=distance_type,
        **kwargs
    )
    return BERT(config)


# Example usage:
if __name__ == "__main__":
    # Create BERT with Euclidean distance layer
    model = create_bert_model(
        distance_type="euclidean",
        vocab_size=30522,
        max_position_embeddings=512,
        n_layer=12,
        n_head=12,
        n_embd=768
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # For MLM task
    targets = input_ids.clone()
    targets[targets == 0] = -100  # Ignore padding tokens
    
    logits, loss, acc = model(input_ids, attention_mask=attention_mask, targets=targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")'''