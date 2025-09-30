import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from .distance_layers import *
from .model_setup import *

optimizer_dict = {'adamw': torch.optim.AdamW}

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    scale_attn_by_inverse_layer_idx: bool = False
    distance: str = "baseline" # "baseline", "euclidean", "manhattan_long", "manhattan_intermediate", "manhattan_fast", "cosine", "cosine_temp_scale_0_1", "cosine_temp_scale_0_3", "cosine_temp_scale_0_5", "cosine_temp_scale_1_0", "cosine_temp_scale_2_0"

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, idx_layer) for idx_layer in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Choose the appropriate head layer based on distance type
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
        # Replace your existing Minkowski entries with:
        elif config.distance == "optimized_minkowski_l1":
            self.lm_head = UltraFastMinkowskiL1(config.n_embd, config.vocab_size)
        elif config.distance == "optimized_minkowski_l2":
            self.lm_head = OptimizedMinkowskiDistLayer(config.n_embd, config.vocab_size, temperature=2.0)
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
        elif config.distance == "minkowski_custom":
            self.lm_head = create_optimized_minkowski_layer(config.n_embd, config.vocab_size, temperature=config.minkowski_temperature)
        elif config.distance == "mahalanobis_diagonal":
            self.lm_head = MahalanobisDistLayerDiagonal(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "mahalanobis_cholesky":
            self.lm_head = MahalanobisDistLayerCholesky(config.n_embd, config.vocab_size, bias=False)
        elif config.distance == "mahalanobis_standard":
            self.lm_head = MahalanobisDistLayerStandard(config.n_embd, config.vocab_size, bias=False)
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
        self.transformer.wte.weight = self.lm_head.weight
        self.n_embd = config.n_embd

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        '''for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                scale = 1/math.sqrt(self.n_embd)/0.0357*0.02
                torch.nn.init.normal_(p, mean=0.0, std=scale/math.sqrt(2 * config.n_layer))'''
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # Standard GPT-2 residual scaling
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Fixed weight initialization to prevent gradient explosion"""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization for better stability
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Much smaller std for embeddings - this is crucial!
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)  # Reduced from 0.01
        elif isinstance(module, LayerNorm):
            # LayerNorm should start with weight=1, bias=0
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Distance layers - use smaller initialization
        elif isinstance(module, (EuclideanDistLayer, ManhattanDistLayerLong, 
                            ManhattanDistLayerIntermediate, ManhattanDistLayerFast, 
                            CosineDistLayer, CosineSimpleDistLayer, CosineTempScaleDistLayer)):
            # Smaller initialization for distance layers
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        
        # Clamp embeddings to prevent extreme values
        tok_emb = torch.clamp(tok_emb, -10, 10)
        pos_emb = torch.clamp(pos_emb, -10, 10)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks with stability checks
        for i, block in enumerate(self.transformer.h):
            x_prev = x
            x = block(x)
            
            # Gradient clipping within the forward pass
            if self.training:
                x = torch.clamp(x, -50, 50)
            
            # Residual connection stability
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"NaN/Inf detected after block {i}, using previous values")
                x = x_prev
                break
        
        x = self.transformer.ln_f(x)
        
        # Final clamp before head
        x = torch.clamp(x, -20, 20)

        if targets is not None:
            # Training case
            if self.config.distance == "baseline":
                logits = self.lm_head(x)
                # Prevent extreme logits
                logits = torch.clamp(logits, -30, 30)
            else:
                # Distance-based processing with stability
                dist_output = self.lm_head(x)
                # Clamp distance outputs
                dist_output = torch.clamp(dist_output, 1e-8, 1e8)
                
                # Safer normalization
                sum_dist = torch.sum(dist_output, dim=-1, keepdim=True)
                sum_dist = torch.clamp(sum_dist, min=1e-8)
                prob = dist_output / sum_dist
                
                # Add smoothing
                alpha = 0.01
                prob = prob + alpha / self.config.vocab_size
                
                # Convert to logits safely
                prob = torch.clamp(prob, 1e-8, 1.0)
                logits = torch.log(prob)
            
            # Reshape for loss computation
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            # Compute loss with label smoothing for stability
            loss = F.cross_entropy(logits, targets, ignore_index=-1, 
                                reduction='none', label_smoothing=0.1)
            
            # Remove any extreme loss values
            loss = torch.clamp(loss, 0, 20)
            loss = torch.mean(loss)
            
            acc = torch.mean((torch.argmax(logits, dim=1) == targets).float())
            
        else:
            # Inference case - only process last position
            if self.config.distance == "baseline":
                logits = self.lm_head(x[:, [-1], :])
                logits = torch.clamp(logits, -30, 30)
            else:
                dist_output = self.lm_head(x[:, [-1], :])
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

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, optimizer_name, weight_decay, learning_rate, betas, rho, gamma, lr_max, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EuclideanDistLayer, ManhattanDistLayerLong, ManhattanDistLayerIntermediate, ManhattanDistLayerFast, CosineDistLayer, CosineSimpleDistLayer, CosineTempScaleDistLayer, OptimizedMinkowskiDistLayer, UltraFastMinkowskiL1, HammingDistLayer, ChebyshevDistLayer, CanberraDistLayer, BrayCurtisDistLayer)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                    
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        opt_func = optimizer_dict[optimizer_name]
        if optimizer_name == 'adamw':
            use_fused = False # seems to lead to NaN
            print(f"using fused AdamW: {use_fused}")
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = opt_func(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        else:
            raise ValueError('Invalid optimizer.')
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 125e12  # Using V100 specs as in original
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

