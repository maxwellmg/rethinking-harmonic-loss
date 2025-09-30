import os

# ===== WANDB AND PROJECT SETTINGS =====
wandb_log = True
# Project and run names now set by command line args in training script
# wandb_project will be overridden by --wandb_project if provided
# wandb_run_name will be auto-generated as {model_type}_{distance}_run

# ===== I/O AND CHECKPOINTING =====
out_dir = 'out'
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# ===== DATA SETTINGS =====
dataset = 'openwebtext'
block_size = 512  # Reduced for BERT compatibility (BERT-base max is 512)

# ===== MODEL ARCHITECTURE =====
# Model types
model_type = 'qwen'  # 'gpt', 'bert', or 'qwen'

# BERT-base and GPT-2 small have similar parameters for fair comparison
n_layer = 12
n_head = 12  
n_embd = 768
dropout = 0.1
bias = True
scale_attn_by_inverse_layer_idx = False

'''# ===== BATCH SIZE AND GRADIENT ACCUMULATION (FOR GPT)=====
batch_size = 16        
gradient_accumulation_steps = 8  # Effective batch size = 128

# ===== LEARNING RATE AND OPTIMIZATION =====
learning_rate = 2e-4 
min_lr = 2e-6
warmup_iters = 500'''

'''# ===== BATCH SIZE AND GRADIENT ACCUMULATION (FOR BERT) =====
batch_size = 38  
gradient_accumulation_steps = 2 '''

# ===== BATCH SIZE AND GRADIENT ACCUMULATION (FOR QWEN) =====
batch_size = 6        # Safe batch size from your test results
gradient_accumulation_steps = 10  # For effective batch size of 60

# ===== LEARNING RATE AND OPTIMIZATION =====
learning_rate = 1e-4   # BERT typically needs lower LR than GPT
min_lr = 1e-6
warmup_iters = 1000    # BERT benefits from longer warmup

# Learning rate decay
decay_lr = True
lr_decay_iters = 10000

# ===== OPTIMIZER SETTINGS =====
optimizer_name = 'adamw'
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999  # Standard AdamW beta2
beta3 = 0.     # For compatibility with your optimizer setup
gamma = 1.     # For compatibility 
rho = 0.1      # For compatibility
lr_max = learning_rate

# ===== GRADIENT CLIPPING =====
grad_clip = 1.0  # Standard value for both architectures

# ===== TRAINING ITERATIONS =====
max_iters = 10000
eval_interval = 1000   # More frequent evaluation to catch issues early
eval_iters = 100     # Reduced for faster evaluation
log_interval = 50    # More frequent logging for debugging

#Testing figures
#max_iters = 20
#eval_interval = 10   # More frequent evaluation to catch issues early
#eval_iters = 10     # Reduced for faster evaluation
#log_interval = 5    # More frequent logging for debugging

# ===== SYSTEM AND PRECISION =====
device = 'cuda'
dtype = 'bfloat16'
compile = True

# ===== DDP SETTINGS =====
backend = 'nccl'

# ===== BERT-SPECIFIC SETTINGS =====
# MLM masking probability (used in get_batch_bert)
mlm_probability = 0.15
mask_token_id = 103  # [MASK] token ID - may need adjustment based on tokenizer
pad_token_id = 0     # Padding token ID

# Vocab sizes - will be overridden by dataset meta.pkl if available
gpt_vocab_size = 50304    # GPT-2 vocab size (padded to multiple of 64)
bert_vocab_size = 30522   # BERT vocab size

# ===== ARCHITECTURE-SPECIFIC OVERRIDES =====
# These can be dynamically adjusted in training script based on model_type
architecture_configs = {
    'gpt': {
        'block_size': 1024,           # GPT can handle longer sequences
        'default_vocab_size': 50304,
        'learning_rate': 2e-4,        # GPT often works well with this LR
        'warmup_iters': 500,
        'dropout': 0.1,
    },
    'bert': {
        'max_position_embeddings': 512,  # BERT's standard max length
        'default_vocab_size': 50304,
        'learning_rate': 1e-4,           # BERT often needs lower LR
        'warmup_iters': 1000,            # BERT benefits from longer warmup
        'dropout': 0.1,
        'type_vocab_size': 2,            # For segment embeddings
        'pad_token_id': 0,
        'batch_size': 38,
        'gradient_accumulation_steps': 2,
    },
    'qwen': {
        'vocab_size': 151936,
        'max_position_embeddings': 1024,
        'n_layer': 24,
        'n_head': 14,
        'n_embd': 896,
        'dropout': 0.0,
        'bias': True,
        'learning_rate': 1e-4,
        'warmup_iters': 1000,
        'batch_size': 6,
        'gradient_accumulation_steps': 10,
        # Qwen-specific parameters
        'intermediate_size': 4864,
        'num_key_value_heads': 2,
        'rms_norm_eps': 1e-6,
        'rope_theta': 1000000.0,
        'block_size': 1024,  # For compatibility
    }
}

# ===== STABILITY AND MONITORING =====
max_grad_norm_threshold = 10.0
max_loss_threshold = 50.0
check_gradients_every = 10
enable_detailed_logging = True

# ===== DISTANCE LAYER SETTINGS =====
# For Minkowski layers
minkowski_temperature = 2.0  # Default L2, can be overridden

# For Mahalanobis layers  
mahalanobis_variant = "diagonal"  # "standard", "diagonal", "cholesky"

# Legacy settings for compatibility
interval = 10
variant = 4

# ===== ENVIRONMENT SETTINGS =====
os.environ["WANDB_MODE"] = "online"
