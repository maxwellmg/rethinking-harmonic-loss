# Updated regular_config.py with optimized PVT configurations based on grid search results
regular_config = {
    'num_classes': 10,  # Default, will be overridden by dataset-specific logic
    
    # Dataset-specific num_classes mapping
    'dataset_num_classes': {
        'MNIST': 10,
        'CIFAR10': 10,
        'CIFAR100': 100
    },
    
    # UPDATED: Dataset-specific learning rates - optimized PVT settings
    'learning_rates': {
        'MLP': {
            'MNIST': 3e-4,
            'CIFAR10': 3e-4,
            'CIFAR100': 3e-4
        },
        'CNN': {
            'MNIST': 3e-4,
            'CIFAR10': 3e-4,
            'CIFAR100': 3e-4
        },
        'ViT': {
            'MNIST': 1e-3,
            'CIFAR10': 1e-3,
            'CIFAR100': 5e-4  # Lower for CIFAR-100
        },
        'PVT': {
            'MNIST': 1e-3,        # Updated from 1e-4 based on grid search
            'CIFAR10': 1e-3,      # Updated from 1e-4 based on grid search (best performing rate)
            'CIFAR100': 5e-4      # Keep lower for CIFAR-100
        },
        'VGG16': {
            'MNIST': 1e-3,
            'CIFAR10': 1e-3,
            'CIFAR100': 1e-3
        },
        'ResNet50': {
            'MNIST': 0.1,
            'CIFAR10': 0.1,
            'CIFAR100': 0.1
        }
    },

    # Epochs config - keeping original values for consistency
    'epochs_config': {
        'MLP': {
            'MNIST': 40,
            'CIFAR10': 40,
            'CIFAR100': 150
        },
        'CNN': {
            'MNIST': 40,
            'CIFAR10': 40,
            'CIFAR100': 150
        },
        'PVT': {
            'MNIST': 80,
            'CIFAR10': 80,        # Keeping original 80 epochs for consistency
            'CIFAR100': 150
        },
        'ResNet50': {
            'MNIST': 100,
            'CIFAR10': 100,
            'CIFAR100': 200
        }
    },

    # Early stopping patience - keeping original values for consistency
    'early_stopping_patience': {
        'MNIST': 15,
        'CIFAR10': 15,        # Keeping original 15 epochs patience for consistency
        'CIFAR100': 25        # More patience for CIFAR-100's longer training
        
        # Optional Removal of Early Stopping Patience for Emissions Comparison
        #'MNIST': 200,
        #'CIFAR10': 200,
        #'CIFAR100': 200
    },

    'min_improvement': 0.01,

    # UPDATED: Optimizer configs - optimized PVT weight decay
    'optimizer_configs': {
        'MLP': {'type': 'Adam', 'weight_decay': 0},
        'CNN': {'type': 'Adam', 'weight_decay': 0},
        'VGG16': {'type': 'SGD', 'momentum': 0.9, 'weight_decay': 1e-4},
        'ResNet50': {'type': 'SGD', 'momentum': 0.9, 'weight_decay': 1e-4},
        'ViT': {'type': 'AdamW', 'weight_decay': 0.05},
        'PVT': {'type': 'AdamW', 'weight_decay': 0.01},  # Updated from 0.05 to 0.01 (best performing)
    },

    # UPDATED: Batch sizes - optimized for PVT performance
    'batch_sizes': {
        'h100': {
            'MLP': {'CIFAR10': 1024, 'MNIST': 2048, 'CIFAR100': 1024},
            'CNN': {'CIFAR10': 4096, 'MNIST': 8192, 'CIFAR100': 512},
            'PVT': {'CIFAR10': 512, 'MNIST': 256, 'CIFAR100': 256},  # Updated CIFAR10 from 256 to 512
            'ViT': {'CIFAR10': 128, 'MNIST': 256, 'CIFAR100': 128},
            'VGG16': {'CIFAR10': 2048, 'MNIST': 4096, 'CIFAR100': 512},
            'ResNet50': {'CIFAR10': 512, 'MNIST': 512, 'CIFAR100': 256},
        },
        'cpu': {
            'MLP': {'CIFAR10': 64, 'MNIST': 128, 'CIFAR100': 32},
            'CNN': {'CIFAR10': 32, 'MNIST': 64, 'CIFAR100': 16},
            'PVT': {'CIFAR10': 8, 'MNIST': 16, 'CIFAR100': 4},
            'ViT': {'CIFAR10': 4, 'MNIST': 8, 'CIFAR100': 2},
            'VGG16': {'CIFAR10': 8, 'MNIST': 16, 'CIFAR100': 4},
            'ResNet50': {'CIFAR10': 4, 'MNIST': 8, 'CIFAR100': 2},
        }
    },

    # UPDATED: Scheduler configurations - CosineAnnealingLR for PVT CIFAR10
    'scheduler_configs': {
        'MLP': {
            'MNIST': None,
            'CIFAR10': None,
            'CIFAR100': {'type': 'StepLR', 'step_size': 50, 'gamma': 0.5}  # Gentle decay
        },
        'CNN': {
            'MNIST': None,
            'CIFAR10': None,
            'CIFAR100': {'type': 'StepLR', 'step_size': 50, 'gamma': 0.5}  # Gentle decay
        },
        'VGG16': {
            'MNIST': {'type': 'StepLR', 'step_size': 30, 'gamma': 0.1},
            'CIFAR10': {'type': 'StepLR', 'step_size': 30, 'gamma': 0.1},
            'CIFAR100': {'type': 'MultiStepLR', 'milestones': [75, 125], 'gamma': 0.1}  # Adjusted milestones
        },
        'ResNet50': {
            'MNIST': {'type': 'StepLR', 'step_size': 30, 'gamma': 0.1},
            'CIFAR10': {'type': 'StepLR', 'step_size': 30, 'gamma': 0.1},
            'CIFAR100': {'type': 'MultiStepLR', 'milestones': [60, 100, 140], 'gamma': 0.2}  # 3-stage decay
        },
        'ViT': {
            'MNIST': None,
            'CIFAR10': None,
            'CIFAR100': {'type': 'CosineAnnealingLR', 'T_max': 150, 'eta_min': 1e-6}  # Full cosine cycle
        },
        'PVT': {
            'MNIST': {'type': 'CosineAnnealingLR', 'T_max': 80, 'eta_min': 1e-6},   # Added based on CIFAR10 success
            'CIFAR10': {'type': 'CosineAnnealingLR', 'T_max': 80, 'eta_min': 1e-6}, # Updated T_max to match 80 epochs
            'CIFAR100': {'type': 'CosineAnnealingLR', 'T_max': 150, 'eta_min': 1e-6}  # Full cosine cycle
        }
    },

    # Rest of your distance layer configs remain the same...
    'dist_layer_configs': {
        'model_specific': {
            'MLP': {
                'hidden_sizes': [512, 256]
            },
            'CNN': {
                # CNN-specific parameters (currently using defaults)
            },
            'VGG16': {
                # VGG16-specific parameters
            },
            'ResNet50': {
                # ResNet50-specific parameters
            },
            'ViT': {
                # ViT-specific parameters
            },
            'PVT': {
                # PVT-specific parameters
            }
        },
        
        'shared_distance_params': {
            'n': 1.0,
            'eps': 1e-4,
            'scale_distances': False,
            
            'euclideanLayer': {
                'n': 1.0,
                'eps': 1e-4
            },
            
            'manhattanLayer': {
                'n': 1.0,
                'eps': 1e-4
            },
            
            'cosineLayer': {
                'n': 1.0,
                'eps': 1e-4,
                'stable': True
            },
            
            'minkowskiLayer': {
                'p': 1.5,
                'n': 1.0,
                'eps': 1e-4
            },
            
            'hammingLayer': {
                'n': 1.0,
                'eps': 1e-4,
                'threshold': 0.5,
                'temperature': 1.0,
                'variant': 'soft'
            },
            
            'chebyshevLayer': {
                'n': 1.0,
                'eps': 1e-4,
                'smooth': False,
                'alpha': 10.0
            },
            
            'canberraLayer': {
                'n': 1.0,
                'eps': 1e-4,
                'variant': 'standard',
                'min_denom': 1e-3,
                'weight_power': 1.0,
                'normalize_weights': True
            },
            
            'bray_curtisLayer': {
                'n': 1.0,
                'eps': 1e-3,
                'variant': 'standard',
                'normalize_inputs': True,
                'min_sum': 1e-3
            },
            
            'mahalanobisLayer': {
                'n': 1.0,
                'eps': 1e-4,
                'variant': 'standard',
                'learn_cov': True,
                'init_identity': True,
                'regularize_cov': True,
                'reg_lambda': 1e-2
            }
        },

        'custom_dist_params': {
            # Baseline
            'baseline': {},

            # Basic distances - use defaults
            'euclidean': {'n': 1.0},
            'manhattan': {'n': 1.0},
            
            # Cosine variants
            'cosine_stable': {'stable': True, 'n': 1.0},
            'cosine_unstable': {'stable': False, 'n': 1.0},
            
            # Minkowski variants with different p values
            #'minkowski_p1.0': {'p': 1.0, 'n': 1.0},      # Manhattan
            'minkowski_p1.5': {'p': 1.5, 'n': 1.0},      # Default
            #'minkowski_p2.0': {'p': 2.0, 'n': 1.0},      # Euclidean
            'minkowski_p3.0': {'p': 3.0, 'n': 1.0},      # Higher order
            
            # Hamming variants
            'hamming_soft': {
                'variant': 'soft', 
                'temperature': 1.0, 
                'threshold': 0.5,
                'n': 1.0
            },
            'hamming_gumbel': {
                'variant': 'gumbel', 
                'temperature': 0.5, 
                'threshold': 0.5,
                'n': 1.0
            },
            'hamming_hard': {
                'variant': 'hard', 
                'temperature': 0.1, 
                'threshold': 0.5,
                'n': 1.0
            },
            
            # Chebyshev variants
            'chebyshev_standard': {
                'smooth': False, 
                'n': 1.0
            },
            'chebyshev_smooth': {
                'smooth': True, 
                'alpha': 10.0,
                'n': 1.0
            },
            
            # Canberra variants
            'canberra_standard': {
                'variant': 'standard',
                'n': 1.0
            },
            'canberra_robust': {
                'variant': 'robust',
                'min_denom': 1e-2,
                'n': 1.0
            },
            'canberra_weighted': {
                'variant': 'weighted',
                'weight_power': 0.5,
                'normalize_weights': True,
                'n': 1.0
            },
            
            # Bray-Curtis variants
            'bray_curtis_standard': {
                'variant': 'standard',
                'n': 1.0
            },
            'bray_curtis_abs': {
                'variant': 'abs',
                'n': 1.0
            },
            'bray_curtis_normalized': {
                'variant': 'normalized',
                'normalize_inputs': True,
                'min_sum': 1e-3,
                'n': 1.0
            },
            
            # Mahalanobis variants
            'mahalanobis_standard': {
                'variant': 'standard',
                'learn_cov': True,
                'init_identity': True,
                'regularize_cov': True,
                'reg_lambda': 1e-2,
                'n': 1.0
            },
            'mahalanobis_diagonal': {
                'variant': 'diagonal',
                'n': 1.0
            },
            'mahalanobis_cholesky': {
                'variant': 'cholesky',
                'n': 1.0
            }
        }
    }
}



# Helper functions for getting dataset-specific configurations
def get_learning_rate(config, model_type, dataset):
    """Get learning rate for specific model and dataset"""
    lr_config = config['learning_rates'][model_type]
    if isinstance(lr_config, dict):
        return lr_config.get(dataset, lr_config.get('CIFAR10', 3e-4))  # Fallback to CIFAR10 rate
    else:
        return lr_config  # Old format compatibility

def get_early_stopping_patience(config, dataset):
    """Get early stopping patience for specific dataset"""
    patience_config = config['early_stopping_patience']
    if isinstance(patience_config, dict):
        return patience_config.get(dataset, 15)  # Default to 15
    else:
        return patience_config  # Old format compatibility

def get_scheduler_config(config, model_type, dataset):
    """Get scheduler configuration for specific model and dataset"""
    scheduler_config = config['scheduler_configs'][model_type]
    if isinstance(scheduler_config, dict):
        return scheduler_config.get(dataset, None)  # Return None if no dataset-specific config
    else:
        return scheduler_config  # Old format compatibility