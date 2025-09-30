import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regular_config import regular_config
from main_run_function import run_experiment

if torch.cuda.is_available():
    hardware = "h100"
    # replace with your available hardware
else:
    hardware = "cpu"

test_config = {
    'hardware': hardware,
    'override_epochs': None,
    'seed_list': [42],
    'lambda_values': [0.0],
    # Options: [0.1, 0.25, 0.5, 0.75, 1, 1.25]
    'datasets': ['MNIST'],
    # Options: ['MNIST', 'CIFAR10', 'CIFAR100']
    'distance_types': ['baseline'],
    # Options: ['baseline', 'euclidean', 'manhattan', 'cosine', 'minkowski', 'chebyshev', 'canberra', 'bray-curtis', 'hamming_mean', 'hamming_median', 'mahalanobis1', 'mahalanobis2']
    'model_types': ['MLP'],
    # Options: ['ViT', 'PVT', 'MLP', 'CNN', 'VGG16', 'ResNet50']

    'distance_layer_types': [
        # Baseline
        'baseline',

        # Basic distances
        'euclidean',
        'manhattan',

        # Cosine variants
        'cosine_stable',
        'cosine_unstable',
        
        # Minkowski with different p values
        #'minkowski_p1.0',    # Manhattan equivalent
        'minkowski_p1.5',    # Default
        #'minkowski_p2.0',    # Euclidean equivalent
        'minkowski_p3.0',    # Higher order
        
        # Hamming variants
        'hamming_soft',
        
        # Chebyshev variants
        'chebyshev_standard',
        'chebyshev_smooth',
        
        # Canberra variants
        'canberra_standard',
        'canberra_robust',
        'canberra_weighted',
        
        # Bray-Curtis variants
        'bray_curtis_standard',
        'bray_curtis_abs',
        'bray_curtis_normalized',
        
        # Mahalanobis variants
        'mahalanobis_standard',
        'mahalanobis_diagonal',
        'mahalanobis_cholesky'
    ]
}

final_config = {**regular_config, **test_config}

if __name__ == '__main__':
    print(f"Starting Run Test")
    
    run_experiment(final_config)
    
    print(f"Completed Run Test")