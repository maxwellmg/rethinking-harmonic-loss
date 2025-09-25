# External Imports
from torch import nn
import pandas as pd
import time
import numpy as np
import torch
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal Imports
from models.models import *
from data.data_loaders import get_loaders
from train_test.training import *
from train_test.testing import *
from regularization.distances import * 
from utils.model_factory import *
from utils.async_emissions_tracker import setup_emissions_tracker
from utils.optimizer_selection import *
from utils.extra_functions import extract_config_params
from utils.seed_utils import set_global_seed
from utils.pca_analysis import PCAInterpretabilityAnalyzer


import warnings

# Convert specific warnings to exceptions to get stack traces
warnings.filterwarnings("error", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("error", category=RuntimeWarning, message="invalid value encountered in scalar divide")

# Save device dynamically to allow for both local and HPC runs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define PCA Analyzer for all Runs
pca_analyzer = PCAInterpretabilityAnalyzer(
    n_components=50,
    standardize=True,
    sample_size=2000,
    device=device
)

# Get Time for Outfile Management
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H%M%S")

# Create emissions_data directory in parent directory (same level as Run)
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emissions_data")
os.makedirs(output_dir, exist_ok=True)

# Helper functions for dataset-specific configurations

def run_experiment(config):
    all_experiment_results = []  # Store results from ALL seeds
    all_saved_models = {}  # Track saved models for interpretability across all seeds
        
    total_experiments = (len(config['seed_list']) * len(config['model_types']) * len(config['datasets']) * 
                        len(config['distance_types']) * len(config['distance_layer_types']) * 
                        len(config['lambda_values']))

    print(f"Starting {total_experiments} total experiments across {len(config['seed_list'])} seeds")

    # Create models directory for saving best models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    min_improvement = config['min_improvement']

    # MAIN SEED LOOP - this should complete all seeds before saving
    for seed_idx, seed in enumerate(config['seed_list']):
        print(f"\n{'='*80}")
        print(f"STARTING SEED {seed} ({seed_idx + 1}/{len(config['seed_list'])})")
        print(f"{'='*80}")
        
        # Only save model weights for the first seed to save memory
        save_weights = (seed_idx == 0)
        if save_weights:
            print("*** SAVING MODEL WEIGHTS (first seed only) ***")
        else:
            print("*** SKIPPING MODEL WEIGHTS (memory conservation) ***")
        
        set_global_seed(seed)

        current_seed = {
            'random_seed': random.getstate()[1][0],
            'numpy_seed': np.random.get_state()[1][0],
            'torch_seed': torch.initial_seed()
        }

        # Reset for each seed
        seed_experiment_results = []
        seed_saved_models = {}

        for model_type in config['model_types']:
            for dataset in config['datasets']:
                # Get the correct number of classes for the dataset
                if 'dataset_num_classes' in config:
                    num_classes = config['dataset_num_classes'][dataset]
                else:
                    num_classes = 100 if dataset == 'CIFAR100' else 10
                
                patience = get_early_stopping_patience(config, dataset)
                print(f"Using {num_classes} classes and patience={patience} for dataset {dataset}")
                
                train_loader, test_loader, in_channels = get_loaders(
                    dataset=dataset, 
                    model_type=model_type, 
                    batch_size=config['batch_sizes'][config['hardware']][model_type][dataset]
                )
                
                for distance_typeLayer in config['distance_layer_types']:
                    if distance_typeLayer != 'baseline':
                        model_factory = ModelFactory(model_type, dataset, config, device, distance_typeLayer, num_classes=num_classes)
                    else:
                        model_factory = ModelFactory(model_type, dataset, config, device, num_classes=num_classes)
                    
                    config_params = extract_config_params(config, distance_typeLayer)
                    
                    for distance_type in config['distance_types']:
                        if distance_type == "baseline":
                            lambda_values = [0.0]
                        else:
                            lambda_values = config['lambda_values']

                        for lamb in lambda_values:
                            print(f"Seed {seed}: Dataset={dataset}, Model={model_type}, Loss={distance_type}, Layer={distance_typeLayer}, λ={lamb}")
                            
                            best_test_acc = 0.0
                            epochs_without_improvement = 0
                            best_model_state = None  # Reset for each experiment

                            model = model_factory.get_fresh_model()
                            optimizer = create_dataset_specific_optimizer(model, model_type, dataset, config)
                            scheduler_config = get_scheduler_config(config, model_type, dataset)
                            scheduler = create_dataset_specific_scheduler(optimizer, scheduler_config)
                            criterion = nn.CrossEntropyLoss()

                            project_name = f"{dataset}_{model_type}_loss-{distance_type}_layer-{distance_typeLayer}_{lamb}"
                            emissions_file_name = f"rerun_9_22_{dataset}_{model_type}_test_seed_{seed}"

                            train_accs = []
                            test_accs = []

                            tracker = setup_emissions_tracker(project_name, emissions_file_name, output_dir)
                            tracker.start()

                            if config['override_epochs'] is not None:
                                max_epochs = config['override_epochs']
                            else:
                                max_epochs = config['epochs_config'][model_type][dataset]
                            
                            for epoch in range(1, max_epochs + 1):
                                train_metrics = train(model, train_loader, optimizer, distance_type, criterion, epoch, lamb=lamb)
                                test_metrics = test(model, test_loader, criterion, num_classes)

                                current_test_acc = test_metrics['test_acc']
                                if current_test_acc > best_test_acc + min_improvement:
                                    best_test_acc = current_test_acc
                                    epochs_without_improvement = 0
                                        
                                    # Save the best model state
                                    best_model_state = {
                                        'model_state_dict': model.state_dict(),
                                        'model_type': model_type,
                                        'distance_type': distance_type,
                                        'distance_layer': distance_typeLayer,
                                        'dataset': dataset,
                                        'lambda': lamb,
                                        'test_acc': current_test_acc,
                                        'epoch': epoch,
                                        'seed': seed  # Add seed to model state
                                    }

                                    # Add Mahalanobis-specific parameters if needed
                                    if 'mahalanobis' in distance_typeLayer.lower():
                                        dist_layer = None
                                        if hasattr(model, 'final_layer') and model.final_layer is not None:
                                            dist_layer = model.final_layer
                                        elif hasattr(model, 'dist_layer') and model.dist_layer is not None:
                                            dist_layer = model.dist_layer

                                        if dist_layer is not None:
                                            current_state_dict = best_model_state['model_state_dict']
                                            
                                            for name, param in dist_layer.named_parameters():
                                                if name in ['cov_inv', 'diag_cov_inv', 'chol_factor']:
                                                    current_state_dict[f'final_layer.{name}'] = param.clone().detach()
                                                    current_state_dict[f'dist_layer.{name}'] = param.clone().detach()

                                            for name, buffer in dist_layer.named_buffers():
                                                if name in ['cov_inv']:
                                                    current_state_dict[f'final_layer.{name}'] = buffer.clone().detach()
                                                    current_state_dict[f'dist_layer.{name}'] = buffer.clone().detach()
                                else:
                                    epochs_without_improvement += 1
                                    
                                if epochs_without_improvement >= patience:
                                    print(f"Early stopping at epoch {epoch}/{max_epochs} (patience={patience})")
                                    break

                                current_lr = optimizer.param_groups[0]['lr']
                                if scheduler is not None:
                                    scheduler.step()

                                train_accs.append(train_metrics['train_acc'])
                                test_accs.append(test_metrics['test_acc'])

                                # Extract distance layer parameters for logging
                                dist_layer_params = {}
                                if hasattr(model, 'dist_layer'):
                                    layer = model.dist_layer
                                    
                                    if hasattr(layer, 'n'):
                                        dist_layer_params['layer_n'] = layer.n
                                    if hasattr(layer, 'eps'):
                                        dist_layer_params['layer_eps'] = layer.eps
                                    
                                    layer_type = type(layer).__name__
                                    
                                    if layer_type == 'MinkowskiDistLayer':
                                        dist_layer_params['layer_p'] = getattr(layer, 'p', None)
                                    elif layer_type == 'HammingDistLayer':
                                        dist_layer_params['layer_threshold'] = getattr(layer, 'threshold', None)
                                        dist_layer_params['layer_temperature'] = getattr(layer, 'temperature', None)
                                        dist_layer_params['layer_variant'] = getattr(layer, 'variant', None)
                                    elif layer_type == 'ChebyshevDistLayer':
                                        dist_layer_params['layer_smooth'] = getattr(layer, 'smooth', None)
                                        dist_layer_params['layer_alpha'] = getattr(layer, 'alpha', None)
                                    elif layer_type == 'CanberraDistLayer':
                                        dist_layer_params['layer_variant'] = getattr(layer, 'variant', None)
                                        dist_layer_params['layer_min_denom'] = getattr(layer, 'min_denom', None)
                                        dist_layer_params['layer_weight_power'] = getattr(layer, 'weight_power', None)
                                        dist_layer_params['layer_normalize_weights'] = getattr(layer, 'normalize_weights', None)
                                    elif layer_type == 'BrayCurtisDistLayer':
                                        dist_layer_params['layer_variant'] = getattr(layer, 'variant', None)
                                        dist_layer_params['layer_normalize_inputs'] = getattr(layer, 'normalize_inputs', None)
                                        dist_layer_params['layer_min_sum'] = getattr(layer, 'min_sum', None)
                                    elif layer_type == 'MahalanobisDistLayer':
                                        dist_layer_params['layer_variant'] = getattr(layer, 'variant', None)
                                        dist_layer_params['layer_learn_cov'] = getattr(layer, 'learn_cov', None)
                                        dist_layer_params['layer_init_identity'] = getattr(layer, 'init_identity', None)
                                        dist_layer_params['layer_regularize_cov'] = getattr(layer, 'regularize_cov', None)
                                        dist_layer_params['layer_reg_lambda'] = getattr(layer, 'reg_lambda', None)
                                    elif layer_type == 'CosineDistLayer':
                                        dist_layer_params['layer_stable'] = getattr(layer, 'stable', None)
                                    
                                    dist_layer_params['layer_class'] = layer_type

                                # Store results with ALL information
                                row = {
                                    'dataset': dataset,
                                    'model_type': model_type,
                                    'lambda': lamb,
                                    'distance_type': distance_type,           
                                    'distance_typeLayer': distance_typeLayer, 
                                    'learning_rate': current_lr,
                                    'num_classes': num_classes,
                                    'early_stopping_patience': patience,
                                    'scheduler_type': type(scheduler).__name__ if scheduler else 'None',
                                    'current_seed': current_seed['torch_seed'],
                                    'random_seed': current_seed['random_seed'],
                                    'numpy_seed': current_seed['numpy_seed'],
                                                                
                                    'pc_0_variance': None,
                                    'pc_1_variance': None,
                                    'pc_2_variance': None,
                                    'pc_3_variance': None,
                                    'pc_4_variance': None,
                                    'pc_5_variance': None,
                                    'pc_6_variance': None,
                                    'pc_7_variance': None,
                                    'pc_8_variance': None,
                                    'pc_9_variance': None,
                                    'pc_10_variance': None,
                                    'pc_20_variance': None,
                                    'pc_50_variance': None,
                                    'intrinsic_dim_90': None,
                                    'intrinsic_dim_95': None,
                                    'effective_rank': None,
                                    'participation_ratio': None,
                                    'total_pca_components': None,

                                    **train_metrics,
                                    **test_metrics,
                                    **config_params,
                                    **dist_layer_params
                                }

                                seed_experiment_results.append(row)

                            try:
                                emissions_data = tracker.stop()
                            except Exception as stop_error:
                                emissions_data = None
                                
                            # PCA Analysis
                            pca_metrics = {}
                            if best_model_state is not None:
                                model_key = f"seed{seed}_{dataset}_{model_type}_{distance_type}_{distance_typeLayer}_lambda{lamb}"
                                #model_save_path = save_model_organized(best_model_state, dataset, model_type, distance_type, distance_typeLayer, lamb, models_dir)

                                try:
                                    print(f"Running PCA analysis for {model_key}")
                                    
                                    if distance_typeLayer != 'baseline':
                                        model_factory = ModelFactory(model_type, dataset, config, device, 
                                                                distance_typeLayer, num_classes=num_classes)
                                    else:
                                        model_factory = ModelFactory(model_type, dataset, config, device, 
                                                                num_classes=num_classes)
                                    
                                    pca_model = model_factory.get_fresh_model()
                                    pca_model.load_state_dict(best_model_state['model_state_dict'])
                                    
                                    features = pca_analyzer.extract_penultimate_features(
                                        pca_model, test_loader, model_type
                                    )
                                    
                                    pca_metrics = pca_analyzer.compute_pca_metrics(features, model_key)
                                    
                                    # Update the last result row with PCA metrics
                                    seed_experiment_results[-1].update({
                                        'intrinsic_dim_90': pca_metrics['intrinsic_dim_90'],
                                        'intrinsic_dim_95': pca_metrics['intrinsic_dim_95'],
                                        'effective_rank': pca_metrics['effective_rank'],
                                        'participation_ratio': pca_metrics['participation_ratio'],
                                        'total_pca_components': pca_metrics['total_components'],
                                    })
                                    
                                except Exception as pca_error:
                                    print(f"✗ PCA analysis failed: {pca_error}")
                                    seed_experiment_results[-1].update({
                                        'intrinsic_dim_90': None,
                                        'intrinsic_dim_95': None,
                                        'effective_rank': None,
                                        'participation_ratio': None,
                                        'total_pca_components': None,
                                    })
                            
                            else:
                                print("No best model available for PCA analysis")
                                seed_experiment_results[-1].update({
                                    'intrinsic_dim_90': None,
                                    'intrinsic_dim_95': None,
                                    'effective_rank': None,
                                    'participation_ratio': None,
                                    'total_pca_components': None,
                                })

                            # Update PCA variance metrics
                            if seed_experiment_results:
                                final_row_idx = len(seed_experiment_results) - 1
                                seed_experiment_results[final_row_idx].update({
                                    'pc_0_variance': pca_metrics.get('pc_0_variance'),
                                    'pc_1_variance': pca_metrics.get('pc_1_variance'),
                                    'pc_2_variance': pca_metrics.get('pc_2_variance'),
                                    'pc_3_variance': pca_metrics.get('pc_3_variance'),
                                    'pc_4_variance': pca_metrics.get('pc_4_variance'),
                                    'pc_5_variance': pca_metrics.get('pc_5_variance'),
                                    'pc_6_variance': pca_metrics.get('pc_6_variance'),
                                    'pc_7_variance': pca_metrics.get('pc_7_variance'),
                                    'pc_8_variance': pca_metrics.get('pc_8_variance'),
                                    'pc_9_variance': pca_metrics.get('pc_9_variance'),
                                    'pc_10_variance': pca_metrics.get('pc_10_variance'),
                                    'pc_20_variance': pca_metrics.get('pc_20_variance'),
                                    'pc_50_variance': pca_metrics.get('pc_50_variance'),
                                    'intrinsic_dim_90': pca_metrics.get('intrinsic_dim_90'),
                                    'intrinsic_dim_95': pca_metrics.get('intrinsic_dim_95'),
                                    'effective_rank': pca_metrics.get('effective_rank'),
                                    'participation_ratio': pca_metrics.get('participation_ratio'),
                                    'total_pca_components': pca_metrics.get('total_components'),
                                })    
                                
                                #if best_model_state is not None:
                                #    torch.save(best_model_state, model_save_path)
                                
                                # Store model info for interpretability analysis
                                #seed_saved_models[model_key] = {
                                #    'model_path': model_save_path,
                                #    'model_state': best_model_state,
                                #    'test_loader': test_loader,
                                #    **best_model_state
                                #}

                            # Calculate AUC metrics
                            if len(train_accs) > 0:
                                train_aucc_normalized = np.trapz([acc/100.0 for acc in train_accs], dx=1) / len(train_accs)
                                test_aucc_normalized = np.trapz([acc/100.0 for acc in test_accs], dx=1) / len(test_accs)
                            else:
                                train_aucc_normalized = 0.0
                                test_aucc_normalized = 0.0

                            seed_experiment_results[-1]['train_aucc'] = round(train_aucc_normalized, 5)
                            seed_experiment_results[-1]['test_aucc'] = round(test_aucc_normalized, 5)
                            
                            # Clear GPU memory
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()

        # Add this seed's results to the overall results
        all_experiment_results.extend(seed_experiment_results)
        #all_saved_models.update(seed_saved_models)
        
        print(f"Completed seed {seed} ({seed_idx + 1}/{len(config['seed_list'])})")
        print(f"Seed {seed} generated {len(seed_experiment_results)} experiment results")

    # SAVE RESULTS AFTER ALL SEEDS COMPLETE
    print(f"\n{'='*80}")
    print(f"ALL SEEDS COMPLETED - SAVING FINAL RESULTS")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(all_experiment_results)
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Update filename to include all seeds, datasets, and models
    seed_range = f"seeds_{min(config['seed_list'])}-{max(config['seed_list'])}"
    dataset_names = "_".join(config['datasets'])
    model_names = "_".join(config['model_types'])
    output_file = os.path.join(outputs_dir, f'rerun_{seed_range}_{dataset_names}_{model_names}_{formatted_time}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Final results saved to: {output_file}")
    print(f"Total experiments completed: {len(all_experiment_results)}")

    # Optional: Run interpretability analysis if you have enough models
    # if len(all_saved_models) >= 2:
    #     print("\nStarting interpretability analysis...")
    #     run_interpretability_analysis(all_saved_models, results_df, outputs_dir, config)
    
    return results_df