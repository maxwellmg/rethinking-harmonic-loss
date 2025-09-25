# interpretability_analysis.py
"""
Standalone module for interpretability analysis using Grad-CAM
Designed to work with distance layer experiments
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import seaborn as sns
from scipy import stats
import json
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Move these imports to top to avoid circular import issues
from data.data_loaders import get_loaders
from utils.model_factory import ModelFactory

def save_model_organized(best_model_state, dataset, model_type, distance_type, distance_typeLayer, lamb, base_models_dir):
    """
    Save model with organized directory structure:
    saved_models/
    ├── CIFAR10/
    │   ├── CNN_DIST/
    │   │   ├── CIFAR10_CNN_DIST_euclidean_EuclideanDistLayer_lambda0.1_best.pth
    │   │   └── CIFAR10_CNN_DIST_cosine_CosineDistLayer_lambda0.2_best.pth
    │   └── MLP_DIST/
    │       └── CIFAR10_MLP_DIST_manhattan_ManhattanDistLayer_lambda0.1_best.pth
    └── CIFAR100/
        └── CNN_DIST/
            └── CIFAR100_CNN_DIST_euclidean_EuclideanDistLayer_lambda0.1_best.pth
    """
    if best_model_state is not None:
        # Create organized directory structure
        dataset_dir = os.path.join(base_models_dir, dataset)
        model_type_dir = os.path.join(dataset_dir, model_type)
        
        # Create directories if they don't exist
        os.makedirs(model_type_dir, exist_ok=True)
        
        # Create filename
        model_key = f"{dataset}_{model_type}_{distance_type}_{distance_typeLayer}_lambda{lamb}"
        model_save_path = os.path.join(model_type_dir, f"{model_key}_best.pth")
        
        # Save the model
        torch.save(best_model_state, model_save_path)
        print(f"Model saved to: {model_save_path}")
        
        return model_save_path
    
    return None
    
class InterpretabilityAnalyzer:
    """Main class for interpretability analysis and visualization"""
    
    def __init__(self, output_dir, device='cuda'):
        self.output_dir = output_dir
        self.device = device
        
        # Create interpretability output directory
        self.interp_dir = os.path.join(output_dir, "interpretability_analysis")
        os.makedirs(self.interp_dir, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        self.samples_dir = os.path.join(self.interp_dir, "individual_samples")
        self.metrics_dir = os.path.join(self.interp_dir, "aggregate_metrics")
        self.stats_dir = os.path.join(self.interp_dir, "statistical_analysis")
        
        for dir_path in [self.samples_dir, self.metrics_dir, self.stats_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_target_layer(self, model, model_type):
        """Get the appropriate target layer for Grad-CAM based on model type"""
        try:
            if 'CNN' in model_type.upper() or 'VGG' in model_type.upper() or 'RESNET' in model_type.upper():
                # For CNN-based models, use last conv layer before distance layer or classifier
                if hasattr(model, 'dist_layer'):
                    # Find the layer just before dist_layer
                    modules = list(model.named_modules())
                    for i, (name, module) in enumerate(modules):
                        if 'dist_layer' in name and i > 0:
                            # Get the previous convolutional layer
                            for j in range(i-1, -1, -1):
                                prev_name, prev_module = modules[j]
                                if isinstance(prev_module, nn.Conv2d):
                                    return prev_module
                
                # Fallback: find last conv layer
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, nn.Conv2d):
                        return module
                        
            elif 'VIT' in model_type.upper() or 'PVT' in model_type.upper():
                # For transformer models, use the last attention layer
                for name, module in reversed(list(model.named_modules())):
                    if 'attn' in name.lower() or 'attention' in name.lower():
                        return module
            
            elif 'MLP' in model_type.upper():
                # For MLP models, use the last linear layer before classifier
                linear_layers = []
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        linear_layers.append(module)
                if len(linear_layers) >= 2:
                    return linear_layers[-2]  # Second to last linear layer
            
            # Default fallback - find last layer that has requires_grad=True
            for module in reversed(list(model.modules())):
                if hasattr(module, 'weight') and module.weight.requires_grad:
                    return module
            
            return list(model.modules())[-2]  # Final fallback
            
        except Exception as e:
            print(f"Error finding target layer for {model_type}: {e}")
            # Ultimate fallback
            modules = list(model.modules())
            if len(modules) >= 2:
                return modules[-2]
            return None
    
    def select_diverse_samples(self, test_loader, num_samples=50, samples_per_class=None):
        """Select diverse samples for consistent comparison across models"""
        samples = []
        
        # Determine samples per class
        if samples_per_class is None:
            num_classes = len(test_loader.dataset.classes) if hasattr(test_loader.dataset, 'classes') else 10
            samples_per_class = max(1, num_samples // num_classes)
        
        class_counts = {}
        
        for batch_idx, (data, targets) in enumerate(test_loader):
            for i, (image, label) in enumerate(zip(data, targets)):
                label_item = label.item()
                
                if class_counts.get(label_item, 0) < samples_per_class:
                    samples.append({
                        'image': image,
                        'label': label_item,
                        'id': f"batch{batch_idx}_img{i}",
                        'batch_idx': batch_idx,
                        'img_idx': i
                    })
                    class_counts[label_item] = class_counts.get(label_item, 0) + 1
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
        
        return samples[:num_samples]
    
    def generate_gradcam_for_model(self, model, model_info, test_samples):
        """Generate Grad-CAM results for a single model across test samples"""
        model.eval()
        results = []
        
        # Get target layer
        target_layer = self.get_target_layer(model, model_info['model_type'])
        if target_layer is None:
            print(f"Warning: Could not find target layer for model {model_info['model_key']}")
            return results
        
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
        except Exception as e:
            print(f"Error creating GradCAM for {model_info['model_key']}: {e}")
            return results
        
        for sample in test_samples:
            try:
                image = sample['image'].to(self.device)
                true_label = sample['label']
                
                # Get model prediction
                with torch.no_grad():
                    output = model(image.unsqueeze(0))
                    predicted_class = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1).max().item()
                
                # Generate Grad-CAM
                targets = [ClassifierOutputTarget(predicted_class)]
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=targets)[0]
                
                # Compute interpretability metrics
                metrics = self.compute_interpretability_metrics(grayscale_cam)
                
                result = {
                    'sample_id': sample['id'],
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'correct_prediction': predicted_class == true_label,
                    'gradcam': grayscale_cam,
                    'original_image': image.cpu().numpy(),
                    **metrics,
                    **model_info  # Include model metadata
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {sample['id']} for model {model_info['model_key']}: {e}")
                continue
        
        return results
    
    def compute_interpretability_metrics(self, cam):
        """Compute various interpretability metrics for a CAM"""
        # Normalize CAM to [0, 1]
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_flat = cam_norm.flatten()
        
        metrics = {}
        
        # Concentration metrics
        metrics['entropy'] = -np.sum(cam_norm * np.log(cam_norm + 1e-8))
        metrics['gini_coefficient'] = self._gini_coefficient(cam_flat)
        metrics['max_activation'] = cam_norm.max()
        metrics['activation_spread'] = np.std(cam_norm)
        metrics['activation_mean'] = np.mean(cam_norm)
        
        # Sparsity metrics
        metrics['sparsity_50'] = np.sum(cam_norm > 0.5) / cam_norm.size
        metrics['sparsity_80'] = np.sum(cam_norm > 0.8) / cam_norm.size
        
        # Focus metrics
        top_10_percent = np.percentile(cam_norm, 90)
        metrics['top_10_percent_mass'] = np.sum(cam_norm[cam_norm >= top_10_percent])
        
        return metrics
    
    def _gini_coefficient(self, x):
        """Compute Gini coefficient for measuring concentration"""
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
    
    def compare_models_interpretability(self, models_dict, test_loader, num_samples=50):
        """Main function to compare interpretability across multiple models"""
        print(f"Starting interpretability comparison for {len(models_dict)} models...")
        
        # Select test samples
        test_samples = self.select_diverse_samples(test_loader, num_samples)
        print(f"Selected {len(test_samples)} diverse test samples")
        
        # Generate results for each model
        all_results = {}
        for model_key, model_info in models_dict.items():
            print(f"Processing model: {model_key}")
            model_results = self.generate_gradcam_for_model(
                model_info['model'], model_info, test_samples
            )
            all_results[model_key] = model_results
        
        # Create visualizations
        self.create_all_visualizations(all_results)
        
        # Save results summary
        self.save_results_summary(all_results)
        
        print(f"Interpretability analysis complete. Results saved to: {self.interp_dir}")
        return all_results
    
    def create_all_visualizations(self, all_results):
        """Create all visualization types"""
        print("Creating visualizations...")
        
        self.create_individual_sample_comparisons(all_results)
        self.create_aggregate_metrics_plots(all_results)
        self.create_statistical_comparisons(all_results)
        self.create_distance_layer_analysis(all_results)
        
    def create_individual_sample_comparisons(self, all_results, max_samples=10):
        """Create side-by-side comparisons for individual samples"""
        if not all_results:
            return
        
        # Get sample IDs that are common across all models
        common_samples = set.intersection(*[
            set(result['sample_id'] for result in results) 
            for results in all_results.values()
        ])
        
        common_samples = list(common_samples)[:max_samples]
        
        for sample_id in common_samples:
            self._plot_single_sample_comparison(all_results, sample_id)
    
    def _plot_single_sample_comparison(self, all_results, sample_id):
        """Plot comparison for a single sample across all models"""
        models = list(all_results.keys())
        n_models = len(models)
        
        if n_models < 2:
            return
        
        fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # Find the sample data for each model
        sample_data = {}
        for model_key in models:
            for result in all_results[model_key]:
                if result['sample_id'] == sample_id:
                    sample_data[model_key] = result
                    break
        
        true_label = None
        for col, model_key in enumerate(models):
            if model_key not in sample_data:
                continue
                
            data = sample_data[model_key]
            true_label = data['true_label']
            
            # Top row: Raw Grad-CAM
            im1 = axes[0, col].imshow(data['gradcam'], cmap='jet', alpha=0.8)
            axes[0, col].set_title(f"{model_key}\nPred: {data['predicted_class']} "
                                 f"(Conf: {data['confidence']:.3f})")
            axes[0, col].axis('off')
            
            # Bottom row: Thresholded (top 20%)
            thresh_cam = data['gradcam'].copy()
            threshold = np.percentile(thresh_cam, 80)
            thresh_cam[thresh_cam < threshold] = 0
            
            im2 = axes[1, col].imshow(thresh_cam, cmap='jet', alpha=0.8)
            axes[1, col].set_title(f"Top 20%\nEntropy: {data['entropy']:.3f}")
            axes[1, col].axis('off')
        
        plt.suptitle(f"Sample {sample_id} (True Label: {true_label})", fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.samples_dir, f"comparison_{sample_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_aggregate_metrics_plots(self, all_results):
        """Create aggregate comparison plots across all samples"""
        # Collect metrics
        metrics_data = self._collect_metrics_data(all_results)
        
        if not metrics_data:
            return
        
        # Define metrics to plot
        metrics_to_plot = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            self._plot_metric_comparison(axes[idx], metrics_data, metric)
        
        plt.suptitle('Interpretability Metrics Comparison Across Models', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.metrics_dir, 'aggregate_metrics_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _collect_metrics_data(self, all_results):
        """Collect and organize metrics data for plotting"""
        metrics_data = {}
        
        for model_key, results in all_results.items():
            if not results:  # Skip empty results
                continue
                
            metrics_data[model_key] = {
                'entropy': [r['entropy'] for r in results],
                'gini_coefficient': [r['gini_coefficient'] for r in results],
                'max_activation': [r['max_activation'] for r in results],
                'activation_spread': [r['activation_spread'] for r in results],
                'model_type': results[0]['model_type'],
                'distance_type': results[0]['distance_type'],
                'distance_layer': results[0]['distance_layer']
            }
        
        return metrics_data
    
    def _plot_metric_comparison(self, ax, metrics_data, metric):
        """Plot comparison for a specific metric"""
        model_names = []
        metric_values = []
        colors = []
        
        for model_key, data in metrics_data.items():
            model_names.append(model_key.replace('_', '\n'))  # Line breaks for readability
            metric_values.append(data[metric])
            
            # Color by distance type
            distance_type = data['distance_type']
            if distance_type == 'baseline':
                colors.append('gray')
            elif 'euclidean' in distance_type.lower():
                colors.append('blue')
            elif 'cosine' in distance_type.lower():
                colors.append('red')
            elif 'manhattan' in distance_type.lower():
                colors.append('green')
            else:
                colors.append('orange')
        
        # Create box plot
        bp = ax.boxplot(metric_values, labels=model_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
    
    def create_statistical_comparisons(self, all_results):
        """Create statistical significance comparison plots"""
        metrics_data = self._collect_metrics_data(all_results)
        
        if len(metrics_data) < 2:
            print("Need at least 2 models for statistical comparison")
            return
        
        # Use entropy as the primary metric for statistical testing
        self._create_significance_matrix(metrics_data, 'entropy')
        
    def _create_significance_matrix(self, metrics_data, metric):
        """Create pairwise significance testing matrix"""
        model_keys = list(metrics_data.keys())
        n_models = len(model_keys)
        
        if n_models < 2:
            return
        
        p_value_matrix = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                try:
                    _, p_val = stats.mannwhitneyu(
                        metrics_data[model_keys[i]][metric], 
                        metrics_data[model_keys[j]][metric], 
                        alternative='two-sided'
                    )
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val
                except Exception as e:
                    print(f"Error in statistical test between {model_keys[i]} and {model_keys[j]}: {e}")
                    p_value_matrix[i, j] = 1.0
                    p_value_matrix[j, i] = 1.0
        
        # Plot significance matrix
        plt.figure(figsize=(max(8, n_models), max(6, n_models)))
        
        # Truncate model names for readability
        short_names = [key.split('_')[-2:] for key in model_keys]  # Take last 2 parts
        short_names = ['_'.join(parts) for parts in short_names]
        
        sns.heatmap(p_value_matrix, 
                   xticklabels=short_names, 
                   yticklabels=short_names,
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0.05,
                   cbar_kws={'label': 'p-value'})
        
        plt.title(f'Statistical Significance Matrix (p-values)\n{metric.title()} Differences Between Models')
        plt.tight_layout()
        
        save_path = os.path.join(self.stats_dir, f'significance_matrix_{metric}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_distance_layer_analysis(self, all_results):
        """Create analysis specific to distance layer types"""
        # Group results by distance layer type
        layer_groups = {}
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            layer_type = results[0]['distance_layer']
            if layer_type not in layer_groups:
                layer_groups[layer_type] = {
                    'entropy': [], 'gini_coefficient': [], 'accuracy': []
                }
            
            for result in results:
                layer_groups[layer_type]['entropy'].append(result['entropy'])
                layer_groups[layer_type]['gini_coefficient'].append(result['gini_coefficient'])
                layer_groups[layer_type]['accuracy'].append(float(result['correct_prediction']))
        
        if len(layer_groups) < 2:
            print("Need at least 2 different distance layer types for comparison")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['entropy', 'gini_coefficient', 'accuracy']
        metric_labels = [
            'Entropy\n(Lower = More Focused)', 
            'Gini Coefficient\n(Higher = More Concentrated)', 
            'Accuracy'
        ]
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            layer_names = list(layer_groups.keys())
            values = [layer_groups[layer][metric] for layer in layer_names]
            
            bp = axes[idx].boxplot(values, labels=layer_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            axes[idx].set_title(label)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Distance Layer Type Comparison', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.metrics_dir, 'distance_layer_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results_summary(self, all_results):
        """Save a summary of results to JSON"""
        summary = {}
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            # Aggregate statistics
            metrics = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
            model_summary = {
                'model_type': results[0]['model_type'],
                'distance_type': results[0]['distance_type'],
                'distance_layer': results[0]['distance_layer'],
                'num_samples': len(results),
                'accuracy': np.mean([r['correct_prediction'] for r in results]),
                'metrics': {}
            }
            
            for metric in metrics:
                values = [r[metric] for r in results]
                model_summary['metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            summary[model_key] = model_summary
        
        # Save to JSON
        summary_path = os.path.join(self.interp_dir, 'interpretability_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results summary saved to: {summary_path}")


def run_interpretability_analysis(saved_models, results_df, outputs_dir, config):
    """
    Main function to run interpretability analysis on saved models
    
    Args:
        saved_models: Dict of saved model info from your training loop
        results_df: Results dataframe from experiments
        outputs_dir: Directory to save outputs
        config: Your experiment config
    """
    # Import inside function to avoid circular import issues
    from data.data_loaders import get_loaders
    from utils.model_factory import ModelFactory
    
    print("Initializing interpretability analysis...")
    
    # Fix 1: Correct constructor call order and add device handling
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = InterpretabilityAnalyzer(outputs_dir, device=device)
    
    # Group models by dataset for comparison
    datasets = set(model_info.get('dataset', 'CIFAR10') for model_info in saved_models.values())
    
    for dataset in datasets:
        print(f"\nAnalyzing interpretability for dataset: {dataset}")
        
        # Filter models for this dataset
        dataset_models = {
            key: info for key, info in saved_models.items() 
            if info.get('dataset', 'CIFAR10') == dataset
        }
        
        if len(dataset_models) < 2:
            print(f"Skipping {dataset} - need at least 2 models for comparison")
            continue
        
        # Load/recreate models for interpretability analysis
        models_dict = {}
        
        for model_key, model_info in dataset_models.items():
            try:
                # Check if model is already loaded in memory
                if 'model' in model_info and model_info['model'] is not None:
                    model = model_info['model']
                    print(f"Using model from memory: {model_key}")
                    
                elif 'model_state' in model_info:
                    # Recreate model from state dict
                    print(f"Recreating model from state: {model_key}")
                    
                    # Get model configuration
                    model_type = model_info.get('model_type', 'CNN_DIST')
                    distance_layer = model_info.get('distance_layer', 'EuclideanDistLayer')
                    
                    # Fix 2: Safer config access
                    if isinstance(config, dict):
                        dataset_num_classes = config.get('dataset_num_classes', {})
                        num_classes = dataset_num_classes.get(dataset, 100 if dataset == 'CIFAR100' else 10)
                    else:
                        # If config is object with attributes
                        if hasattr(config, 'dataset_num_classes') and hasattr(config.dataset_num_classes, dataset):
                            num_classes = getattr(config.dataset_num_classes, dataset)
                        else:
                            num_classes = 100 if dataset == 'CIFAR100' else 10
                    
                    # Create model factory
                    if model_type in ['MLP_DIST', 'CNN_DIST', 'ViT_DIST', 'PVT_DIST', 'VGG16_DIST', 'ResNet50_DIST']:
                        model_factory = ModelFactory(
                            model_type, dataset, config, 
                            torch.device(device),
                            distance_layer, num_classes=num_classes
                        )
                    else:
                        model_factory = ModelFactory(
                            model_type, dataset, config,
                            torch.device(device),
                            num_classes=num_classes
                        )
                    
                    # Create fresh model and load state
                    model = model_factory.get_fresh_model()
                    model.load_state_dict(model_info['model_state'])
                    model.eval()
                    
                else:
                    print(f"No model or model_state found for {model_key}, skipping...")
                    continue
                
                # Add to models dictionary
                models_dict[model_key] = {
                    'model': model,
                    'model_key': model_key,
                    'model_type': model_info.get('model_type', 'CNN_DIST'),
                    'distance_type': model_info.get('distance_type', 'baseline'),
                    'distance_layer': model_info.get('distance_layer', 'EuclideanDistLayer'),
                    'dataset': dataset,
                    'test_acc': model_info.get('test_acc', 0.0),
                    'lambda': model_info.get('lambda', 0.0)
                }
                
                print(f"✓ Loaded model: {model_key} (Acc: {model_info.get('test_acc', 0.0):.3f}%)")
                
            except Exception as e:
                print(f"✗ Error loading model {model_key}: {e}")
                continue
        
        if len(models_dict) >= 2:
            # Get test loader for this dataset
            try:
                _, test_loader, _ = get_loaders(
                    dataset=dataset,
                    model_type=list(models_dict.values())[0]['model_type'],  # Use first model's type
                    batch_size=32  # Smaller batch size for interpretability
                )
                
                print(f"Comparing interpretability across {len(models_dict)} models...")
                
                # Fix 3: Use correct method name
                comparison_results = analyzer.compare_models_interpretability(
                    models_dict, 
                    test_loader, 
                    num_samples=50  # Adjust based on your needs
                )
                
                # Fix 4: Visualizations and summary are created inside compare_models_interpretability
                # No need to call them separately
                
                print(f"✅ Interpretability analysis for {dataset} complete!")
                print(f"   Results saved to: {analyzer.interp_dir}")
                
            except Exception as e:
                print(f"✗ Error during interpretability analysis for {dataset}: {e}")
                continue
        else:
            print(f"✗ Insufficient loaded models for {dataset} ({len(models_dict)} models)")
    
    print("\n" + "="*60)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("="*60)


def load_models_for_interpretability(results_df, model_save_dir, device='cuda'):
    """
    Helper function to load saved models for interpretability analysis
    This function should be customized based on your model saving strategy
    """
    models_dict = {}
    
    # Group by unique model configurations and find best performing ones
    grouped = results_df.groupby(['dataset', 'model_type', 'distance_type', 'distance_layer'])
    
    for group_key, group_df in grouped:
        dataset, model_type, distance_type, distance_layer = group_key
        
        # Find best performing model in this group
        best_row = group_df.loc[group_df['test_acc'].idxmax()]
        
        model_key = f"{dataset}_{model_type}_{distance_type}_{distance_layer}"
        
        # Construct model file path (customize this based on your saving convention)
        model_filename = f"{dataset}_{model_type}_{distance_type}_{distance_layer}_lambda{best_row['lambda']}_best.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                # You'll need to recreate the model architecture and load weights
                # This is a placeholder - implement based on your ModelFactory
                model = None  # Load your model here
                
                models_dict[model_key] = {
                    'model': model,
                    'model_key': model_key,
                    'model_type': model_type,
                    'distance_type': distance_type,
                    'distance_layer': distance_layer,
                    'dataset': dataset,
                    'test_acc': best_row['test_acc'],
                    'lambda': best_row['lambda']
                }
                
            except Exception as e:
                print(f"Error loading model {model_key}: {e}")
        else:
            print(f"Model file not found: {model_path}")
    
    return models_dict
    
'''import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import seaborn as sns
from scipy import stats
import json
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class InterpretabilityAnalyzer:
    """Main class for interpretability analysis and visualization"""
    
    def __init__(self, output_dir, device='cuda'):
        self.output_dir = output_dir
        self.device = device
        
        # Create interpretability output directory
        self.interp_dir = os.path.join(output_dir, "interpretability_analysis")
        os.makedirs(self.interp_dir, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        self.samples_dir = os.path.join(self.interp_dir, "individual_samples")
        self.metrics_dir = os.path.join(self.interp_dir, "aggregate_metrics")
        self.stats_dir = os.path.join(self.interp_dir, "statistical_analysis")
        
        for dir_path in [self.samples_dir, self.metrics_dir, self.stats_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_target_layer(self, model, model_type):
        """Get the appropriate target layer for Grad-CAM based on model type"""
        if 'CNN' in model_type.upper() or 'VGG' in model_type.upper() or 'RESNET' in model_type.upper():
            # For CNN-based models, use last conv layer before distance layer or classifier
            if hasattr(model, 'dist_layer'):
                # Find the layer just before dist_layer
                modules = list(model.named_modules())
                for i, (name, module) in enumerate(modules):
                    if 'dist_layer' in name and i > 0:
                        # Get the previous convolutional layer
                        for j in range(i-1, -1, -1):
                            prev_name, prev_module = modules[j]
                            if isinstance(prev_module, nn.Conv2d):
                                return prev_module
            
            # Fallback: find last conv layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    return module
                    
        elif 'VIT' in model_type.upper() or 'PVT' in model_type.upper():
            # For transformer models, use the last attention layer
            for name, module in reversed(list(model.named_modules())):
                if 'attn' in name.lower() or 'attention' in name.lower():
                    return module
        
        elif 'MLP' in model_type.upper():
            # For MLP models, use the last linear layer before classifier
            linear_layers = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    linear_layers.append(module)
            if len(linear_layers) >= 2:
                return linear_layers[-2]  # Second to last linear layer
        
        # Default fallback - find last layer that has requires_grad=True
        for module in reversed(list(model.modules())):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                return module
        
        return list(model.modules())[-2]  # Final fallback
    
    def select_diverse_samples(self, test_loader, num_samples=50, samples_per_class=None):
        """Select diverse samples for consistent comparison across models"""
        samples = []
        
        # Determine samples per class
        if samples_per_class is None:
            num_classes = len(test_loader.dataset.classes) if hasattr(test_loader.dataset, 'classes') else 10
            samples_per_class = max(1, num_samples // num_classes)
        
        class_counts = {}
        
        for batch_idx, (data, targets) in enumerate(test_loader):
            for i, (image, label) in enumerate(zip(data, targets)):
                label_item = label.item()
                
                if class_counts.get(label_item, 0) < samples_per_class:
                    samples.append({
                        'image': image,
                        'label': label_item,
                        'id': f"batch{batch_idx}_img{i}",
                        'batch_idx': batch_idx,
                        'img_idx': i
                    })
                    class_counts[label_item] = class_counts.get(label_item, 0) + 1
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
        
        return samples[:num_samples]
    
    def generate_gradcam_for_model(self, model, model_info, test_samples):
        """Generate Grad-CAM results for a single model across test samples"""
        model.eval()
        results = []
        
        # Get target layer
        target_layer = self.get_target_layer(model, model_info['model_type'])
        if target_layer is None:
            print(f"Warning: Could not find target layer for model {model_info['model_key']}")
            return results
        
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
        except Exception as e:
            print(f"Error creating GradCAM for {model_info['model_key']}: {e}")
            return results
        
        for sample in test_samples:
            try:
                image = sample['image'].to(self.device)
                true_label = sample['label']
                
                # Get model prediction
                with torch.no_grad():
                    output = model(image.unsqueeze(0))
                    predicted_class = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1).max().item()
                
                # Generate Grad-CAM
                targets = [ClassifierOutputTarget(predicted_class)]
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=targets)[0]
                
                # Compute interpretability metrics
                metrics = self.compute_interpretability_metrics(grayscale_cam)
                
                result = {
                    'sample_id': sample['id'],
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'correct_prediction': predicted_class == true_label,
                    'gradcam': grayscale_cam,
                    'original_image': image.cpu().numpy(),
                    **metrics,
                    **model_info  # Include model metadata
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {sample['id']} for model {model_info['model_key']}: {e}")
                continue
        
        return results
    
    def compute_interpretability_metrics(self, cam):
        """Compute various interpretability metrics for a CAM"""
        # Normalize CAM to [0, 1]
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_flat = cam_norm.flatten()
        
        metrics = {}
        
        # Concentration metrics
        metrics['entropy'] = -np.sum(cam_norm * np.log(cam_norm + 1e-8))
        metrics['gini_coefficient'] = self._gini_coefficient(cam_flat)
        metrics['max_activation'] = cam_norm.max()
        metrics['activation_spread'] = np.std(cam_norm)
        metrics['activation_mean'] = np.mean(cam_norm)
        
        # Sparsity metrics
        metrics['sparsity_50'] = np.sum(cam_norm > 0.5) / cam_norm.size
        metrics['sparsity_80'] = np.sum(cam_norm > 0.8) / cam_norm.size
        
        # Focus metrics
        top_10_percent = np.percentile(cam_norm, 90)
        metrics['top_10_percent_mass'] = np.sum(cam_norm[cam_norm >= top_10_percent])
        
        return metrics
    
    def _gini_coefficient(self, x):
        """Compute Gini coefficient for measuring concentration"""
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
    
    def compare_models_interpretability(self, models_dict, test_loader, num_samples=50):
        """Main function to compare interpretability across multiple models"""
        print(f"Starting interpretability comparison for {len(models_dict)} models...")
        
        # Select test samples
        test_samples = self.select_diverse_samples(test_loader, num_samples)
        print(f"Selected {len(test_samples)} diverse test samples")
        
        # Generate results for each model
        all_results = {}
        for model_key, model_info in models_dict.items():
            print(f"Processing model: {model_key}")
            model_results = self.generate_gradcam_for_model(
                model_info['model'], model_info, test_samples
            )
            all_results[model_key] = model_results
        
        # Create visualizations
        self.create_all_visualizations(all_results)
        
        # Save results summary
        self.save_results_summary(all_results)
        
        print(f"Interpretability analysis complete. Results saved to: {self.interp_dir}")
        return all_results
    
    def create_all_visualizations(self, all_results):
        """Create all visualization types"""
        print("Creating visualizations...")
        
        self.create_individual_sample_comparisons(all_results)
        self.create_aggregate_metrics_plots(all_results)
        self.create_statistical_comparisons(all_results)
        self.create_distance_layer_analysis(all_results)
        
    def create_individual_sample_comparisons(self, all_results, max_samples=10):
        """Create side-by-side comparisons for individual samples"""
        if not all_results:
            return
        
        # Get sample IDs that are common across all models
        common_samples = set.intersection(*[
            set(result['sample_id'] for result in results) 
            for results in all_results.values()
        ])
        
        common_samples = list(common_samples)[:max_samples]
        
        for sample_id in common_samples:
            self._plot_single_sample_comparison(all_results, sample_id)
    
    def _plot_single_sample_comparison(self, all_results, sample_id):
        """Plot comparison for a single sample across all models"""
        models = list(all_results.keys())
        n_models = len(models)
        
        if n_models < 2:
            return
        
        fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # Find the sample data for each model
        sample_data = {}
        for model_key in models:
            for result in all_results[model_key]:
                if result['sample_id'] == sample_id:
                    sample_data[model_key] = result
                    break
        
        true_label = None
        for col, model_key in enumerate(models):
            if model_key not in sample_data:
                continue
                
            data = sample_data[model_key]
            true_label = data['true_label']
            
            # Top row: Raw Grad-CAM
            im1 = axes[0, col].imshow(data['gradcam'], cmap='jet', alpha=0.8)
            axes[0, col].set_title(f"{model_key}\nPred: {data['predicted_class']} "
                                 f"(Conf: {data['confidence']:.3f})")
            axes[0, col].axis('off')
            
            # Bottom row: Thresholded (top 20%)
            thresh_cam = data['gradcam'].copy()
            threshold = np.percentile(thresh_cam, 80)
            thresh_cam[thresh_cam < threshold] = 0
            
            im2 = axes[1, col].imshow(thresh_cam, cmap='jet', alpha=0.8)
            axes[1, col].set_title(f"Top 20%\nEntropy: {data['entropy']:.3f}")
            axes[1, col].axis('off')
        
        plt.suptitle(f"Sample {sample_id} (True Label: {true_label})", fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.samples_dir, f"comparison_{sample_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_aggregate_metrics_plots(self, all_results):
        """Create aggregate comparison plots across all samples"""
        # Collect metrics
        metrics_data = self._collect_metrics_data(all_results)
        
        if not metrics_data:
            return
        
        # Define metrics to plot
        metrics_to_plot = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            self._plot_metric_comparison(axes[idx], metrics_data, metric)
        
        plt.suptitle('Interpretability Metrics Comparison Across Models', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.metrics_dir, 'aggregate_metrics_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _collect_metrics_data(self, all_results):
        """Collect and organize metrics data for plotting"""
        metrics_data = {}
        
        for model_key, results in all_results.items():
            if not results:  # Skip empty results
                continue
                
            metrics_data[model_key] = {
                'entropy': [r['entropy'] for r in results],
                'gini_coefficient': [r['gini_coefficient'] for r in results],
                'max_activation': [r['max_activation'] for r in results],
                'activation_spread': [r['activation_spread'] for r in results],
                'model_type': results[0]['model_type'],
                'distance_type': results[0]['distance_type'],
                'distance_layer': results[0]['distance_layer']
            }
        
        return metrics_data
    
    def _plot_metric_comparison(self, ax, metrics_data, metric):
        """Plot comparison for a specific metric"""
        model_names = []
        metric_values = []
        colors = []
        
        for model_key, data in metrics_data.items():
            model_names.append(model_key.replace('_', '\n'))  # Line breaks for readability
            metric_values.append(data[metric])
            
            # Color by distance type
            distance_type = data['distance_type']
            if distance_type == 'baseline':
                colors.append('gray')
            elif 'euclidean' in distance_type.lower():
                colors.append('blue')
            elif 'cosine' in distance_type.lower():
                colors.append('red')
            elif 'manhattan' in distance_type.lower():
                colors.append('green')
            else:
                colors.append('orange')
        
        # Create box plot
        bp = ax.boxplot(metric_values, labels=model_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
    
    def create_statistical_comparisons(self, all_results):
        """Create statistical significance comparison plots"""
        metrics_data = self._collect_metrics_data(all_results)
        
        if len(metrics_data) < 2:
            print("Need at least 2 models for statistical comparison")
            return
        
        # Use entropy as the primary metric for statistical testing
        self._create_significance_matrix(metrics_data, 'entropy')
        
    def _create_significance_matrix(self, metrics_data, metric):
        """Create pairwise significance testing matrix"""
        model_keys = list(metrics_data.keys())
        n_models = len(model_keys)
        
        if n_models < 2:
            return
        
        p_value_matrix = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                try:
                    _, p_val = stats.mannwhitneyu(
                        metrics_data[model_keys[i]][metric], 
                        metrics_data[model_keys[j]][metric], 
                        alternative='two-sided'
                    )
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val
                except Exception as e:
                    print(f"Error in statistical test between {model_keys[i]} and {model_keys[j]}: {e}")
                    p_value_matrix[i, j] = 1.0
                    p_value_matrix[j, i] = 1.0
        
        # Plot significance matrix
        plt.figure(figsize=(max(8, n_models), max(6, n_models)))
        
        # Truncate model names for readability
        short_names = [key.split('_')[-2:] for key in model_keys]  # Take last 2 parts
        short_names = ['_'.join(parts) for parts in short_names]
        
        sns.heatmap(p_value_matrix, 
                   xticklabels=short_names, 
                   yticklabels=short_names,
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0.05,
                   cbar_kws={'label': 'p-value'})
        
        plt.title(f'Statistical Significance Matrix (p-values)\n{metric.title()} Differences Between Models')
        plt.tight_layout()
        
        save_path = os.path.join(self.stats_dir, f'significance_matrix_{metric}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_distance_layer_analysis(self, all_results):
        """Create analysis specific to distance layer types"""
        # Group results by distance layer type
        layer_groups = {}
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            layer_type = results[0]['distance_layer']
            if layer_type not in layer_groups:
                layer_groups[layer_type] = {
                    'entropy': [], 'gini_coefficient': [], 'accuracy': []
                }
            
            for result in results:
                layer_groups[layer_type]['entropy'].append(result['entropy'])
                layer_groups[layer_type]['gini_coefficient'].append(result['gini_coefficient'])
                layer_groups[layer_type]['accuracy'].append(float(result['correct_prediction']))
        
        if len(layer_groups) < 2:
            print("Need at least 2 different distance layer types for comparison")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['entropy', 'gini_coefficient', 'accuracy']
        metric_labels = [
            'Entropy\n(Lower = More Focused)', 
            'Gini Coefficient\n(Higher = More Concentrated)', 
            'Accuracy'
        ]
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            layer_names = list(layer_groups.keys())
            values = [layer_groups[layer][metric] for layer in layer_names]
            
            bp = axes[idx].boxplot(values, labels=layer_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            axes[idx].set_title(label)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Distance Layer Type Comparison', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.metrics_dir, 'distance_layer_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results_summary(self, all_results):
        """Save a summary of results to JSON"""
        summary = {}
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            # Aggregate statistics
            metrics = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
            model_summary = {
                'model_type': results[0]['model_type'],
                'distance_type': results[0]['distance_type'],
                'distance_layer': results[0]['distance_layer'],
                'num_samples': len(results),
                'accuracy': np.mean([r['correct_prediction'] for r in results]),
                'metrics': {}
            }
            
            for metric in metrics:
                values = [r[metric] for r in results]
                model_summary['metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            summary[model_key] = model_summary
        
        # Save to JSON
        summary_path = os.path.join(self.interp_dir, 'interpretability_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results summary saved to: {summary_path}")


def run_interpretability_analysis(saved_models, results_df, outputs_dir, config):
    """
    Main function to run interpretability analysis on saved models
    
    Args:
        saved_models: Dict of saved model info from your training loop
        results_df: Results dataframe from experiments
        outputs_dir: Directory to save outputs
        config: Your experiment config
    """
    from data.data_loaders import get_loaders  # Import your data loader
    from utils.model_factory import ModelFactory  # Import your model factory
    
    print("Initializing interpretability analysis...")
    
    # Initialize analyzer
    analyzer = InterpretabilityAnalyzer(config, outputs_dir)
    
    # Group models by dataset for comparison
    datasets = set(model_info.get('dataset', 'CIFAR10') for model_info in saved_models.values())
    
    for dataset in datasets:
        print(f"\nAnalyzing interpretability for dataset: {dataset}")
        
        # Filter models for this dataset
        dataset_models = {
            key: info for key, info in saved_models.items() 
            if info.get('dataset', 'CIFAR10') == dataset
        }
        
        if len(dataset_models) < 2:
            print(f"Skipping {dataset} - need at least 2 models for comparison")
            continue
        
        # Load/recreate models for interpretability analysis
        models_dict = {}
        
        for model_key, model_info in dataset_models.items():
            try:
                # Check if model is already loaded in memory
                if 'model' in model_info and model_info['model'] is not None:
                    model = model_info['model']
                    print(f"Using model from memory: {model_key}")
                    
                elif 'model_state' in model_info:
                    # Recreate model from state dict
                    print(f"Recreating model from state: {model_key}")
                    
                    # Get model configuration
                    model_type = model_info.get('model_type', 'CNN_DIST')
                    distance_layer = model_info.get('distance_layer', 'EuclideanDistLayer')
                    
                    # Get number of classes for this dataset
                    if hasattr(config, 'dataset_num_classes') and dataset in config['dataset_num_classes']:
                        num_classes = config['dataset_num_classes'][dataset]
                    else:
                        num_classes = 100 if dataset == 'CIFAR100' else 10
                    
                    # Create model factory
                    if model_type in ['MLP_DIST', 'CNN_DIST', 'ViT_DIST', 'PVT_DIST', 'VGG16_DIST', 'ResNet50_DIST']:
                        model_factory = ModelFactory(
                            model_type, dataset, config, 
                            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                            distance_layer, num_classes=num_classes
                        )
                    else:
                        model_factory = ModelFactory(
                            model_type, dataset, config,
                            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                            num_classes=num_classes
                        )
                    
                    # Create fresh model and load state
                    model = model_factory.get_fresh_model()
                    model.load_state_dict(model_info['model_state'])
                    model.eval()
                    
                else:
                    print(f"No model or model_state found for {model_key}, skipping...")
                    continue
                
                # Add to models dictionary
                models_dict[model_key] = {
                    'model': model,
                    'model_key': model_key,
                    'model_type': model_info.get('model_type', 'CNN_DIST'),
                    'distance_type': model_info.get('distance_type', 'baseline'),
                    'distance_layer': model_info.get('distance_layer', 'EuclideanDistLayer'),
                    'dataset': dataset,
                    'test_acc': model_info.get('test_acc', 0.0),
                    'lambda': model_info.get('lambda', 0.0)
                }
                
                print(f"✓ Loaded model: {model_key} (Acc: {model_info.get('test_acc', 0.0):.3f}%)")
                
            except Exception as e:
                print(f"✗ Error loading model {model_key}: {e}")
                continue
        
        if len(models_dict) >= 2:
            # Get test loader for this dataset
            try:
                _, test_loader, _ = get_loaders(
                    dataset=dataset,
                    model_type=list(models_dict.values())[0]['model_type'],  # Use first model's type
                    batch_size=32  # Smaller batch size for interpretability
                )
                
                print(f"Comparing interpretability across {len(models_dict)} models...")
                
                # Run interpretability comparison
                comparison_results = analyzer.generate_gradcam_comparison(
                    models_dict, 
                    test_loader, 
                    num_samples=50  # Adjust based on your needs
                )
                
                # Create visualizations
                analyzer.create_comparison_visualizations(comparison_results)
                
                # Save summary
                analyzer.save_results_summary(comparison_results)
                
                print(f"✅ Interpretability analysis for {dataset} complete!")
                print(f"   Results saved to: {analyzer.interp_dir}")
                
            except Exception as e:
                print(f"✗ Error during interpretability analysis for {dataset}: {e}")
                continue
        else:
            print(f"✗ Insufficient loaded models for {dataset} ({len(models_dict)} models)")
    
    print("\n" + "="*60)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("="*60)


def load_models_for_interpretability(results_df, model_save_dir, device='cuda'):
    """
    Helper function to load saved models for interpretability analysis
    This function should be customized based on your model saving strategy
    """
    models_dict = {}
    
    # Group by unique model configurations and find best performing ones
    grouped = results_df.groupby(['dataset', 'model_type', 'distance_type', 'distance_typeLayer'])
    
    for group_key, group_df in grouped:
        dataset, model_type, distance_type, distance_layer = group_key
        
        # Find best performing model in this group
        best_row = group_df.loc[group_df['test_acc'].idxmax()]
        
        model_key = f"{dataset}_{model_type}_{distance_type}_{distance_layer}"
        
        # Construct model file path (customize this based on your saving convention)
        model_filename = f"{dataset}_{model_type}_{distance_type}_{distance_layer}_lambda{best_row['lambda']}_best.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                # You'll need to recreate the model architecture and load weights
                # This is a placeholder - implement based on your ModelFactory
                model = None  # Load your model here
                
                models_dict[model_key] = {
                    'model': model,
                    'model_key': model_key,
                    'model_type': model_type,
                    'distance_type': distance_type,
                    'distance_layer': distance_layer,
                    'dataset': dataset,
                    'test_acc': best_row['test_acc'],
                    'lambda': best_row['lambda']
                }
                
            except Exception as e:
                print(f"Error loading model {model_key}: {e}")
        else:
            print(f"Model file not found: {model_path}")
    
    return models_dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import seaborn as sns
from scipy import stats
import json
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class InterpretabilityAnalyzer:
    """Main class for interpretability analysis and visualization"""
    
    def __init__(self, output_dir, device='cuda'):
        self.output_dir = output_dir
        self.device = device
        
        # Create interpretability output directory
        self.interp_dir = os.path.join(output_dir, "interpretability_analysis")
        os.makedirs(self.interp_dir, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        self.samples_dir = os.path.join(self.interp_dir, "individual_samples")
        self.metrics_dir = os.path.join(self.interp_dir, "aggregate_metrics")
        self.stats_dir = os.path.join(self.interp_dir, "statistical_analysis")
        
        for dir_path in [self.samples_dir, self.metrics_dir, self.stats_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_target_layer(self, model, model_type):
        """Get the appropriate target layer for Grad-CAM based on model type"""
        if 'CNN' in model_type.upper() or 'VGG' in model_type.upper() or 'RESNET' in model_type.upper():
            # For CNN-based models, use last conv layer before distance layer or classifier
            if hasattr(model, 'dist_layer'):
                # Find the layer just before dist_layer
                modules = list(model.named_modules())
                for i, (name, module) in enumerate(modules):
                    if 'dist_layer' in name and i > 0:
                        # Get the previous convolutional layer
                        for j in range(i-1, -1, -1):
                            prev_name, prev_module = modules[j]
                            if isinstance(prev_module, nn.Conv2d):
                                return prev_module
            
            # Fallback: find last conv layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    return module
                    
        elif 'VIT' in model_type.upper() or 'PVT' in model_type.upper():
            # For transformer models, use the last attention layer
            for name, module in reversed(list(model.named_modules())):
                if 'attn' in name.lower() or 'attention' in name.lower():
                    return module
        
        elif 'MLP' in model_type.upper():
            # For MLP models, use the last linear layer before classifier
            linear_layers = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    linear_layers.append(module)
            if len(linear_layers) >= 2:
                return linear_layers[-2]  # Second to last linear layer
        
        # Default fallback - find last layer that has requires_grad=True
        for module in reversed(list(model.modules())):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                return module
        
        return list(model.modules())[-2]  # Final fallback
    
    def select_diverse_samples(self, test_loader, num_samples=50, samples_per_class=None):
        """Select diverse samples for consistent comparison across models"""
        samples = []
        
        # Determine samples per class
        if samples_per_class is None:
            num_classes = len(test_loader.dataset.classes) if hasattr(test_loader.dataset, 'classes') else 10
            samples_per_class = max(1, num_samples // num_classes)
        
        class_counts = {}
        
        for batch_idx, (data, targets) in enumerate(test_loader):
            for i, (image, label) in enumerate(zip(data, targets)):
                label_item = label.item()
                
                if class_counts.get(label_item, 0) < samples_per_class:
                    samples.append({
                        'image': image,
                        'label': label_item,
                        'id': f"batch{batch_idx}_img{i}",
                        'batch_idx': batch_idx,
                        'img_idx': i
                    })
                    class_counts[label_item] = class_counts.get(label_item, 0) + 1
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
        
        return samples[:num_samples]
    
    def generate_gradcam_for_model(self, model, model_info, test_samples):
        """Generate Grad-CAM results for a single model across test samples"""
        model.eval()
        results = []
        
        # Get target layer
        target_layer = self.get_target_layer(model, model_info['model_type'])
        if target_layer is None:
            print(f"Warning: Could not find target layer for model {model_info['model_key']}")
            return results
        
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
        except Exception as e:
            print(f"Error creating GradCAM for {model_info['model_key']}: {e}")
            return results
        
        for sample in test_samples:
            try:
                image = sample['image'].to(self.device)
                true_label = sample['label']
                
                # Get model prediction
                with torch.no_grad():
                    output = model(image.unsqueeze(0))
                    predicted_class = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1).max().item()
                
                # Generate Grad-CAM
                targets = [ClassifierOutputTarget(predicted_class)]
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=targets)[0]
                
                # Compute interpretability metrics
                metrics = self.compute_interpretability_metrics(grayscale_cam)
                
                result = {
                    'sample_id': sample['id'],
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'correct_prediction': predicted_class == true_label,
                    'gradcam': grayscale_cam,
                    'original_image': image.cpu().numpy(),
                    **metrics,
                    **model_info  # Include model metadata
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {sample['id']} for model {model_info['model_key']}: {e}")
                continue
        
        return results
    
    def compute_interpretability_metrics(self, cam):
        """Compute various interpretability metrics for a CAM"""
        # Normalize CAM to [0, 1]
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_flat = cam_norm.flatten()
        
        metrics = {}
        
        # Concentration metrics
        metrics['entropy'] = -np.sum(cam_norm * np.log(cam_norm + 1e-8))
        metrics['gini_coefficient'] = self._gini_coefficient(cam_flat)
        metrics['max_activation'] = cam_norm.max()
        metrics['activation_spread'] = np.std(cam_norm)
        metrics['activation_mean'] = np.mean(cam_norm)
        
        # Sparsity metrics
        metrics['sparsity_50'] = np.sum(cam_norm > 0.5) / cam_norm.size
        metrics['sparsity_80'] = np.sum(cam_norm > 0.8) / cam_norm.size
        
        # Focus metrics
        top_10_percent = np.percentile(cam_norm, 90)
        metrics['top_10_percent_mass'] = np.sum(cam_norm[cam_norm >= top_10_percent])
        
        return metrics
    
    def _gini_coefficient(self, x):
        """Compute Gini coefficient for measuring concentration"""
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
    
    def compare_models_interpretability(self, models_dict, test_loader, num_samples=50):
        """Main function to compare interpretability across multiple models"""
        print(f"Starting interpretability comparison for {len(models_dict)} models...")
        
        # Select test samples
        test_samples = self.select_diverse_samples(test_loader, num_samples)
        print(f"Selected {len(test_samples)} diverse test samples")
        
        # Generate results for each model
        all_results = {}
        for model_key, model_info in models_dict.items():
            print(f"Processing model: {model_key}")
            model_results = self.generate_gradcam_for_model(
                model_info['model'], model_info, test_samples
            )
            all_results[model_key] = model_results
        
        # Create visualizations
        self.create_all_visualizations(all_results)
        
        # Save results summary
        self.save_results_summary(all_results)
        
        print(f"Interpretability analysis complete. Results saved to: {self.interp_dir}")
        return all_results
    
    def create_all_visualizations(self, all_results):
        """Create all visualization types"""
        print("Creating visualizations...")
        
        self.create_individual_sample_comparisons(all_results)
        self.create_aggregate_metrics_plots(all_results)
        self.create_statistical_comparisons(all_results)
        self.create_distance_layer_analysis(all_results)
        
    def create_individual_sample_comparisons(self, all_results, max_samples=10):
        """Create side-by-side comparisons for individual samples"""
        if not all_results:
            return
        
        # Get sample IDs that are common across all models
        common_samples = set.intersection(*[
            set(result['sample_id'] for result in results) 
            for results in all_results.values()
        ])
        
        common_samples = list(common_samples)[:max_samples]
        
        for sample_id in common_samples:
            self._plot_single_sample_comparison(all_results, sample_id)
    
    def _plot_single_sample_comparison(self, all_results, sample_id):
        """Plot comparison for a single sample across all models"""
        models = list(all_results.keys())
        n_models = len(models)
        
        if n_models < 2:
            return
        
        fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # Find the sample data for each model
        sample_data = {}
        for model_key in models:
            for result in all_results[model_key]:
                if result['sample_id'] == sample_id:
                    sample_data[model_key] = result
                    break
        
        true_label = None
        for col, model_key in enumerate(models):
            if model_key not in sample_data:
                continue
                
            data = sample_data[model_key]
            true_label = data['true_label']
            
            # Top row: Raw Grad-CAM
            im1 = axes[0, col].imshow(data['gradcam'], cmap='jet', alpha=0.8)
            axes[0, col].set_title(f"{model_key}\nPred: {data['predicted_class']} "
                                 f"(Conf: {data['confidence']:.3f})")
            axes[0, col].axis('off')
            
            # Bottom row: Thresholded (top 20%)
            thresh_cam = data['gradcam'].copy()
            threshold = np.percentile(thresh_cam, 80)
            thresh_cam[thresh_cam < threshold] = 0
            
            im2 = axes[1, col].imshow(thresh_cam, cmap='jet', alpha=0.8)
            axes[1, col].set_title(f"Top 20%\nEntropy: {data['entropy']:.3f}")
            axes[1, col].axis('off')
        
        plt.suptitle(f"Sample {sample_id} (True Label: {true_label})", fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.samples_dir, f"comparison_{sample_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_aggregate_metrics_plots(self, all_results):
        """Create aggregate comparison plots across all samples"""
        # Collect metrics
        metrics_data = self._collect_metrics_data(all_results)
        
        if not metrics_data:
            return
        
        # Define metrics to plot
        metrics_to_plot = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            self._plot_metric_comparison(axes[idx], metrics_data, metric)
        
        plt.suptitle('Interpretability Metrics Comparison Across Models', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.metrics_dir, 'aggregate_metrics_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _collect_metrics_data(self, all_results):
        """Collect and organize metrics data for plotting"""
        metrics_data = {}
        
        for model_key, results in all_results.items():
            if not results:  # Skip empty results
                continue
                
            metrics_data[model_key] = {
                'entropy': [r['entropy'] for r in results],
                'gini_coefficient': [r['gini_coefficient'] for r in results],
                'max_activation': [r['max_activation'] for r in results],
                'activation_spread': [r['activation_spread'] for r in results],
                'model_type': results[0]['model_type'],
                'distance_type': results[0]['distance_type'],
                'distance_layer': results[0]['distance_layer']
            }
        
        return metrics_data
    
    def _plot_metric_comparison(self, ax, metrics_data, metric):
        """Plot comparison for a specific metric"""
        model_names = []
        metric_values = []
        colors = []
        
        for model_key, data in metrics_data.items():
            model_names.append(model_key.replace('_', '\n'))  # Line breaks for readability
            metric_values.append(data[metric])
            
            # Color by distance type
            distance_type = data['distance_type']
            if distance_type == 'baseline':
                colors.append('gray')
            elif 'euclidean' in distance_type.lower():
                colors.append('blue')
            elif 'cosine' in distance_type.lower():
                colors.append('red')
            elif 'manhattan' in distance_type.lower():
                colors.append('green')
            else:
                colors.append('orange')
        
        # Create box plot
        bp = ax.boxplot(metric_values, labels=model_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
    
    def create_statistical_comparisons(self, all_results):
        """Create statistical significance comparison plots"""
        metrics_data = self._collect_metrics_data(all_results)
        
        if len(metrics_data) < 2:
            print("Need at least 2 models for statistical comparison")
            return
        
        # Use entropy as the primary metric for statistical testing
        self._create_significance_matrix(metrics_data, 'entropy')
        
    def _create_significance_matrix(self, metrics_data, metric):
        """Create pairwise significance testing matrix"""
        model_keys = list(metrics_data.keys())
        n_models = len(model_keys)
        
        if n_models < 2:
            return
        
        p_value_matrix = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                try:
                    _, p_val = stats.mannwhitneyu(
                        metrics_data[model_keys[i]][metric], 
                        metrics_data[model_keys[j]][metric], 
                        alternative='two-sided'
                    )
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val
                except Exception as e:
                    print(f"Error in statistical test between {model_keys[i]} and {model_keys[j]}: {e}")
                    p_value_matrix[i, j] = 1.0
                    p_value_matrix[j, i] = 1.0
        
        # Plot significance matrix
        plt.figure(figsize=(max(8, n_models), max(6, n_models)))
        
        # Truncate model names for readability
        short_names = [key.split('_')[-2:] for key in model_keys]  # Take last 2 parts
        short_names = ['_'.join(parts) for parts in short_names]
        
        sns.heatmap(p_value_matrix, 
                   xticklabels=short_names, 
                   yticklabels=short_names,
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0.05,
                   cbar_kws={'label': 'p-value'})
        
        plt.title(f'Statistical Significance Matrix (p-values)\n{metric.title()} Differences Between Models')
        plt.tight_layout()
        
        save_path = os.path.join(self.stats_dir, f'significance_matrix_{metric}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_distance_layer_analysis(self, all_results):
        """Create analysis specific to distance layer types"""
        # Group results by distance layer type
        layer_groups = {}
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            layer_type = results[0]['distance_layer']
            if layer_type not in layer_groups:
                layer_groups[layer_type] = {
                    'entropy': [], 'gini_coefficient': [], 'accuracy': []
                }
            
            for result in results:
                layer_groups[layer_type]['entropy'].append(result['entropy'])
                layer_groups[layer_type]['gini_coefficient'].append(result['gini_coefficient'])
                layer_groups[layer_type]['accuracy'].append(float(result['correct_prediction']))
        
        if len(layer_groups) < 2:
            print("Need at least 2 different distance layer types for comparison")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['entropy', 'gini_coefficient', 'accuracy']
        metric_labels = [
            'Entropy\n(Lower = More Focused)', 
            'Gini Coefficient\n(Higher = More Concentrated)', 
            'Accuracy'
        ]
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            layer_names = list(layer_groups.keys())
            values = [layer_groups[layer][metric] for layer in layer_names]
            
            bp = axes[idx].boxplot(values, labels=layer_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            axes[idx].set_title(label)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Distance Layer Type Comparison', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.metrics_dir, 'distance_layer_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results_summary(self, all_results):
        """Save a summary of results to JSON"""
        summary = {}
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            # Aggregate statistics
            metrics = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
            model_summary = {
                'model_type': results[0]['model_type'],
                'distance_type': results[0]['distance_type'],
                'distance_layer': results[0]['distance_layer'],
                'num_samples': len(results),
                'accuracy': np.mean([r['correct_prediction'] for r in results]),
                'metrics': {}
            }
            
            for metric in metrics:
                values = [r[metric] for r in results]
                model_summary['metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            summary[model_key] = model_summary
        
        # Save to JSON
        summary_path = os.path.join(self.interp_dir, 'interpretability_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results summary saved to: {summary_path}")


def run_interpretability_analysis(saved_models, results_df, outputs_dir, config):
    """
    Main function to run interpretability analysis on saved models
    
    Args:
        saved_models: Dict of saved model info from your training loop
        results_df: Results dataframe from experiments
        outputs_dir: Directory to save outputs
        config: Your experiment config
    """
    from data.data_loaders import get_loaders  # Import your data loader
    from utils.model_factory import ModelFactory  # Import your model factory
    
    print("Initializing interpretability analysis...")
    
    # Initialize analyzer
    #analyzer = InterpretabilityEvaluator(config, outputs_dir)
    analyzer = InterpretabilityAnalyzer(config, outputs_dir)
    
    # Group models by dataset for comparison
    datasets = set(model_info.get('dataset', 'CIFAR10') for model_info in saved_models.values())
    
    for dataset in datasets:
        print(f"\nAnalyzing interpretability for dataset: {dataset}")
        
        # Filter models for this dataset
        dataset_models = {
            key: info for key, info in saved_models.items() 
            if info.get('dataset', 'CIFAR10') == dataset
        }
        
        if len(dataset_models) < 2:
            print(f"Skipping {dataset} - need at least 2 models for comparison")
            continue
        
        # Load/recreate models for interpretability analysis
        models_dict = {}
        
        for model_key, model_info in dataset_models.items():
            try:
                # Check if model is already loaded in memory
                if 'model' in model_info and model_info['model'] is not None:
                    model = model_info['model']
                    print(f"Using model from memory: {model_key}")
                    
                elif 'model_state' in model_info:
                    # Recreate model from state dict
                    print(f"Recreating model from state: {model_key}")
                    
                    # Get model configuration
                    model_type = model_info.get('model_type', 'CNN_DIST')
                    distance_layer = model_info.get('distance_layer', 'EuclideanDistLayer')
                    
                    # Get number of classes for this dataset
                    if hasattr(config, 'dataset_num_classes') and dataset in config['dataset_num_classes']:
                        num_classes = config['dataset_num_classes'][dataset]
                    else:
                        num_classes = 100 if dataset == 'CIFAR100' else 10
                    
                    # Create model factory
                    if model_type in ['MLP_DIST', 'CNN_DIST', 'ViT_DIST', 'PVT_DIST', 'VGG16_DIST', 'ResNet50_DIST']:
                        model_factory = ModelFactory(
                            model_type, dataset, config, 
                            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                            distance_layer, num_classes=num_classes
                        )
                    else:
                        model_factory = ModelFactory(
                            model_type, dataset, config,
                            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                            num_classes=num_classes
                        )
                    
                    # Create fresh model and load state
                    model = model_factory.get_fresh_model()
                    model.load_state_dict(model_info['model_state'])
                    model.eval()
                    
                else:
                    print(f"No model or model_state found for {model_key}, skipping...")
                    continue
                
                # Add to models dictionary
                models_dict[model_key] = {
                    'model': model,
                    'model_key': model_key,
                    'model_type': model_info.get('model_type', 'CNN_DIST'),
                    'distance_type': model_info.get('distance_type', 'baseline'),
                    'distance_layer': model_info.get('distance_layer', 'EuclideanDistLayer'),
                    'dataset': dataset,
                    'test_acc': model_info.get('test_acc', 0.0),
                    'lambda': model_info.get('lambda', 0.0)
                }
                
                print(f"✓ Loaded model: {model_key} (Acc: {model_info.get('test_acc', 0.0):.3f}%)")
                
            except Exception as e:
                print(f"✗ Error loading model {model_key}: {e}")
                continue
        
        if len(models_dict) >= 2:
            # Get test loader for this dataset
            try:
                _, test_loader, _ = get_loaders(
                    dataset=dataset,
                    model_type=list(models_dict.values())[0]['model_type'],  # Use first model's type
                    batch_size=32  # Smaller batch size for interpretability
                )
                
                print(f"Comparing interpretability across {len(models_dict)} models...")
                
                # Run interpretability comparison
                comparison_results = analyzer.generate_gradcam_comparison(
                    models_dict, 
                    test_loader, 
                    num_samples=50  # Adjust based on your needs
                )
                
                # Create visualizations
                analyzer.create_comparison_visualizations(comparison_results)
                
                # Save summary
                analyzer.save_results_summary(comparison_results)
                
                print(f"✅ Interpretability analysis for {dataset} complete!")
                print(f"   Results saved to: {analyzer.interp_dir}")
                
            except Exception as e:
                print(f"✗ Error during interpretability analysis for {dataset}: {e}")
                continue
        else:
            print(f"✗ Insufficient loaded models for {dataset} ({len(models_dict)} models)")
    
    print("\n" + "="*60)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("="*60)


def load_models_for_interpretability(results_df, model_save_dir, device='cuda'):
    """
    Helper function to load saved models for interpretability analysis
    This function should be customized based on your model saving strategy
    """
    models_dict = {}
    
    # Group by unique model configurations and find best performing ones
    grouped = results_df.groupby(['dataset', 'model_type', 'distance_type', 'distance_typeLayer'])
    
    for group_key, group_df in grouped:
        dataset, model_type, distance_type, distance_layer = group_key
        
        # Find best performing model in this group
        best_row = group_df.loc[group_df['test_acc'].idxmax()]
        
        model_key = f"{dataset}_{model_type}_{distance_type}_{distance_layer}"
        
        # Construct model file path (customize this based on your saving convention)
        model_filename = f"{dataset}_{model_type}_{distance_type}_{distance_layer}_lambda{best_row['lambda']}_best.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                # You'll need to recreate the model architecture and load weights
                # This is a placeholder - implement based on your ModelFactory
                model = None  # Load your model here
                
                models_dict[model_key] = {
                    'model': model,
                    'model_key': model_key,
                    'model_type': model_type,
                    'distance_type': distance_type,
                    'distance_layer': distance_layer,
                    'dataset': dataset,
                    'test_acc': best_row['test_acc'],
                    'lambda': best_row['lambda']
                }
                
            except Exception as e:
                print(f"Error loading model {model_key}: {e}")
        else:
            print(f"Model file not found: {model_path}")
    
    return models_dict

# Additional imports for Grad-CAM

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import seaborn as sns
from scipy import stats
import json
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class InterpretabilityAnalyzer:
    """Main class for interpretability analysis and visualization"""
    
    def __init__(self, output_dir, device='cuda'):
        self.output_dir = output_dir
        self.device = device
        
        # Create interpretability output directory
        self.interp_dir = os.path.join(output_dir, "interpretability_analysis")
        os.makedirs(self.interp_dir, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        self.samples_dir = os.path.join(self.interp_dir, "individual_samples")
        self.metrics_dir = os.path.join(self.interp_dir, "aggregate_metrics")
        self.stats_dir = os.path.join(self.interp_dir, "statistical_analysis")
        
        for dir_path in [self.samples_dir, self.metrics_dir, self.stats_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_target_layer(self, model, model_type):
        """Get the appropriate target layer for Grad-CAM based on model type"""
        if 'CNN' in model_type.upper() or 'VGG' in model_type.upper() or 'RESNET' in model_type.upper():
            # For CNN-based models, use last conv layer before distance layer or classifier
            if hasattr(model, 'dist_layer'):
                # Find the layer just before dist_layer
                modules = list(model.named_modules())
                for i, (name, module) in enumerate(modules):
                    if 'dist_layer' in name and i > 0:
                        # Get the previous convolutional layer
                        for j in range(i-1, -1, -1):
                            prev_name, prev_module = modules[j]
                            if isinstance(prev_module, nn.Conv2d):
                                return prev_module
            
            # Fallback: find last conv layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    return module
                    
        elif 'VIT' in model_type.upper() or 'PVT' in model_type.upper():
            # For transformer models, use the last attention layer
            for name, module in reversed(list(model.named_modules())):
                if 'attn' in name.lower() or 'attention' in name.lower():
                    return module
        
        elif 'MLP' in model_type.upper():
            # For MLP models, use the last linear layer before classifier
            linear_layers = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    linear_layers.append(module)
            if len(linear_layers) >= 2:
                return linear_layers[-2]  # Second to last linear layer
        
        # Default fallback - find last layer that has requires_grad=True
        for module in reversed(list(model.modules())):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                return module
        
        return list(model.modules())[-2]  # Final fallback
    
    def select_diverse_samples(self, test_loader, num_samples=50, samples_per_class=None):
        """Select diverse samples for consistent comparison across models"""
        samples = []
        
        # Determine samples per class
        if samples_per_class is None:
            num_classes = len(test_loader.dataset.classes) if hasattr(test_loader.dataset, 'classes') else 10
            samples_per_class = max(1, num_samples // num_classes)
        
        class_counts = {}
        
        for batch_idx, (data, targets) in enumerate(test_loader):
            for i, (image, label) in enumerate(zip(data, targets)):
                label_item = label.item()
                
                if class_counts.get(label_item, 0) < samples_per_class:
                    samples.append({
                        'image': image,
                        'label': label_item,
                        'id': f"batch{batch_idx}_img{i}",
                        'batch_idx': batch_idx,
                        'img_idx': i
                    })
                    class_counts[label_item] = class_counts.get(label_item, 0) + 1
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
        
        return samples[:num_samples]
    
    def generate_gradcam_for_model(self, model, model_info, test_samples):
        """Generate Grad-CAM results for a single model across test samples"""
        model.eval()
        results = []
        
        # Get target layer
        target_layer = self.get_target_layer(model, model_info['model_type'])
        if target_layer is None:
            print(f"Warning: Could not find target layer for model {model_info['model_key']}")
            return results
        
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
        except Exception as e:
            print(f"Error creating GradCAM for {model_info['model_key']}: {e}")
            return results
        
        for sample in test_samples:
            try:
                image = sample['image'].to(self.device)
                true_label = sample['label']
                
                # Get model prediction
                with torch.no_grad():
                    output = model(image.unsqueeze(0))
                    predicted_class = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1).max().item()
                
                # Generate Grad-CAM
                targets = [ClassifierOutputTarget(predicted_class)]
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=targets)[0]
                
                # Compute interpretability metrics
                metrics = self.compute_interpretability_metrics(grayscale_cam)
                
                result = {
                    'sample_id': sample['id'],
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'correct_prediction': predicted_class == true_label,
                    'gradcam': grayscale_cam,
                    'original_image': image.cpu().numpy(),
                    **metrics,
                    **model_info  # Include model metadata
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {sample['id']} for model {model_info['model_key']}: {e}")
                continue
        
        return results
    
    def compute_interpretability_metrics(self, cam):
        """Compute various interpretability metrics for a CAM"""
        # Normalize CAM to [0, 1]
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_flat = cam_norm.flatten()
        
        metrics = {}
        
        # Concentration metrics
        metrics['entropy'] = -np.sum(cam_norm * np.log(cam_norm + 1e-8))
        metrics['gini_coefficient'] = self._gini_coefficient(cam_flat)
        metrics['max_activation'] = cam_norm.max()
        metrics['activation_spread'] = np.std(cam_norm)
        metrics['activation_mean'] = np.mean(cam_norm)
        
        # Sparsity metrics
        metrics['sparsity_50'] = np.sum(cam_norm > 0.5) / cam_norm.size
        metrics['sparsity_80'] = np.sum(cam_norm > 0.8) / cam_norm.size
        
        # Focus metrics
        top_10_percent = np.percentile(cam_norm, 90)
        metrics['top_10_percent_mass'] = np.sum(cam_norm[cam_norm >= top_10_percent])
        
        return metrics
    
    def _gini_coefficient(self, x):
        """Compute Gini coefficient for measuring concentration"""
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
    
    def compare_models_interpretability(self, models_dict, test_loader, num_samples=50):
        """Main function to compare interpretability across multiple models"""
        print(f"Starting interpretability comparison for {len(models_dict)} models...")
        
        # Select test samples
        test_samples = self.select_diverse_samples(test_loader, num_samples)
        print(f"Selected {len(test_samples)} diverse test samples")
        
        # Generate results for each model
        all_results = {}
        for model_key, model_info in models_dict.items():
            print(f"Processing model: {model_key}")
            model_results = self.generate_gradcam_for_model(
                model_info['model'], model_info, test_samples
            )
            all_results[model_key] = model_results
        
        # Create visualizations
        self.create_all_visualizations(all_results)
        
        # Save results summary
        self.save_results_summary(all_results)
        
        print(f"Interpretability analysis complete. Results saved to: {self.interp_dir}")
        return all_results
    
    def create_all_visualizations(self, all_results):
        """Create all visualization types"""
        print("Creating visualizations...")
        
        self.create_individual_sample_comparisons(all_results)
        self.create_aggregate_metrics_plots(all_results)
        self.create_statistical_comparisons(all_results)
        self.create_distance_layer_analysis(all_results)
        
    def create_individual_sample_comparisons(self, all_results, max_samples=10):
        """Create side-by-side comparisons for individual samples"""
        if not all_results:
            return
        
        # Get sample IDs that are common across all models
        common_samples = set.intersection(*[
            set(result['sample_id'] for result in results) 
            for results in all_results.values()
        ])
        
        common_samples = list(common_samples)[:max_samples]
        
        for sample_id in common_samples:
            self._plot_single_sample_comparison(all_results, sample_id)
    
    def _plot_single_sample_comparison(self, all_results, sample_id):
        """Plot comparison for a single sample across all models"""
        models = list(all_results.keys())
        n_models = len(models)
        
        if n_models < 2:
            return
        
        fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # Find the sample data for each model
        sample_data = {}
        for model_key in models:
            for result in all_results[model_key]:
                if result['sample_id'] == sample_id:
                    sample_data[model_key] = result
                    break
        
        true_label = None
        for col, model_key in enumerate(models):
            if model_key not in sample_data:
                continue
                
            data = sample_data[model_key]
            true_label = data['true_label']
            
            # Top row: Raw Grad-CAM
            im1 = axes[0, col].imshow(data['gradcam'], cmap='jet', alpha=0.8)
            axes[0, col].set_title(f"{model_key}\nPred: {data['predicted_class']} "
                                 f"(Conf: {data['confidence']:.3f})")
            axes[0, col].axis('off')
            
            # Bottom row: Thresholded (top 20%)
            thresh_cam = data['gradcam'].copy()
            threshold = np.percentile(thresh_cam, 80)
            thresh_cam[thresh_cam < threshold] = 0
            
            im2 = axes[1, col].imshow(thresh_cam, cmap='jet', alpha=0.8)
            axes[1, col].set_title(f"Top 20%\nEntropy: {data['entropy']:.3f}")
            axes[1, col].axis('off')
        
        plt.suptitle(f"Sample {sample_id} (True Label: {true_label})", fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.samples_dir, f"comparison_{sample_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_aggregate_metrics_plots(self, all_results):
        """Create aggregate comparison plots across all samples"""
        # Collect metrics
        metrics_data = self._collect_metrics_data(all_results)
        
        if not metrics_data:
            return
        
        # Define metrics to plot
        metrics_to_plot = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            self._plot_metric_comparison(axes[idx], metrics_data, metric)
        
        plt.suptitle('Interpretability Metrics Comparison Across Models', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.metrics_dir, 'aggregate_metrics_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _collect_metrics_data(self, all_results):
        """Collect and organize metrics data for plotting"""
        metrics_data = {}
        
        for model_key, results in all_results.items():
            if not results:  # Skip empty results
                continue
                
            metrics_data[model_key] = {
                'entropy': [r['entropy'] for r in results],
                'gini_coefficient': [r['gini_coefficient'] for r in results],
                'max_activation': [r['max_activation'] for r in results],
                'activation_spread': [r['activation_spread'] for r in results],
                'model_type': results[0]['model_type'],
                'distance_type': results[0]['distance_type'],
                'distance_layer': results[0]['distance_layer']
            }
        
        return metrics_data
    
    def _plot_metric_comparison(self, ax, metrics_data, metric):
        """Plot comparison for a specific metric"""
        model_names = []
        metric_values = []
        colors = []
        
        for model_key, data in metrics_data.items():
            model_names.append(model_key.replace('_', '\n'))  # Line breaks for readability
            metric_values.append(data[metric])
            
            # Color by distance type
            distance_type = data['distance_type']
            if distance_type == 'baseline':
                colors.append('gray')
            elif 'euclidean' in distance_type.lower():
                colors.append('blue')
            elif 'cosine' in distance_type.lower():
                colors.append('red')
            elif 'manhattan' in distance_type.lower():
                colors.append('green')
            else:
                colors.append('orange')
        
        # Create box plot
        bp = ax.boxplot(metric_values, labels=model_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
    
    def create_statistical_comparisons(self, all_results):
        """Create statistical significance comparison plots"""
        metrics_data = self._collect_metrics_data(all_results)
        
        if len(metrics_data) < 2:
            print("Need at least 2 models for statistical comparison")
            return
        
        # Use entropy as the primary metric for statistical testing
        self._create_significance_matrix(metrics_data, 'entropy')
        
    def _create_significance_matrix(self, metrics_data, metric):
        """Create pairwise significance testing matrix"""
        model_keys = list(metrics_data.keys())
        n_models = len(model_keys)
        
        if n_models < 2:
            return
        
        p_value_matrix = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                try:
                    _, p_val = stats.mannwhitneyu(
                        metrics_data[model_keys[i]][metric], 
                        metrics_data[model_keys[j]][metric], 
                        alternative='two-sided'
                    )
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val
                except Exception as e:
                    print(f"Error in statistical test between {model_keys[i]} and {model_keys[j]}: {e}")
                    p_value_matrix[i, j] = 1.0
                    p_value_matrix[j, i] = 1.0
        
        # Plot significance matrix
        plt.figure(figsize=(max(8, n_models), max(6, n_models)))
        
        # Truncate model names for readability
        short_names = [key.split('_')[-2:] for key in model_keys]  # Take last 2 parts
        short_names = ['_'.join(parts) for parts in short_names]
        
        sns.heatmap(p_value_matrix, 
                   xticklabels=short_names, 
                   yticklabels=short_names,
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0.05,
                   cbar_kws={'label': 'p-value'})
        
        plt.title(f'Statistical Significance Matrix (p-values)\n{metric.title()} Differences Between Models')
        plt.tight_layout()
        
        save_path = os.path.join(self.stats_dir, f'significance_matrix_{metric}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_distance_layer_analysis(self, all_results):
        """Create analysis specific to distance layer types"""
        # Group results by distance layer type
        layer_groups = {}
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            layer_type = results[0]['distance_layer']
            if layer_type not in layer_groups:
                layer_groups[layer_type] = {
                    'entropy': [], 'gini_coefficient': [], 'accuracy': []
                }
            
            for result in results:
                layer_groups[layer_type]['entropy'].append(result['entropy'])
                layer_groups[layer_type]['gini_coefficient'].append(result['gini_coefficient'])
                layer_groups[layer_type]['accuracy'].append(float(result['correct_prediction']))
        
        if len(layer_groups) < 2:
            print("Need at least 2 different distance layer types for comparison")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['entropy', 'gini_coefficient', 'accuracy']
        metric_labels = [
            'Entropy\n(Lower = More Focused)', 
            'Gini Coefficient\n(Higher = More Concentrated)', 
            'Accuracy'
        ]
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            layer_names = list(layer_groups.keys())
            values = [layer_groups[layer][metric] for layer in layer_names]
            
            bp = axes[idx].boxplot(values, labels=layer_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            axes[idx].set_title(label)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Distance Layer Type Comparison', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.metrics_dir, 'distance_layer_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results_summary(self, all_results):
        """Save a summary of results to JSON"""
        summary = {}
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            # Aggregate statistics
            metrics = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
            model_summary = {
                'model_type': results[0]['model_type'],
                'distance_type': results[0]['distance_type'],
                'distance_layer': results[0]['distance_layer'],
                'num_samples': len(results),
                'accuracy': np.mean([r['correct_prediction'] for r in results]),
                'metrics': {}
            }
            
            for metric in metrics:
                values = [r[metric] for r in results]
                model_summary['metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            summary[model_key] = model_summary
        
        # Save to JSON
        summary_path = os.path.join(self.interp_dir, 'interpretability_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results summary saved to: {summary_path}")


# Simple alias for backward compatibility
# InterpretabilityAnalyzer = InterpretabilityEvaluator


def load_models_for_interpretability(results_df, model_save_dir, device='cuda'):
    """
    Helper function to load saved models for interpretability analysis
    This function should be customized based on your model saving strategy
    """
    models_dict = {}
    
    # Group by unique model configurations and find best performing ones
    grouped = results_df.groupby(['dataset', 'model_type', 'distance_type', 'distance_typeLayer'])
    
    for group_key, group_df in grouped:
        dataset, model_type, distance_type, distance_layer = group_key
        
        # Find best performing model in this group
        best_row = group_df.loc[group_df['test_acc'].idxmax()]
        
        model_key = f"{dataset}_{model_type}_{distance_type}_{distance_layer}"
        
        # Construct model file path (customize this based on your saving convention)
        model_filename = f"{dataset}_{model_type}_{distance_type}_{distance_layer}_lambda{best_row['lambda']}_best.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                # You'll need to recreate the model architecture and load weights
                # This is a placeholder - implement based on your ModelFactory
                model = None  # Load your model here
                
                models_dict[model_key] = {
                    'model': model,
                    'model_key': model_key,
                    'model_type': model_type,
                    'distance_type': distance_type,
                    'distance_layer': distance_layer,
                    'dataset': dataset,
                    'test_acc': best_row['test_acc'],
                    'lambda': best_row['lambda']
                }
                
            except Exception as e:
                print(f"Error loading model {model_key}: {e}")
        else:
            print(f"Model file not found: {model_path}")
    
    return models_dict

import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import seaborn as sns
from scipy import stats
import json

class InterpretabilityEvaluator:
    """Comprehensive interpretability evaluation using Grad-CAM"""
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.interpretability_results = []
        
        # Create interpretability output directory
        self.interp_dir = os.path.join(output_dir, "interpretability_analysis")
        os.makedirs(self.interp_dir, exist_ok=True)
        
    def get_target_layer(self, model, model_type):
        """Get the appropriate target layer for Grad-CAM based on model type"""
        if 'CNN' in model_type or 'VGG' in model_type or 'ResNet' in model_type:
            # For CNN-based models, use last conv layer before distance layer or classifier
            if hasattr(model, 'dist_layer'):
                # Find the layer just before dist_layer
                modules = list(model.named_modules())
                for i, (name, module) in enumerate(modules):
                    if 'dist_layer' in name and i > 0:
                        # Get the previous convolutional layer
                        for j in range(i-1, -1, -1):
                            prev_name, prev_module = modules[j]
                            if isinstance(prev_module, nn.Conv2d):
                                return prev_module
            # Fallback: find last conv layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    return module
                    
        elif 'ViT' in model_type or 'PVT' in model_type:
            # For transformer models, use the last attention layer
            for name, module in reversed(list(model.named_modules())):
                if 'attn' in name.lower() or 'attention' in name.lower():
                    return module
                    
        # Default fallback
        return list(model.modules())[-2]  # Second to last layer
    
    def compute_localization_metrics(self, cam, ground_truth_mask=None):
        """Compute interpretability metrics for the CAM"""
        metrics = {}
        
        # Normalize CAM to [0, 1]
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Concentration metrics
        metrics['entropy'] = -np.sum(cam_norm * np.log(cam_norm + 1e-8))
        metrics['gini_coefficient'] = self._gini_coefficient(cam_norm.flatten())
        metrics['max_activation'] = cam_norm.max()
        metrics['activation_spread'] = np.std(cam_norm)
        
        # Localization quality (if ground truth available)
        if ground_truth_mask is not None:
            gt_norm = (ground_truth_mask > 0).astype(float)
            metrics['intersection_over_union'] = self._compute_iou(cam_norm > 0.5, gt_norm)
            metrics['pointing_game_acc'] = float(cam_norm[gt_norm.argmax()] > 0.5)
        
        return metrics
    
    def _gini_coefficient(self, x):
        """Compute Gini coefficient for measuring concentration"""
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
    
    def _compute_iou(self, pred_mask, gt_mask):
        """Compute Intersection over Union"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / (union + 1e-8)
    
    def generate_gradcam_comparison(self, models_dict, test_loader, num_samples=50):
        """Generate Grad-CAM visualizations for model comparison"""
        
        # Select diverse test samples
        test_samples = self._select_diverse_samples(test_loader, num_samples)
        
        comparison_results = []
        
        for sample_idx, (image, true_label, image_id) in enumerate(test_samples):
            sample_results = {
                'sample_id': image_id,
                'true_label': true_label,
                'models': {}
            }
            
            for model_key, model_info in models_dict.items():
                model = model_info['model']
                model_type = model_info['model_type']
                distance_type = model_info['distance_type']
                distance_layer = model_info['distance_layer']
                
                try:
                    # Get model prediction
                    model.eval()
                    with torch.no_grad():
                        output = model(image.unsqueeze(0).to(device))
                        predicted_class = output.argmax(dim=1).item()
                        confidence = torch.softmax(output, dim=1).max().item()
                    
                    # Generate Grad-CAM
                    target_layer = self.get_target_layer(model, model_type)
                    if target_layer is None:
                        continue
                        
                    cam = GradCAM(model=model, target_layers=[target_layer])
                    targets = [ClassifierOutputTarget(predicted_class)]
                    
                    grayscale_cam = cam(input_tensor=image.unsqueeze(0).to(device), 
                                      targets=targets)[0]
                    
                    # Compute interpretability metrics
                    interp_metrics = self.compute_localization_metrics(grayscale_cam)
                    
                    # Store results
                    sample_results['models'][model_key] = {
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'correct_prediction': predicted_class == true_label,
                        'gradcam': grayscale_cam,
                        'model_type': model_type,
                        'distance_type': distance_type,
                        'distance_layer': distance_layer,
                        **interp_metrics
                    }
                    
                except Exception as e:
                    print(f"Error processing {model_key} on sample {sample_idx}: {e}")
                    continue
            
            comparison_results.append(sample_results)
        
        return comparison_results
    
    def _select_diverse_samples(self, test_loader, num_samples):
        """Select diverse samples for consistent comparison"""
        samples = []
        samples_per_class = max(1, num_samples // len(test_loader.dataset.classes))
        class_counts = {}
        
        for batch_idx, (data, targets) in enumerate(test_loader):
            for i, (image, label) in enumerate(zip(data, targets)):
                label_item = label.item()
                if class_counts.get(label_item, 0) < samples_per_class:
                    samples.append((image, label_item, f"batch{batch_idx}_img{i}"))
                    class_counts[label_item] = class_counts.get(label_item, 0) + 1
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
        
        return samples[:num_samples]
    
    def create_comparison_visualizations(self, comparison_results, save_individual=True):
        """Create comprehensive visualization comparing models"""
        
        # 1. Individual sample comparisons
        if save_individual:
            self._create_individual_sample_plots(comparison_results)
        
        # 2. Aggregate metric comparisons
        self._create_aggregate_metric_plots(comparison_results)
        
        # 3. Statistical significance tests
        self._create_statistical_comparison_plots(comparison_results)
        
        # 4. Distance layer specific analysis
        self._create_distance_layer_analysis(comparison_results)
    
    def _create_individual_sample_plots(self, comparison_results):
        """Create side-by-side Grad-CAM comparisons for individual samples"""
        
        for sample_idx, sample_data in enumerate(comparison_results[:10]):  # Limit for space
            models = sample_data['models']
            if len(models) < 2:
                continue
                
            n_models = len(models)
            fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
            if n_models == 1:
                axes = axes.reshape(2, 1)
            
            for col, (model_key, model_data) in enumerate(models.items()):
                # Original activation
                axes[0, col].imshow(model_data['gradcam'], cmap='jet', alpha=0.7)
                axes[0, col].set_title(f"{model_key}\nPred: {model_data['predicted_class']} "
                                     f"(Conf: {model_data['confidence']:.3f})")
                axes[0, col].axis('off')
                
                # Thresholded activation (top 20%)
                thresh_cam = model_data['gradcam'].copy()
                thresh_cam[thresh_cam < np.percentile(thresh_cam, 80)] = 0
                axes[1, col].imshow(thresh_cam, cmap='jet', alpha=0.7)
                axes[1, col].set_title(f"Top 20% Activations\nEntropy: {model_data['entropy']:.3f}")
                axes[1, col].axis('off')
            
            plt.suptitle(f"Sample {sample_data['sample_id']} (True: {sample_data['true_label']})")
            plt.tight_layout()
            plt.savefig(os.path.join(self.interp_dir, f"gradcam_comparison_sample_{sample_idx}.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def _create_aggregate_metric_plots(self, comparison_results):
        """Create aggregate comparison plots across all samples"""
        
        # Collect all metrics
        all_metrics = {}
        for sample_data in comparison_results:
            for model_key, model_data in sample_data['models'].items():
                if model_key not in all_metrics:
                    all_metrics[model_key] = {
                        'entropy': [], 'gini_coefficient': [], 'max_activation': [],
                        'activation_spread': [], 'accuracy': [], 'confidence': [],
                        'model_type': model_data['model_type'],
                        'distance_type': model_data['distance_type'],
                        'distance_layer': model_data['distance_layer']
                    }
                
                all_metrics[model_key]['entropy'].append(model_data['entropy'])
                all_metrics[model_key]['gini_coefficient'].append(model_data['gini_coefficient'])
                all_metrics[model_key]['max_activation'].append(model_data['max_activation'])
                all_metrics[model_key]['activation_spread'].append(model_data['activation_spread'])
                all_metrics[model_key]['accuracy'].append(float(model_data['correct_prediction']))
                all_metrics[model_key]['confidence'].append(model_data['confidence'])
        
        # Create comparison plots
        metrics_to_plot = ['entropy', 'gini_coefficient', 'max_activation', 'activation_spread']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            model_names = []
            metric_values = []
            colors = []
            
            for model_key, metrics in all_metrics.items():
                model_names.append(model_key)
                metric_values.append(metrics[metric])
                # Color by distance type
                if metrics['distance_type'] == 'baseline':
                    colors.append('gray')
                elif 'euclidean' in metrics['distance_type']:
                    colors.append('blue')
                elif 'cosine' in metrics['distance_type']:
                    colors.append('red')
                else:
                    colors.append('green')
            
            # Box plot
            bp = axes[idx].boxplot(metric_values, labels=model_names, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Interpretability Metrics Comparison Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(self.interp_dir, 'aggregate_metrics_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_comparison_plots(self, comparison_results):
        """Create statistical significance comparison plots"""
        
        # Collect metrics for statistical testing
        model_metrics = {}
        for sample_data in comparison_results:
            for model_key, model_data in sample_data['models'].items():
                if model_key not in model_metrics:
                    model_metrics[model_key] = []
                model_metrics[model_key].append(model_data['entropy'])  # Use entropy as main metric
        
        # Create pairwise comparison matrix
        model_keys = list(model_metrics.keys())
        n_models = len(model_keys)
        p_value_matrix = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                _, p_val = stats.mannwhitneyu(model_metrics[model_keys[i]], 
                                            model_metrics[model_keys[j]], 
                                            alternative='two-sided')
                p_value_matrix[i, j] = p_val
                p_value_matrix[j, i] = p_val
        
        # Plot significance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(p_value_matrix, 
                   xticklabels=model_keys, 
                   yticklabels=model_keys,
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0.05)
        plt.title('Statistical Significance Matrix (p-values)\nEntropy Differences Between Models')
        plt.tight_layout()
        plt.savefig(os.path.join(self.interp_dir, 'statistical_significance_matrix.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_distance_layer_analysis(self, comparison_results):
        """Create analysis specific to distance layer types"""
        
        # Group by distance layer type
        layer_groups = {}
        for sample_data in comparison_results:
            for model_key, model_data in sample_data['models'].items():
                layer_type = model_data['distance_layer']
                if layer_type not in layer_groups:
                    layer_groups[layer_type] = {'entropy': [], 'gini': [], 'accuracy': []}
                
                layer_groups[layer_type]['entropy'].append(model_data['entropy'])
                layer_groups[layer_type]['gini'].append(model_data['gini_coefficient'])
                layer_groups[layer_type]['accuracy'].append(float(model_data['correct_prediction']))
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['entropy', 'gini', 'accuracy']
        metric_labels = ['Entropy (Lower = More Focused)', 'Gini Coefficient (Higher = More Concentrated)', 'Accuracy']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            layer_names = list(layer_groups.keys())
            values = [layer_groups[layer][metric] for layer in layer_names]
            
            bp = axes[idx].boxplot(values, labels=layer_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            axes[idx].set_title(label)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Distance Layer Type Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(self.interp_dir, 'distance_layer_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


# Modified run_experiment function with integrated interpretability analysis
def run_experiment_with_interpretability(config):
    """Enhanced experiment runner with interpretability evaluation"""
    
    # Run original experiments
    results_df = run_experiment(config)
    
    # Initialize interpretability evaluator
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    interp_evaluator = InterpretabilityEvaluator(config, output_dir)
    
    print("\n" + "="*50)
    print("STARTING INTERPRETABILITY ANALYSIS")
    print("="*50)
    
    # For each unique combination, load best model and evaluate interpretability
    best_models = {}
    
    for dataset in config['datasets']:
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        # Get test loader for this dataset
        _, test_loader, _ = get_loaders(
            dataset=dataset, 
            model_type=config['model_types'][0],  # Use first model type for loader config
            batch_size=32  # Smaller batch for interpretability
        )
        
        # Find best model for each configuration
        for _, group in dataset_results.groupby(['model_type', 'distance_type', 'distance_typeLayer']):
            best_row = group.loc[group['test_acc'].idxmax()]
            
            model_key = f"{best_row['dataset']}_{best_row['model_type']}_{best_row['distance_type']}_{best_row['distance_typeLayer']}"
            
            # Recreate the best model (you'll need to implement model recreation logic)
            # This is a placeholder - you'll need to adapt based on your model saving strategy
            try:
                # You might want to save models during training and load them here
                # For now, this is a conceptual framework
                
                model_info = {
                    'model': None,  # Load your saved model here
                    'model_type': best_row['model_type'],
                    'distance_type': best_row['distance_type'], 
                    'distance_layer': best_row['distance_typeLayer'],
                    'test_acc': best_row['test_acc']
                }
                
                best_models[model_key] = model_info
                
            except Exception as e:
                print(f"Could not load model {model_key}: {e}")
                continue
        
        # Generate interpretability comparison for this dataset
        if len(best_models) >= 2:  # Need at least 2 models to compare
            print(f"Analyzing interpretability for {dataset}...")
            
            comparison_results = interp_evaluator.generate_gradcam_comparison(
                best_models, test_loader, num_samples=30
            )
            
            interp_evaluator.create_comparison_visualizations(comparison_results)
            
            print(f"Interpretability analysis saved to: {interp_evaluator.interp_dir}")
    
    return results_df'''