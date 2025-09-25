import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from collections import defaultdict
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path so we can import from models directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate GradCAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.numpy()
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

class ModelGradCAMComparator:
    def __init__(self, saved_models_path="../saved_models"):
        self.saved_models_path = Path(saved_models_path)
        self.models_data = defaultdict(lambda: defaultdict(list))
        self.distance_types = []
        
        # Dataset configurations - matching your exact training normalization
        self.dataset_configs = {
            'CIFAR10': {
                'num_classes': 10,
                'input_size': (32, 32),
                'channels': 3,
                # Different normalization per architecture
                'normalization': {
                    'CNN': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
                    'PVT': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
                    'ResNet50': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
                }
            },
            'CIFAR100': {
                'num_classes': 100,
                'input_size': (32, 32),
                'channels': 3,
                'normalization': {
                    'CNN': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
                    'PVT': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
                    'ResNet50': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}
                }
            },
            'MNIST': {
                'num_classes': 10,
                'input_size': (28, 28),
                'channels': 1,
                'normalization': {
                    'CNN': {'mean': [0.5], 'std': [0.5]},
                    'PVT': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'convert_to_rgb': True},
                    'ResNet50': {'mean': [0.1307], 'std': [0.3081]}
                }
            }
        }
        
    def discover_models(self):
        """Discover all model files in the directory structure"""
        print("Discovering models...")
        
        for dataset_dir in self.saved_models_path.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name.upper()
            print(f"Found dataset: {dataset_name}")
            
            for arch_dir in dataset_dir.iterdir():
                if not arch_dir.is_dir():
                    continue
                    
                arch_name = arch_dir.name
                print(f"  Found architecture: {arch_name}")
                
                for model_file in arch_dir.glob("*.pth"):
                    # Parse distance type from filename
                    filename = model_file.name
                    
                    # Expected format: DATASET_ARCH_baseline_DISTANCE_lambda0.0_best.pth
                    parts = filename.split('_')
                    
                    # Find the distance type after "baseline_"
                    try:
                        baseline_idx = parts.index('baseline')
                        
                        if baseline_idx + 1 < len(parts):
                            distance_type = parts[baseline_idx + 1]
                            
                            # Handle special cases with variants
                            if distance_type in ['cosine', 'minkowski', 'hamming', 'chebyshev', 'canberra', 'bray', 'mahalanobis']:
                                # Check if there's a variant (e.g., cosine_stable, minkowski_p1.5)
                                if baseline_idx + 2 < len(parts) and not parts[baseline_idx + 2].startswith('lambda'):
                                    variant = parts[baseline_idx + 2]
                                    distance_type = f"{distance_type}_{variant}"
                            elif distance_type == 'baseline':
                                # This is the baseline model (no distance layer)
                                distance_type = 'baseline'
                            
                        else:
                            print(f"    Warning: Could not parse distance type from {filename}")
                            continue
                    except ValueError:
                        print(f"    Warning: 'baseline' not found in filename {filename}")
                        continue
                    
                    model_info = {
                        'path': model_file,
                        'dataset': dataset_name,
                        'architecture': arch_name,
                        'distance_type': distance_type,
                        'filename': filename
                    }
                    
                    self.models_data[dataset_name][arch_name].append(model_info)
                    
                    if distance_type not in self.distance_types:
                        self.distance_types.append(distance_type)
        
        print(f"\nDiscovered distance types: {self.distance_types}")
        return self.models_data
    
    def get_target_layer_name(self, model, architecture):
        """Get the target layer name for GradCAM based on your specific architectures"""
        if architecture == 'CNN':
            # For SimpleCNN: target conv2 (the last conv layer before flattening)
            return 'conv2'
        
        elif architecture == 'ResNet50':
            # For ResNet50Wrapper: target the last conv layer in layer4
            return 'model.layer4.2.conv3'  # Last conv in the last ResNet block
        
        elif architecture == 'PVT':
            # For PVTWrapper: target the last layer of the backbone
            # This might need adjustment based on the exact PVT model structure
            return 'backbone.norm'  # or 'backbone.stages.3' - may need to check actual structure
        
        else:
            # Fallback: find the last conv layer
            conv_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    conv_layers.append(name)
            
            return conv_layers[-1] if conv_layers else None
    
    def create_model_architecture(self, architecture, dataset_name, distance_type):
        """Create model architecture using your existing ModelFactory"""
        # Import your existing utilities
        from utils.model_factory import ModelFactory
        from regular_config import regular_config
        
        config = regular_config  # Use your actual config
        
        # Map architecture names to your model types (NO _DIST suffixes)
        if architecture == 'CNN':
            model_type = 'CNN'
        elif architecture == 'ResNet50':
            model_type = 'ResNet50'
        elif architecture == 'PVT':
            model_type = 'PVT'
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Get number of classes for this dataset
        if 'dataset_num_classes' in config:
            num_classes = config['dataset_num_classes'][dataset_name]
        else:
            num_classes = 100 if dataset_name == 'CIFAR100' else 10
        
        # Create device (use CPU for GradCAM to avoid memory issues)
        device = torch.device('cpu')
        
        # Handle baseline vs distance layer models - use EXACT names from your config
        if distance_type == 'baseline':
            distance_layer_type = 'baseline'
        else:
            # Map parsed distance types to your exact config distance_layer_types
            distance_layer_type = distance_type  # Start with the parsed type
            
            # Handle special mappings for compound types
            if distance_type.startswith('bray_curtis'):
                # bray_curtis_standard -> bray_curtis_standard (keep as is)
                pass  
            elif distance_type.startswith('minkowski_p'):
                # minkowski_p1.5 -> minkowski_p1.5 (keep as is)
                pass
            elif distance_type == 'cosine_stable':
                distance_layer_type = 'cosine_stable'
            elif distance_type == 'cosine_unstable':
                distance_layer_type = 'cosine_unstable'
            elif distance_type.startswith('hamming_'):
                # hamming_soft -> hamming_soft (keep as is)
                pass
            elif distance_type.startswith('chebyshev_'):
                # chebyshev_standard -> chebyshev_standard (keep as is)
                pass
            elif distance_type.startswith('canberra_'):
                # canberra_standard -> canberra_standard (keep as is)
                pass
            elif distance_type.startswith('mahalanobis_'):
                # mahalanobis_standard -> mahalanobis_standard (keep as is)
                # mahalanobis_diagonal -> mahalanobis_diagonal (keep as is)
                # mahalanobis_cholesky -> mahalanobis_cholesky (keep as is)
                pass
            elif distance_type == 'euclidean':
                distance_layer_type = 'euclidean'
            elif distance_type == 'manhattan':
                distance_layer_type = 'manhattan'
            else:
                # If we don't recognize it, try to use it as-is
                print(f"    Warning: Unknown distance type {distance_type}, using as-is")
                distance_layer_type = distance_type
        
        # Create ModelFactory and get model
        try:
            model_factory = ModelFactory(
                model_type=model_type,
                dataset=dataset_name,
                config=config,
                device=device,
                distance_typeLayer=distance_layer_type,
                num_classes=num_classes
            )
            
            model = model_factory.get_fresh_model()
            return model
            
        except Exception as e:
            print(f"    Error creating model with ModelFactory: {e}")
            print(f"    Attempted: model_type={model_type}, distance_layer={distance_layer_type}")
            return None
    
    def load_model(self, model_path, architecture, dataset_name, distance_type):
        """Load model with weights using your exact model structure"""
        import time
        
        try:
            # Get dataset configuration
            config = self.dataset_configs[dataset_name]
            
            # Create model architecture
            model = self.create_model_architecture(architecture, dataset_name, distance_type)
            
            if model is None:
                return None
            
            # Load weights
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Debug: Print model structure vs state dict keys
            print(f"    Model layers: {list(model.named_modules())[:5]}...")  # First 5 layers
            print(f"    State dict keys: {list(state_dict.keys())[:5]}...")  # First 5 keys
            
            # Handle the final_layer vs dist_layer mismatch
            model_keys = set(model.state_dict().keys())
            saved_keys = set(state_dict.keys())
            
            # Fix the mismatch: if saved model has final_layer but created model expects dist_layer
            if distance_type != 'baseline':
                # For distance layer models, copy final_layer weights to dist_layer
                final_layer_keys = [k for k in saved_keys if k.startswith('final_layer.')]
                dist_layer_keys = [k for k in model_keys if k.startswith('dist_layer.')]
                
                if final_layer_keys and dist_layer_keys:
                    print(f"    Mapping final_layer -> dist_layer for distance model")
                    # Copy final_layer weights to dist_layer in state dict
                    for final_key in final_layer_keys:
                        dist_key = final_key.replace('final_layer.', 'dist_layer.')
                        if dist_key in model_keys:
                            state_dict[dist_key] = state_dict[final_key]
                    
                    # Remove final_layer keys that don't exist in the created model
                    for final_key in final_layer_keys:
                        if final_key not in model_keys:
                            del state_dict[final_key]
            
            missing_in_model = set(state_dict.keys()) - model_keys
            missing_in_saved = model_keys - set(state_dict.keys())
            
            if missing_in_model:
                print(f"    Keys in saved model but not in created model: {list(missing_in_model)[:3]}...")
            if missing_in_saved:
                print(f"    Keys in created model but not in saved model: {list(missing_in_saved)[:3]}...")
            
            model.load_state_dict(state_dict)
            model.eval()
            
            print(f"    Successfully loaded model!")
            time.sleep(2)  # Added delay for observation
            
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            time.sleep(2)  # Added delay for observation
            return None
    
    def get_sample_data(self, dataset_name, architecture, num_samples=10):
        """Get sample data for GradCAM analysis with correct normalization"""
        config = self.dataset_configs[dataset_name]
        norm_config = config['normalization'][architecture]
        
        # Handle MNIST PVT special case (convert to RGB)
        if dataset_name == 'MNIST' and architecture == 'PVT':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(norm_config['mean'], norm_config['std'])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_config['mean'], norm_config['std'])
            ])
        
        # Load dataset
        if dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'MNIST':
            dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Get random samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=1, shuffle=False)
        
        return dataloader, dataset.classes if hasattr(dataset, 'classes') else None
    
    def generate_gradcam_comparison(self, dataset_name, architecture, output_dir="gradcam_results"):
        """Generate GradCAM comparisons for all distance variants"""
        print(f"\nGenerating GradCAM for {dataset_name} - {architecture}")
        
        models_info = self.models_data[dataset_name][architecture]
        if len(models_info) < 2:
            print(f"Not enough models to compare (found {len(models_info)})")
            return None
        
        os.makedirs(f"{output_dir}/{dataset_name}_{architecture}", exist_ok=True)
        
        # Load all models
        models = {}
        gradcams = {}
        
        for model_info in models_info:
            distance_type = model_info['distance_type']
            print(f"  Loading {distance_type} model...")
            
            model = self.load_model(
                model_info['path'], 
                architecture, 
                dataset_name, 
                distance_type
            )
            
            if model is None:
                continue
            
            # Get target layer for GradCAM
            target_layer = self.get_target_layer_name(model, architecture)
            if target_layer is None:
                print(f"    Warning: Could not find target layer for {distance_type}")
                continue
            
            models[distance_type] = model
            gradcams[distance_type] = GradCAM(model, target_layer)
            print(f"    Using target layer: {target_layer}")
        
        if not models:
            print("No models successfully loaded")
            return None
        
        # Get sample data
        dataloader, class_names = self.get_sample_data(dataset_name, architecture, num_samples=5)
        
        # Generate GradCAM for each sample
        sample_results = []
        
        for sample_idx, (input_tensor, target) in enumerate(dataloader):
            print(f"  Processing sample {sample_idx + 1}...")
            
            sample_result = {
                'sample_idx': sample_idx,
                'true_label': target.item(),
                'gradcams': {},
                'predictions': {}
            }
            
            # Generate GradCAM for each model
            for distance_type, gradcam in gradcams.items():
                try:
                    # Get prediction
                    with torch.no_grad():
                        output = models[distance_type](input_tensor)
                        predicted_class = output.argmax(dim=1).item()
                        confidence = F.softmax(output, dim=1).max().item()
                    
                    # Generate GradCAM
                    cam = gradcam.generate_cam(input_tensor, predicted_class)
                    
                    sample_result['gradcams'][distance_type] = cam
                    sample_result['predictions'][distance_type] = {
                        'predicted_class': predicted_class,
                        'confidence': confidence
                    }
                    
                except Exception as e:
                    print(f"    Error generating GradCAM for {distance_type}: {e}")
            
            sample_results.append(sample_result)
            
            # Create visualization
            self._create_gradcam_visualization(
                input_tensor, sample_result, dataset_name, architecture, 
                sample_idx, class_names, output_dir
            )
        
        # Cleanup
        for gradcam in gradcams.values():
            gradcam.cleanup()
        
        return sample_results
    
    def _create_gradcam_visualization(self, input_tensor, sample_result, dataset_name, 
                                    architecture, sample_idx, class_names, output_dir):
        """Create GradCAM visualization comparing all distance types"""
        
        gradcams = sample_result['gradcams']
        predictions = sample_result['predictions']
        
        if not gradcams:
            return
        
        num_models = len(gradcams)
        fig, axes = plt.subplots(2, num_models + 1, figsize=(4 * (num_models + 1), 8))
        
        if num_models == 1:
            axes = axes.reshape(2, -1)
        
        # Original image - handle different tensor shapes more robustly
        input_squeezed = input_tensor.squeeze()
        
        # Handle different input shapes
        if input_squeezed.ndim == 2:  # (H, W) - single channel like MNIST
            original_img = input_squeezed.numpy()
            original_img = np.expand_dims(original_img, axis=2)  # Make (H, W, 1)
        elif input_squeezed.ndim == 3:  # (C, H, W) 
            if input_squeezed.shape[0] == 1:  # Single channel (1, H, W)
                original_img = input_squeezed.squeeze(0).numpy()  # (H, W)
                original_img = np.expand_dims(original_img, axis=2)  # (H, W, 1)
            else:  # Multi-channel (C, H, W)
                original_img = input_squeezed.permute(1, 2, 0).numpy()  # (H, W, C)
        else:
            raise ValueError(f"Unexpected input tensor shape: {input_squeezed.shape}")
        
        print(f"Debug: Original image shape after processing: {original_img.shape}")
        
        # Denormalize for display
        config = self.dataset_configs[dataset_name]
        norm_config = config['normalization'][architecture]
        mean = np.array(norm_config['mean'])
        std = np.array(norm_config['std'])
        
        if len(mean) == 1:  # MNIST (except PVT)
            if architecture == 'PVT':
                # PVT MNIST is already RGB
                for i in range(3):
                    original_img[:, :, i] = original_img[:, :, i] * std[i] + mean[i]
            else:
                # Regular MNIST
                original_img = original_img * std[0] + mean[0]
                original_img = np.repeat(original_img, 3, axis=2)  # Convert to RGB for display
        else:
            # CIFAR datasets or MNIST PVT
            for i in range(3):
                original_img[:, :, i] = original_img[:, :, i] * std[i] + mean[i]
        
        original_img = np.clip(original_img, 0, 1)
        
        # Display original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title(f'Original\nTrue: {sample_result["true_label"]}')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Display GradCAM for each model
        for idx, (distance_type, cam) in enumerate(gradcams.items()):
            col_idx = idx + 1
            
            pred_info = predictions[distance_type]
            
            # Resize CAM to input size
            cam_resized = cv2.resize(cam, original_img.shape[:2])
            
            # Create heatmap overlay
            heatmap = plt.cm.jet(cam_resized)[:, :, :3]
            overlay = 0.6 * original_img + 0.4 * heatmap
            
            # Display GradCAM
            axes[0, col_idx].imshow(overlay)
            axes[0, col_idx].set_title(f'{distance_type}\nPred: {pred_info["predicted_class"]}\nConf: {pred_info["confidence"]:.3f}')
            axes[0, col_idx].axis('off')
            
            # Display pure heatmap
            axes[1, col_idx].imshow(cam_resized, cmap='jet')
            axes[1, col_idx].set_title(f'{distance_type} Heatmap')
            axes[1, col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataset_name}_{architecture}/sample_{sample_idx}_gradcam.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_gradcam_similarities(self, sample_results):
        """Analyze similarities between GradCAM heatmaps"""
        similarities = []
        
        for sample_result in sample_results:
            gradcams = sample_result['gradcams']
            distance_types = list(gradcams.keys())
            
            for i in range(len(distance_types)):
                for j in range(i + 1, len(distance_types)):
                    dist1, dist2 = distance_types[i], distance_types[j]
                    cam1, cam2 = gradcams[dist1], gradcams[dist2]
                    
                    # Compute similarity metrics
                    correlation = np.corrcoef(cam1.flatten(), cam2.flatten())[0, 1]
                    
                    # Structural similarity (simplified)
                    mse = np.mean((cam1 - cam2) ** 2)
                    
                    similarities.append({
                        'sample_idx': sample_result['sample_idx'],
                        'comparison': f"{dist1}_vs_{dist2}",
                        'correlation': correlation,
                        'mse': mse
                    })
        
        return pd.DataFrame(similarities)
    
    def run_full_gradcam_analysis(self):
        """Run GradCAM analysis for specified dataset-architecture pairs"""
        self.discover_models()
        
        all_results = {}
        
        # Filter datasets and architectures based on configuration
        datasets_to_process = [d for d in self.models_data.keys() if d in DATASETS_TO_ANALYZE]
        
        print(f"Configured to analyze datasets: {DATASETS_TO_ANALYZE}")
        print(f"Configured to analyze architectures: {ARCHITECTURES_TO_ANALYZE}")
        print(f"Found datasets in saved_models: {list(self.models_data.keys())}")
        print(f"Will process: {datasets_to_process}")
        
        for dataset in datasets_to_process:
            architectures_to_process = [a for a in self.models_data[dataset].keys() if a in ARCHITECTURES_TO_ANALYZE]
            print(f"Found architectures for {dataset}: {list(self.models_data[dataset].keys())}")
            print(f"Will process architectures for {dataset}: {architectures_to_process}")
            
            for architecture in architectures_to_process:
                print(f"\n{'='*50}")
                print(f"Analyzing {dataset} - {architecture}")
                print(f"{'='*50}")
                
                try:
                    results = self.generate_gradcam_comparison(dataset, architecture)
                    if results:
                        # Analyze similarities with enhanced metrics
                        similarity_df, distance_types = self.analyze_gradcam_similarities(results)
                        
                        # Create comprehensive visualizations
                        analysis_dir = self.plot_interpretability_analysis(
                            similarity_df, distance_types, results, 
                            dataset, architecture, "gradcam_results"
                        )
                        
                        all_results[f"{dataset}_{architecture}"] = {
                            'gradcam_results': results,
                            'similarity_analysis': similarity_df,
                            'distance_types': distance_types,
                            'analysis_directory': analysis_dir
                        }
                        
                        # Save detailed similarity analysis
                        similarity_df.to_csv(f"gradcam_results/{dataset}_{architecture}_detailed_similarities.csv", index=False)
                        
                        # Create and save correlation matrix as CSV
                        corr_matrix = self.create_correlation_matrix(similarity_df, distance_types, 'correlation')
                        corr_df = pd.DataFrame(corr_matrix, index=distance_types, columns=distance_types)
                        corr_df.to_csv(f"gradcam_results/{dataset}_{architecture}_correlation_matrix.csv")
                        
                        # Print summary with matrix format
                        print(f"\n=== CORRELATION MATRIX SUMMARY ===")
                        print(f"Dataset: {dataset}, Architecture: {architecture}")
                        print(f"Distance types analyzed: {len(distance_types)}")
                        print(f"Valid correlations: {similarity_df['correlation'].notna().sum()}")
                        print(f"Analysis saved to: {analysis_dir}")
                        
                        # Print top correlations
                        valid_similarities = similarity_df.dropna(subset=['correlation'])
                        if not valid_similarities.empty:
                            top_positive = valid_similarities.nlargest(5, 'correlation')
                            top_negative = valid_similarities.nsmallest(5, 'correlation')
                            
                            print(f"\nTop 5 Positive Correlations:")
                            for _, row in top_positive.iterrows():
                                print(f"  {row['comparison']}: {row['correlation']:.3f}")
                            
                            print(f"\nTop 5 Negative Correlations:")
                            for _, row in top_negative.iterrows():
                                print(f"  {row['comparison']}: {row['correlation']:.3f}")
                
                except Exception as e:
                    print(f"Error analyzing {dataset}-{architecture}: {e}")
                    import traceback
                    traceback.print_exc()
        
        return all_results

# Example usage
if __name__ == "__main__":
    print("Starting GradCAM interpretability analysis...")
    print("Note: Make sure models/models.py is in your path or adjust the import")
    
    comparator = ModelGradCAMComparator()
    results = comparator.run_full_gradcam_analysis()
    
    print(f"\nGradCAM analysis complete! Results saved to gradcam_results/")
    print("Check the generated visualizations to compare model interpretability across different distance layers.")
    
    if results:
        print("\nSummary of comparisons:")
        for key, result in results.items():
            dataset, arch = key.split('_', 1)
            similarities = result['similarity_analysis']
            if not similarities.empty:
                avg_corr = similarities.groupby('comparison')['correlation'].mean()
                print(f"\n{dataset} - {arch}:")
                for comp, corr in avg_corr.items():
                    print(f"  {comp}: {corr:.3f} average correlation")
    else:
        print("No comparison results generated. Check your model files and file structure.")