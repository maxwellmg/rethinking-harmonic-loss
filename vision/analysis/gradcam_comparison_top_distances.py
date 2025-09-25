# Local Imports
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from utils.model_factory import ModelFactory
#from custom_config_runs.regular_config import regular_config
DATASETS_TO_ANALYZE = ['MNIST', 'CIFAR10',  'CIFAR100'] # ['MNIST', 'CIFAR10', 'CIFAR100']
ARCHITECTURES_TO_ANALYZE = ['CNN', 'ResNet50', 'PVT'] #['CNN', 'ResNet50', 'PVT']

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
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Local Imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_factory import ModelFactory
from custom_config_runs.regular_config import regular_config

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
        target_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
                target_found = True
                print(f"    Successfully registered hooks for layer: {name}")
                break
        
        if not target_found:
            print(f"    ERROR: Target layer '{self.target_layer_name}' not found in model!")
            print(f"    Available layers: {[name for name, _ in self.model.named_modules()][:5]}...")
    
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

        # Check if hooks captured data
        if self.gradients is None:
            raise ValueError(f"Gradients not captured - check if target layer '{self.target_layer_name}' exists")
        if self.activations is None:
            raise ValueError(f"Activations not captured - check if target layer '{self.target_layer_name}' exists")
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Debug: Check intermediate values
        print(f"  Gradients shape: {gradients.shape}, has NaN: {torch.isnan(gradients).any()}")
        print(f"  Activations shape: {activations.shape}, has NaN: {torch.isnan(activations).any()}")

        # Ensure we have 3D tensors [channels, height, width]
        if gradients.dim() != 3 or activations.dim() != 3:
            raise ValueError(f"Expected 3D activations/gradients [C,H,W], got gradients: {gradients.shape}, activations: {activations.shape}")
        
        if gradients.shape[1] <= 1 or gradients.shape[2] <= 1:
            raise ValueError(f"Spatial dimensions too small for meaningful GradCAM: {gradients.shape}")
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        print(f"  Weights has NaN: {torch.isnan(weights).any()}")
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        print(f"  CAM before ReLU has NaN: {torch.isnan(cam).any()}")
        
        # Apply ReLU
        cam = F.relu(cam)
        print(f"  CAM after ReLU has NaN: {torch.isnan(cam).any()}")
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        print(f"  CAM min: {cam_min}, max: {cam_max}")
        
        if cam_max == cam_min:
            print(f"  WARNING: CAM is constant (min=max={cam_max}), returning zeros")
            return np.zeros_like(cam.numpy())
        
        cam = cam - cam_min
        cam = cam / cam_max
        print(f"  Final CAM has NaN: {torch.isnan(cam).any()}")
        
        return cam.numpy()

    def generate_cam_debug(self, input_tensor, class_idx=None):
        """Generate GradCAM heatmap with detailed debugging"""
        print(f"    DEBUG: Starting GradCAM generation...")
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        print(f"    DEBUG: Model output shape: {output.shape}")
        print(f"    DEBUG: Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        print(f"    DEBUG: Target class: {class_idx}")
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        print(f"    DEBUG: Class score: {class_score.item():.4f}")
        
        class_score.backward(retain_graph=True)

        # Check if hooks captured data
        if self.gradients is None:
            raise ValueError(f"Gradients not captured - check if target layer '{self.target_layer_name}' exists")
        if self.activations is None:
            raise ValueError(f"Activations not captured - check if target layer '{self.target_layer_name}' exists")
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        print(f"    DEBUG: Gradients shape: {gradients.shape}")
        print(f"    DEBUG: Gradients range: [{gradients.min().item():.6f}, {gradients.max().item():.6f}]")
        print(f"    DEBUG: Gradients std: {gradients.std().item():.6f}")
        print(f"    DEBUG: Activations shape: {activations.shape}")
        print(f"    DEBUG: Activations range: [{activations.min().item():.6f}, {activations.max().item():.6f}]")
        print(f"    DEBUG: Activations std: {activations.std().item():.6f}")

        # Ensure we have 3D tensors [channels, height, width]
        if gradients.dim() != 3 or activations.dim() != 3:
            raise ValueError(f"Expected 3D activations/gradients [C,H,W], got gradients: {gradients.shape}, activations: {activations.shape}")
        
        if gradients.shape[1] <= 1 or gradients.shape[2] <= 1:
            raise ValueError(f"Spatial dimensions too small for meaningful GradCAM: {gradients.shape}")
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        print(f"    DEBUG: Weights shape: {weights.shape}")
        print(f"    DEBUG: Weights range: [{weights.min().item():.6f}, {weights.max().item():.6f}]")
        print(f"    DEBUG: Weights std: {weights.std().item():.6f}")
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        print(f"    DEBUG: CAM before ReLU range: [{cam.min().item():.6f}, {cam.max().item():.6f}]")
        print(f"    DEBUG: CAM before ReLU std: {cam.std().item():.6f}")
        
        # Apply ReLU
        cam = F.relu(cam)
        print(f"    DEBUG: CAM after ReLU range: [{cam.min().item():.6f}, {cam.max().item():.6f}]")
        print(f"    DEBUG: CAM after ReLU std: {cam.std().item():.6f}")
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        print(f"    DEBUG: CAM min: {cam_min.item():.6f}, max: {cam_max.item():.6f}")
        
        if cam_max == cam_min:
            print(f"    DEBUG: WARNING - CAM is constant! This explains the flat heatmap.")
            return np.zeros_like(cam.numpy())
        
        cam = cam - cam_min
        cam = cam / cam_max
        print(f"    DEBUG: Final normalized CAM range: [{cam.min().item():.6f}, {cam.max().item():.6f}]")
        
        return cam.numpy()
    
    def check_activation_dimensions(self, input_tensor):
        """Debug method to check activation dimensions at target layer"""
        self.model.eval()
        
        def debug_hook(module, input, output):
            print(f"    Activation shape at {self.target_layer_name}: {output.shape}")
            return output
        
        # Register temporary hook
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                handle = module.register_forward_hook(debug_hook)
                break
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Remove temporary hook
        handle.remove()
    
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
            return 'conv2'
        
        elif architecture == 'ResNet50':
            #return 'model.layer4.2.conv3'
            return 'model.layer4.1.conv3' 
        
        elif architecture == 'PVT':
            # For PVT, we need CONVOLUTIONAL layers that still have spatial dimensions
            # LayerNorm layers in transformers work on flattened representations
            possible_targets = [
                # Convolutional layers with spatial dimensions (in order of preference)
                'backbone.stages.2.downsample.proj',  # Conv2d layer in stage 2
                'backbone.stages.2.blocks.1.mlp.dwconv',  # Depthwise conv in stage 2
                'backbone.stages.2.blocks.0.mlp.dwconv',  # Depthwise conv in stage 2
                'backbone.stages.1.downsample.proj',  # Conv2d layer in stage 1  
                'backbone.stages.1.blocks.1.mlp.dwconv',  # Depthwise conv in stage 1
                'backbone.stages.0.blocks.1.mlp.dwconv',  # Depthwise conv in stage 0
                'backbone.patch_embed.proj',  # Initial patch embedding conv
            ]
            
            model_layer_names = [name for name, _ in model.named_modules()]
            
            for target in possible_targets:
                if target in model_layer_names:
                    print(f"    Found PVT target layer: {target}")
                    return target
            
            # Debug fallback - find all layers with spatial dimensions
            print(f"    Testing layers for spatial dimensions...")
            for name, module in model.named_modules():
                if any(keyword in name for keyword in ['stages.1', 'stages.2']) and \
                any(layer_type in name for layer_type in ['norm', 'proj', 'dwconv']):
                    print(f"    Candidate layer: {name}")
            
            # If none found, return the first reasonable option
            if 'backbone.stages.2.norm' in model_layer_names:
                return 'backbone.stages.2.norm'
            elif 'backbone.stages.1.norm' in model_layer_names:
                return 'backbone.stages.1.norm'
            
            print(f"    ERROR: No suitable target layer found for PVT")
            return None
        
        else:
            # Fallback for unknown architectures
            conv_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    conv_layers.append(name)
            
            return conv_layers[-1] if conv_layers else None
            
    def create_model_architecture(self, architecture, dataset_name, distance_type):
        """Create model architecture using your existing ModelFactory"""
        # Import your existing utilities
        
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
            time.sleep(0.5)  # Added delay for observation
            
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            time.sleep(5)  # Added delay for observation
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

        # Get sample data first to get correct input dimensions
        dataloader, class_names = self.get_sample_data(dataset_name, architecture, num_samples=1)
        sample_input, _ = next(iter(dataloader))  # Get one sample for dimension checking
        
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

            # DEBUG: Print actual model structure for PVT
            if architecture == 'PVT':
                print(f"  DEBUG: PVT model structure for {distance_type}:")
                for name, module in model.named_modules():
                    if any(keyword in name.lower() for keyword in ['backbone', 'norm', 'stage', 'conv', 'attention']):
                        print(f"    {name}: {type(module).__name__}")
                print("  ----")
            
            # Get target layer for GradCAM
            target_layer = self.get_target_layer_name(model, architecture)
            if target_layer is None:
                print(f"    Warning: Could not find target layer for {distance_type}")
                continue
            
             # Create GradCAM and check dimensions
            gradcam = GradCAM(model, target_layer)
            
            print(f"    Checking activation dimensions for {distance_type}...")
            gradcam.check_activation_dimensions(sample_input)
            
            models[distance_type] = model
            gradcams[distance_type] = gradcam
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
                    cam = gradcam.generate_cam_debug(input_tensor, predicted_class)
                    
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
        """Analyze similarities between GradCAM heatmaps with enhanced metrics"""
        similarities = []
        distance_types = set()
        
        for sample_result in sample_results:
            gradcams = sample_result['gradcams']
            distance_types.update(gradcams.keys())
            distance_type_list = list(gradcams.keys())
            
            for i in range(len(distance_type_list)):
                for j in range(i + 1, len(distance_type_list)):
                    dist1, dist2 = distance_type_list[i], distance_type_list[j]
                    cam1, cam2 = gradcams[dist1], gradcams[dist2]
                    
                    # Check if either CAM is flat (constant)
                    if cam1.std() == 0 or cam2.std() == 0:
                        print(f"  Skipping {dist1} vs {dist2}: one or both CAMs are constant")
                        continue
                    
                    # Compute multiple similarity metrics
                    correlation = np.corrcoef(cam1.flatten(), cam2.flatten())[0, 1]
                    mse = np.mean((cam1 - cam2) ** 2)
                    
                    # Structural similarity (simplified SSIM)
                    mean1, mean2 = cam1.mean(), cam2.mean()
                    var1, var2 = cam1.var(), cam2.var()
                    covar = np.mean((cam1 - mean1) * (cam2 - mean2))
                    ssim = (2 * mean1 * mean2 + 1e-6) * (2 * covar + 1e-6) / ((mean1**2 + mean2**2 + 1e-6) * (var1 + var2 + 1e-6))
                    
                    # Cosine similarity
                    cos_sim = np.dot(cam1.flatten(), cam2.flatten()) / (np.linalg.norm(cam1.flatten()) * np.linalg.norm(cam2.flatten()))
                    
                    similarities.append({
                        'sample_idx': sample_result['sample_idx'],
                        'comparison': f"{dist1}_vs_{dist2}",
                        'dist1': dist1,
                        'dist2': dist2,
                        'correlation': correlation,
                        'mse': mse,
                        'ssim': ssim,
                        'cosine_similarity': cos_sim
                    })
        
        return pd.DataFrame(similarities), sorted(list(distance_types))
    
    def create_correlation_matrix(self, similarity_df, distance_types, metric='correlation'):
        """Create correlation matrix from pairwise similarities"""
        n = len(distance_types)
        matrix = np.full((n, n), np.nan)
        
        # Fill diagonal with 1.0 (self-similarity)
        np.fill_diagonal(matrix, 1.0)
        
        # Fill upper and lower triangles
        for _, row in similarity_df.iterrows():
            if pd.isna(row[metric]):
                continue
                
            dist1, dist2 = row['dist1'], row['dist2']
            if dist1 in distance_types and dist2 in distance_types:
                i, j = distance_types.index(dist1), distance_types.index(dist2)
                matrix[i, j] = row[metric]
                matrix[j, i] = row[metric]  # Symmetric
        
        return matrix

    # Add this comprehensive visualization method:
    def plot_interpretability_analysis(self, similarity_df, distance_types, sample_results, 
                                    dataset_name, architecture, output_dir):
        """Create comprehensive interpretability analysis visualizations"""
        analysis_dir = f"{output_dir}/{dataset_name}_{architecture}_analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = self.create_correlation_matrix(similarity_df, distance_types, 'correlation')
        
        mask = np.isnan(corr_matrix)
        sns.heatmap(corr_matrix, 
                    xticklabels=distance_types, 
                    yticklabels=distance_types,
                    annot=True, 
                    fmt='.3f', 
                    cmap='RdBu_r', 
                    center=0,
                    mask=mask,
                    square=True,
                    cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title(f'GradCAM Correlation Matrix\n{dataset_name} - {architecture}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{analysis_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Multiple Similarity Metrics Comparison
        metrics = ['correlation', 'ssim', 'cosine_similarity']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(metrics):
            matrix = self.create_correlation_matrix(similarity_df, distance_types, metric)
            mask = np.isnan(matrix)
            
            sns.heatmap(matrix, 
                        xticklabels=distance_types, 
                        yticklabels=distance_types,
                        annot=True, 
                        fmt='.3f', 
                        cmap='RdBu_r', 
                        center=0,
                        mask=mask,
                        square=True,
                        ax=axes[idx],
                        cbar_kws={'label': metric.replace('_', ' ').title()})
            
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_xticklabels(distance_types, rotation=45, ha='right')
            axes[idx].set_yticklabels(distance_types, rotation=0)
        
        plt.suptitle(f'Similarity Metrics Comparison\n{dataset_name} - {architecture}')
        plt.tight_layout()
        plt.savefig(f"{analysis_dir}/similarity_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Hierarchical Clustering
        if len(distance_types) > 2:
            plt.figure(figsize=(12, 8))
            
            # Use correlation matrix for clustering
            corr_matrix = self.create_correlation_matrix(similarity_df, distance_types, 'correlation')
            
            # Convert correlation to distance (1 - correlation)
            distance_matrix = 1 - corr_matrix
            
            # Remove NaN values for clustering
            valid_indices = ~np.isnan(distance_matrix).any(axis=1)
            if valid_indices.sum() > 2:
                valid_types = [distance_types[i] for i in range(len(distance_types)) if valid_indices[i]]
                valid_distance_matrix = distance_matrix[valid_indices][:, valid_indices]
                
                # Hierarchical clustering
                condensed_distances = []
                n = len(valid_types)
                for i in range(n):
                    for j in range(i+1, n):
                        condensed_distances.append(valid_distance_matrix[i, j])
                
                if len(condensed_distances) > 0:
                    linkage_matrix = linkage(condensed_distances, method='ward')
                    dendrogram(linkage_matrix, labels=valid_types, orientation='top')
                    plt.title(f'Hierarchical Clustering of Distance Types\n{dataset_name} - {architecture}')
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Distance')
                    plt.tight_layout()
                    plt.savefig(f"{analysis_dir}/hierarchical_clustering.png", dpi=300, bbox_inches='tight')
            
            plt.close()
        
        # 4. Distribution Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Correlation distribution
        valid_corr = similarity_df['correlation'].dropna()
        if len(valid_corr) > 0:
            axes[0, 0].hist(valid_corr, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(valid_corr.mean(), color='red', linestyle='--', label=f'Mean: {valid_corr.mean():.3f}')
            axes[0, 0].set_xlabel('Correlation Coefficient')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Correlations')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # MSE distribution
        valid_mse = similarity_df['mse'].dropna()
        if len(valid_mse) > 0:
            axes[0, 1].hist(valid_mse, bins=20, alpha=0.7, edgecolor='black', color='orange')
            axes[0, 1].axvline(valid_mse.mean(), color='red', linestyle='--', label=f'Mean: {valid_mse.mean():.6f}')
            axes[0, 1].set_xlabel('Mean Squared Error')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of MSE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation vs MSE scatter
        if len(valid_corr) > 0 and len(valid_mse) > 0:
            scatter_df = similarity_df.dropna(subset=['correlation', 'mse'])
            axes[1, 0].scatter(scatter_df['correlation'], scatter_df['mse'], alpha=0.6)
            axes[1, 0].set_xlabel('Correlation Coefficient')
            axes[1, 0].set_ylabel('Mean Squared Error')
            axes[1, 0].set_title('Correlation vs MSE')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Sample consistency analysis
        sample_consistency = []
        for dist_type in distance_types:
            type_correlations = []
            for sample_result in sample_results:
                if dist_type in sample_result['gradcams']:
                    # Get correlations for this distance type across all comparisons
                    sample_corrs = similarity_df[
                        (similarity_df['sample_idx'] == sample_result['sample_idx']) &
                        ((similarity_df['dist1'] == dist_type) | (similarity_df['dist2'] == dist_type))
                    ]['correlation'].dropna()
                    if len(sample_corrs) > 0:
                        type_correlations.extend(sample_corrs.tolist())
            
            if len(type_correlations) > 0:
                sample_consistency.append({
                    'distance_type': dist_type,
                    'mean_correlation': np.mean(type_correlations),
                    'std_correlation': np.std(type_correlations)
                })
        
        if sample_consistency:
            consistency_df = pd.DataFrame(sample_consistency)
            axes[1, 1].bar(range(len(consistency_df)), consistency_df['mean_correlation'], 
                        yerr=consistency_df['std_correlation'], alpha=0.7, capsize=5)
            axes[1, 1].set_xlabel('Distance Type')
            axes[1, 1].set_ylabel('Mean Correlation ± Std')
            axes[1, 1].set_title('Consistency Across Samples')
            axes[1, 1].set_xticks(range(len(consistency_df)))
            axes[1, 1].set_xticklabels(consistency_df['distance_type'], rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Statistical Analysis\n{dataset_name} - {architecture}')
        plt.tight_layout()
        plt.savefig(f"{analysis_dir}/statistical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Summary Statistics Table
        summary_stats = []
        
        for metric in ['correlation', 'mse', 'ssim', 'cosine_similarity']:
            valid_values = similarity_df[metric].dropna()
            if len(valid_values) > 0:
                summary_stats.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Count': len(valid_values),
                    'Mean': valid_values.mean(),
                    'Std': valid_values.std(),
                    'Min': valid_values.min(),
                    'Max': valid_values.max(),
                    'Median': valid_values.median()
                })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv(f"{analysis_dir}/summary_statistics.csv", index=False)
            
            # Create a nice table visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('tight')
            ax.axis('off')
            
            table_data = summary_df.round(4).values
            table = ax.table(cellText=table_data, 
                            colLabels=summary_df.columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            plt.title(f'Summary Statistics\n{dataset_name} - {architecture}', pad=20)
            plt.savefig(f"{analysis_dir}/summary_table.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        return analysis_dir

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
                    results = self._focused(dataset, architecture)
                    if results:
                        # Analyze similarities
                        similarity_df = self.analyze_gradcam_similarities(results)
                        
                        all_results[f"{dataset}_{architecture}"] = {
                            'gradcam_results': results,
                            'similarity_analysis': similarity_df
                        }
                        
                        # Save similarity analysis
                        similarity_df.to_csv(f"gradcam_results/{dataset}_{architecture}_similarities.csv", index=False)
                        
                        # Print summary
                        if not similarity_df.empty:
                            print(f"Average correlations:")
                            avg_corr = similarity_df.groupby('comparison')['correlation'].mean()
                            for comp, corr in avg_corr.items():
                                print(f"  {comp}: {corr:.3f}")
                
                except Exception as e:
                    print(f"Error analyzing {dataset}-{architecture}: {e}")
        
        return all_results
    
    def get_top_performing_distances(self, models, dataloader, num_top=3):
        """Identify top performing distance types by average confidence across test samples"""
        print("  Evaluating model performance to identify top distances...")
        
        performance_scores = {}
        sample_count = 0
        
        # Test on a subset of samples to get confidence scores
        for input_tensor, target in dataloader:
            sample_count += 1
            if sample_count > 50:  # Limit evaluation to 50 samples for efficiency
                break
                
            for distance_type, model in models.items():
                with torch.no_grad():
                    output = model(input_tensor)
                    confidence = F.softmax(output, dim=1).max().item()
                    
                    if distance_type not in performance_scores:
                        performance_scores[distance_type] = []
                    performance_scores[distance_type].append(confidence)
        
        # Calculate average confidence for each distance type
        avg_performance = {}
        for distance_type, scores in performance_scores.items():
            avg_performance[distance_type] = np.mean(scores)
        
        # Sort by performance and get top performers (excluding euclidean which is always included)
        sorted_distances = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        
        # Always include euclidean, then get top 3 others
        top_distances = ['euclidean']
        for dist_type, score in sorted_distances:
            if dist_type != 'euclidean' and len(top_distances) < 4:
                top_distances.append(dist_type)
        
        performance_info = [(d, avg_performance.get(d, 0)) for d in top_distances]
        print(f"  Selected distances (avg confidence): {[(d, f'{score:.3f}') for d, score in performance_info]}")
        return top_distances
    
    def get_best_samples_across_all_distances(self, dataset_name, architecture, target_distances, models):
        """Get samples that perform well across ALL distance types, not just one"""
        print(f"  Finding samples with high confidence across all distances...")
        
        # Get full test dataset
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
            target_classes = list(range(10))
            class_names = dataset.classes
        elif dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
            # Select 10 representative classes for CIFAR-100
            target_classes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # One from each superclass
            class_names = [dataset.classes[i] for i in target_classes]
        elif dataset_name == 'MNIST':
            dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            target_classes = list(range(10))
            class_names = [str(i) for i in range(10)]
        
        # Collect candidates for each class
        class_candidates = {cls: [] for cls in target_classes}
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        sample_count = 0
        for input_tensor, target in dataloader:
            target_class = target.item()
            if target_class not in target_classes:
                continue
                
            # Evaluate this sample across all distances
            confidences = {}
            all_correct = True
            
            for distance_type in target_distances:
                if distance_type not in models:
                    continue
                    
                with torch.no_grad():
                    output = models[distance_type](input_tensor)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = F.softmax(output, dim=1).max().item()
                    
                    confidences[distance_type] = confidence
                    
                    # Check if prediction is correct
                    if predicted_class != target_class:
                        all_correct = False
            
            # Only consider samples that all models predict correctly
            if all_correct and len(confidences) == len(target_distances):
                # Calculate average confidence across all distances
                avg_confidence = np.mean(list(confidences.values()))
                min_confidence = min(confidences.values())
                
                class_candidates[target_class].append({
                    'input_tensor': input_tensor,
                    'confidences': confidences,
                    'avg_confidence': avg_confidence,
                    'min_confidence': min_confidence,
                    'sample_idx': sample_count
                })
            
            sample_count += 1
            if sample_count > 2000:  # Limit search to avoid excessive computation
                break
        
        # Select best sample for each class (prioritize high minimum confidence)
        selected_samples = []
        selected_class_names = []
        selected_confidences = []
        
        for class_idx in target_classes:
            candidates = class_candidates[class_idx]
            if not candidates:
                print(f"    Warning: No candidates found for class {class_idx}")
                continue
                
            # Sort by minimum confidence first (ensures all distances are confident), 
            # then by average confidence
            best_candidate = max(candidates, key=lambda x: (x['min_confidence'], x['avg_confidence']))
            
            selected_samples.append(best_candidate['input_tensor'])
            selected_class_names.append(class_names[class_idx] if isinstance(class_names, list) else str(class_idx))
            selected_confidences.append(best_candidate['confidences'])
            
            print(f"    Class {class_idx}: avg_conf={best_candidate['avg_confidence']:.3f}, "
                f"min_conf={best_candidate['min_confidence']:.3f}")
        
        return selected_samples, selected_class_names, target_classes, selected_confidences

    def create_enhanced_gradcam_table(self, dataset_name, architecture, models, gradcams, output_dir):
        """Create enhanced GradCAM table with Y-axis labels and X-axis confidence scores"""
        print(f"  Creating enhanced GradCAM comparison table...")
        
        # Get top performing distances
        dataloader, _ = self.get_sample_data(dataset_name, architecture, num_samples=50)
        top_distances = self.get_top_performing_distances(models, dataloader, num_top=3)
        
        # Get best samples across all distances
        class_samples, class_names, target_classes, sample_confidences = self.get_best_samples_across_all_distances(
            dataset_name, architecture, top_distances, models)
        
        if not class_samples:
            print("  Error: No class samples found")
            return
        
        num_classes = len(class_samples)
        num_rows = 8  # 2 for euclidean + 6 for top 3 (2 each)
        
        # Create the visualization with extra space for labels
        fig, axes = plt.subplots(num_rows, num_classes, figsize=(3 * num_classes, 2.5 * num_rows))
        
        if num_classes == 1:
            axes = axes.reshape(-1, 1)
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Process each distance type (euclidean + top 3)
        for dist_idx, distance_type in enumerate(top_distances):
            if distance_type not in gradcams:
                print(f"  Warning: {distance_type} not available in gradcams")
                continue
                
            gradcam = gradcams[distance_type]
            model = models[distance_type]
            
            # Calculate row indices for this distance type (2 rows each)
            base_row = dist_idx * 2
            
            # Process each class sample
            for class_idx, (input_tensor, class_name) in enumerate(zip(class_samples, class_names)):
                try:
                    # Add this debug line first:
                    print(f"    Processing {distance_type} for class {class_name} (architecture: {architecture})")
                    
                    try:
                        original_img = self._prepare_image_for_display(input_tensor, dataset_name, architecture)
                        print(f"    Image prep successful: {original_img.shape}")
                    except Exception as img_error:
                        print(f"    Image prep FAILED for {architecture}-{dataset_name}: {img_error}")
                        # Fill with blank and continue
                        axes[base_row, class_idx].axis('off')
                        axes[base_row + 1, class_idx].axis('off')
                        continue

                    # Get prediction and generate GradCAM
                    try:
                        with torch.no_grad():
                            output = model(input_tensor)
                            predicted_class = output.argmax(dim=1).item()
                            confidence = F.softmax(output, dim=1).max().item()
                        
                        cam = gradcam.generate_cam_debug(input_tensor, predicted_class)
                        
                        # Debug the image preparation step specifically:

                        # Add these debug prints:
                        print(f"    VIZ DEBUG: CAM shape after generation: {cam.shape}")
                        print(f"    VIZ DEBUG: CAM range after generation: [{cam.min():.6f}, {cam.max():.6f}]")
                        print(f"    VIZ DEBUG: CAM std after generation: {cam.std():.6f}")

                    except Exception as gradcam_error:
                        print(f"    GradCAM generation FAILED for {distance_type}: {gradcam_error}")
                        # Fill with blank and continue
                        axes[base_row, class_idx].axis('off')
                        axes[base_row + 1, class_idx].axis('off')
                        continue

                    # After the resize:
                    cam_resized = cv2.resize(cam, original_img.shape[:2])
                    print(f"    VIZ DEBUG: CAM shape after resize: {cam_resized.shape}")
                    print(f"    VIZ DEBUG: CAM range after resize: [{cam_resized.min():.6f}, {cam_resized.max():.6f}]")
                    print(f"    VIZ DEBUG: CAM std after resize: {cam_resized.std():.6f}")

                    # After colormap application:
                    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
                    print(f"    VIZ DEBUG: Heatmap shape: {heatmap.shape}")
                    print(f"    VIZ DEBUG: Heatmap range: [{heatmap.min():.6f}, {heatmap.max():.6f}]")

                    try:
                        original_img = self._prepare_image_for_display(input_tensor, dataset_name, architecture)
                        print(f"    Image prep successful: {original_img.shape}")
                    except Exception as img_error:
                        print(f"    Image prep FAILED for {architecture}-{dataset_name}: {img_error}")
                        continue  # Skip this sample and move to next
                    
                    # Resize CAM to match image
                    try:
                        # Resize CAM to match image
                        cam_resized = cv2.resize(cam, original_img.shape[:2])
                        print(f"    VIZ DEBUG: CAM shape after resize: {cam_resized.shape}")
                        print(f"    VIZ DEBUG: CAM range after resize: [{cam_resized.min():.6f}, {cam_resized.max():.6f}]")
                        print(f"    VIZ DEBUG: CAM std after resize: {cam_resized.std():.6f}")
                        
                        # Create heatmap
                        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
                        print(f"    VIZ DEBUG: Heatmap shape: {heatmap.shape}")
                        print(f"    VIZ DEBUG: Heatmap range: [{heatmap.min():.6f}, {heatmap.max():.6f}]")
                        
                        # Row 1: Overlay (heatmap + original)
                        overlay = 0.6 * original_img + 0.4 * heatmap
                        axes[base_row, class_idx].imshow(overlay)
                        axes[base_row, class_idx].axis('off')
                        
                        # Row 2: Pure heatmap
                        axes[base_row + 1, class_idx].imshow(cam_resized, cmap='jet')
                        axes[base_row + 1, class_idx].axis('off')
                        
                        # Add class label on top row for each class
                        if dist_idx == 0:
                            axes[base_row, class_idx].set_title(f'{class_name}', 
                                                            fontsize=10, pad=5, weight='bold')
                        
                        # Add confidence score
                        if class_idx < len(sample_confidences):
                            conf_for_this_distance = sample_confidences[class_idx].get(distance_type, confidence)
                            
                            fig.text(
                                (class_idx + 0.5) / num_classes, 
                                1 - (base_row + 0.5) / num_rows - 0.02,
                                f'{conf_for_this_distance:.3f}',
                                ha='center', va='center', fontsize=9, 
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
                            )
                        
                        print(f"    ✓ Successfully processed {distance_type} for class {class_name}")
                        
                    except Exception as viz_error:
                        print(f"    Visualization processing FAILED for {distance_type}: {viz_error}")
                        # Fill with blank
                        axes[base_row, class_idx].axis('off')
                        axes[base_row + 1, class_idx].axis('off')
                        continue
                    
                except Exception as e:
                    print(f"    Unexpected error processing {distance_type} for class {class_name}: {e}")
                    # Fill with blank
                    axes[base_row, class_idx].axis('off')
                    axes[base_row + 1, class_idx].axis('off')
                    cam_resized = cv2.resize(cam, original_img.shape[:2])
                    
                    # Row 1: Overlay (heatmap + original)
                    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
                    overlay = 0.6 * original_img + 0.4 * heatmap
                    axes[base_row, class_idx].imshow(overlay)
                    axes[base_row, class_idx].axis('off')
                    
                    # Row 2: Pure heatmap
                    axes[base_row + 1, class_idx].imshow(cam_resized, cmap='jet')
                    axes[base_row + 1, class_idx].axis('off')
                    
                    # Add class label and confidence on top row for each class
                    if dist_idx == 0:
                        axes[base_row, class_idx].set_title(f'{class_name}', 
                                                        fontsize=10, pad=5, weight='bold')
                    
                    # Add confidence score in the middle (between overlay and heatmap rows)
                    if class_idx < len(sample_confidences):
                        conf_for_this_distance = sample_confidences[class_idx].get(distance_type, confidence)
                        
                        # Add text annotation for confidence between the two rows
                        fig.text(
                            # X position: center of this column
                            (class_idx + 0.5) / num_classes, 
                            # Y position: between the two rows for this distance
                            1 - (base_row + 0.5) / num_rows - 0.02,
                            f'{conf_for_this_distance:.3f}',
                            ha='center', va='center', fontsize=9, 
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
                        )
                    
                except Exception as e:
                    print(f"    Error processing {distance_type} for class {class_name}: {e}")
                    # Fill with blank
                    axes[base_row, class_idx].axis('off')
                    axes[base_row + 1, class_idx].axis('off')
        
        # Add Y-axis labels for each distance type and row type
        for dist_idx, distance_type in enumerate(top_distances):
            base_row = dist_idx * 2
            
            # Clean up distance type name for display
            display_name = distance_type.replace('_', ' ').title()
            if len(display_name) > 12:  # Truncate long names
                display_name = display_name[:12] + '...'
            
            # Add labels for both rows of this distance type
            fig.text(0.02, 1 - (base_row + 0.25) / num_rows, f'{display_name}\nOverlay', 
                    ha='left', va='center', fontsize=10, weight='bold', rotation=0)
            fig.text(0.02, 1 - (base_row + 0.75) / num_rows, f'{display_name}\nHeatmap', 
                    ha='left', va='center', fontsize=10, weight='bold', rotation=0)
        
        # Add a confidence label in the middle
        fig.text(0.5, 0.95, 'Confidence Scores', ha='center', va='top', 
                fontsize=12, weight='bold')
        
        plt.suptitle(f'Enhanced GradCAM Analysis: {dataset_name} - {architecture}', 
                    fontsize=16, y=0.98)
        
        # Adjust layout to make room for labels
        plt.subplots_adjust(left=0.15, right=0.98, top=0.93, bottom=0.02, 
                        hspace=0.1, wspace=0.05)
        
        # Save the visualization
        output_path = f"{output_dir}/{dataset_name}_{architecture}/enhanced_gradcam_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved enhanced table to: {output_path}")
        
        # Also save a summary of the selected samples and their confidences
        summary_path = f"{output_dir}/{dataset_name}_{architecture}/sample_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Sample Selection Summary for {dataset_name} - {architecture}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Selected Distance Types: {top_distances}\n\n")
            
            for i, (class_name, confidences) in enumerate(zip(class_names, sample_confidences)):
                f.write(f"Class {target_classes[i]} ({class_name}):\n")
                for dist_type, conf in confidences.items():
                    f.write(f"  {dist_type}: {conf:.4f}\n")
                avg_conf = np.mean(list(confidences.values()))
                min_conf = min(confidences.values())
                f.write(f"  Average: {avg_conf:.4f}, Minimum: {min_conf:.4f}\n\n")

    # Update the main generation method to use the enhanced version
    def generate_gradcam_comparison_enhanced(self, dataset_name, architecture, output_dir="gradcam_results"):
        """Generate enhanced GradCAM comparison table with better sample selection and labels"""
        print(f"\nGenerating Enhanced GradCAM for {dataset_name} - {architecture}")
        
        models_info = self.models_data[dataset_name][architecture]
        if len(models_info) < 2:
            print(f"Not enough models to compare (found {len(models_info)})")
            return None
        
        os.makedirs(f"{output_dir}/{dataset_name}_{architecture}", exist_ok=True)
        
        # Load all models and create GradCAMs
        models = {}
        gradcams = {}
        
        # Get sample data first for dimension checking
        dataloader, class_names = self.get_sample_data(dataset_name, architecture, num_samples=1)
        sample_input, _ = next(iter(dataloader))
        
        for model_info in models_info:
            distance_type = model_info['distance_type']
            print(f"  Loading {distance_type} model...")
            
            model = self.load_model(model_info['path'], architecture, dataset_name, distance_type)
            if model is None:
                continue
            
            # ADD THIS: Test layers for the first model (ResNet50 only)
            if architecture == 'ResNet50' and len(models) == 0:  # Only test once
                print(f"\n  Testing target layers for {architecture}...")
                self.test_resnet_layers_for_gradcam(model, sample_input)
                print(f"  Continuing with original layer choice...")
            
            target_layer = self.get_target_layer_name(model, architecture)
            if target_layer is None:
                print(f"    Warning: Could not find target layer for {distance_type}")
                continue
            
            gradcam = GradCAM(model, target_layer)
            models[distance_type] = model
            gradcams[distance_type] = gradcam
        
        if not models:
            print("No models successfully loaded")
            return None
        
        # Create the enhanced comparison table
        self.create_enhanced_gradcam_table(dataset_name, architecture, models, gradcams, output_dir)
        
        # Cleanup
        for gradcam in gradcams.values():
            gradcam.cleanup()
        
        return True

    def test_resnet_layers_for_gradcam(self, model, input_tensor):
        """Test different ResNet layers to find the best one for GradCAM"""
        candidate_layers = [
            'model.layer4.2.conv3',      # Your current choice
            'model.layer4.2.conv2',      # Alternative in same block
            'model.layer4.1.conv3',      # Previous block
            'model.layer4.0.conv3',      # First block in layer4
            'model.layer3.5.conv3',      # Previous layer
            'model.avgpool',             # Global average pooling
            'model.layer4.2.bn3',        # Batch norm after conv3
        ]
        
        print(f"Testing ResNet target layers for meaningful GradCAM...")
        
        for layer_name in candidate_layers:
            print(f"\nTesting layer: {layer_name}")
            
            # Check if layer exists
            layer_exists = False
            for name, module in model.named_modules():
                if name == layer_name:
                    layer_exists = True
                    print(f"  ✓ Layer exists: {type(module).__name__}")
                    break
            
            if not layer_exists:
                print(f"  ✗ Layer not found")
                continue
            
            try:
                # Create temporary GradCAM
                temp_gradcam = GradCAM(model, layer_name)
                
                # Test with debug version
                cam = temp_gradcam.generate_cam_debug(input_tensor)
                
                # Analyze results
                cam_std = cam.std()
                cam_range = cam.max() - cam.min()
                
                print(f"  Results: std={cam_std:.6f}, range={cam_range:.6f}")
                
                if cam_std > 0.01 and cam_range > 0.1:
                    print(f"  ✓ This layer produces meaningful gradients!")
                else:
                    print(f"  ✗ This layer produces flat/weak gradients")
                
                temp_gradcam.cleanup()
                
            except Exception as e:
                print(f"  ✗ Error: {e}")

    def _prepare_image_for_display(self, input_tensor, dataset_name, architecture):
        """Helper method to prepare image tensor for display"""
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
        return original_img

    # Modify the generate_gradcam_comparison method to call the new table creation
    
    def run_focused_gradcam_analysis(self):
        """Run focused GradCAM analysis for specified dataset-architecture pairs"""
        self.discover_models()
        
        # Filter datasets and architectures based on configuration
        datasets_to_process = [d for d in self.models_data.keys() if d in DATASETS_TO_ANALYZE]
        
        print(f"Configured to analyze datasets: {DATASETS_TO_ANALYZE}")
        print(f"Configured to analyze architectures: {ARCHITECTURES_TO_ANALYZE}")
        print(f"Found datasets in saved_models: {list(self.models_data.keys())}")
        print(f"Will process: {datasets_to_process}")
        
        success_count = 0
        total_combinations = 0
        
        for dataset in datasets_to_process:
            architectures_to_process = [a for a in self.models_data[dataset].keys() if a in ARCHITECTURES_TO_ANALYZE]
            print(f"Found architectures for {dataset}: {list(self.models_data[dataset].keys())}")
            print(f"Will process architectures for {dataset}: {architectures_to_process}")
            
            for architecture in architectures_to_process:
                total_combinations += 1
                print(f"\n{'='*50}")
                print(f"Creating focused analysis for {dataset} - {architecture}")
                print(f"{'='*50}")
                
                try:
                    #result = self.generate_gradcam_comparison_focused(dataset, architecture)
                    result = self.generate_gradcam_comparison_enhanced(dataset, architecture)
                    if result:
                        success_count += 1
                        print(f"✓ Successfully created focused table for {dataset}-{architecture}")
                    else:
                        print(f"✗ Failed to create focused table for {dataset}-{architecture}")
                        
                except Exception as e:
                    print(f"✗ Error analyzing {dataset}-{architecture}: {e}")
                    import traceback
                    traceback.print_exc()
        
        return success_count, total_combinations


if __name__ == "__main__":
    print("Starting Enhanced GradCAM interpretability analysis...")
    print("Note: Make sure models/models.py is in your path or adjust the import")
    print("This will create enhanced comparison tables showing:")
    print("- Euclidean distance (always included)")
    print("- Top 3 performing distances by confidence")
    print("- 8 rows total (2 per distance: overlay + heatmap)")
    print("- 10 columns showing different classes")
    print("- Y-axis labels for distance types")
    print("- X-axis confidence scores")
    print("- Optimized sample selection across all distances")
    
    comparator = ModelGradCAMComparator()
    success_count, total_combinations = comparator.run_focused_gradcam_analysis()
    
    print(f"\nEnhanced GradCAM analysis complete!")
    print(f"Successfully processed: {success_count}/{total_combinations} dataset-architecture combinations")
    print(f"Results saved to gradcam_results/")
    print("Check the 'enhanced_gradcam_table.png' files in each dataset-architecture folder.")
    print("Also check 'sample_summary.txt' files for confidence details.")
    
    if success_count == 0:
        print("\nNo tables were generated. Check:")
        print("- Model files exist in the expected directory structure")
        print("- DATASETS_TO_ANALYZE and ARCHITECTURES_TO_ANALYZE match your available models")
        print("- Model loading is working correctly")
    elif success_count < total_combinations:
        print(f"\nSome combinations failed ({total_combinations - success_count} failed)")
        print("Check the error messages above for details")
    else:
        print(f"\nAll combinations processed successfully!")
        print("The enhanced tables show interpretability comparisons with:")
        print("- Performance-based distance selection")
        print("- Cross-distance optimized class representatives")
        print("- Clean 8×10 grid layout with comprehensive labels")






'''def get_class_representative_samples(self, dataset_name, architecture, target_distances):
        """Get one representative sample for each class"""
        print(f"  Collecting representative samples for each class...")
        
        # Get full test dataset
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
            target_classes = list(range(10))
            class_names = dataset.classes
        elif dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
            # Select 10 representative classes for CIFAR-100
            target_classes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # One from each superclass
            class_names = [dataset.classes[i] for i in target_classes]
        elif dataset_name == 'MNIST':
            dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            target_classes = list(range(10))
            class_names = [str(i) for i in range(10)]
        
        # Find one good representative for each target class
        class_samples = {}
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        for input_tensor, target in dataloader:
            target_class = target.item()
            if target_class in target_classes and target_class not in class_samples:
                class_samples[target_class] = input_tensor
                
            # Stop when we have all classes
            if len(class_samples) == len(target_classes):
                break
        
        # Organize samples in order of target_classes
        ordered_samples = []
        ordered_class_names = []
        for class_idx in target_classes:
            if class_idx in class_samples:
                ordered_samples.append(class_samples[class_idx])
                ordered_class_names.append(class_names[class_idx] if isinstance(class_names, list) else str(class_idx))
        
        return ordered_samples, ordered_class_names, target_classes

    def create_focused_gradcam_table(self, dataset_name, architecture, models, gradcams, output_dir):
        """Create the new 8-row focused comparison table"""
        print(f"  Creating focused GradCAM comparison table...")
        
        # Get top performing distances
        dataloader, _ = self.get_sample_data(dataset_name, architecture, num_samples=50)
        top_distances = self.get_top_performing_distances(models, dataloader, num_top=3)
        
        # Get class representative samples
        class_samples, class_names, target_classes = self.get_class_representative_samples(
            dataset_name, architecture, top_distances)
        
        if not class_samples:
            print("  Error: No class samples found")
            return
        
        num_classes = len(class_samples)
        num_rows = 8  # 2 for euclidean + 6 for top 3 (2 each)
        
        # Create the visualization
        fig, axes = plt.subplots(num_rows, num_classes, figsize=(2.5 * num_classes, 2 * num_rows))
        
        if num_classes == 1:
            axes = axes.reshape(-1, 1)
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        row_labels = []
        
        # Process each distance type (euclidean + top 3)
        for dist_idx, distance_type in enumerate(top_distances):
            if distance_type not in gradcams:
                print(f"  Warning: {distance_type} not available in gradcams")
                continue
                
            gradcam = gradcams[distance_type]
            model = models[distance_type]
            
            # Calculate row indices for this distance type (2 rows each)
            base_row = dist_idx * 2
            
            # Add row labels
            row_labels.extend([f"{distance_type}_overlay", f"{distance_type}_heatmap"])
            
            # Process each class sample
            for class_idx, (input_tensor, class_name) in enumerate(zip(class_samples, class_names)):
                try:
                    # Get prediction and generate GradCAM
                    with torch.no_grad():
                        output = model(input_tensor)
                        predicted_class = output.argmax(dim=1).item()
                        confidence = F.softmax(output, dim=1).max().item()
                    
                    cam = gradcam.generate_cam(input_tensor, predicted_class)
                    
                    # Prepare original image for display
                    original_img = self._prepare_image_for_display(input_tensor, dataset_name, architecture)
                    
                    # Resize CAM to match image
                    cam_resized = cv2.resize(cam, original_img.shape[:2])
                    
                    # Row 1: Overlay (heatmap + original)
                    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
                    overlay = 0.6 * original_img + 0.4 * heatmap
                    axes[base_row, class_idx].imshow(overlay)
                    axes[base_row, class_idx].axis('off')
                    
                    # Row 2: Pure heatmap
                    axes[base_row + 1, class_idx].imshow(cam_resized, cmap='jet')
                    axes[base_row + 1, class_idx].axis('off')
                    
                    # Add title only on first row for each class
                    if dist_idx == 0:
                        axes[base_row, class_idx].set_title(f'{class_name}\nP:{predicted_class} C:{confidence:.2f}', 
                                                        fontsize=8, pad=5)
                    
                except Exception as e:
                    print(f"    Error processing {distance_type} for class {class_name}: {e}")
                    # Fill with blank
                    axes[base_row, class_idx].axis('off')
                    axes[base_row + 1, class_idx].axis('off')
        
        # Add row labels on the left
        for row_idx, label in enumerate(row_labels):
            if row_idx < num_rows:
                axes[row_idx, 0].set_ylabel(label, rotation=0, ha='right', va='center', 
                                        fontsize=9, labelpad=20)
        
        plt.suptitle(f'Focused GradCAM Analysis: {dataset_name} - {architecture}', 
                    fontsize=14, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, top=0.93)  # Make room for row labels
        
        # Save the visualization
        output_path = f"{output_dir}/{dataset_name}_{architecture}/focused_gradcam_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved focused table to: {output_path}")
        
        def generate_gradcam_comparison_focused(self, dataset_name, architecture, output_dir="gradcam_results"):
        """Generate focused GradCAM comparison table"""
        print(f"\nGenerating Focused GradCAM for {dataset_name} - {architecture}")
        
        models_info = self.models_data[dataset_name][architecture]
        if len(models_info) < 2:
            print(f"Not enough models to compare (found {len(models_info)})")
            return None
        
        os.makedirs(f"{output_dir}/{dataset_name}_{architecture}", exist_ok=True)
        
        # Load all models and create GradCAMs
        models = {}
        gradcams = {}
        
        # Get sample data first for dimension checking
        dataloader, class_names = self.get_sample_data(dataset_name, architecture, num_samples=1)
        sample_input, _ = next(iter(dataloader))
        
        for model_info in models_info:
            distance_type = model_info['distance_type']
            print(f"  Loading {distance_type} model...")
            
            model = self.load_model(model_info['path'], architecture, dataset_name, distance_type)
            if model is None:
                continue
            
            target_layer = self.get_target_layer_name(model, architecture)
            if target_layer is None:
                print(f"    Warning: Could not find target layer for {distance_type}")
                continue
            
            gradcam = GradCAM(model, target_layer)
            models[distance_type] = model
            gradcams[distance_type] = gradcam
        
        if not models:
            print("No models successfully loaded")
            return None
        
        # Create the focused comparison table
        self.create_focused_gradcam_table(dataset_name, architecture, models, gradcams, output_dir)
        
        # Cleanup
        for gradcam in gradcams.values():
            gradcam.cleanup()
        
        return True'''