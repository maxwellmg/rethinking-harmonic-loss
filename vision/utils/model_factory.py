import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal Imports
from models.models import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelFactory:
    def __init__(self, model_type, dataset, config, device, distance_typeLayer=None, num_classes=None):
        self.model_type = model_type
        self.dataset = dataset
        self.config = config
        self.device = device
        self.distance_typeLayer = distance_typeLayer
        # CHANGE 1: Add num_classes parameter with fallback logic
        self.num_classes = num_classes or self._get_num_classes_from_config()
        self.initial_state = None
        self.model = None

        # Set default distance_typeLayer if not specified
        
        if self.distance_typeLayer is None:
            self.distance_typeLayer = "baseline"
        
        # DEBUG: Print what we're creating (you can remove this later)
        print(f"ModelFactory: Creating {model_type} with distance_typeLayer='{self.distance_typeLayer}'")
        
        # Pre-calculate model parameters
        self._setup_model_params()
    
    def _get_num_classes_from_config(self):
        """Get number of classes from config with dataset-specific logic"""
        # CHANGE 2: Use dataset-specific num_classes if available
        if 'dataset_num_classes' in self.config:
            return self.config['dataset_num_classes'].get(self.dataset, 10)
        # Fallback logic
        elif self.dataset == 'CIFAR100':
            return 100
        elif self.dataset in ['MNIST', 'CIFAR10']:
            return 10
        else:
            # Ultimate fallback
            return self.config.get('num_classes', 10)
        
    def _setup_model_params(self):
        """Pre-calculate model architecture parameters"""
        if self.dataset == "MNIST":
            if self.model_type in ['ViT']:
                self.input_height, self.input_width, self.in_channels = 224, 224, 3
            elif self.model_type in ['PVT']:
                self.input_height, self.input_width, self.in_channels = 32, 32, 3
            else:
                self.input_height, self.input_width, self.in_channels = 28, 28, 1
                
        elif self.dataset == "CIFAR10":
            self.input_height, self.input_width, self.in_channels = 32, 32, 3
            
        # CHANGE 3: Add CIFAR100 support
        elif self.dataset == "CIFAR100":
            self.input_height, self.input_width, self.in_channels = 32, 32, 3
        
        else:
            # Default fallback for unknown datasets
            self.input_height, self.input_width, self.in_channels = 32, 32, 3

    def _get_distance_params(self, distance_typeLayer):
        """Extract distance parameters from condensed config with model-specific overrides"""
        if 'dist_layer_configs' not in self.config:
            return {}
        
        # Get shared distance parameters (used by all models)
        shared_params = self.config['dist_layer_configs'].get('shared_distance_params', {})
        
        # Get global parameters
        result = {
            'n': shared_params.get('n', 1.0),
            'eps': shared_params.get('eps', 1e-4),
            'scale_distances': shared_params.get('scale_distances', False)
        }
        
        # Extract base distance type from variant name
        base_distance_type = distance_typeLayer
        if '_' in distance_typeLayer:
            parts = distance_typeLayer.split('_')
            if parts[0] in ['cosine', 'minkowski', 'hamming', 'chebyshev', 'canberra', 'bray', 'mahalanobis']:
                if parts[0] == 'bray':
                    base_distance_type = 'bray_curtis'
                else:
                    base_distance_type = parts[0]
        
        # Look for distance-specific parameters with "Layer" suffix in shared params
        layer_key = base_distance_type + 'Layer'
        if layer_key in shared_params:
            result.update(shared_params[layer_key])
        elif base_distance_type in shared_params:
            result.update(shared_params[base_distance_type])
        
        # Apply model-specific distance overrides
        model_overrides = self.config['dist_layer_configs'].get('model_distance_overrides', {})
        if self.model_type in model_overrides:
            model_distance_overrides = model_overrides[self.model_type].get('distance_overrides', {})
            if base_distance_type in model_distance_overrides:
                result.update(model_distance_overrides[base_distance_type])
        
        # Apply any custom overrides for the specific variant (highest priority)
        if 'custom_dist_params' in self.config and distance_typeLayer in self.config['custom_dist_params']:
            result.update(self.config['custom_dist_params'][distance_typeLayer])
        
        return result

    def _get_model_specific_params(self, model_type):
        """Get model-specific parameters like hidden_sizes"""
        if 'dist_layer_configs' not in self.config:
            return {}
        
        model_specific = self.config['dist_layer_configs'].get('model_specific', {})
        return model_specific.get(model_type, {})

    def create_model(self):
        """Create a fresh model instance using unified classes with comprehensive parameters"""
        
        # Set default distance_typeLayer if not specified
        if self.distance_typeLayer is None:
            self.distance_typeLayer = "baseline"
        
        # Get distance-specific parameters (with model-specific overrides)
        dist_params = self._get_distance_params(self.distance_typeLayer)
        
        # Get model-specific parameters
        model_params = self._get_model_specific_params(self.model_type)
        
        # Build common model kwargs for distance parameters
        distance_kwargs = {
            'distance_layer_type': self.distance_typeLayer,
            'n': dist_params.get('n', 1.0),
            'scale_distances': dist_params.get('scale_distances', False)
        }
        
        # Add distance-specific parameters (same as before)
        base_distance_type = self.distance_typeLayer
        if '_' in self.distance_typeLayer:
            parts = self.distance_typeLayer.split('_')
            if parts[0] in ['cosine', 'minkowski', 'hamming', 'chebyshev', 'canberra', 'bray', 'mahalanobis']:
                if parts[0] == 'bray':
                    base_distance_type = 'bray_curtis'
                else:
                    base_distance_type = parts[0]
        
        # Add distance-specific parameters
        if base_distance_type == 'euclidean':
            distance_kwargs['eps'] = dist_params.get('eps', 1e-4)
        elif base_distance_type == 'manhattan':
            distance_kwargs['eps'] = dist_params.get('eps', 1e-4)
        elif base_distance_type == 'minkowski':
            distance_kwargs.update({
                'minkowski_p': dist_params.get('p', 1.5),
                'eps': dist_params.get('eps', 1e-4)
            })
        elif base_distance_type == 'hamming':
            distance_kwargs.update({
                'hamming_threshold': dist_params.get('threshold', 0.5),
                'hamming_temperature': dist_params.get('temperature', 1.0),
                'hamming_variant': dist_params.get('variant', 'soft'),
                'eps': dist_params.get('eps', 1e-4)
            })
        elif base_distance_type == 'chebyshev':
            distance_kwargs.update({
                'chebyshev_smooth': dist_params.get('smooth', False),
                'chebyshev_alpha': dist_params.get('alpha', 10.0),
                'eps': dist_params.get('eps', 1e-4)
            })
        elif base_distance_type == 'canberra':
            distance_kwargs.update({
                'canberra_variant': dist_params.get('variant', 'standard'),
                'canberra_min_denom': dist_params.get('min_denom', 1e-3),
                'canberra_weight_power': dist_params.get('weight_power', 1.0),
                'canberra_normalize_weights': dist_params.get('normalize_weights', True),
                'eps': dist_params.get('eps', 1e-4)
            })
        elif base_distance_type == 'bray_curtis':
            distance_kwargs.update({
                'bray_curtis_variant': dist_params.get('variant', 'standard'),
                'bray_curtis_normalize_inputs': dist_params.get('normalize_inputs', True),
                'bray_curtis_min_sum': dist_params.get('min_sum', 1e-3),
                'eps': dist_params.get('eps', 1e-3)
            })
        elif base_distance_type == 'mahalanobis':
            # Parse the variant from distance_typeLayer
            if 'diagonal' in self.distance_typeLayer.lower():
                variant = 'diagonal'
            elif 'cholesky' in self.distance_typeLayer.lower():
                variant = 'cholesky'
            else:
                variant = 'standard'  # default for mahalanobis_standard or plain mahalanobis
            
            print(f"ModelFactory: Creating Mahalanobis {variant} variant")
            
            distance_kwargs.update({
                'mahalanobis_variant': variant,
                'mahalanobis_learn_cov': dist_params.get('learn_cov', True),
                'mahalanobis_init_identity': dist_params.get('init_identity', True),
                'mahalanobis_regularize_cov': dist_params.get('regularize_cov', True),
                'mahalanobis_reg_lambda': dist_params.get('reg_lambda', 1e-2),
                'eps': dist_params.get('eps', 1e-4)
            })
        elif base_distance_type == 'cosine':
            distance_kwargs.update({
                'cosine_stable': dist_params.get('stable', True),
                'eps': dist_params.get('eps', 1e-4)
            })
        
        # Create model based on type using unified classes
        if self.model_type == 'MLP':
            mlp_input_size = self.input_height * self.input_width * self.in_channels
            
            # Get MLP-specific params
            hidden_sizes = model_params.get('hidden_sizes', [512, 256])
            
            model_kwargs = {
                'input_size': mlp_input_size,
                'num_classes': self.num_classes,  # CHANGE 4: Use self.num_classes
                'hidden_sizes': hidden_sizes,
                **distance_kwargs
            }
            return SimpleImageMLP(**model_kwargs).to(self.device)
            
        elif self.model_type == 'CNN':
            # Could add CNN-specific params here
            # conv_channels = model_params.get('conv_channels', [32, 64])
            # feature_dim = model_params.get('feature_dim', 128)
            
            model_kwargs = {
                'in_channels': self.in_channels,
                'num_classes': self.num_classes,  # CHANGE 4: Use self.num_classes
                'input_height': self.input_height,
                'input_width': self.input_width,
                **distance_kwargs
            }
            return SimpleCNN(**model_kwargs).to(self.device)
            
        elif self.model_type == 'VGG16':
            # Could add VGG16-specific params here
            # dropout_rate = model_params.get('dropout_rate', 0.5)
            
            model_kwargs = {
                'num_classes': self.num_classes,  # CHANGE 4: Use self.num_classes
                'in_channels': self.in_channels,
                'input_height': self.input_height,
                'input_width': self.input_width,
                **distance_kwargs
            }
            return VGG16Wrapper(**model_kwargs).to(self.device)
            
        elif self.model_type == 'ResNet50':
            # Could add ResNet50-specific params here
            # use_resnet18_for_small = model_params.get('use_resnet18_for_small', False)
            
            model_kwargs = {
                'num_classes': self.num_classes,  # CHANGE 4: Use self.num_classes
                'in_channels': self.in_channels,
                'input_height': self.input_height,
                'input_width': self.input_width,
                **distance_kwargs
            }
            return ResNet50Wrapper(**model_kwargs).to(self.device)
            
        elif self.model_type == 'ViT':
            # Could add ViT-specific params here
            # model_name = model_params.get('model_name', 'vit_tiny_patch16_224')
            
            model_kwargs = {
                'num_classes': self.num_classes,  # CHANGE 4: Use self.num_classes
                'in_channels': self.in_channels,
                **distance_kwargs
            }
            return ViTWrapper(**model_kwargs).to(self.device)
            
        elif self.model_type == 'PVT':
            # Could add PVT-specific params here
            # model_name = model_params.get('model_name', 'pvt_v2_b0')
            
            model_kwargs = {
                'num_classes': self.num_classes,  # CHANGE 4: Use self.num_classes
                'in_channels': self.in_channels,
                'input_height': self.input_height,
                'input_width': self.input_width,
                **distance_kwargs
            }
            return PVTWrapper(**model_kwargs).to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_fresh_model(self):
        """Get a model with freshly initialized parameters"""
        if self.model is None:
            # Create the model for the first time
            self.model = self.create_model()
            # Store the initial state for quick resets
            self.initial_state = {name: param.clone() for name, param in self.model.named_parameters()}
        else:
            # Reset to initial parameters (much faster than recreation)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(self.initial_state[name])
        
        return self.model