import torch
import torch.nn as nn
import numpy as np
from models.distlayers import *
import timm
from torch import nn, optim

#from models.vgg_resnet import VGG16Wrapper, ResNet50Wrapper, SimplifiedVGG16ForCIFAR10
#from models.pvt_vit import ViTWrapper, PVTWrapper

class SimpleImageMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[512, 256], 
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Distance-specific parameters (same as before)
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Build shared layers
        layers = []
        prev_size = input_size
        
        # Hidden layers (same for both baseline and distance)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.layers = nn.Sequential(*layers)  # All hidden layers
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(prev_size, num_classes)
            self.dist_layer = None
        else:
            # Distance layer
            self.final_layer = self._create_distance_layer(
                self.distance_layer_type, prev_size, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer  # For compatibility
        
    def _create_distance_layer(self, distance_type, feature_dim, num_classes, n, eps,
                              minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                              chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                              canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                              bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                              mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                              mahalanobis_reg_lambda, cosine_stable):
        """Create distance layer with all parameters"""
        
        # Extract base distance type from variant name
        base_distance_type = distance_type
        if '_' in distance_type:
            parts = distance_type.split('_')
            if parts[0] in ['cosine', 'minkowski', 'hamming', 'chebyshev', 'canberra', 'bray', 'mahalanobis']:
                if parts[0] == 'bray':
                    base_distance_type = 'bray_curtis'
                else:
                    base_distance_type = parts[0]
        
        if base_distance_type == "euclidean":
            return EuclideanDistLayer(feature_dim, num_classes, n=n, eps=eps)
        elif base_distance_type == "manhattan":
            return ManhattanDistLayer(feature_dim, num_classes, n=n, eps=eps)
        elif base_distance_type == "cosine":
            return CosineDistLayer(feature_dim, num_classes, n=n, eps=eps, stable=cosine_stable)
        elif base_distance_type == "minkowski":
            return MinkowskiDistLayer(feature_dim, num_classes, p=minkowski_p, n=n, eps=eps)
        elif base_distance_type == "hamming":
            return HammingDistLayer(feature_dim, num_classes, n=n, eps=eps,
                                   threshold=hamming_threshold, temperature=hamming_temperature,
                                   variant=hamming_variant)
        elif base_distance_type == "chebyshev":
            return ChebyshevDistLayer(feature_dim, num_classes, n=n, eps=eps,
                                     smooth=chebyshev_smooth, alpha=chebyshev_alpha)
        elif base_distance_type == "canberra":
            return CanberraDistLayer(feature_dim, num_classes, n=n, eps=eps,
                                    variant=canberra_variant, min_denom=canberra_min_denom,
                                    weight_power=canberra_weight_power, 
                                    normalize_weights=canberra_normalize_weights)
        elif base_distance_type == "bray_curtis":
            return BrayCurtisDistLayer(feature_dim, num_classes, n=n, eps=eps,
                                      variant=bray_curtis_variant,
                                      normalize_inputs=bray_curtis_normalize_inputs,
                                      min_sum=bray_curtis_min_sum)
        elif base_distance_type == "mahalanobis":
            return MahalanobisDistLayer(feature_dim, num_classes, n=n, eps=eps,
                                       variant=mahalanobis_variant,
                                       learn_cov=mahalanobis_learn_cov,
                                       init_identity=mahalanobis_init_identity,
                                       regularize_cov=mahalanobis_regularize_cov,
                                       reg_lambda=mahalanobis_reg_lambda)
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}")
        
    def forward(self, x, return_embedding=False):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        embedding = self.layers(x)
        
        # FIXED: Final layer processing
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            logits = self.final_layer(embedding)
        else:
            # FIXED: Distance layer
            distances = self.final_layer(embedding, scale=self.scale_distances)
            logits = -distances  # Use negative distances as logits
        
        if return_embedding:
            return logits, embedding
        return logits


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_height, input_width,
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Distance-specific parameters (same as MLP)
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Same convolutional layers for all versions
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Use dummy input to infer the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_height, input_width)
            x = self.pool(torch.relu(self.conv1(dummy_input)))
            x = self.pool(torch.relu(self.conv2(x)))
            self.flattened_size = x.view(1, -1).shape[1]
        
        # Feature extraction layer (same for all versions)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(128, num_classes)
            self.dist_layer = None
        else:
            # Distance layer (reuse MLP's distance layer creation)
            self.final_layer = SimpleImageMLP._create_distance_layer(
                self, distance_layer_type, 128, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer  # For compatibility
    
    def forward(self, x, return_embedding=False):
        # Same convolutional processing for all versions
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        embedding = torch.relu(self.fc1(x))
        
        # FIXED: Final layer processing  
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            logits = self.final_layer(embedding)
        else:
            # FIXED: Distance layer
            distances = self.final_layer(embedding, scale=self.scale_distances)
            logits = -distances  # Use negative distances as logits
        
        if return_embedding:
            return logits, embedding
        return logits


class VGG16Wrapper(nn.Module):
    def __init__(self, num_classes, in_channels, input_height, input_width,
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Distance parameters (same as before)
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Store input parameters
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create appropriate VGG architecture based on input size (same for all)
        if input_height == 32 and input_width == 32 and in_channels == 3:
            self.features, self.pre_classifier, feature_dim = self._create_cifar10_architecture()
            self.is_simplified = True
            self.is_mnist = False
        elif input_height == 28 and input_width == 28 and in_channels == 1:
            self.features, self.pre_classifier, feature_dim = self._create_mnist_architecture()
            self.is_simplified = True
            self.is_mnist = True
        else:
            self.features, self.pre_classifier, feature_dim = self._create_standard_architecture(
                in_channels, num_classes)
            self.is_simplified = False
            self.is_mnist = False
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(feature_dim, num_classes)
            self.dist_layer = None
        else:
            # Distance layer (reuse MLP's distance layer creation)
            self.final_layer = SimpleImageMLP._create_distance_layer(
                self, distance_layer_type, feature_dim, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer  # For compatibility
    
    def _create_cifar10_architecture(self):
        """Create CIFAR-10 specific VGG architecture (32x32 input)"""
        features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 -> 1x1
        )
        
        pre_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        
        return features, pre_classifier, 256
    
    def _create_mnist_architecture(self):
        """Create MNIST specific VGG architecture (28x28 input, fewer pooling)"""
        features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            
            # Block 3 - Fewer layers to prevent over-reduction
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7 -> 3x3 (with some rounding)
            
            # Block 4 - Reduced complexity
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # No more pooling to avoid dimension issues
        )
        
        pre_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Adaptive pooling to ensure 1x1 output
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        return features, pre_classifier, 128
    
    def _create_standard_architecture(self, in_channels, num_classes):
        """Create standard VGG16 for larger inputs"""
        import torchvision.models as models
        vgg_model = models.vgg16(weights=None)
        
        # Modify first layer if needed
        if in_channels != 3:
            vgg_model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        features = vgg_model.features
        
        # Create pre-classifier (without final layer)
        pre_classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-1])
        
        # Get embedding dimension from the pre-classifier
        feature_dim = vgg_model.classifier[-1].in_features
        
        return features, pre_classifier, feature_dim
        
    def forward(self, x, return_embedding=False):
        # Extract features (same for all versions)
        features = self.features(x)
        
        if self.is_mnist:
            # For MNIST, use the adaptive pooling approach
            embedding = self.pre_classifier(features)
        else:
            # For CIFAR-10 and others, flatten then process
            if self.is_simplified:
                # Simplified VGG (CIFAR-10)
                flattened = features.view(features.size(0), -1)
                embedding = self.pre_classifier(flattened)
            else:
                # Standard VGG
                features = nn.functional.adaptive_avg_pool2d(features, (7, 7))
                flattened = torch.flatten(features, 1)
                embedding = self.pre_classifier(flattened)
        
        # FIXED: Final layer processing
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            logits = self.final_layer(embedding)
        else:
            # FIXED: Distance layer
            distances = self.final_layer(embedding, scale=self.scale_distances)
            logits = -distances  # Use negative distances as logits
        
        if return_embedding:
            return logits, embedding
        return logits

"""class ResNet50Wrapper(nn.Module):
    def __init__(self, num_classes, in_channels, input_height, input_width,
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Distance parameters (same as before)
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Store architecture info
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create ResNet50 model (same for all versions)
        import torchvision.models as models
        self.model = models.resnet50(weights=None)
        
        # Apply the same modifications for small inputs
        if input_height <= 32 or input_width <= 32:
            if in_channels == 1:  # MNIST
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.small_input = True
                self.is_mnist = True
            else:  # CIFAR-10
                self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.small_input = True
                self.is_mnist = False
            
            # Remove max pooling completely for small images
            self.model.maxpool = nn.Identity()
        else:
            # For standard size inputs, just modify input channels if needed
            if in_channels != 3:
                self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.small_input = False
            self.is_mnist = (in_channels == 1)
        
        # Get the feature dimension (ResNet50 has 2048 features before the final FC layer)
        self.feature_dim = self.model.fc.in_features  # This should be 2048 for ResNet50
        
        # Remove the original final layer - we'll replace it
        self.model.fc = nn.Identity()
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(self.feature_dim, num_classes)
            self.dist_layer = None
        else:
            # FIXED: Create distance layer directly instead of using MLP method
            self.final_layer = self._create_distance_layer(
                distance_layer_type, self.feature_dim, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer  # For compatibility

    def _create_distance_layer(self, distance_layer_type, in_features, out_features, 
                              n, eps, minkowski_p, hamming_threshold, hamming_temperature, 
                              hamming_variant, chebyshev_smooth, chebyshev_alpha, 
                              canberra_variant, canberra_min_denom, canberra_weight_power, 
                              canberra_normalize_weights, bray_curtis_variant, 
                              bray_curtis_normalize_inputs, bray_curtis_min_sum, 
                              mahalanobis_variant, mahalanobis_learn_cov, 
                              mahalanobis_init_identity, mahalanobis_regularize_cov, 
                              mahalanobis_reg_lambda, cosine_stable):
        #Create distance layer - copied from your distance layers
        
        # Import distance layers
        from distlayers import (
            EuclideanDistLayer, ManhattanDistLayer, CosineDistLayer, MinkowskiDistLayer,
            HammingDistLayer, ChebyshevDistLayer, CanberraDistLayer, BrayCurtisDistLayer,
            MahalanobisDistLayer
        )
        
        distance_type = distance_layer_type.lower()
        
        if distance_type == "euclidean":
            return EuclideanDistLayer(in_features, out_features, n=n, eps=eps)
        elif distance_type == "manhattan":
            return ManhattanDistLayer(in_features, out_features, n=n, eps=eps)
        elif distance_type.startswith("cosine"):
            stable = cosine_stable if "stable" in distance_type else not cosine_stable
            return CosineDistLayer(in_features, out_features, n=n, eps=eps, stable=stable)
        elif distance_type.startswith("minkowski"):
            return MinkowskiDistLayer(in_features, out_features, p=minkowski_p, n=n, eps=eps)
        elif distance_type.startswith("hamming"):
            return HammingDistLayer(in_features, out_features, n=n, eps=eps,
                                  threshold=hamming_threshold, temperature=hamming_temperature,
                                  variant=hamming_variant)
        elif distance_type.startswith("chebyshev"):
            return ChebyshevDistLayer(in_features, out_features, n=n, eps=eps,
                                    smooth=chebyshev_smooth, alpha=chebyshev_alpha)
        elif distance_type.startswith("canberra"):
            return CanberraDistLayer(in_features, out_features, n=n, eps=eps,
                                   variant=canberra_variant, min_denom=canberra_min_denom,
                                   weight_power=canberra_weight_power, 
                                   normalize_weights=canberra_normalize_weights)
        elif distance_type.startswith("bray_curtis"):
            return BrayCurtisDistLayer(in_features, out_features, n=n, eps=eps,
                                     variant=bray_curtis_variant,
                                     normalize_inputs=bray_curtis_normalize_inputs,
                                     min_sum=bray_curtis_min_sum)
        elif distance_type.startswith("mahalanobis"):
            return MahalanobisDistLayer(in_features, out_features, n=n, eps=eps,
                                      variant=mahalanobis_variant,
                                      learn_cov=mahalanobis_learn_cov,
                                      init_identity=mahalanobis_init_identity,
                                      regularize_cov=mahalanobis_regularize_cov,
                                      reg_lambda=mahalanobis_reg_lambda)
        else:
            raise ValueError(f"Unknown distance layer type: {distance_type}")
        
    def forward(self, x, return_embedding=False):
        # Forward through ResNet50 backbone (same for all versions)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # This is Identity() for small inputs
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        # Global average pooling
        embedding = self.model.avgpool(x)
        embedding = torch.flatten(embedding, 1)
        
        # CRITICAL FIX: Final layer processing
        if self.distance_layer_type == "baseline":
            # Regular linear layer - returns logits directly
            logits = self.final_layer(embedding)
        else:
            # FIXED: Distance layer returns similarity scores (positive values)
            # The distance layers already return (distance)^(-n) which are positive similarities
            # Just use these directly as logits after ensuring numerical stability
            similarities = self.final_layer(embedding, scale=self.scale_distances)
            
            # Clamp to prevent numerical issues and take log to convert to logits
            similarities = torch.clamp(similarities, min=1e-8)
            logits = torch.log(similarities)
        
        if return_embedding:
            return logits, embedding
        return logits"""

class ResNet50Wrapper(nn.Module):
    def __init__(self, num_classes, in_channels, input_height, input_width,
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Distance parameters (same as before)
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Store architecture info
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create ResNet50 model (same for all versions)
        import torchvision.models as models
        self.model = models.resnet50(weights=None)
        
        # Apply the same modifications for small inputs
        if input_height <= 32 or input_width <= 32:
            if in_channels == 1:  # MNIST
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.small_input = True
                self.is_mnist = True
            else:  # CIFAR-10
                self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.small_input = True
                self.is_mnist = False
            
            # Remove max pooling completely for small images
            self.model.maxpool = nn.Identity()
        else:
            # For standard size inputs, just modify input channels if needed
            if in_channels != 3:
                self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.small_input = False
            self.is_mnist = (in_channels == 1)
        
        # Get the feature dimension (ResNet50 has 2048 features before the final FC layer)
        self.feature_dim = self.model.fc.in_features  # This should be 2048 for ResNet50
        
        # Remove the original final layer - we'll replace it
        self.model.fc = nn.Identity()
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(self.feature_dim, num_classes)
            self.dist_layer = None
        else:
            # Distance layer (reuse MLP's distance layer creation)
            self.final_layer = SimpleImageMLP._create_distance_layer(
                self, distance_layer_type, self.feature_dim, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer  # For compatibility
        
    """def forward(self, x, return_embedding=False):
        # Forward through ResNet50 backbone (same for all versions)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # This is Identity() for small inputs
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        # Global average pooling
        embedding = self.model.avgpool(x)
        embedding = torch.flatten(embedding, 1)
        
        # The model.fc is now Identity(), so embedding is the final feature vector
        
        # FIXED: Final layer processing
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            logits = self.final_layer(embedding)
        else:
            # FIXED: Distance layer returns similarities, just take log for logits
            similarities = self.final_layer(embedding, scale=self.scale_distances)
            logits = torch.log(torch.clamp(similarities, min=1e-8))
        
        if return_embedding:
            return logits, embedding
        return logits"""
    def forward(self, x, return_embedding=False):
        # Forward through ResNet50 backbone (same for all versions)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # This is Identity() for small inputs
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        # Global average pooling
        embedding = self.model.avgpool(x)
        embedding = torch.flatten(embedding, 1)
        
        # FIXED: Normalize ResNet50 features before distance computation
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            logits = self.final_layer(embedding)
        else:
            # HYPOTHESIS: ResNet50 features need normalization for distance layers
            # Normalize features to unit norm (like what MLP might implicitly have)
            embedding_normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            # Apply distance layer to normalized features
            # prior use:
            # distances = self.final_layer(embedding_normalized, scale=self.scale_distances)
            # logits = -distances  # Use negative distances as logits

            similarities = self.final_layer(embedding, scale=self.scale_distances)
            similarities = torch.clamp(similarities, min=1e-8)
            logits = torch.log(similarities)
        
        if return_embedding:
            return logits, embedding
        return logits



class ViTWrapper(nn.Module):
    def __init__(self, num_classes, in_channels,
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Distance-specific parameters
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create ViT model without final classification head (same for all versions)
        try:
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                            num_classes=0, in_chans=in_channels)
        except Exception as e:
            raise RuntimeError(f"Failed to create ViT model with {in_channels} input channels: {e}")
        
        # Get the feature dimension more robustly
        self.feature_dim = self._get_feature_dimension()
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(self.feature_dim, num_classes)
            self.dist_layer = None
        else:
            # Distance layer (reuse MLP's distance layer creation)
            self.final_layer = SimpleImageMLP._create_distance_layer(
                self, distance_layer_type, self.feature_dim, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer  # For compatibility
        
    def _get_feature_dimension(self):
        """Robustly determine feature dimension"""
        with torch.no_grad():
            # Use appropriate input size for ViT (224x224)
            dummy_input = torch.zeros(1, self.in_channels, 224, 224)
            try:
                features = self.backbone(dummy_input)
                
                if len(features.shape) == 3:  # [batch, sequence_length, embed_dim]
                    pooled_features = features.mean(dim=1)
                    feature_dim = pooled_features.shape[1]
                elif len(features.shape) == 2:  # [batch, features]
                    feature_dim = features.shape[1]
                else:
                    raise ValueError(f"Unexpected ViT feature shape: {features.shape}")
                    
                print(f"ViT feature dimension: {feature_dim}")
                return feature_dim
                
            except Exception as e:
                raise RuntimeError(f"Failed to determine ViT feature dimension: {e}")
        
    def forward(self, x, return_embedding=False):
        try:
            # Extract features using backbone (same for all versions)
            features = self.backbone(x)
            
            # Handle different feature shapes robustly
            if len(features.shape) == 3:  # [batch, sequence_length, embed_dim]
                embedding = features.mean(dim=1)
            elif len(features.shape) == 2:  # [batch, features]
                embedding = features
            else:
                raise ValueError(f"Unexpected ViT feature shape in forward: {features.shape}")
            
            # FIXED: Final layer processing
            if self.distance_layer_type == "baseline":
                # Regular linear layer
                logits = self.final_layer(embedding)
            else:
                # FIXED: Distance layer with proper gradient flow
                distances = self.final_layer(embedding, scale=self.scale_distances)
                
                # Use negative distances as logits (closer = higher probability)
                logits = -distances
                
                # Optional: Add debug info during first few iterations
                if self.training and torch.rand(1).item() < 0.01:  # Print 1% of the time
                    print(f"Distance stats - Min: {distances.min():.6f}, Max: {distances.max():.6f}")
                    print(f"Logit stats - Min: {logits.min():.6f}, Max: {logits.max():.6f}")
            
            if return_embedding:
                return logits, embedding
            return logits
            
        except Exception as e:
            raise RuntimeError(f"ViT forward pass failed: {e}")

class PVTWrapper(nn.Module):
    def __init__(self, num_classes, in_channels, input_height=32, input_width=32,
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Same distance-specific parameters as ViT
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create PVT model without final classification head (same for all versions)
        try:
            # Try to create with img_size if supported
            try:
                self.backbone = timm.create_model('pvt_v2_b0', pretrained=False,
                                                num_classes=0, in_chans=in_channels,
                                                img_size=(input_height, input_width))
            except:
                # Fallback without img_size
                self.backbone = timm.create_model('pvt_v2_b0', pretrained=False,
                                                num_classes=0, in_chans=in_channels)
        except Exception as e:
            raise RuntimeError(f"Failed to create PVT model with {in_channels} input channels: {e}")
        
        # Get the feature dimension more robustly
        self.feature_dim = self._get_feature_dimension()
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(self.feature_dim, num_classes)
            self.dist_layer = None
        else:
            # Distance layer (reuse MLP's distance layer creation)
            self.final_layer = SimpleImageMLP._create_distance_layer(
                self, distance_layer_type, self.feature_dim, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer
        
    def _get_feature_dimension(self):
        """Robustly determine feature dimension"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            try:
                features = self.backbone(dummy_input)
                
                if len(features.shape) == 4:  # [batch, channels, height, width]
                    pooled_features = features.mean(dim=[2, 3])
                    feature_dim = pooled_features.shape[1]
                elif len(features.shape) == 2:  # [batch, features]
                    feature_dim = features.shape[1]
                else:
                    raise ValueError(f"Unexpected PVT feature shape: {features.shape}")
                    
                print(f"PVT feature dimension: {feature_dim}")
                return feature_dim
                
            except Exception as e:
                raise RuntimeError(f"Failed to determine PVT feature dimension: {e}")
        
    '''Original
    def forward(self, x, return_embedding=False):
        try:
            # Extract features (same for all versions)
            features = self.backbone(x)
            
            # Handle different feature shapes robustly
            if len(features.shape) == 4:  # [batch, channels, height, width]
                embedding = features.mean(dim=[2, 3])
            elif len(features.shape) == 2:  # [batch, features]
                embedding = features
            else:
                raise ValueError(f"Unexpected PVT feature shape in forward: {features.shape}")
            
            # FIXED: Final layer processing
            if self.distance_layer_type == "baseline":
                # Regular linear layer
                logits = self.final_layer(embedding)
            else:
                # FIXED: Distance layer
                distances = self.final_layer(embedding, scale=self.scale_distances)
                logits = -distances  # Use negative distances as logits
            
            if return_embedding:
                return logits, embedding
            return logits
            
        except Exception as e:
            raise RuntimeError(f"PVT forward pass failed: {e}")'''

    #Just the logit fix
    '''def forward(self, x, return_embedding=False):
        try:
            # Extract features (same for all versions)
            features = self.backbone(x)
            
            # Handle different feature shapes robustly
            if len(features.shape) == 4:  # [batch, channels, height, width]
                embedding = features.mean(dim=[2, 3])
            elif len(features.shape) == 2:  # [batch, features]
                embedding = features
            else:
                raise ValueError(f"Unexpected PVT feature shape in forward: {features.shape}")
            
            # DEBUGGING: Print feature statistics to understand the problem
            if hasattr(self, 'debug_count'):
                self.debug_count += 1
            else:
                self.debug_count = 1
            
            if self.debug_count <= 3:  # Only print first few batches
                print(f"\nPVT Debug - Batch {self.debug_count}:")
                print(f"Embedding shape: {embedding.shape}")
                print(f"Embedding mean: {embedding.mean().item():.6f}")
                print(f"Embedding std: {embedding.std().item():.6f}")
                print(f"Embedding min: {embedding.min().item():.6f}")
                print(f"Embedding max: {embedding.max().item():.6f}")
                print(f"Embedding norm: {torch.norm(embedding, dim=1).mean().item():.6f}")
            
            # SIMPLIFIED FIX: PVT features are already well-behaved
            if self.distance_layer_type == "baseline":
                # Regular linear layer - returns logits directly
                logits = self.final_layer(embedding)
            else:
                # Since PVT features are already normalized, just fix the logits conversion
                similarities = self.final_layer(embedding, scale=self.scale_distances)
                similarities = torch.clamp(similarities, min=1e-8)
                logits = torch.log(similarities)
            
            if return_embedding:
                return logits, embedding
            return logits
            
        except Exception as e:
            raise RuntimeError(f"PVT forward pass failed: {e}")'''

    def forward(self, x, return_embedding=False):
        try:
            # Extract features (same for all versions)
            features = self.backbone(x)
            
            # Handle different feature shapes robustly
            if len(features.shape) == 4:  # [batch, channels, height, width]
                embedding = features.mean(dim=[2, 3])
            elif len(features.shape) == 2:  # [batch, features]
                embedding = features
            else:
                raise ValueError(f"Unexpected PVT feature shape in forward: {features.shape}")
            
            # DEBUGGING: Print feature statistics to understand the problem
            if hasattr(self, 'debug_count'):
                self.debug_count += 1
            else:
                self.debug_count = 1
            
            if self.debug_count <= 3:  # Only print first few batches
                print(f"\nPVT Debug - Batch {self.debug_count}:")
                print(f"Embedding shape: {embedding.shape}")
                print(f"Embedding mean: {embedding.mean().item():.6f}")
                print(f"Embedding std: {embedding.std().item():.6f}")
                print(f"Embedding min: {embedding.min().item():.6f}")
                print(f"Embedding max: {embedding.max().item():.6f}")
                print(f"Embedding norm: {torch.norm(embedding, dim=1).mean().item():.6f}")
            
            # TARGETED DEBUGGING: Compare distance distributions
            if self.distance_layer_type == "baseline":
                logits = self.final_layer(embedding)
            else:
                # Get raw similarities and analyze them
                similarities = self.final_layer(embedding, scale=self.scale_distances)
                
                # Debug: Check similarity statistics (only first few batches)
                if self.debug_count <= 3:
                    print(f"Similarity stats - Batch {self.debug_count}:")
                    print(f"  Min: {similarities.min().item():.8f}")
                    print(f"  Max: {similarities.max().item():.8f}")
                    print(f"  Mean: {similarities.mean().item():.8f}")
                    print(f"  Std: {similarities.std().item():.8f}")
                    
                    # Check if similarities are too uniform (bad for classification)
                    sim_range = similarities.max() - similarities.min()
                    print(f"  Range: {sim_range.item():.8f}")
                    
                    # Check for problematic values
                    if similarities.min() <= 0:
                        print(f"  WARNING: Non-positive similarities detected!")
                
                similarities = torch.clamp(similarities, min=1e-8)
                logits = torch.log(similarities)
            
            if return_embedding:
                return logits, embedding
            return logits
            
        except Exception as e:
            raise RuntimeError(f"PVT forward pass failed: {e}")

    """ # Preprocessing with logit fix
    def forward(self, x, return_embedding=False):
        try:
            # Extract features (same for all versions)
            features = self.backbone(x)
            
            # Handle different feature shapes robustly
            if len(features.shape) == 4:  # [batch, channels, height, width]
                embedding = features.mean(dim=[2, 3])
            elif len(features.shape) == 2:  # [batch, features]
                embedding = features
            else:
                raise ValueError(f"Unexpected PVT feature shape in forward: {features.shape}")
            
            # DEBUGGING: Print feature statistics to understand the problem
            if hasattr(self, 'debug_count'):
                self.debug_count += 1
            else:
                self.debug_count = 1
            
            if self.debug_count <= 3:  # Only print first few batches
                print(f"\nPVT Debug - Batch {self.debug_count}:")
                print(f"Embedding shape: {embedding.shape}")
                print(f"Embedding mean: {embedding.mean().item():.6f}")
                print(f"Embedding std: {embedding.std().item():.6f}")
                print(f"Embedding min: {embedding.min().item():.6f}")
                print(f"Embedding max: {embedding.max().item():.6f}")
                print(f"Embedding norm: {torch.norm(embedding, dim=1).mean().item():.6f}")
            
            # CRITICAL FIX: PVT-specific feature preprocessing
            if self.distance_layer_type == "baseline":
                # Regular linear layer - returns logits directly
                logits = self.final_layer(embedding)
            else:
                # HYPOTHESIS: PVT features need aggressive preprocessing
                
                # 1. Check for problematic values
                if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                    print("WARNING: PVT embedding has NaN/Inf values!")
                    embedding = torch.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 2. Robust feature normalization for transformer features
                # Center the features
                embedding_centered = embedding - embedding.mean(dim=1, keepdim=True)
                
                # Scale by robust standard deviation (avoid division by zero)
                embedding_std = torch.std(embedding_centered, dim=1, keepdim=True)
                embedding_std = torch.clamp(embedding_std, min=1e-6)
                embedding_normalized = embedding_centered / embedding_std
                
                # 3. Additional clipping for extreme values (transformers can have outliers)
                embedding_clipped = torch.clamp(embedding_normalized, min=-5.0, max=5.0)
                
                # 4. Apply distance layer to preprocessed features
                similarities = self.final_layer(embedding_clipped, scale=self.scale_distances)
                
                # 5. FIXED: Convert similarities to logits properly
                similarities = torch.clamp(similarities, min=1e-8)
                logits = torch.log(similarities)
            
            if return_embedding:
                return logits, embedding
            return logits
            
        except Exception as e:
            raise RuntimeError(f"PVT forward pass failed: {e}")"""

'''class PVTWrapper(nn.Module):
    def __init__(self, num_classes, in_channels, input_height=32, input_width=32,
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Same distance-specific parameters as ViT
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create PVT model without final classification head (same for all versions)
        try:
            # Try to create with img_size if supported
            try:
                self.backbone = timm.create_model('pvt_v2_b0', pretrained=False,
                                                num_classes=0, in_chans=in_channels,
                                                img_size=(input_height, input_width))
            except:
                # Fallback without img_size
                self.backbone = timm.create_model('pvt_v2_b0', pretrained=False,
                                                num_classes=0, in_chans=in_channels)
        except Exception as e:
            raise RuntimeError(f"Failed to create PVT model with {in_channels} input channels: {e}")
        
        # Get the feature dimension more robustly
        self.feature_dim = self._get_feature_dimension()
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(self.feature_dim, num_classes)
            self.dist_layer = None
        else:
            # FIXED: Create distance layer directly instead of using non-existent MLP method
            self.final_layer = self._create_distance_layer(
                distance_layer_type, self.feature_dim, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer

    def _create_distance_layer(self, distance_layer_type, in_features, out_features, 
                              n, eps, minkowski_p, hamming_threshold, hamming_temperature, 
                              hamming_variant, chebyshev_smooth, chebyshev_alpha, 
                              canberra_variant, canberra_min_denom, canberra_weight_power, 
                              canberra_normalize_weights, bray_curtis_variant, 
                              bray_curtis_normalize_inputs, bray_curtis_min_sum, 
                              mahalanobis_variant, mahalanobis_learn_cov, 
                              mahalanobis_init_identity, mahalanobis_regularize_cov, 
                              mahalanobis_reg_lambda, cosine_stable):
        """Create distance layer - same as ResNet50"""
        
        # Import distance layers
        from regularization.distances import (
            EuclideanDistLayer, ManhattanDistLayer, CosineDistLayer, MinkowskiDistLayer,
            HammingDistLayer, ChebyshevDistLayer, CanberraDistLayer, BrayCurtisDistLayer,
            MahalanobisDistLayer
        )
        
        distance_type = distance_layer_type.lower()
        
        if distance_type == "euclidean":
            return EuclideanDistLayer(in_features, out_features, n=n, eps=eps)
        elif distance_type == "manhattan":
            return ManhattanDistLayer(in_features, out_features, n=n, eps=eps)
        elif distance_type.startswith("cosine"):
            stable = cosine_stable if "stable" in distance_type else not cosine_stable
            return CosineDistLayer(in_features, out_features, n=n, eps=eps, stable=stable)
        elif distance_type.startswith("minkowski"):
            return MinkowskiDistLayer(in_features, out_features, p=minkowski_p, n=n, eps=eps)
        elif distance_type.startswith("hamming"):
            return HammingDistLayer(in_features, out_features, n=n, eps=eps,
                                  threshold=hamming_threshold, temperature=hamming_temperature,
                                  variant=hamming_variant)
        elif distance_type.startswith("chebyshev"):
            return ChebyshevDistLayer(in_features, out_features, n=n, eps=eps,
                                    smooth=chebyshev_smooth, alpha=chebyshev_alpha)
        elif distance_type.startswith("canberra"):
            return CanberraDistLayer(in_features, out_features, n=n, eps=eps,
                                   variant=canberra_variant, min_denom=canberra_min_denom,
                                   weight_power=canberra_weight_power, 
                                   normalize_weights=canberra_normalize_weights)
        elif distance_type.startswith("bray_curtis"):
            return BrayCurtisDistLayer(in_features, out_features, n=n, eps=eps,
                                     variant=bray_curtis_variant,
                                     normalize_inputs=bray_curtis_normalize_inputs,
                                     min_sum=bray_curtis_min_sum)
        elif distance_type.startswith("mahalanobis"):
            return MahalanobisDistLayer(in_features, out_features, n=n, eps=eps,
                                      variant=mahalanobis_variant,
                                      learn_cov=mahalanobis_learn_cov,
                                      init_identity=mahalanobis_init_identity,
                                      regularize_cov=mahalanobis_regularize_cov,
                                      reg_lambda=mahalanobis_reg_lambda)
        else:
            raise ValueError(f"Unknown distance layer type: {distance_type}")
        
    def _get_feature_dimension(self):
        """Robustly determine feature dimension"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            try:
                features = self.backbone(dummy_input)
                
                if len(features.shape) == 4:  # [batch, channels, height, width]
                    pooled_features = features.mean(dim=[2, 3])
                    feature_dim = pooled_features.shape[1]
                elif len(features.shape) == 2:  # [batch, features]
                    feature_dim = features.shape[1]
                else:
                    raise ValueError(f"Unexpected PVT feature shape: {features.shape}")
                    
                print(f"PVT feature dimension: {feature_dim}")
                return feature_dim
                
            except Exception as e:
                raise RuntimeError(f"Failed to determine PVT feature dimension: {e}")
        
    def forward(self, x, return_embedding=False):
        try:
            # Extract features (same for all versions)
            features = self.backbone(x)
            
            # Handle different feature shapes robustly
            if len(features.shape) == 4:  # [batch, channels, height, width]
                embedding = features.mean(dim=[2, 3])
            elif len(features.shape) == 2:  # [batch, features]
                embedding = features
            else:
                raise ValueError(f"Unexpected PVT feature shape in forward: {features.shape}")
            
            # CRITICAL FIX: Final layer processing
            if self.distance_layer_type == "baseline":
                # Regular linear layer - returns logits directly
                logits = self.final_layer(embedding)
            else:
                # FIXED: Distance layer returns similarity scores (positive values)
                # Convert similarities to logits using log
                similarities = self.final_layer(embedding, scale=self.scale_distances)
                
                # Ensure similarities are positive and add epsilon for numerical stability
                similarities = torch.clamp(similarities, min=1e-8)
                
                # Convert similarities to logits (log of probabilities)
                # Normalize to create probability distribution
                probs = similarities / torch.sum(similarities, dim=-1, keepdim=True)
                logits = torch.log(probs + 1e-8)
            
            if return_embedding:
                return logits, embedding
            return logits
            
        except Exception as e:
            raise RuntimeError(f"PVT forward pass failed: {e}")
            
    def forward(self, x, return_embedding=False):
        try:
            # Extract features (same for all versions)
            features = self.backbone(x)
            
            # Handle different feature shapes robustly
            if len(features.shape) == 4:  # [batch, channels, height, width]
                embedding = features.mean(dim=[2, 3])
            elif len(features.shape) == 2:  # [batch, features]
                embedding = features
            else:
                raise ValueError(f"Unexpected PVT feature shape in forward: {features.shape}")
            
            # DEBUGGING: Print feature statistics to understand the problem
            if hasattr(self, 'debug_count'):
                self.debug_count += 1
            else:
                self.debug_count = 1
            
            if self.debug_count <= 3:  # Only print first few batches
                print(f"\nPVT Debug - Batch {self.debug_count}:")
                print(f"Embedding shape: {embedding.shape}")
                print(f"Embedding mean: {embedding.mean().item():.6f}")
                print(f"Embedding std: {embedding.std().item():.6f}")
                print(f"Embedding min: {embedding.min().item():.6f}")
                print(f"Embedding max: {embedding.max().item():.6f}")
                print(f"Embedding norm: {torch.norm(embedding, dim=1).mean().item():.6f}")
            
            # CRITICAL FIX: PVT-specific feature preprocessing
            if self.distance_layer_type == "baseline":
                # Regular linear layer - returns logits directly
                logits = self.final_layer(embedding)
            else:
                # HYPOTHESIS: PVT features need aggressive preprocessing
                
                # 1. Check for problematic values
                if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                    print("WARNING: PVT embedding has NaN/Inf values!")
                    embedding = torch.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 2. Robust feature normalization for transformer features
                # Center the features
                embedding_centered = embedding - embedding.mean(dim=1, keepdim=True)
                
                # Scale by robust standard deviation (avoid division by zero)
                embedding_std = torch.std(embedding_centered, dim=1, keepdim=True)
                embedding_std = torch.clamp(embedding_std, min=1e-6)
                embedding_normalized = embedding_centered / embedding_std
                
                # 3. Additional clipping for extreme values (transformers can have outliers)
                embedding_clipped = torch.clamp(embedding_normalized, min=-5.0, max=5.0)
                
                # 4. Apply distance layer to preprocessed features
                similarities = self.final_layer(embedding_clipped, scale=self.scale_distances)
                
                # 5. FIXED: Convert similarities to logits properly
                similarities = torch.clamp(similarities, min=1e-8)
                logits = torch.log(similarities)
            
            if return_embedding:
                return logits, embedding
            return logits
            
        except Exception as e:
            raise RuntimeError(f"PVT forward pass failed: {e}")'''

'''class PVTWrapper(nn.Module):
    def __init__(self, num_classes, in_channels, input_height=32, input_width=32,
                 distance_layer_type="baseline", n=1., eps=1e-4,
                 # Same distance-specific parameters as ViT
                 minkowski_p=1.5,
                 hamming_threshold=0.5,
                 hamming_temperature=1.0,
                 hamming_variant="soft",
                 chebyshev_smooth=False,
                 chebyshev_alpha=10.0,
                 canberra_variant="standard",
                 canberra_min_denom=1e-3,
                 canberra_weight_power=1.0,
                 canberra_normalize_weights=True,
                 bray_curtis_variant="standard",
                 bray_curtis_normalize_inputs=True,
                 bray_curtis_min_sum=1e-3,
                 mahalanobis_variant="standard",
                 mahalanobis_learn_cov=True,
                 mahalanobis_init_identity=True,
                 mahalanobis_regularize_cov=True,
                 mahalanobis_reg_lambda=1e-2,
                 cosine_stable=True,
                 scale_distances=False):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.distance_layer_type = distance_layer_type.lower()
        self.scale_distances = scale_distances
        
        # Create PVT model without final classification head (same for all versions)
        try:
            # Try to create with img_size if supported
            try:
                self.backbone = timm.create_model('pvt_v2_b0', pretrained=False,
                                                num_classes=0, in_chans=in_channels,
                                                img_size=(input_height, input_width))
            except:
                # Fallback without img_size
                self.backbone = timm.create_model('pvt_v2_b0', pretrained=False,
                                                num_classes=0, in_chans=in_channels)
        except Exception as e:
            raise RuntimeError(f"Failed to create PVT model with {in_channels} input channels: {e}")
        
        # Get the feature dimension more robustly
        self.feature_dim = self._get_feature_dimension()
        
        # Create final layer based on distance_layer_type
        if self.distance_layer_type == "baseline":
            # Regular linear layer
            self.final_layer = nn.Linear(self.feature_dim, num_classes)
            self.dist_layer = None
        else:
            # Distance layer (reuse MLP's distance layer creation)
            self.final_layer = SimpleImageMLP._create_distance_layer(
                self, distance_layer_type, self.feature_dim, num_classes, n, eps,
                minkowski_p, hamming_threshold, hamming_temperature, hamming_variant,
                chebyshev_smooth, chebyshev_alpha, canberra_variant, canberra_min_denom,
                canberra_weight_power, canberra_normalize_weights, bray_curtis_variant,
                bray_curtis_normalize_inputs, bray_curtis_min_sum, mahalanobis_variant,
                mahalanobis_learn_cov, mahalanobis_init_identity, mahalanobis_regularize_cov,
                mahalanobis_reg_lambda, cosine_stable
            )
            self.dist_layer = self.final_layer
        
    def _get_feature_dimension(self):
        """Robustly determine feature dimension"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            try:
                features = self.backbone(dummy_input)
                
                if len(features.shape) == 4:  # [batch, channels, height, width]
                    pooled_features = features.mean(dim=[2, 3])
                    feature_dim = pooled_features.shape[1]
                elif len(features.shape) == 2:  # [batch, features]
                    feature_dim = features.shape[1]
                else:
                    raise ValueError(f"Unexpected PVT feature shape: {features.shape}")
                    
                print(f"PVT feature dimension: {feature_dim}")
                return feature_dim
                
            except Exception as e:
                raise RuntimeError(f"Failed to determine PVT feature dimension: {e}")
        
    def forward(self, x, return_embedding=False):
        try:
            # Extract features (same for all versions)
            features = self.backbone(x)
            
            # Handle different feature shapes robustly
            if len(features.shape) == 4:  # [batch, channels, height, width]
                embedding = features.mean(dim=[2, 3])
            elif len(features.shape) == 2:  # [batch, features]
                embedding = features
            else:
                raise ValueError(f"Unexpected PVT feature shape in forward: {features.shape}")
            
            # FIXED: Final layer processing
            if self.distance_layer_type == "baseline":
                # Regular linear layer
                logits = self.final_layer(embedding)
            else:
                # FIXED: Distance layer
                distances = self.final_layer(embedding, scale=self.scale_distances)
                logits = -distances  # Use negative distances as logits
            
            if return_embedding:
                return logits, embedding
            return logits
            
        except Exception as e:
            raise RuntimeError(f"PVT forward pass failed: {e}")'''