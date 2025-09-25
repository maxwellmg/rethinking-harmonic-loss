import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class PCAInterpretabilityAnalyzer:
    """
    Standalone PCA analysis module for vision model interpretability.
    
    Integrates with your existing experiment workflow to analyze saved models
    and compute standardized interpretability metrics.
    """
    
    def __init__(self, n_components: int = 50, standardize: bool = True, 
                 sample_size: int = None, device: str = 'cuda'):
        """
        Args:
            n_components: Number of PCA components (standardized across experiments)
            standardize: Whether to standardize features before PCA
            sample_size: Max samples to use for PCA (None = use all)
            device: Device for model inference
        """
        self.n_components = n_components
        self.standardize = standardize
        self.sample_size = sample_size
        self.device = device
        self.results = {}
        
    def extract_penultimate_features(self, model: torch.nn.Module, 
                                   dataloader: torch.utils.data.DataLoader, 
                                   model_type: str) -> np.ndarray:
        """
        Extract penultimate layer features for different model architectures.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader for feature extraction
            model_type: Model type string for architecture-specific handling
            
        Returns:
            features: Numpy array of shape (n_samples, n_features)
        """
        model.eval()
        model = model.to(self.device)
        features_list = []
        
        # Architecture-specific feature extraction
        if model_type.startswith('MLP'):
            # SimpleImageMLP: extract from self.layers (before final_layer)
            def hook_fn(module, input, output):
                features_list.append(output.detach().cpu())
            
            handle = model.layers.register_forward_hook(hook_fn)
            
        elif model_type.startswith('CNN'):
            # SimpleCNN: extract from self.fc1 (before final_layer)
            def hook_fn(module, input, output):
                features_list.append(output.detach().cpu())
            
            handle = model.fc1.register_forward_hook(hook_fn)
            
        elif model_type.startswith('VGG16'):
            # VGG16Wrapper: extract from self.pre_classifier (before final_layer)
            def hook_fn(module, input, output):
                if len(output.shape) > 2:
                    output = torch.flatten(output, 1)
                features_list.append(output.detach().cpu())
            
            handle = model.pre_classifier.register_forward_hook(hook_fn)
            
        elif model_type.startswith('ResNet50'):
            # ResNet50Wrapper: extract after avgpool (before final_layer)
            def hook_fn(module, input, output):
                if len(output.shape) > 2:
                    output = torch.flatten(output, 1)
                features_list.append(output.detach().cpu())
            
            # Hook onto the avgpool layer (last before final_layer)
            handle = model.model.avgpool.register_forward_hook(hook_fn)
            
        elif model_type.startswith('ViT'):
            # ViTWrapper: extract from self.backbone (before final_layer)
            def hook_fn(module, input, output):
                if len(output.shape) == 3:  # [batch, seq_len, embed_dim]
                    output = output.mean(dim=1)  # Pool over sequence
                elif len(output.shape) > 2:
                    output = torch.flatten(output, 1)
                features_list.append(output.detach().cpu())
            
            handle = model.backbone.register_forward_hook(hook_fn)
            
        elif model_type.startswith('PVT'):
            # PVTWrapper: extract from self.backbone (before final_layer)
            def hook_fn(module, input, output):
                if len(output.shape) == 4:  # [batch, channels, height, width]
                    output = output.mean(dim=[2, 3])  # Global average pool
                elif len(output.shape) > 2:
                    output = torch.flatten(output, 1)
                features_list.append(output.detach().cpu())
            
            handle = model.backbone.register_forward_hook(hook_fn)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Extract features
        try:
            with torch.no_grad():
                sample_count = 0
                for batch_idx, (data, _) in enumerate(dataloader):
                    if self.sample_size and sample_count >= self.sample_size:
                        break
                        
                    data = data.to(self.device)
                    _ = model(data)  # Forward pass triggers hook
                    sample_count += data.size(0)
                    
                    if batch_idx % 50 == 0:
                        print(f"Processed batch {batch_idx}/{len(dataloader)} "
                              f"(samples: {sample_count})")
                        
        finally:
            handle.remove()
            
        # Concatenate features
        if not features_list:
            raise RuntimeError("No features extracted - check hook registration")
            
        features = torch.cat(features_list, dim=0).numpy()
        print(f"Extracted features shape: {features.shape} for {model_type}")
        
        return features
    
    def compute_pca_metrics(self, features: np.ndarray, model_key: str) -> Dict[str, float]:
        """
        Compute PCA and interpretability metrics.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            model_key: Unique identifier for the model
            
        Returns:
            Dictionary containing PCA metrics
        """
        print(f"Computing PCA for {model_key}...")
        
        # Standardize if requested
        if self.standardize:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        # Fit PCA with standardized components
        n_components = min(self.n_components, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(features)
        
        # Extract explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumsum_variance = np.cumsum(explained_variance_ratio)
        
        # Compute interpretability metrics
        metrics = {
            # Cumulative variance explained up to each component (0-10, 20, 50)
            'pc_0_variance': float(np.sum(explained_variance_ratio[:1])) if len(explained_variance_ratio) >= 1 else None,
            'pc_1_variance': float(np.sum(explained_variance_ratio[:2])) if len(explained_variance_ratio) >= 2 else None,
            'pc_2_variance': float(np.sum(explained_variance_ratio[:3])) if len(explained_variance_ratio) >= 3 else None,
            'pc_3_variance': float(np.sum(explained_variance_ratio[:4])) if len(explained_variance_ratio) >= 4 else None,
            'pc_4_variance': float(np.sum(explained_variance_ratio[:5])) if len(explained_variance_ratio) >= 5 else None,
            'pc_5_variance': float(np.sum(explained_variance_ratio[:6])) if len(explained_variance_ratio) >= 6 else None,
            'pc_6_variance': float(np.sum(explained_variance_ratio[:7])) if len(explained_variance_ratio) >= 7 else None,
            'pc_7_variance': float(np.sum(explained_variance_ratio[:8])) if len(explained_variance_ratio) >= 8 else None,
            'pc_8_variance': float(np.sum(explained_variance_ratio[:9])) if len(explained_variance_ratio) >= 9 else None,
            'pc_9_variance': float(np.sum(explained_variance_ratio[:10])) if len(explained_variance_ratio) >= 10 else None,
            'pc_10_variance': float(np.sum(explained_variance_ratio[:11])) if len(explained_variance_ratio) >= 11 else None,
            'pc_20_variance': float(np.sum(explained_variance_ratio[:21])) if len(explained_variance_ratio) >= 21 else None,
            'pc_50_variance': float(np.sum(explained_variance_ratio[:51])) if len(explained_variance_ratio) >= 51 else None,
            
            # Intrinsic dimensionality
            'intrinsic_dim_90': int(np.argmax(cumsum_variance >= 0.90) + 1) if np.any(cumsum_variance >= 0.90) else n_components,
            'intrinsic_dim_95': int(np.argmax(cumsum_variance >= 0.95) + 1) if np.any(cumsum_variance >= 0.95) else n_components,
            
            # Concentration measures
            'effective_rank': float(self._compute_effective_rank(explained_variance_ratio)),
            'participation_ratio': float(self._compute_participation_ratio(explained_variance_ratio)),
            
            # Individual components
            'first_component_variance': float(explained_variance_ratio[0]),
            'second_component_variance': float(explained_variance_ratio[1]) if len(explained_variance_ratio) > 1 else 0.0,
            
            # Total components computed
            'total_components': n_components
        }

        # Store results
        self.results[model_key] = {
            'pca': pca,
            'explained_variance_ratio': explained_variance_ratio,
            'cumsum_variance': cumsum_variance,
            'metrics': metrics,
            'feature_shape': features.shape
        }
        
        return metrics
    
    def _compute_effective_rank(self, explained_variance_ratio: np.ndarray) -> float:
        """Compute effective rank (Shannon entropy of eigenvalue distribution)."""
        eps = 1e-12
        p = explained_variance_ratio + eps
        p = p / np.sum(p)  # Normalize
        effective_rank = np.exp(-np.sum(p * np.log(p)))
        return effective_rank
    
    def _compute_participation_ratio(self, explained_variance_ratio: np.ndarray) -> float:
        """Compute participation ratio."""
        return 1.0 / np.sum(explained_variance_ratio ** 2)
    
    def analyze_saved_models(self, saved_models: Dict, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all saved models and add PCA metrics to results DataFrame.
        
        Args:
            saved_models: Dictionary of saved models from your experiment
            results_df: Existing results DataFrame
            
        Returns:
            Enhanced DataFrame with PCA interpretability metrics
        """
        print("="*60)
        print("STARTING PCA INTERPRETABILITY ANALYSIS")
        print("="*60)
        
        pca_results = []
        
        for model_key, model_info in saved_models.items():
            try:
                print(f"\nProcessing: {model_key}")
                
                # Reconstruct model from saved state
                model_state = model_info['model_state']
                model = self._reconstruct_model(model_state)
                
                # Extract features
                test_loader = model_info['test_loader']
                model_type = model_state['model_type']
                
                features = self.extract_penultimate_features(model, test_loader, model_type)
                
                # Compute PCA metrics
                metrics = self.compute_pca_metrics(features, model_key)
                
                # Add model identification info
                metrics.update({
                    'dataset': model_state['dataset'],
                    'model_type': model_state['model_type'],
                    'distance_type': model_state['distance_type'],
                    'distance_layer': model_state['distance_layer'],
                    'lambda': model_state['lambda'],
                    'test_acc': model_state['test_acc'],
                    'epoch': model_state['epoch']
                })
                
                pca_results.append(metrics)
                
                # Print key results
                print(f"✓ First-2-PC Variance: {metrics['first_2_components_variance']:.1%}")
                print(f"✓ Effective Rank: {metrics['effective_rank']:.2f}")
                print(f"✓ Intrinsic Dim (90%): {metrics['intrinsic_dim_90']}")
                
            except Exception as e:
                print(f"✗ Failed to process {model_key}: {e}")
                continue
        
        if not pca_results:
            print("No models successfully processed!")
            return results_df
        
        # Create PCA results DataFrame
        pca_df = pd.DataFrame(pca_results)
        
        # Merge with existing results
        merge_columns = ['dataset', 'model_type', 'distance_type', 'distance_typeLayer', 'lambda']
        
        # Handle column name differences
        if 'distance_typeLayer' in results_df.columns:
            pca_df['distance_typeLayer'] = pca_df['distance_layer']
        
        # Merge on common columns
        enhanced_df = results_df.merge(
            pca_df[['dataset', 'model_type', 'distance_type', 'distance_layer', 'lambda'] + 
                   [col for col in pca_df.columns if col.endswith('_variance') or 
                    col.startswith('intrinsic_') or col in ['effective_rank', 'participation_ratio']]],
            left_on=merge_columns,
            right_on=['dataset', 'model_type', 'distance_type', 'distance_layer', 'lambda'],
            how='left',
            suffixes=('', '_pca')
        )
        
        print(f"\n✓ Successfully analyzed {len(pca_results)} models")
        print(f"✓ Enhanced results shape: {enhanced_df.shape}")
        
        return enhanced_df
    
    def _reconstruct_model(self, model_state: Dict) -> torch.nn.Module:
        """
        Reconstruct model from saved state dictionary.
        
        Args:
            model_state: Dictionary containing model info and state_dict
            
        Returns:
            Reconstructed PyTorch model
        """
        from utils.model_factory import ModelFactory
        
        # Extract model parameters
        model_type = model_state['model_type']
        dataset = model_state['dataset']
        distance_layer = model_state.get('distance_layer', 'baseline')
        
        # Determine number of classes
        num_classes = 100 if dataset == 'CIFAR100' else 10
        
        # Create model using your existing factory
        # (You'll need to import your config or create a minimal one)
        config = {'hardware': 'cuda'}  # Minimal config for model creation
        
        try:
            if distance_layer != 'baseline':
                model_factory = ModelFactory(model_type, dataset, config, self.device, 
                                           distance_layer, num_classes=num_classes)
            else:
                model_factory = ModelFactory(model_type, dataset, config, self.device, 
                                           num_classes=num_classes)
            
            model = model_factory.get_fresh_model()
            
            # Load saved weights
            model.load_state_dict(model_state['model_state_dict'])
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to reconstruct {model_type} model: {e}")
    
    def save_interpretability_summary(self, enhanced_df: pd.DataFrame, output_dir: str, 
                                    timestamp: str) -> str:
        """
        Save interpretability analysis summary.
        
        Args:
            enhanced_df: DataFrame with PCA metrics
            output_dir: Directory to save results
            timestamp: Timestamp for filename
            
        Returns:
            Path to saved summary file
        """
        # Create interpretability summary
        summary_columns = [
            'dataset', 'model_type', 'distance_type', 'distance_typeLayer', 'lambda',
            'test_acc', 'first_2_components_variance', 'effective_rank', 
            'intrinsic_dim_90', 'first_component_variance'
        ]
        
        available_columns = [col for col in summary_columns if col in enhanced_df.columns]
        summary_df = enhanced_df[available_columns].copy()
        
        # Convert variance to percentage for readability
        if 'first_2_components_variance' in summary_df.columns:
            summary_df['first_2_pc_variance_pct'] = (summary_df['first_2_components_variance'] * 100).round(1)
        
        # Sort by interpretability score
        if 'first_2_components_variance' in summary_df.columns:
            summary_df = summary_df.sort_values('first_2_components_variance', ascending=False)
        
        # Save summary
        summary_file = os.path.join(output_dir, f'interpretability_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"✓ Interpretability summary saved: {summary_file}")
        
        # Print top results
        print("\n" + "="*60)
        print("TOP INTERPRETABILITY RESULTS (First 2 Components Variance)")
        print("="*60)
        
        if 'first_2_pc_variance_pct' in summary_df.columns:
            top_results = summary_df.head(10)
            for _, row in top_results.iterrows():
                print(f"{row['dataset']:<10} {row['model_type']:<10} {row['distance_type']:<12} "
                      f"{row.get('distance_typeLayer', 'N/A'):<15} λ={row['lambda']:<6.3f} "
                      f"Acc:{row.get('test_acc', 0):<6.2f}% PC2:{row['first_2_pc_variance_pct']:<6.1f}%")
        
        return summary_file

# Integration function for your main experiment runner
def run_pca_interpretability_analysis(saved_models: Dict, results_df: pd.DataFrame, 
                                    output_dir: str, config: Dict, timestamp: str = None) -> pd.DataFrame:
    """
    Main integration function to run PCA interpretability analysis.
    
    This function integrates with your existing experiment workflow.
    
    Args:
        saved_models: Dictionary of saved models from experiment
        results_df: Existing results DataFrame
        output_dir: Directory for outputs
        config: Experiment configuration
        timestamp: Optional timestamp for filenames
        
    Returns:
        Enhanced DataFrame with PCA interpretability metrics
    """
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Initialize analyzer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = PCAInterpretabilityAnalyzer(
        n_components=50,  # Standardized across all experiments
        standardize=True,
        sample_size=2000,  # Limit for efficiency
        device=device
    )
    
    # Run analysis
    enhanced_df = analyzer.analyze_saved_models(saved_models, results_df)
    
    # Save enhanced results
    enhanced_file = os.path.join(output_dir, f'results_with_interpretability_{timestamp}.csv')
    enhanced_df.to_csv(enhanced_file, index=False)
    print(f"✓ Enhanced results saved: {enhanced_file}")
    
    # Save interpretability summary
    analyzer.save_interpretability_summary(enhanced_df, output_dir, timestamp)
    
    return enhanced_df