import os
import glob
import torch
import pandas as pd
from interpretability_analysis import InterpretabilityAnalyzer

def load_and_analyze_saved_models(model_save_dir, outputs_dir, config, results_df=None):
    """
    Load saved .pth model files and run interpretability analysis
    
    Args:
        model_save_dir: Directory containing saved .pth files
        outputs_dir: Directory to save interpretability outputs
        config: Your experiment config
        results_df: Optional - results dataframe to get model metadata
    """
    print("Loading saved models for interpretability analysis...")
    
    # Import required modules
    try:
        from data.data_loaders import get_loaders
        from utils.model_factory import ModelFactory
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Find all .pth files
    pth_files = glob.glob(os.path.join(model_save_dir, "*.pth"))
    print(f"Found {len(pth_files)} .pth files")
    
    if len(pth_files) < 2:
        print("❌ Need at least 2 saved models for comparison")
        return
    
    # Parse model information from filenames
    models_info = {}
    
    for pth_file in pth_files:
        filename = os.path.basename(pth_file)
        print(f"Processing: {filename}")
        
        # Try to parse filename (adjust this based on your naming convention)
        # Expected format: {dataset}_{model_type}_{distance_type}_{distance_layer}_lambda{lambda}_best.pth
        # or similar patterns
        
        try:
            # Remove .pth extension
            name_parts = filename.replace('.pth', '').split('_')
            
            # This is a flexible parser - adjust based on your actual filenames
            model_info = parse_model_filename(filename, results_df)
            
            if model_info:
                model_key = f"{model_info['dataset']}_{model_info['model_type']}_{model_info['distance_type']}"
                models_info[model_key] = {
                    'file_path': pth_file,
                    **model_info
                }
                print(f"  ✅ Parsed: {model_key}")
            else:
                print(f"  ⚠️  Could not parse filename: {filename}")
                
        except Exception as e:
            print(f"  ❌ Error parsing {filename}: {e}")
    
    if len(models_info) < 2:
        print(f"❌ Only parsed {len(models_info)} valid models. Need at least 2.")
        return
    
    # Group by dataset
    datasets = set(info['dataset'] for info in models_info.values())
    
    # Initialize analyzer
    analyzer = InterpretabilityAnalyzer(outputs_dir, device=device)
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Analyzing dataset: {dataset}")
        print(f"{'='*50}")
        
        # Filter models for this dataset
        dataset_models = {k: v for k, v in models_info.items() if v['dataset'] == dataset}
        
        if len(dataset_models) < 2:
            print(f"Skipping {dataset} - only {len(dataset_models)} model(s)")
            continue
        
        # Load models
        loaded_models = {}
        
        for model_key, model_info in dataset_models.items():
            try:
                print(f"Loading model: {model_key}")
                
                # Create model architecture
                model = create_model_architecture(model_info, config, device)
                
                # Load weights
                state_dict = torch.load(model_info['file_path'], map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                
                print(f"  ✅ Loaded successfully")
                
                # Test forward pass
                test_input = torch.randn(1, 3, 32, 32).to(device)  # Adjust size as needed
                with torch.no_grad():
                    output = model(test_input)
                print(f"  ✅ Forward pass works, output shape: {output.shape}")
                
                loaded_models[model_key] = {
                    'model': model,
                    'model_key': model_key,
                    'model_type': model_info['model_type'],
                    'distance_type': model_info.get('distance_type', 'baseline'),
                    'distance_layer': model_info.get('distance_layer', 'baseline'),
                    'dataset': dataset,
                    'test_acc': model_info.get('test_acc', 0.0),
                    'lambda': model_info.get('lambda', 0.0)
                }
                
            except Exception as e:
                print(f"  ❌ Failed to load {model_key}: {e}")
                continue
        
        if len(loaded_models) >= 2:
            try:
                # Get test loader
                print(f"Getting test loader for {dataset}...")
                _, test_loader, _ = get_loaders(
                    dataset=dataset,
                    model_type=list(loaded_models.values())[0]['model_type'],
                    batch_size=32
                )
                
                print(f"Running interpretability analysis on {len(loaded_models)} models...")
                
                # Run analysis
                results = analyzer.compare_models_interpretability(
                    loaded_models, 
                    test_loader, 
                    num_samples=25  # Start with fewer samples for testing
                )
                
                print(f"✅ Interpretability analysis complete for {dataset}!")
                print(f"   Results saved to: {analyzer.interp_dir}")
                
            except Exception as e:
                print(f"❌ Analysis failed for {dataset}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ Only loaded {len(loaded_models)} models for {dataset}")


def parse_model_filename(filename, results_df=None):
    """
    Parse model filename to extract metadata
    Adjust this function based on your actual filename format
    """
    try:
        # Remove .pth extension
        base_name = filename.replace('.pth', '')
        
        # Try different parsing strategies
        parts = base_name.split('_')
        
        # Strategy 1: Standard format
        # Expected: dataset_modeltype_distancetype_distancelayer_lambda0.1_best.pth
        if len(parts) >= 4:
            dataset = parts[0]
            model_type = parts[1]
            
            # Look for lambda value
            lambda_val = 0.0
            for part in parts:
                if part.startswith('lambda'):
                    try:
                        lambda_val = float(part.replace('lambda', ''))
                    except:
                        pass
            
            # Determine distance info
            distance_type = 'baseline'
            distance_layer = 'baseline'
            
            if 'DIST' in model_type:
                if len(parts) >= 3:
                    distance_type = parts[2] if parts[2] != 'best' else 'euclidean'
                if len(parts) >= 4:
                    distance_layer = parts[3] if parts[3] not in ['best', f'lambda{lambda_val}'] else 'EuclideanDistLayer'
            
            # Get test accuracy from results_df if available
            test_acc = 0.0
            if results_df is not None:
                try:
                    matching_rows = results_df[
                        (results_df['dataset'] == dataset) & 
                        (results_df['model_type'] == model_type) &
                        (abs(results_df['lambda'] - lambda_val) < 0.001)
                    ]
                    if not matching_rows.empty:
                        test_acc = matching_rows['test_acc'].iloc[0]
                except:
                    pass
            
            return {
                'dataset': dataset,
                'model_type': model_type,
                'distance_type': distance_type,
                'distance_layer': distance_layer,
                'lambda': lambda_val,
                'test_acc': test_acc
            }
    
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
    
    return None


def create_model_architecture(model_info, config, device):
    """
    Create model architecture based on model_info
    Adjust this based on your ModelFactory implementation
    """
    from utils.model_factory import ModelFactory
    
    model_type = model_info['model_type']
    dataset = model_info['dataset']
    distance_layer = model_info.get('distance_layer', 'EuclideanDistLayer')
    
    # Determine number of classes
    if isinstance(config, dict):
        dataset_num_classes = config.get('dataset_num_classes', {})
        num_classes = dataset_num_classes.get(dataset, 100 if dataset == 'CIFAR100' else 10)
    else:
        if hasattr(config, 'dataset_num_classes'):
            num_classes = getattr(config.dataset_num_classes, dataset, 100 if dataset == 'CIFAR100' else 10)
        else:
            num_classes = 100 if dataset == 'CIFAR100' else 10
    
    # Create model factory
    if model_type in ['MLP_DIST', 'CNN_DIST', 'ViT_DIST', 'PVT_DIST', 'VGG16_DIST', 'ResNet50_DIST']:
        model_factory = ModelFactory(
            model_type, dataset, config, 
            device, distance_layer, num_classes=num_classes
        )
    else:
        model_factory = ModelFactory(
            model_type, dataset, config, device, num_classes=num_classes
        )
    
    return model_factory.get_fresh_model()


def simple_filename_test(model_save_dir):
    """Simple test to see what filenames look like"""
    pth_files = glob.glob(os.path.join(model_save_dir, "*.pth"))
    
    print("Found .pth files:")
    for pth_file in pth_files[:10]:  # Show first 10
        filename = os.path.basename(pth_file)
        print(f"  {filename}")
        
        # Try to parse
        parsed = parse_model_filename(filename)
        if parsed:
            print(f"    → Parsed as: {parsed}")
        else:
            print(f"    → Could not parse")
        print()


# Usage examples:

# 1. First, test filename parsing
# simple_filename_test("path/to/your/saved/models")

# 2. Run the full analysis
# load_and_analyze_saved_models(
#     model_save_dir="path/to/your/saved/models",
#     outputs_dir=outputs_dir,
#     config=config,
#     results_df=results_df  # Optional, helps get test accuracies
# )