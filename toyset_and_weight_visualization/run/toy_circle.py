import sys
import torch
import json
import os
import glob
import pandas as pd

# Set matplotlib backend based on environment
import matplotlib
if os.environ.get('DISPLAY') is None:  # No display available (headless server)
    matplotlib.use('Agg')
    print("Using Agg backend (headless mode)")
else:
    print("Using default backend (display available)")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math

# Import CodeCarbon for emissions tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
    print("CodeCarbon available for emissions tracking")
except ImportError:
    print("Warning: CodeCarbon not available. Install with 'pip install codecarbon'")
    CODECARBON_AVAILABLE = False

sys.path.append("../")

from utils.driver import train_single_model

# CONSOLIDATED CONFIGURATIONS - All models and distance types
configurations = [
    # Standard models (explicitly setting distance_type to None)
    ('standard_MLP', 0.0, None),
    ('standard_MLP', 0.5, None),
    ('standard_transformer', 0.0, None),
    
    # Harmonic MLPs - All distance types
    ('H_MLP', 0.0, 'euclidean'),
    ('H_MLP', 0.0, 'manhattan'),
    ('H_MLP', 0.0, 'cosine'),
    ('H_MLP', 0.0, 'cosine_stable'),
    ('H_MLP', 0.0, 'minkowski'),
    ('H_MLP', 0.0, 'hamming'),
    ('H_MLP', 0.0, 'hamming_soft'),
    ('H_MLP', 0.0, 'hamming_gumbel'),
    ('H_MLP', 0.0, 'chebyshev'),
    ('H_MLP', 0.0, 'chebyshev_soft'),
    ('H_MLP', 0.0, 'canberra_standard'),
    ('H_MLP', 0.0, 'canberra_weighted'),
    ('H_MLP', 0.0, 'canberra_robust'),
    ('H_MLP', 0.0, 'bray_curtis_standard'),
    ('H_MLP', 0.0, 'bray_curtis_abs'),
    ('H_MLP', 0.0, 'bray_curtis_norm'),
    ('H_MLP', 0.0, 'mahalanobis'),
    ('H_MLP', 0.0, 'mahalanobis_cholesky'),
    ('H_MLP', 0.0, 'mahalanobis_diagonal'),
    
    # Harmonic Transformers - All distance types
    ('H_transformer', 0.0, 'euclidean'),
    ('H_transformer', 0.0, 'manhattan'),
    ('H_transformer', 0.0, 'cosine'),
    ('H_transformer', 0.0, 'cosine_stable'),
    ('H_transformer', 0.0, 'minkowski'),
    ('H_transformer', 0.0, 'hamming'),
    ('H_transformer', 0.0, 'hamming_soft'),
    ('H_transformer', 0.0, 'hamming_gumbel'),
    ('H_transformer', 0.0, 'chebyshev'),
    ('H_transformer', 0.0, 'chebyshev_soft'),
    ('H_transformer', 0.0, 'canberra_standard'),
    ('H_transformer', 0.0, 'canberra_weighted'),
    ('H_transformer', 0.0, 'canberra_robust'),
    ('H_transformer', 0.0, 'bray_curtis_standard'),
    ('H_transformer', 0.0, 'bray_curtis_abs'),
    ('H_transformer', 0.0, 'bray_curtis_norm'),
    ('H_transformer', 0.0, 'mahalanobis'),
    ('H_transformer', 0.0, 'mahalanobis_cholesky'),
    ('H_transformer', 0.0, 'mahalanobis_diagonal')
]

print(f"Loaded {len(configurations)} configurations")
# Verify all configurations have 3 elements
for i, config in enumerate(configurations):
    if len(config) != 3:
        print(f"Warning: Configuration {i} has {len(config)} elements: {config}")
    else:
        model_id, weight_decay, distance_type = config
        print(f"Config {i}: {model_id}, {weight_decay}, {distance_type}")
        if i >= 5:  # Only show first few to avoid spam
            print(f"... and {len(configurations)-6} more configurations")
            break

def setup_emissions_tracker(config_name, emissions_dir):
    """Setup emissions tracker for carbon footprint monitoring"""
    if not CODECARBON_AVAILABLE:
        return None
    
    try:
        # Ensure the emissions_data directory exists
        os.makedirs(emissions_dir, exist_ok=True)
                 
        tracker = EmissionsTracker(
            project_name=f"Circle_Case_Study_{config_name}",
            output_dir=emissions_dir,
            output_file=f"emissions_{config_name}.csv",  # Simplified filename
            save_to_file=True,
            log_level="error"
        )
        return tracker
    except Exception as e:
        print(f"Warning: Could not setup emissions tracker: {e}")
        return None

def consolidate_emissions_data(emissions_dir, results, backup_individual_files=True):
    """Read individual emissions CSV files and consolidate with accuracy data"""
    if not CODECARBON_AVAILABLE:
        print("CodeCarbon not available - skipping emissions consolidation")
        return None
        
    try:
        # Find all emissions CSV files in the directory
        emission_files = glob.glob(os.path.join(emissions_dir, "emissions_*.csv"))
        
        # Filter out any existing consolidated files
        emission_files = [f for f in emission_files if 'consolidated' not in os.path.basename(f)]
        
        if not emission_files:
            print("No individual emissions files found to consolidate")
            return None
        
        print(f"Found {len(emission_files)} emissions files to consolidate")
        consolidated_data = []
        successful_files = []
        
        for file_path in emission_files:
            try:
                # Extract config name from filename
                filename = os.path.basename(file_path)
                config_name = filename.replace("emissions_", "").replace(".csv", "")
                
                print(f"Processing emissions file: {filename} -> config: {config_name}")
                
                # Read the emissions CSV file
                df = pd.read_csv(file_path)
                
                if not df.empty:
                    # Get the last row (most recent measurement) for each config
                    last_row = df.iloc[-1].copy()
                    
                    # Add configuration info and results
                    last_row['configuration'] = config_name
                    result_data = results.get(config_name, {'train_accuracy': None, 'test_accuracy': None})
                    last_row['final_train_accuracy'] = result_data['train_accuracy']
                    last_row['final_test_accuracy'] = result_data['test_accuracy']
                    
                    consolidated_data.append(last_row)
                    successful_files.append(file_path)
                    print(f"✓ Successfully processed {config_name}")
                else:
                    print(f"⚠ Empty emissions file: {filename}")
                    
            except Exception as e:
                print(f"✗ Could not process emissions file {file_path}: {e}")
                continue
        
        if consolidated_data:
            # Create consolidated DataFrame
            consolidated_df = pd.DataFrame(consolidated_data)
            
            # Reorder columns to put configuration and accuracy first
            cols = consolidated_df.columns.tolist()
            priority_cols = ['configuration', 'final_train_accuracy', 'final_test_accuracy']
            other_cols = [col for col in cols if col not in priority_cols]
            consolidated_df = consolidated_df[priority_cols + other_cols]
            
            # Save consolidated file
            consolidated_file = os.path.join(emissions_dir, "emissions_consolidated.csv")
            consolidated_df.to_csv(consolidated_file, index=False)
            
            print(f"\n✓ Consolidated emissions data saved to: {consolidated_file}")
            print(f"✓ Consolidated {len(consolidated_data)} configurations")
            
            # Only clean up individual files if consolidation was successful AND backup_individual_files is False
            if not backup_individual_files and len(successful_files) > 0:
                print("\nCleaning up individual emissions files...")
                for file_path in successful_files:
                    try:
                        os.remove(file_path)
                        print(f"Removed: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")
            else:
                print(f"\nKeeping {len(successful_files)} individual emissions files as backup")
            
            return consolidated_df
        else:
            print("No valid emissions data found to consolidate")
            return None
            
    except Exception as e:
        print(f"Error consolidating emissions data: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_emissions_summary(consolidated_df):
    """Display a summary of the emissions data"""
    if consolidated_df is None or consolidated_df.empty:
        print("No emissions data to display")
        return
    
    print(f"\nEmissions Summary ({len(consolidated_df)} configurations):")
    print("=" * 120)
    
    # Define which columns to display (common CodeCarbon columns)
    display_cols = ['configuration', 'final_train_accuracy', 'final_test_accuracy']
    
    # Add common emissions columns if they exist
    possible_cols = ['duration', 'emissions', 'emissions_rate', 'cpu_power', 'gpu_power', 
                    'ram_power', 'energy_consumed']
    
    for col in possible_cols:
        if col in consolidated_df.columns:
            display_cols.append(col)
    
    # Create header with fixed widths
    col_widths = {
        'configuration': 30,
        'final_train_accuracy': 15,
        'final_test_accuracy': 15,
        'duration': 12,
        'emissions': 12,
        'emissions_rate': 15,
        'cpu_power': 12,
        'gpu_power': 12,
        'ram_power': 12,
        'energy_consumed': 15
    }
    
    header = ""
    for col in display_cols:
        width = col_widths.get(col, 12)
        header += f"{col.replace('_', ' ').title():<{width}} "
    
    print(header)
    print("-" * len(header))
    
    # Display data rows
    for _, row in consolidated_df.iterrows():
        row_str = ""
        for col in display_cols:
            width = col_widths.get(col, 12)
            value = row.get(col, 'N/A')
            
            if col in ['final_train_accuracy', 'final_test_accuracy'] and isinstance(value, (int, float)):
                formatted_value = f"{value:.4f}"
            elif col in ['duration', 'emissions', 'emissions_rate', 'cpu_power', 'gpu_power', 
                        'ram_power', 'energy_consumed'] and isinstance(value, (int, float)):
                if value < 0.001:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            # Truncate if too long
            if len(formatted_value) > width - 1:
                formatted_value = formatted_value[:width-1]
            
            row_str += f"{formatted_value:<{width}} "
        
        print(row_str)
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("Summary Statistics:")
    
    if 'final_train_accuracy' in consolidated_df.columns:
        train_acc_valid = consolidated_df['final_train_accuracy'].dropna()
        if len(train_acc_valid) > 0:
            print(f"Train Accuracy - Mean: {train_acc_valid.mean():.4f}, Std: {train_acc_valid.std():.4f}, Min: {train_acc_valid.min():.4f}, Max: {train_acc_valid.max():.4f}")
    
    if 'final_test_accuracy' in consolidated_df.columns:
        test_acc_valid = consolidated_df['final_test_accuracy'].dropna()
        if len(test_acc_valid) > 0:
            print(f"Test Accuracy  - Mean: {test_acc_valid.mean():.4f}, Std: {test_acc_valid.std():.4f}, Min: {test_acc_valid.min():.4f}, Max: {test_acc_valid.max():.4f}")
    
    if 'emissions' in consolidated_df.columns:
        emissions_valid = consolidated_df['emissions'].dropna()
        if len(emissions_valid) > 0:
            print(f"Emissions (kg)  - Total: {emissions_valid.sum():.6f}, Mean: {emissions_valid.mean():.6f}, Max: {emissions_valid.max():.6f}")

# TRAINING LOOP WITH EMISSIONS TRACKING
print(f"Running {len(configurations)} configurations with emissions tracking...")

# Setup emissions directory
emissions_dir = os.path.join("../emissions_data")

# Store results for emissions consolidation
results = {}

for i, config in enumerate(configurations):
    # Handle different tuple lengths safely
    if len(config) == 3:
        model_id, weight_decay, distance_type = config
    elif len(config) == 2:
        model_id, weight_decay = config
        distance_type = None
    else:
        print(f"Warning: Unexpected configuration format: {config}")
        continue
        
    print(f"\nConfiguration {i+1}/{len(configurations)}: {model_id}, weight_decay={weight_decay}, distance_type={distance_type}")
    
    # Create unique configuration name
    if distance_type is not None:
        config_name = f'{model_id}_{distance_type}_{weight_decay}'
    else:
        config_name = f'{model_id}_{weight_decay}'
    
    # Setup emissions tracker for this configuration
    tracker = setup_emissions_tracker(config_name, emissions_dir)
    
    if tracker:
        print(f"Starting emissions tracker for {config_name}")
        tracker.start()
    
    try:
        param_dict = {
            'seed': 50,
            'data_id': 'circle',
            'data_size': 1000,
            'train_ratio': 0.8,
            'model_id': model_id,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'embd_dim': 16,
            'n_exp': 1,
            'lr': 0.001,
            'weight_decay': weight_decay,
            'num_epochs': 2000,  # Full training
            'distance_type': distance_type  # Always include distance_type
        }
        
        result_dict = train_single_model(param_dict)
        
        train_acc = result_dict['results']['train_accuracies']
        test_acc = result_dict['results']['test_accuracies']
        
        # Store final accuracies for emissions consolidation
        results[config_name] = {
            'train_accuracy': train_acc[-1] if train_acc else None,
            'test_accuracy': test_acc[-1] if test_acc else None
        }
        
        model = result_dict['model']
        
        # Save model and results - CONSISTENT NAMING
        filename_suffix = config_name
        
        torch.save(model.state_dict(), f'../results/case_study_{filename_suffix}.pt')
        with open(f'../results/case_study_{filename_suffix}_results.json', 'w') as f:
            json.dump(result_dict['results'], f, indent=4)
        
        print(f"✓ Training completed - Train Acc: {results[config_name]['train_accuracy']:.4f}, Test Acc: {results[config_name]['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"✗ Error training {config_name}: {str(e)}")
        results[config_name] = {'train_accuracy': None, 'test_accuracy': None}
        import traceback
        traceback.print_exc()
    
    # Stop emissions tracking
    if tracker:
        try:
            tracker.stop()
            print(f"Stopped emissions tracker for {config_name}")
        except Exception as stop_error:
            print(f"Warning: Could not stop emissions tracker for {config_name}: {stop_error}")

print(f"\nTraining complete! Trained {len([r for r in results.values() if r['train_accuracy'] is not None])}/{len(configurations)} configurations successfully.")

# Consolidate emissions data (keeping individual files as backup)
if CODECARBON_AVAILABLE:
    print("\nConsolidating emissions data...")
    consolidated_df = consolidate_emissions_data(emissions_dir, results, backup_individual_files=True)
    display_emissions_summary(consolidated_df)
else:
    print("\nEmissions tracking not available (CodeCarbon not installed)")

print("\nNow creating visualizations...")

# VISUALIZATION FUNCTION
def create_adaptive_visualization_with_max_cols(config_list, max_cols=6, save_prefix="circle_case_study"):
    """
    Create visualization with maximum number of columns per row.
    If you have more configs than max_cols, it will create multiple rows.
    """
    num_configs = len(config_list)
    plt.rcParams.update({'font.size': 10})  # Smaller font for many plots

    # Check if we're on a server
    is_headless = os.environ.get('DISPLAY') is None
    print(f"Running in {'headless' if is_headless else 'interactive'} mode")
    print(f"Current matplotlib backend: {matplotlib.get_backend()}")
    
    if num_configs <= max_cols:
        # Single row of configs
        num_rows = 2  # accuracy + embeddings
        num_cols = num_configs
        fig_width = max(15, num_configs * 3.0)  # Increased width
        fig_height = 10  # Increased height
    else:
        # Multiple rows of configs
        config_rows = math.ceil(num_configs / max_cols)
        num_rows = config_rows * 2  # Each config row has accuracy + embeddings
        num_cols = min(num_configs, max_cols)
        fig_width = max(15, num_cols * 3.0)  # Increased width
        fig_height = config_rows * 5.5  # More spacing per config row
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Adjust subplot spacing to prevent overlap
    plt.subplots_adjust(
        left=0.05,      # Left margin
        bottom=0.08,    # Bottom margin  
        right=0.95,     # Right margin
        top=0.92,       # Top margin
        wspace=0.25,    # Width spacing between subplots
        hspace=0.45     # Height spacing between subplots (increased)
    )

    distance_names = {
        'euclidean': 'Euclidean',
        'manhattan': 'Manhattan',
        'cosine': 'Cosine',
        'cosine_stable': 'Cosine (Stable)',
        'minkowski': 'Minkowski',
        'hamming': 'Hamming (Hard)',
        'hamming_soft': 'Hamming (Soft)',
        'hamming_gumbel': 'Hamming (Gumbel)',
        'chebyshev': 'Chebyshev',
        'chebyshev_soft': 'Chebyshev (Smooth)',
        'canberra_standard': 'Canberra (Regular)',
        'canberra_robust': 'Canberra (Robust)',
        'canberra_weighted': 'Canberra (Weighted)',
        'bray_curtis_standard': 'Bray-Curtis (Regular)',
        'bray_curtis_abs': 'Bray-Curtis (Absolute)',
        'bray_curtis_norm': 'Bray-Curtis (Normalized)',
        'mahalanobis': 'Mahalanobis (Regular)',
        'mahalanobis_cholesky': 'Mahalanobis (Cholesky)',
        'mahalanobis_diagonal': 'Mahalanobis (Diagonal)',
    }
    
    for i, config in enumerate(config_list):
        # Calculate position in grid
        config_row = i // max_cols
        config_col = i % max_cols
        
        model_id, weight_decay, distance_type = config
        
        if distance_type is not None:
            filename_suffix = f'{model_id}_{distance_type}_{weight_decay}'
        else:
            filename_suffix = f'{model_id}_{weight_decay}'
        
        # ACCURACY PLOT (top row for each config row)
        acc_subplot_idx = config_row * 2 * max_cols + config_col + 1
        plt.subplot(num_rows, max_cols, acc_subplot_idx)
        
        try:
            # CONSISTENT FILE LOADING - matches the save names
            with open(f'../results/case_study_{filename_suffix}_results.json', 'r') as f:
                results_data = json.load(f)
            train_acc = results_data['train_accuracies']
            test_acc = results_data['test_accuracies']
            plt.plot(train_acc, label='train', linewidth=1.5)
            plt.plot(test_acc, label='test', linewidth=1.5)
            plt.legend(fontsize=8)
        except FileNotFoundError:
            print(f"Warning: Results file not found for {filename_suffix}")
            plt.text(0.5, 0.5, 'No results\nfound', ha='center', va='center', transform=plt.gca().transAxes)
            
        # Title logic with cleaner distance names
        if model_id == 'standard_MLP':
            if weight_decay == 0:
                plt.title('Standard MLP\nweight decay=0', fontsize=9)
            else:
                plt.title(f'Standard MLP\nweight decay={weight_decay}', fontsize=9)
        elif model_id == 'standard_transformer':
            plt.title('Standard\nTransformer', fontsize=9)
        elif model_id == 'H_MLP':
            clean_name = distance_names.get(distance_type, distance_type)
            plt.title(f'H-MLP\n({clean_name})', fontsize=9)
        elif model_id == 'H_transformer':
            clean_name = distance_names.get(distance_type, distance_type)
            plt.title(f'H-Trans\n({clean_name})', fontsize=9)
        else:
            plt.title(f'{model_id}\n{distance_type or "default"}', fontsize=9)
        
        plt.xlabel('Epoch', fontsize=8)
        if config_col == 0:
            plt.ylabel('Accuracy', fontsize=8)
        plt.ylim(-0.1, 1.1)
        plt.xticks([0, 1000], fontsize=8)
        plt.yticks([0, 0.5, 1], fontsize=8)
        
        # EMBEDDING PLOT (bottom row for each config row)
        emb_subplot_idx = (config_row * 2 + 1) * max_cols + config_col + 1
        plt.subplot(num_rows, max_cols, emb_subplot_idx)
        
        try:
            # CONSISTENT FILE LOADING - matches the save names
            weights = torch.load(f'../results/case_study_{filename_suffix}.pt', map_location='cpu')
            if 'embedding.weight' in weights:
                rep = weights['embedding.weight'].cpu().numpy()
            elif 'embedding' in weights:
                rep = weights['embedding'].cpu().numpy()
            else:
                # Find embedding weights in state dict
                embedding_keys = [k for k in weights.keys() if 'embedding' in k and 'weight' in k]
                if embedding_keys:
                    embedding_key = embedding_keys[0]
                    rep = weights[embedding_key].cpu().numpy()
                else:
                    raise KeyError("No embedding weights found")
                
            pca = PCA(n_components=2)
            rep_pca = pca.fit_transform(rep)
            plt.scatter(rep_pca[:,0], rep_pca[:,1], c='b', s=8, alpha=0.7)
            
            # Add numbers for first 31 points (modular arithmetic vocab)
            for j in range(min(31, len(rep_pca))):
                plt.text(rep_pca[j,0], rep_pca[j,1], str(j), fontsize=6, ha='center', va='center')
            
            plt.xlabel('PC0', fontsize=8)
            if config_col == 0:
                plt.ylabel('PC1', fontsize=8)
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            
            ev_text = f"EV: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.0f}%"
            plt.text(0.8, 0.1, ev_text, fontsize=8, transform=plt.gca().transAxes, ha='center', va='center')
            
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Model file issue for {filename_suffix}: {e}")
            plt.text(0.5, 0.5, f'No embedding\nfound\n{type(e).__name__}', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=8)
    
    # Create figures directory if it doesn't exist
    figures_dir = '../figures'
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Figures directory created/exists: {os.path.exists(figures_dir)}")
    
    # Save files with error handling
    pdf_path = f'{figures_dir}/{save_prefix}_{num_configs}_configs.pdf'
    png_path = f'{figures_dir}/{save_prefix}_{num_configs}_configs.png'
    
    try:
        # Use subplots_adjust instead of tight_layout for better control
        # tight_layout can sometimes cause overlapping with complex subplot arrangements
        plt.savefig(pdf_path, bbox_inches='tight', dpi=150, pad_inches=0.3)
        plt.savefig(png_path, bbox_inches='tight', dpi=150, pad_inches=0.3)
        
        # Only show if we have a display
        if not is_headless:
            plt.show()
        else:
            print("Skipping plt.show() in headless mode")
            
        plt.close()  # Always close to free memory
        
        # Verify files were created
        if os.path.exists(pdf_path):
            print(f"✓ PDF saved: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
        else:
            print(f"✗ PDF not found: {pdf_path}")
            
        if os.path.exists(png_path):
            print(f"✓ PNG saved: {png_path} ({os.path.getsize(png_path)} bytes)")
        else:
            print(f"✗ PNG not found: {png_path}")
            
    except Exception as e:
        print(f"Error saving plots: {e}")
        import traceback
        traceback.print_exc()


# CREATE VISUALIZATIONS with different column limits
print("Creating visualization with max 6 columns per row...")
create_adaptive_visualization_with_max_cols(configurations, max_cols=6, save_prefix="consolidated_circle_case_study_6col")

print("Creating visualization with max 4 columns per row...")
create_adaptive_visualization_with_max_cols(configurations, max_cols=4, save_prefix="consolidated_circle_case_study_4col")

print("All visualizations complete!")