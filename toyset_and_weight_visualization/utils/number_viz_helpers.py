import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from codecarbon import EmissionsTracker
from matplotlib.backends.backend_pdf import PdfPages

CODECARBON_AVAILABLE = True

def setup_emissions_tracker(distance_name, emissions_dir, dataset):
    """Setup emissions tracker for carbon footprint monitoring"""
    if not CODECARBON_AVAILABLE:
        return None
    
    try:
        os.makedirs(emissions_dir, exist_ok=True)
                 
        tracker = EmissionsTracker(
            project_name=f"{dataset}_Distance_Layer_{distance_name}",
            output_dir=emissions_dir,
            output_file=f"emissions_{dataset}_{distance_name}.csv",
            save_to_file=True,
            log_level="error"
        )
        return tracker
    except Exception as e:
        print(f"Warning: Could not setup emissions tracker: {e}")
        return None

def create_combined_visualization_cifar(results):
    # Create three different PDF visualizations for CIFAR-10 weights:
    # 1. Channel separation (RGB channels shown separately)
    # 2. RGB composite (full color images)
    # 3. Grayscale average (averaged channels, similar to MNIST approach)
    
    # Extract accuracy from the results dictionary structure
    sorted_results = sorted(
        [(name, data['accuracy']) for name, data in results.items() 
         if data['accuracy'] is not None], 
        key=lambda x: x[1], reverse=True
    )
    
    plt.rcParams.update({'font.size': 10})
    
    print("Creating CIFAR-10 weight visualizations...")
    
    # VISUALIZATION 1: CHANNEL SEPARATION
    print("Creating channel separation visualization...")
    with PdfPages("../figures/cifar10_harmonic_weights_channels.pdf") as pdf:
        for distance_name, accuracy in sorted_results:
            try:
                if distance_name == "baseline":
                    model_path = f"../results/cifar10_baseline.pth"
                else:
                    model_path = f"../results/cifar10_harmonic_{distance_name}.pth"
                
                weights = torch.load(model_path, map_location='cpu')
                
                # Create figure with more space for RGB visualization
                fig, axes = plt.subplots(10, 3, figsize=(8, 20))
                plt.suptitle(f"CIFAR-10 Channels: {distance_name.title()} Weight Visualization (Accuracy: {accuracy:.2f}%)", 
                             fontsize=12)
                
                for i in range(10):  # 10 classes
                    # Get weights for class i and reshape to CIFAR-10 dimensions
                    weight = weights['fc1.weight'][i].reshape(3, 32, 32).detach().cpu().numpy()
                    
                    # Normalize weights for better visualization
                    weight_normalized = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
                    
                    # Show each color channel separately
                    channel_names = ['Red', 'Green', 'Blue']
                    for channel in range(3):
                        ax = axes[i, channel]
                        
                        # Apply threshold for binary visualization
                        channel_weight = np.where(weight_normalized[channel] < 0.5, 0, 1)
                        
                        ax.imshow(channel_weight, cmap='gray', vmin=0, vmax=1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # Add labels
                        if i == 0:
                            ax.set_title(f'{channel_names[channel]}', fontsize=10)
                        if channel == 0:
                            ax.set_ylabel(f'Class {i}', fontsize=10)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not add {distance_name} to channel separation PDF: {e}")
    
    print("Channel separation visualization saved to: ../figures/cifar10_harmonic_weights_channels.pdf")
    
    # VISUALIZATION 2: RGB COMPOSITE
    print("Creating RGB composite visualization...")
    with PdfPages("../figures/cifar10_harmonic_weights_rgb.pdf") as pdf:
        for distance_name, accuracy in sorted_results:
            try:
                # Handle different model file naming for baseline vs distance layers
                if distance_name == "baseline":
                    model_path = f"../results/cifar10_baseline.pth"
                else:
                    model_path = f"../results/cifar10_harmonic_{distance_name}.pth"
                
                weights = torch.load(model_path, map_location='cpu')
                
                # Create figure for RGB composite visualization
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                plt.suptitle(f"CIFAR-10 RGB: {distance_name.title()} Weight Visualization (Accuracy: {accuracy:.2f}%)", 
                             fontsize=12)
                
                for i in range(10):  # 10 classes
                    row = i // 5
                    col = i % 5
                    ax = axes[row, col]
                    
                    # Get weights for class i and reshape to CIFAR-10 dimensions
                    weight = weights['fc1.weight'][i].reshape(3, 32, 32).detach().cpu().numpy()
                    
                    # Transpose to (H, W, C) for matplotlib
                    weight_rgb = np.transpose(weight, (1, 2, 0))
                    
                    # Normalize each channel independently for better visualization
                    for c in range(3):
                        channel_min = weight_rgb[:, :, c].min()
                        channel_max = weight_rgb[:, :, c].max()
                        if channel_max > channel_min:
                            weight_rgb[:, :, c] = (weight_rgb[:, :, c] - channel_min) / (channel_max - channel_min)
                        else:
                            weight_rgb[:, :, c] = 0.5
                    
                    # Clip values to [0, 1] range
                    weight_rgb = np.clip(weight_rgb, 0, 1)
                    
                    ax.imshow(weight_rgb)
                    ax.set_title(f'Class {i}', fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not add {distance_name} to RGB composite PDF: {e}")
    
    print("RGB composite visualization saved to: ../figures/cifar10_harmonic_weights_rgb.pdf")
    
    # VISUALIZATION 3: GRAYSCALE AVERAGE
    print("Creating grayscale average visualization...")
    with PdfPages("../figures/cifar10_harmonic_weights_grayscale.pdf") as pdf:
        for distance_name, accuracy in sorted_results:
            try:
                # Handle different model file naming for baseline vs distance layers
                if distance_name == "baseline":
                    model_path = f"../results/cifar10_baseline.pth"
                else:
                    model_path = f"../results/cifar10_harmonic_{distance_name}.pth"
                
                weights = torch.load(model_path, map_location='cpu')
                
                # Create figure similar to original MNIST layout
                fig, ax = plt.subplots(2, 5, figsize=(10, 4))
                plt.suptitle(f"CIFAR-10 Grayscale: {distance_name.title()} Weight Visualization (Accuracy: {accuracy:.2f}%)", 
                             fontsize=10)
                
                for i in range(10):
                    plt.subplot(2, 5, i + 1)
                    
                    # Get weights for class i and reshape to CIFAR-10 dimensions
                    weight = weights['fc1.weight'][i].reshape(3, 32, 32).detach().cpu().numpy()
                    
                    # Average across color channels to create grayscale
                    weight_gray = np.mean(weight, axis=0)
                    
                    # Apply threshold similar to original (adjusted for different scale)
                    weight_thresh = np.where(weight_gray < np.percentile(weight_gray, 50), 0, 1)
                    
                    plt.imshow(weight_thresh, cmap='gray')
                    plt.title(f'Class {i}', fontsize=8)
                    plt.xticks([])
                    plt.yticks([])
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not add {distance_name} to grayscale PDF: {e}")
    
    print("Grayscale visualization saved to: ../figures/cifar10_harmonic_weights_grayscale.pdf")
    
    # SUMMARY
    print("\n" + "="*60)
    print("CIFAR-10 Weight Visualization Complete!")
    print("="*60)
    print("Three PDF files have been created:")
    print("1. Channel Separation: ../figures/cifar10_harmonic_weights_channels.pdf")
    print("2. RGB Composite:     ../figures/cifar10_harmonic_weights_rgb.pdf")
    print("3. Grayscale Average: ../figures/cifar10_harmonic_weights_grayscale.pdf")
    print("="*60)



def create_combined_visualization_mnist(results):
    """Create a combined PDF with all distance layer visualizations"""
    
    sorted_results = sorted(
        [(name, data['accuracy']) for name, data in results.items() 
         if data['accuracy'] is not None], 
        key=lambda x: x[1], reverse=True
    )
    
    plt.rcParams.update({'font.size': 10})  # Even smaller for combined view
    
    with PdfPages("../figures/mnist_harmonic_weights_combined.pdf") as pdf:
        for distance_name, accuracy in sorted_results:
            try:
                # Handle different model file naming for baseline vs distance layers
                if distance_name == "baseline":
                    model_path = f"../results/mnist_baseline.pth"
                else:
                    model_path = f"../results/mnist_harmonic_{distance_name}.pth"
                
                weights = torch.load(model_path, map_location='cpu')
                
                fig, ax = plt.subplots(2, 5, figsize=(10, 4))
                plt.suptitle(f"MNIST: {distance_name.title()} Weight Visualization (Accuracy: {accuracy:.2f}%)", 
                             fontsize=10)
                
                for i in range(10):
                    plt.subplot(2, 5, i + 1)
                    weight = weights['fc1.weight'][i].reshape(28, 28).detach().cpu().numpy()
                    weight = np.where(weight < 0.01, 1, 0)
                    plt.imshow(weight)
                    plt.xticks([])
                    plt.yticks([])
                
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not add {distance_name} to combined PDF: {e}")

def consolidate_emissions_data(emissions_dir, results, dataset):
    """Read individual emissions CSV files and consolidate with accuracy data"""
    if not CODECARBON_AVAILABLE:
        return None
        
    try:
        emission_files = glob.glob(os.path.join(emissions_dir, f"emissions_{dataset}_*.csv"))
        
        print(f"DEBUG: Looking for pattern: emissions_{dataset}_*.csv")
        print(f"DEBUG: Found files: {emission_files}")
        
        if not emission_files:
            print(f"No {dataset} emissions files found to consolidate")
            return None
        
        consolidated_data = []
        
        for file_path in emission_files:
            try:
                filename = os.path.basename(file_path)
                print(f"DEBUG: Processing filename: {filename}")
                
                distance_name = filename.replace(f"emissions_{dataset}_", "").replace(".csv", "")
                print(f"DEBUG: Extracted distance name: '{distance_name}'")
                
                df = pd.read_csv(file_path)
                
                if not df.empty:
                    last_row = df.iloc[-1].copy()
                    
                    last_row['distance_layer'] = distance_name
                    last_row['dataset'] = dataset
                    result_data = results.get(distance_name, {'accuracy': None, 'final_epoch': None})
                    last_row['accuracy'] = result_data['accuracy']
                    last_row['final_epoch'] = result_data['final_epoch']
                    
                    consolidated_data.append(last_row)
                    print(f"DEBUG: Successfully processed {distance_name}")
                    
            except Exception as e:
                print(f"Warning: Could not process emissions file {file_path}: {e}")
                continue
        
        if consolidated_data:
            consolidated_df = pd.DataFrame(consolidated_data)
            
            cols = consolidated_df.columns.tolist()
            priority_cols = ['dataset', 'distance_layer', 'accuracy', 'final_epoch']
            other_cols = [col for col in cols if col not in priority_cols]
            existing_priority_cols = [col for col in priority_cols if col in cols]
            consolidated_df = consolidated_df[existing_priority_cols + other_cols]

            consolidated_file = os.path.join(emissions_dir, f"emissions_consolidated_{dataset.lower()}.csv")
            consolidated_df.to_csv(consolidated_file, index=False)
            
            if os.path.exists(consolidated_file):
                file_size = os.path.getsize(consolidated_file)
                print(f"\nConsolidated {dataset} emissions data saved to: {consolidated_file}")
                print(f"File size: {file_size} bytes")
            else:
                print(f"ERROR: Consolidated file was not created at {consolidated_file}")
                return None
            
            print("Cleaning up individual emissions files...")
            for file_path in emission_files:
                try:
                    os.remove(file_path)
                    print(f"Removed: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
            
            return consolidated_df
        else:
            print("No valid emissions data found")
            return None
            
    except Exception as e:
        print(f"Warning: Could not consolidate emissions data: {e}")
        return None
        
def display_emissions_summary(consolidated_df):
    """Display a summary of the emissions data"""
    if consolidated_df is None or consolidated_df.empty:
        print("No emissions data to display")
        return
    
    print("\nEmissions Summary:")
    
    display_cols = ['distance_layer', 'accuracy', 'final_epoch']
    
    possible_cols = ['duration', 'emissions', 'emissions_rate', 'cpu_power', 'gpu_power', 
                    'ram_power', 'energy_consumed', 'timestamp']
    
    for col in possible_cols:
        if col in consolidated_df.columns:
            display_cols.append(col)

    header = ""
    col_widths = {}
    for col in display_cols:
        if col == 'distance_layer':
            width = 20
        elif col in ['accuracy', 'final_epoch']:
            width = 12
        elif col == 'timestamp':
            width = 20
        else:
            width = 12
        col_widths[col] = width
        header += f"{col.replace('_', ' ').title():<{width}} "
    
    print(header)
    
    for _, row in consolidated_df.iterrows():
        row_str = ""
        for col in display_cols:
            width = col_widths[col]
            value = row.get(col, 'N/A')
            
            if col == 'accuracy' and isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}%"
            elif col == 'final_epoch' and isinstance(value, (int, float)):
                formatted_value = f"{int(value)}"
            elif col in ['duration', 'emissions', 'emissions_rate', 'cpu_power', 'gpu_power', 
                        'ram_power', 'energy_consumed'] and isinstance(value, (int, float)):
                if value < 0.001:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.4f}"
            elif col == 'timestamp':
                formatted_value = str(value)[:19] if str(value) != 'N/A' else 'N/A'
            else:
                formatted_value = str(value)
            
            row_str += f"{formatted_value:<{width}} "
        
        print(row_str)