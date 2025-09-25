import sys
sys.path.append("../")

from utils.driver import set_seed
set_seed(57)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

CODECARBON_AVAILABLE = True

from utils.model import *
from utils.number_viz_helpers import *

#dataset = "CIFAR10"
dataset = "MNIST"

# Define distance metrics and their corresponding layer classes
distance_configs = {
    'euclidean': EuclideanDistLayer,
    'manhattan': ManhattanDistLayer,
    'cosine': CosineDistLayer,
    'cosine_stable': CosineDistLayerStable,
    'minkowski': MinkowskiDistLayer,
    'bray_curtis_standard': BrayCurtisDistLayer,
    'bray_curtis_abs': BrayCurtisDistLayerAbs,
    'bray_curtis_norm': BrayCurtisDistLayerNormalized,
    'mahalanobis': MahalanobisDistLayer,
    'mahalanobis_cholesky': MahalanobisDistLayerCholesky,
    'mahalanobis_diagonal': MahalanobisDistLayerDiagonal,
    'hamming': HammingDistLayer,
    'hamming_soft': HammingDistLayerSoft,
    'hamming_gumbel': HammingDistLayerGumbel,
    'chebyshev': ChebyshevDistLayer,
    'chebyshev_soft': ChebyshevDistLayerSmooth,
    'canberra_standard': CanberraDistLayer,
    'canberra_weighted': CanberraDistLayerWeighted,
    'canberra_robust': CanberraDistLayerRobust
}

class SimpleNN(nn.Module):
    def __init__(self, distance_layer_class, harmonic=False):
        super(SimpleNN, self).__init__()
        self.harmonic = harmonic
        if harmonic:
            layer_name = distance_layer_class.__name__
            
            if dataset == "CIFAR10":
                #if layer_name == 'MinkowskiDistLayer':
                    # Changed from 28*28 (784) to 32*32*3 (3072) for CIFAR-10
                    #self.fc1 = distance_layer_class(32 * 32 * 3, 10, p=2.0, n=32., eps=1e-3)
                if layer_name == 'MinkowskiDistLayer':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, p=1.5, n=1.0, eps=1e-1)  # Changed
                #elif layer_name == 'MahalanobisDistLayer':
                    #self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=32., learn_cov=False, eps=1e-3)
                elif layer_name == 'MahalanobisDistLayer':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=0.1, learn_cov=False, eps=1e-1)  # Changed
                elif layer_name == 'MahalanobisDistLayerCholesky':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=1.0, eps=1e-2)
                #elif layer_name == 'MahalanobisDistLayerDiagonal':
                    #self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=32., eps=1e-3)
                elif layer_name == 'MahalanobisDistLayerDiagonal':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=0.1, eps=1e-1)
                elif layer_name == 'HammingDistLayer':
                    try:
                        self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=1.0, eps=1e-2)
                    except:
                        self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=1.0, threshold=0.0, eps=1e-2)
                elif layer_name == 'HammingDistLayerSoft':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=1.0, temperature=10.0, eps=1e-2)
                elif layer_name == 'HammingDistLayerGumbel':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=1.0, temperature=50.0, eps=1e-2)
                elif layer_name == 'CanberraDistLayer':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=1.0, eps=1e-1)
                elif layer_name == 'CanberraDistLayerRobust':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=1.0, min_denom=1e-1, eps=1e-2)
                elif layer_name == 'BrayCurtisDistLayer':
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=0.1, eps=1e-1)  # Changed
                else:
                    # Default case - changed n parameter from 28 to 32
                    self.fc1 = distance_layer_class(32 * 32 * 3, 10, n=32.)

            elif dataset == "MNIST":
                if layer_name == 'MinkowskiDistLayer':
                    self.fc1 = distance_layer_class(28 * 28, 10, p=1.5, n=28., eps=1e-3)
                elif layer_name == 'MahalanobisDistLayer':
                    self.fc1 = distance_layer_class(28 * 28, 10, n=28., learn_cov=False, eps=1e-3)
                elif layer_name == 'MahalanobisDistLayerCholesky':
                    # Try with much smaller n and higher eps for stability
                    self.fc1 = distance_layer_class(28 * 28, 10, n=1.0, eps=1e-2)
                elif layer_name == 'MahalanobisDistLayerDiagonal':
                    self.fc1 = distance_layer_class(28 * 28, 10, n=28., eps=1e-3)
                elif layer_name == 'HammingDistLayer':
                    # Try with much smaller n and no threshold
                    try:
                        self.fc1 = distance_layer_class(28 * 28, 10, n=1.0, eps=1e-2)
                    except:
                        # Fallback if threshold parameter is required
                        self.fc1 = distance_layer_class(28 * 28, 10, n=1.0, threshold=0.0, eps=1e-2)
                elif layer_name == 'HammingDistLayerSoft':
                    self.fc1 = distance_layer_class(28 * 28, 10, n=1.0, temperature=10.0, eps=1e-2)
                elif layer_name == 'HammingDistLayerGumbel':
                    # Try with very high temperature and small n
                    self.fc1 = distance_layer_class(28 * 28, 10, n=1.0, temperature=50.0, eps=1e-2)
                elif layer_name == 'CanberraDistLayer':
                    # Try with much smaller n for stability
                    self.fc1 = distance_layer_class(28 * 28, 10, n=1.0, eps=1e-1)
                elif layer_name == 'CanberraDistLayerRobust':
                    # Use much higher min_denom and smaller n
                    self.fc1 = distance_layer_class(28 * 28, 10, n=1.0, min_denom=1e-1, eps=1e-2)
                else:
                    # Default case for layers that work with n parameter
                    self.fc1 = distance_layer_class(28 * 28, 10, n=28.)
        else:
            if dataset == "CIFAR10":
                self.fc1 = nn.Linear(32 * 32 * 3, 10)
            elif dataset == "MNIST":
                self.fc1 = nn.Linear(28 * 28, 10)
        #nn.init.normal_(self.fc1.weight, mean=0, std=1/28.)
        
        # Updated weight initialization std from 1/28 to 1/32
        nn.init.normal_(self.fc1.weight, mean=0, std=1/32.)

    def forward(self, x):
        if dataset == "CIFAR10":
            # Changed from 28*28 to 32*32*3 for flattening
            x = x.view(-1, 32 * 32 * 3)  # Flatten the input
            x = self.fc1(x)
            if self.harmonic:
                prob = x/torch.sum(x, dim=1, keepdim=True)
                logits = (-1)*torch.log(prob)
                return logits
            return x
        elif dataset == "MNIST":
            x = x.view(-1, 28 * 28)  # Flatten the input
            x = self.fc1(x)
            if self.harmonic:
                prob = x/torch.sum(x, dim=1, keepdim=True)
                logits = (-1)*torch.log(prob)
                return logits
            return x

# 3. Update data loading section in train_and_evaluate function
def train_and_evaluate(distance_name, distance_layer_class=None, is_baseline=False):
    """Train and evaluate model with specific distance layer or baseline"""
    if is_baseline:
        print("\n")
        print(f"Training baseline model (standard linear layer)")
        print("\n")
    else:
        print("\n")
        print(f"Training with {distance_name} distance layer")
        print("\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Original MNIST Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    #max_epochs = 10

    # Potential CIFAR Hyperparameters
    # batch_size = 64  # Could increase to 128 or 256
    # learning_rate = 0.001  # Might need adjustment
    max_epochs = 10  # Might need more epochs for CIFAR-10
    
    # Load datasets
    if dataset == "CIFAR10":
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
        ])

        # Alternative simpler transform (similar to original MNIST approach):
        # transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])

        train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    elif dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Initialize the model, loss function, and optimizer
    if is_baseline:
        model = SimpleNN(None, harmonic=False).to(device)
    else:
        model = SimpleNN(distance_layer_class, harmonic=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    patience = 10
    min_delta = 1e-4
    best_loss = 1e9
    epochs_no_improve = 0
    final_epoch = 0

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(data)
            
            # Different loss computation for baseline vs harmonic models
            if is_baseline:
                loss = criterion(outputs, targets)
            else:
                loss = outputs[range(targets.size(0)), targets].mean()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        final_epoch = epoch + 1  # Store the current epoch number (1-indexed)
        
        # Early stopping check
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}], Loss: {running_loss / len(train_loader):.4f}")
        
        if epochs_no_improve >= patience:
            print(f"Stopping training. No improvement for {patience} epochs.")
            break
    
    print(f"Training completed at epoch {final_epoch}")
    
    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(data)
            
            # Different evaluation for baseline vs harmonic models
            if is_baseline:
                _, predicted = torch.max(outputs, 1)
            else:
                outputs = (-1)*outputs
                _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / len(test_dataset) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save the model
    if dataset == "CIFAR10":
        if is_baseline:
            model_path = f"../results/cifar10_baseline.pth"
        else:
            model_path = f"../results/cifar10_harmonic_{distance_name}.pth"
        torch.save(model.state_dict(), model_path)
    elif dataset == "MNIST":
        if is_baseline:
            model_path = f"../results/mnist_baseline.pth"
        else:
            model_path = f"../results/mnist_harmonic_{distance_name}.pth"
        torch.save(model.state_dict(), model_path)
    
    # Create weight visualization
    '''if is_baseline:
        create_weight_visualization("baseline", model_path, accuracy)
    else:
        create_weight_visualization(distance_name, model_path, accuracy)'''
    
    return accuracy, final_epoch

# 5. Update results file name
def main():
    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)
    
    # Setup emissions directory
    emissions_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emissions_data")
    
    # Store results
    results = {}
    
    # First, run the baseline model
    print("Starting baseline model training...")
    tracker = setup_emissions_tracker("baseline", emissions_dir, dataset=dataset)  # Individual tracking
    
    if tracker:
        print("Starting emissions tracker for baseline")
        tracker.start()
    
    try:
        accuracy, final_epoch = train_and_evaluate("baseline", None, is_baseline=True)
        results["baseline"] = {'accuracy': accuracy, 'final_epoch': final_epoch}
        
    except Exception as e:
        print(f"Error training baseline model: {str(e)}")
        results["baseline"] = {'accuracy': None, 'final_epoch': None}
    
    # Stop emissions tracking for baseline
    if tracker:
        try:
            tracker.stop()
            print("Stopped emissions tracker for baseline")
        except Exception as stop_error:
            print(f"Warning: Could not stop emissions tracker for baseline: {stop_error}")
    
    # Loop through all distance configurations
    for distance_name, distance_layer_class in distance_configs.items():
        # Setup individual tracker for each distance layer
        tracker = setup_emissions_tracker(distance_name, emissions_dir, dataset=dataset)
        
        if tracker:
            print(f"Starting emissions tracker for {distance_name}")
            tracker.start()
        
        try:
            accuracy, final_epoch = train_and_evaluate(distance_name, distance_layer_class)
            results[distance_name] = {'accuracy': accuracy, 'final_epoch': final_epoch}
            
        except Exception as e:
            print(f"Error training with {distance_name}: {str(e)}")
            results[distance_name] = {'accuracy': None, 'final_epoch': None}
        
        # Stop emissions tracking
        if tracker:
            try:
                tracker.stop()
                print(f"Stopped emissions tracker for {distance_name}")
            except Exception as stop_error:
                print(f"Warning: Could not stop emissions tracker for {distance_name}: {stop_error}")
    
    # Print summary of results
    print("\n")
    print("SUMMARY OF RESULTS")
    print("\n")
    
    # Sort results by accuracy (descending)
    sorted_results = sorted([(name, data['accuracy']) for name, data in results.items() if data['accuracy'] is not None], 
                           key=lambda x: x[1], reverse=True)
    
    print(f"{'Distance Layer':<25} {'Accuracy (%)':<12} {'Final Epoch':<12}")
    print("-" * 52)
    
    for distance_name, accuracy in sorted_results:
        final_epoch = results[distance_name]['final_epoch']
        print(f"{distance_name:<25} {accuracy:>8.2f}%     {final_epoch:>8}")
    
    # Print failed experiments
    failed = [name for name, data in results.items() if data['accuracy'] is None]
    if failed:
        print(f"\nFailed experiments: {', '.join(failed)}")
    
    # Save results to file
    with open(f"../results/{dataset}_distance_comparison_results.txt", "w") as f:  # Changed filename
        f.write("CIFAR-10 Distance Layer Comparison Results\n") 
        f.write("\n\n")
        f.write(f"{'Distance Layer':<25} {'Accuracy (%)':<12} {'Final Epoch':<12}\n")
        f.write("\n")
        
        for distance_name, accuracy in sorted_results:
            final_epoch = results[distance_name]['final_epoch']
            f.write(f"{distance_name:<25} {accuracy:>8.2f}%     {final_epoch:>8}\n")
        
        if failed:
            f.write(f"\nFailed experiments: {', '.join(failed)}\n")

    print("\nCreating combined visualization...")
    if dataset == "CIFAR10":
        create_combined_visualization_cifar(results)
    elif dataset == "MNIST":
        create_combined_visualization_mnist(results)
    
    # Consolidate emissions data and clean up individual files
    if CODECARBON_AVAILABLE:
        consolidated_df = consolidate_emissions_data(emissions_dir, results, dataset)
        display_emissions_summary(consolidated_df)
    else:
        print("\nEmissions tracking not available (CodeCarbon not installed)")

if __name__ == "__main__":
    main()