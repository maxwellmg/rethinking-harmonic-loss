import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# Simple MLP/CNN transforms - no need to resize to 224x224
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ViT transforms - requires 224x224
transform_mnist_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

transform_cifar_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# PVT transforms
transform_mnist_pvt = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# VGG/ResNet transforms - training from scratch with dataset-specific normalization
# MNIST transforms for VGG/ResNet
transform_mnist_vgg_resnet_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

transform_mnist_vgg_resnet_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# CIFAR-10 transforms for VGG/ResNet
transform_cifar10_vgg_resnet_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 stats
])

transform_cifar10_vgg_resnet_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR-100 transforms for VGG/ResNet (with stronger augmentation for more complex dataset)
transform_cifar100_vgg_resnet_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),  # Slightly more rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 stats
])

transform_cifar100_vgg_resnet_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# CIFAR-100 transforms for other models
transform_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

def get_loaders(dataset, model_type, batch_size):
    """
    Get data loaders with appropriate transforms for different model types
    
    Args:
        dataset: 'MNIST', 'CIFAR10', or 'CIFAR100'
        model_type: 'MLP', 'CNN', 'VGG16', 'ResNet50', 'ViT', 'PVT', or their _DIST variants
        batch_size: batch size for data loaders
    
    Returns:
        train_loader, test_loader, in_channels
    """
    
    # Extract base model type (remove _DIST suffix if present)
    base_model_type = model_type.replace('_DIST', '')
    
    if dataset == 'MNIST':
        data_path = './data_MNIST'
        download_needed = not os.path.exists(os.path.join(data_path, 'processed'))
        
        # Select transforms based on base model type
        if base_model_type in ['VGG16', 'ResNet50']:
            train_transform = transform_mnist_vgg_resnet_train
            test_transform = transform_mnist_vgg_resnet_test
            in_channels = 1  # Single channel for MNIST
            
        elif base_model_type == 'ViT':
            train_transform = transform_mnist_vit
            test_transform = transform_mnist_vit
            in_channels = 3  # Converted to 3 channels for ViT
            
        elif base_model_type == 'PVT':
            train_transform = transform_mnist_pvt
            test_transform = transform_mnist_pvt
            in_channels = 3  # Converted to 3 channels for PVT
            
        else:  # MLP, CNN, or other models (and their _DIST variants)
            train_transform = transform_mnist
            test_transform = transform_mnist
            in_channels = 1
        
        # Load datasets
        train_set = datasets.MNIST(data_path, train=True, download=download_needed, transform=train_transform)
        test_set = datasets.MNIST(data_path, train=False, download=download_needed, transform=test_transform)
        
    elif dataset == 'CIFAR10':
        data_path = './data_CIFAR10'
        download_needed = not os.path.exists(os.path.join(data_path, 'cifar-10-batches-py'))

        # Select transforms based on base model type
        if base_model_type in ['VGG16', 'ResNet50']:
            train_transform = transform_cifar10_vgg_resnet_train
            test_transform = transform_cifar10_vgg_resnet_test
            
        elif base_model_type == 'ViT':
            train_transform = transform_cifar_vit
            test_transform = transform_cifar_vit
            
        elif base_model_type in ['PVT', 'MLP', 'CNN']:
            train_transform = transform_cifar
            test_transform = transform_cifar
            
        else:
            train_transform = transform_cifar
            test_transform = transform_cifar
        
        in_channels = 3  # CIFAR-10 always has 3 channels
        
        # Load datasets
        train_set = datasets.CIFAR10(data_path, train=True, download=download_needed, transform=train_transform)
        test_set = datasets.CIFAR10(data_path, train=False, download=download_needed, transform=test_transform)
        
    elif dataset == 'CIFAR100':
        data_path = './data_CIFAR100'
        download_needed = not os.path.exists(os.path.join(data_path, 'cifar-100-python'))

        # Select transforms based on base model type
        if base_model_type in ['VGG16', 'ResNet50']:
            train_transform = transform_cifar100_vgg_resnet_train
            test_transform = transform_cifar100_vgg_resnet_test
            
        elif base_model_type == 'ViT':
            train_transform = transform_cifar_vit  # Can reuse CIFAR ViT transform
            test_transform = transform_cifar_vit
            
        elif base_model_type in ['PVT', 'MLP', 'CNN']:
            train_transform = transform_cifar100
            test_transform = transform_cifar100
            
        else:
            train_transform = transform_cifar100
            test_transform = transform_cifar100
        
        in_channels = 3  # CIFAR-100 always has 3 channels
        
        # Load datasets
        train_set = datasets.CIFAR100(data_path, train=True, download=download_needed, transform=train_transform)
        test_set = datasets.CIFAR100(data_path, train=False, download=download_needed, transform=test_transform)
        
    else:
        print(f"Unsupported dataset: {dataset}")
        raise ValueError("Unsupported dataset")

    # DataLoader configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset == 'CIFAR100':
        # For H100, use more aggressive settings
        num_workers = 32 if torch.cuda.is_available() else 0  # Increase from 16
        prefetch_factor = 16 if torch.cuda.is_available() else None  # Increase from 8
    else:
        num_workers = 16 if torch.cuda.is_available() else 0
        prefetch_factor = 8 if torch.cuda.is_available() else None
    

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor, 
        drop_last=True   
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, in_channels


'''import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# Simple MLP/CNN transforms - no need to resize to 224x224
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ViT transforms - requires 224x224
transform_mnist_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

transform_cifar_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# PVT transforms
transform_mnist_pvt = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# VGG/ResNet transforms - training from scratch with dataset-specific normalization
# MNIST transforms for VGG/ResNet
transform_mnist_vgg_resnet_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

transform_mnist_vgg_resnet_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# CIFAR-10 transforms for VGG/ResNet
transform_cifar_vgg_resnet_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 stats
])

transform_cifar_vgg_resnet_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def get_loaders(dataset, model_type, batch_size):
    """
    Get data loaders with appropriate transforms for different model types
    
    Args:
        dataset: 'MNIST' or 'CIFAR10'
        model_type: 'MLP', 'CNN', 'VGG16', 'ResNet50', 'ViT', 'PVT', or their _DIST variants
        batch_size: batch size for data loaders
    
    Returns:
        train_loader, test_loader, in_channels
    """
    
    # Extract base model type (remove _DIST suffix if present)
    base_model_type = model_type.replace('_DIST', '')
    
    if dataset == 'MNIST':
        data_path = './data_MNIST'
        download_needed = not os.path.exists(os.path.join(data_path, 'processed'))
        
        # Select transforms based on base model type
        if base_model_type in ['VGG16', 'ResNet50']:
            train_transform = transform_mnist_vgg_resnet_train
            test_transform = transform_mnist_vgg_resnet_test
            in_channels = 1  # Single channel for MNIST
            
        elif base_model_type == 'ViT':
            train_transform = transform_mnist_vit
            test_transform = transform_mnist_vit
            in_channels = 3  # Converted to 3 channels for ViT
            
        elif base_model_type == 'PVT':
            train_transform = transform_mnist_pvt
            test_transform = transform_mnist_pvt
            in_channels = 3  # Converted to 3 channels for PVT
            
        else:  # MLP, CNN, or other models (and their _DIST variants)
            train_transform = transform_mnist
            test_transform = transform_mnist
            in_channels = 1
        
        # Load datasets
        train_set = datasets.MNIST(data_path, train=True, download=download_needed, transform=train_transform)
        test_set = datasets.MNIST(data_path, train=False, download=download_needed, transform=test_transform)
        
    elif dataset == 'CIFAR10':
        data_path = './data_CIFAR'
        download_needed = not os.path.exists(os.path.join(data_path, 'cifar-10-batches-py'))

        # Select transforms based on base model type
        if base_model_type in ['VGG16', 'ResNet50']:
            train_transform = transform_cifar_vgg_resnet_train
            test_transform = transform_cifar_vgg_resnet_test
            
        elif base_model_type == 'ViT':
            train_transform = transform_cifar_vit
            test_transform = transform_cifar_vit
            
        elif base_model_type in ['PVT', 'MLP', 'CNN']:
            train_transform = transform_cifar
            test_transform = transform_cifar
            
        else:
            train_transform = transform_cifar
            test_transform = transform_cifar
        
        in_channels = 3  # CIFAR-10 always has 3 channels
        
        # Load datasets
        train_set = datasets.CIFAR10(data_path, train=True, download=download_needed, transform=train_transform)
        test_set = datasets.CIFAR10(data_path, train=False, download=download_needed, transform=test_transform) 
        
    else:
        print(f"Unsupported dataset: {dataset}")
        raise ValueError("Unsupported dataset")

    # DataLoader configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 16 if torch.cuda.is_available() else 0
    prefetch_factor = 8 if torch.cuda.is_available() else None

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor, 
        drop_last=True   
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, in_channels
'''