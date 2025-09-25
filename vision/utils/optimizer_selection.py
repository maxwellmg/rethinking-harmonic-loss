import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

def get_learning_rate(config, model_type, dataset):
    """Get learning rate for specific model and dataset"""
    lr_config = config['learning_rates'][model_type]
    if isinstance(lr_config, dict):
        return lr_config.get(dataset, lr_config.get('CIFAR10', 3e-4))  # Fallback to CIFAR10 rate
    else:
        return lr_config  # Old format compatibility

def get_early_stopping_patience(config, dataset):
    """Get early stopping patience for specific dataset"""
    patience_config = config['early_stopping_patience']
    if isinstance(patience_config, dict):
        return patience_config.get(dataset, 15)  # Default to 15
    else:
        return patience_config  # Old format compatibility

def get_scheduler_config(config, model_type, dataset):
    """Get scheduler configuration for specific model and dataset"""
    scheduler_configs = config['scheduler_configs']
    if model_type in scheduler_configs:
        scheduler_config = scheduler_configs[model_type]
        if isinstance(scheduler_config, dict):
            return scheduler_config.get(dataset, None)  # Return None if no dataset-specific config
        else:
            return scheduler_config  # Old format compatibility
    return None

def create_dataset_specific_scheduler(optimizer, scheduler_config):
    """Create scheduler based on configuration"""
    if scheduler_config is None:
        return None
    
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == 'StepLR':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, 
                     step_size=scheduler_config['step_size'], 
                     gamma=scheduler_config['gamma'])
    
    elif scheduler_type == 'MultiStepLR':
        from torch.optim.lr_scheduler import MultiStepLR
        return MultiStepLR(optimizer,
                          milestones=scheduler_config['milestones'],
                          gamma=scheduler_config['gamma'])
    
    elif scheduler_type == 'CosineAnnealingLR':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer,
                                T_max=scheduler_config['T_max'],
                                eta_min=scheduler_config.get('eta_min', 0))
    
    elif scheduler_type == 'ExponentialLR':
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, gamma=scheduler_config['gamma'])
    
    else:
        print(f"Warning: Unknown scheduler type {scheduler_type}, returning None")
        return None

def create_dataset_specific_optimizer(model, model_type, dataset, config):
    """Create optimizer with dataset-specific learning rate"""
    # Get dataset-specific learning rate
    learning_rate = get_learning_rate(config, model_type, dataset)
    
    # Get optimizer configuration
    if model_type in config['optimizer_configs']:
        opt_config = config['optimizer_configs'][model_type]
        opt_type = opt_config['type']
        
        if opt_type == 'Adam':
            return torch.optim.Adam(model.parameters(), 
                                   lr=learning_rate,
                                   weight_decay=opt_config.get('weight_decay', 0))
        
        elif opt_type == 'AdamW':
            return torch.optim.AdamW(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=opt_config.get('weight_decay', 0.01))
        
        elif opt_type == 'SGD':
            return torch.optim.SGD(model.parameters(),
                                  lr=learning_rate,
                                  momentum=opt_config.get('momentum', 0.9),
                                  weight_decay=opt_config.get('weight_decay', 1e-4))
        
        else:
            print(f"Warning: Unknown optimizer type {opt_type}, using Adam")
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    else:
        print(f"Warning: No optimizer config for {model_type}, using Adam")
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
