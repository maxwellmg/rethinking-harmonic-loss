def extract_config_params(config, distance_typeLayer):
    """Extract distance layer parameters from config for logging"""
    config_params = {}
    
    if 'custom_dist_params' in config and distance_typeLayer in config['custom_dist_params']:
        params = config['custom_dist_params'][distance_typeLayer]
        
        # Add config prefix to distinguish from actual layer values
        for key, value in params.items():
            config_params[f'config_{key}'] = value
    
    return config_params