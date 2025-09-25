import torch

def cleanup_memory(model, optimizer):
    """
    Clean up memory after experiment
    
    Args:
        model: PyTorch model to clean up
        optimizer: PyTorch optimizer to clean up
    """
    try:
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Error during memory cleanup: {e}")