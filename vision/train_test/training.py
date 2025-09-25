import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal Imports
from embedding_metrics import compute_embedding_similarity_optimized
#from regularization.distances import *
from regularization.optimized_regularized_distances import *

#distance_list = ['baseline', 'euclidean', 'manhattan', 'cosine', 'minkowski', 'chebyshev', 'canberra', 'bray-curtis', 'hamming', 'mahalanobis']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DISTANCE_FUNCTIONS = {
    'baseline': lambda x: torch.tensor(0.0, device=x.device),
    'euclidean': euclidean_distance_optimized,
    'manhattan': manhattan_distance_optimized,
    'cosine': cosine_distance_optimized,
    'minkowski': minkowski_distance_optimized,
    'chebyshev': chebyshev_distance_optimized,
    'canberra': canberra_distance_optimized,
    'bray-curtis': bray_curtis_distance,
    'hamming_mean': hamming_median_distance,
    'hamming_median': hamming_mean_distance,
    'mahalanobis1': mahalanobis_distance_optimized,
    'mahalanobis2': mahalanobis_distance_accurate
}

def train(model, loader, optimizer, distance_type, criterion, epoch, lamb):
    model.train()
    
    # Use tensors to avoid GPU-CPU sync
    total_loss_epoch = 0.0
    total_correct = 0
    total_samples = 0
    total_reg_tensor = torch.tensor(0.0, device=device)  # Keep on GPU
    total_emb_norm_tensor = torch.tensor(0.0, device=device)  # Keep on GPU
    batch_count = 0
    total_similarity_tensor = torch.tensor(0.0, device=device)  # Keep on GPU

    # Get distance function once, outside the loop
    distance_func = DISTANCE_FUNCTIONS.get(distance_type, DISTANCE_FUNCTIONS['baseline'])
    apply_regularization = distance_type != 'baseline' and lamb > 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output, embedding = model(data, return_embedding=True)
        loss = criterion(output, target)

        if apply_regularization and embedding.numel() > 0:
            # Compute regularization term
            reg = distance_func(embedding)
            total_loss = loss + lamb * reg
            
            # Accumulate on GPU (no sync)
            total_reg_tensor += reg
            total_emb_norm_tensor += torch.mean(torch.norm(embedding, dim=1))
            
            # Compute embedding similarity
            sim_score = compute_embedding_similarity_optimized(embedding)
            if isinstance(sim_score, torch.Tensor):
                total_similarity_tensor += sim_score
            else:
                total_similarity_tensor += torch.tensor(sim_score, device=device)
        else:
            total_loss = loss
            # For baseline, add zeros to maintain tensor operations
            total_reg_tensor += torch.tensor(0.0, device=device)
            if embedding.numel() > 0:
                total_emb_norm_tensor += torch.mean(torch.norm(embedding, dim=1))
                sim_score = compute_embedding_similarity_optimized(embedding)
                if isinstance(sim_score, torch.Tensor):
                    total_similarity_tensor += sim_score
                else:
                    total_similarity_tensor += torch.tensor(sim_score, device=device)

        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()  # Only sync at end of training step
        
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += target.size(0)
        batch_count += 1

    # Only sync with CPU at the very end
    train_acc = 100.0 * total_correct / total_samples
    avg_loss = total_loss_epoch / batch_count
    avg_reg = total_reg_tensor.item() / batch_count  # Single sync at end
    avg_emb_norm = total_emb_norm_tensor.item() / batch_count  # Single sync at end
    avg_similarity = total_similarity_tensor.item() / batch_count  # Single sync at end

    return {
        'epoch': epoch,
        'train_loss': round(avg_loss, 5),
        'train_acc': round(train_acc, 5),
        'reg_term': round(avg_reg, 5),
        'embedding_norm': round(avg_emb_norm, 5),
        'embedding_similarity': round(avg_similarity, 5)
    }


# Suppress NumPy warnings since we're only using baseline
'''warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
'''


def safe_embedding_similarity(embedding):
    """Safely compute embedding similarity with proper error handling"""
    if embedding.numel() == 0 or embedding.size(0) <= 1:
        return torch.tensor(0.0, device=embedding.device)
    
    try:
        sim_score = compute_embedding_similarity_optimized(embedding)
        if isinstance(sim_score, torch.Tensor):
            if torch.isnan(sim_score) or torch.isinf(sim_score):
                return torch.tensor(0.0, device=embedding.device)
            return sim_score
        else:
            if np.isnan(sim_score) or np.isinf(sim_score):
                return torch.tensor(0.0, device=embedding.device)
            return torch.tensor(sim_score, device=embedding.device)
    except Exception:
        return torch.tensor(0.0, device=embedding.device)

def safe_embedding_norm(embedding):
    """Safely compute embedding norm with proper error handling"""
    if embedding.numel() == 0:
        return torch.tensor(0.0, device=embedding.device)
    
    try:
        emb_norms = torch.norm(embedding, dim=1)
        if emb_norms.numel() == 0:
            return torch.tensor(0.0, device=embedding.device)
        
        mean_norm = torch.mean(emb_norms)
        if torch.isnan(mean_norm) or torch.isinf(mean_norm):
            return torch.tensor(0.0, device=embedding.device)
        
        return mean_norm
    except Exception:
        return torch.tensor(0.0, device=embedding.device)

def dist_train(model, loader, optimizer, distance_type, criterion, epoch, lamb):
    model.train()
    
    # Simplified variables since we're only doing baseline
    total_loss_epoch = 0.0
    total_correct = 0
    total_samples = 0
    total_emb_norm_tensor = torch.tensor(0.0, device=device)
    total_similarity_tensor = torch.tensor(0.0, device=device)
    batch_count = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output, embedding = model(data, return_embedding=True)
        loss = criterion(output, target)
        
        # For baseline, no regularization - just the loss
        total_loss = loss

        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()
        
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += target.size(0)
        batch_count += 1
        
        # Safe embedding metrics computation
        total_emb_norm_tensor += safe_embedding_norm(embedding)
        total_similarity_tensor += safe_embedding_similarity(embedding)

    # Compute final metrics
    train_acc = 100.0 * total_correct / total_samples
    avg_loss = total_loss_epoch / batch_count
    avg_emb_norm = total_emb_norm_tensor.item() / batch_count
    avg_similarity = total_similarity_tensor.item() / batch_count

    return {
        'epoch': epoch,
        'train_loss': round(avg_loss, 5),
        'train_acc': round(train_acc, 5),
        'reg_term': 0.0,  # Always 0 for baseline
        'embedding_norm': round(avg_emb_norm, 5),
        'embedding_similarity': round(avg_similarity, 5)
    }

# Even simpler version if you don't need embedding metrics at all
# Also update train_minimal to be extra safe
def train_minimal(model, loader, optimizer, distance_type, criterion, epoch, lamb):
    """Minimal train function for baseline only - no embedding metrics"""
    model.train()
    
    total_loss_epoch = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        # Simple forward pass - no need for embeddings
        if hasattr(model, 'forward') and 'return_embedding' in model.forward.__code__.co_varnames:
            output = model(data, return_embedding=False)
        else:
            output = model(data)
            
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += target.size(0)
        batch_count += 1

    train_acc = 100.0 * total_correct / total_samples
    avg_loss = total_loss_epoch / batch_count

    return {
        'epoch': epoch,
        'train_loss': round(avg_loss, 5),
        'train_acc': round(train_acc, 5),
        'reg_term': 0.0,
        'embedding_norm': 0.0,
        'embedding_similarity': 0.0
    }