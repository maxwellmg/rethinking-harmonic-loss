import torch
import sys
import os
import warnings
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_embedding_similarity(embedding):
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    sim_matrix = torch.matmul(embedding, embedding.T)
    B = sim_matrix.size(0)
    # Remove diagonal (self-similarity)
    sim_matrix = sim_matrix - torch.eye(B, device=embedding.device)
    return sim_matrix.abs().mean().item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_embedding_similarity_optimized(embedding):
    if embedding.size(0) <= 1:
        return torch.tensor(0.0, device=embedding.device)
    
    # Normalize embeddings
    normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    # Compute similarity matrix more efficiently
    sim_matrix = torch.mm(normalized, normalized.t())
    
    # Remove diagonal without creating identity matrix (more memory efficient)
    batch_size = sim_matrix.size(0)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=embedding.device)
    
    # Return tensor (no .item() call) - keep on GPU
    return sim_matrix[mask].abs().mean()

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