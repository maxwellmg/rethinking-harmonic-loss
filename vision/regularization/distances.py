import torch
from torch import nn, optim


def no_regularization(embedding):
    reg = torch.tensor(0.0, device=embedding.device)
    return reg

def euclidean_distance(embedding):
    reg = torch.mean(torch.sqrt(torch.mean(embedding**2, dim=0)))
    return reg

def manhattan_distance(embedding):
    reg = torch.mean(torch.sum(torch.abs(embedding), dim=0))
    return reg

def cosine_distance(embedding):
    # Compute pairwise cosine distances
    norm_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    cosine_sim = torch.mm(norm_embedding, norm_embedding.t())
    # Remove diagonal and compute mean distance
    mask = ~torch.eye(cosine_sim.size(0), dtype=torch.bool, device=embedding.device)
    reg = 1 - cosine_sim[mask].mean()
    return reg

def canberra_distance(embedding):
    numerator = torch.abs(embedding)  # Fixed: use 'embedding' not 'model.embedding'
    denominator = torch.abs(embedding) + 1e-8  # Fixed: use 'embedding' not 'model.embedding'
    reg = torch.mean((numerator / denominator).sum(dim=0))
    return reg

def bray_curtis_distance(embedding):
    # Compare each row with the mean of all rows
    mean_embedding = torch.mean(embedding, dim=0, keepdim=True)
    numerator_bc = torch.sum(torch.abs(embedding - mean_embedding), dim=1)
    denominator_bc = torch.sum(torch.abs(embedding) + torch.abs(mean_embedding), dim=1) + 1e-8
    reg = torch.mean(numerator_bc / denominator_bc)
    return reg

def minkowski_distance(embedding, p=3):
    # Minkowski distance with p=3 (you can adjust p as needed)
    reg = torch.mean(torch.pow(torch.sum(torch.pow(torch.abs(embedding), p), dim=0), 1/p))
    return reg

def chebyshev_distance(embedding):
    # Chebyshev distance (L-infinity norm)
    reg = torch.mean(torch.max(torch.abs(embedding), dim=0)[0])
    return reg

def hamming_distance(embedding):
    # For continuous embeddings, we'll threshold and compare
    # This is an approximation since Hamming is typically for discrete data
    threshold = torch.mean(embedding)
    binary_embedding = (embedding > threshold).float()
    mean_binary = torch.mean(binary_embedding, dim=0, keepdim=True)
    reg = torch.mean(torch.sum(torch.abs(binary_embedding - mean_binary), dim=1))
    return reg

def mahalanobis_distance(embedding):
    # Simplified Mahalanobis distance
    mean_embedding = torch.mean(embedding, dim=0, keepdim=True)
    centered = embedding - mean_embedding
    # Use identity covariance matrix for simplicity (or compute actual covariance)
    cov_inv = torch.eye(embedding.shape[1], dtype=embedding.dtype, device=embedding.device)
    distances = torch.sum((centered @ cov_inv) * centered, dim=1)
    reg = torch.mean(torch.sqrt(distances + 1e-8))
    return reg