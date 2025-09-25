import torch
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalized distance functions - still GPU-efficient
def euclidean_distance_optimized(embedding):
    # Normalize by embedding dimension for consistent scaling
    raw_distance = torch.mean(torch.norm(embedding, p=2, dim=1))
    return raw_distance / embedding.shape[1]

def manhattan_distance_optimized(embedding):
    raw_distance = torch.mean(torch.norm(embedding, p=1, dim=1))
    return raw_distance / embedding.shape[1]

def cosine_distance_optimized(embedding):
    normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
    similarity_matrix = torch.mm(normalized, normalized.t())
    n = embedding.size(0)
    total_sim = torch.sum(similarity_matrix) - n  # subtract diagonal
    raw_distance = 1.0 - total_sim / (n * (n - 1))
    # Cosine distance is already normalized [0,2], but we can scale by a factor
    return raw_distance / 2.0  # Now [0,1] range

def minkowski_distance_optimized(embedding, p=3):
    raw_distance = torch.mean(torch.norm(embedding, p=p, dim=1))
    return raw_distance / embedding.shape[1]

def chebyshev_distance_optimized(embedding):
    raw_distance = torch.mean(torch.norm(embedding, p=float('inf'), dim=1))
    # Chebyshev is max of absolute differences, normalize differently
    return raw_distance / torch.sqrt(torch.tensor(embedding.shape[1], dtype=torch.float))

def canberra_distance_optimized(embedding):
    abs_embedding = torch.abs(embedding)
    raw_distance = torch.mean(torch.sum(abs_embedding / (abs_embedding + 1e-8), dim=1))
    return raw_distance / embedding.shape[1]

def bray_curtis_distance(embedding):
    mean_embedding = torch.mean(embedding, dim=0, keepdim=True)
    numerator_bc = torch.sum(torch.abs(embedding - mean_embedding), dim=1)
    denominator_bc = torch.sum(torch.abs(embedding) + torch.abs(mean_embedding), dim=1) + 1e-8
    raw_distance = torch.mean(numerator_bc / denominator_bc)
    # Bray-Curtis is already [0,1], minimal normalization needed
    return raw_distance

def hamming_median_distance(embedding):
    threshold = torch.median(embedding)
    binary_embedding = (embedding > threshold).float()
    mean_binary = torch.mean(binary_embedding, dim=0)
    diff = torch.abs(binary_embedding - mean_binary)
    raw_distance = torch.mean(torch.sum(diff, dim=1))
    return raw_distance / embedding.shape[1]

def hamming_mean_distance(embedding):
    threshold = torch.mean(embedding)
    binary_embedding = (embedding > threshold).float()
    mean_binary = torch.mean(binary_embedding, dim=0, keepdim=True)
    raw_distance = torch.mean(torch.sum(torch.abs(binary_embedding - mean_binary), dim=1))
    return raw_distance / embedding.shape[1]

def mahalanobis_distance_optimized(embedding):
    batch_size, embed_dim = embedding.shape
    
    if batch_size == 1:
        return torch.tensor(0.0, device=embedding.device)
    
    mean_embedding = torch.mean(embedding, dim=0)
    centered = embedding - mean_embedding
    distances_squared = torch.sum(centered * centered, dim=1)
    raw_distance = torch.mean(torch.sqrt(distances_squared + 1e-8))
    return raw_distance / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float))

def mahalanobis_distance_accurate(embedding):
    batch_size, embed_dim = embedding.shape
    
    if batch_size == 1:
        return torch.tensor(0.0, device=embedding.device)
    
    mean_embedding = torch.mean(embedding, dim=0)
    centered = embedding - mean_embedding
    cov = torch.mm(centered.t(), centered) / (batch_size - 1)
    cov_inv = torch.linalg.pinv(cov + 1e-6 * torch.eye(embed_dim, device=embedding.device))
    distances_squared = torch.sum((centered @ cov_inv) * centered, dim=1)
    raw_distance = torch.mean(torch.sqrt(distances_squared + 1e-8))
    return raw_distance / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float))