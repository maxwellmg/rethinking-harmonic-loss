import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import math
from dataset import *
from visualization import *
from distlayers import *

import sys
# import keyboard

from tqdm import tqdm

import torch.nn.functional as F

class CustomLosses:
    @staticmethod
    def contrastive_loss(output, target, margin=1.0):
        """
        Computes contrastive loss with pairwise distance computation.

        Args:
            output (Tensor): Embeddings of shape (batch_size, embedding_dim).
            target (Tensor): Labels of shape (batch_size,).
            margin (float): Margin for contrastive loss.

        Returns:
            Tensor: Contrastive loss value.
        """
        batch_size = output.size(0)

        # Compute pairwise squared Euclidean distances
        distances = torch.cdist(output, output, p=2)  # Shape: (batch_size, batch_size)

        # Mask to extract positive pairs (same class)
        target_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # Shape: (batch_size, batch_size)
        pos_mask = target_matrix.fill_diagonal_(False)  # Exclude self-comparisons

        # Mask to extract negative pairs (different class)
        neg_mask = ~target_matrix

        # Compute positive loss
        pos_distances = distances * pos_mask  # Only keep positive pairs
        pos_loss = torch.sum(pos_distances ** 2) / (pos_mask.sum() + 1e-8)  # Prevent division by zero

        # Compute negative loss
        neg_distances = distances * neg_mask  # Only keep negative pairs
        neg_loss = torch.sum(F.relu(margin - neg_distances) ** 2) / (neg_mask.sum() + 1e-8)

        # Total contrastive loss
        loss = pos_loss + neg_loss
        return loss

    
    @staticmethod
    def constant_margin_loss(output, target, margin=1.0):
        distances = torch.norm(output, dim=1)
        loss = torch.mean(torch.clamp(margin - distances, min=0.0))
        return loss
    
    @staticmethod
    def spherical_loss(output, target, epsilon=1e-4):
        """
        Compute the spherical softmax loss as defined in the paper.
        
        Args:
            output: Tensor of shape (batch_size, num_classes) - pre-activations (logits)
            target: Tensor of shape (batch_size,) - target class indices
            epsilon: float - small constant for numerical stability
            
        Returns:
            The spherical softmax loss
        """
        # First L2 normalize the outputs along the class dimension (spherical projection)
        normalized_output = F.normalize(output, p=2, dim=1)  # σ_i = o_i / ||o||
        
        # Compute squared normalized outputs plus epsilon
        squared_normalized = normalized_output.pow(2) + epsilon  # σ_i² + ϵ
        
        # Compute spherical softmax probabilities
        probabilities = squared_normalized / squared_normalized.sum(dim=1, keepdim=True)
        
        # Gather the probabilities of the target classes
        target_probs = probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Compute negative log likelihood
        loss = -torch.log(target_probs)
        
        return loss.mean()

    
    @staticmethod
    def soft_nearest_neighbor_loss(output, target, temperature=0.1):
        """
        Robust implementation of soft nearest neighbor loss.
        
        Args:
            output: Tensor of shape (batch_size, embedding_dim) - the embeddings
            target: Tensor of shape (batch_size,) - the class labels
            temperature: float - temperature parameter
            
        Returns:
            The soft nearest neighbor loss as a scalar tensor
        """
        batch_size = output.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=output.device)
        
        # Compute pairwise squared Euclidean distances in a numerically stable way
        output_normalized = F.normalize(output, p=2, dim=1)
        cos_similarity = torch.mm(output_normalized, output_normalized.t())
        distances = 2 - 2 * cos_similarity  # converts cosine similarity to squared Euclidean
        
        # Compute exponential terms with temperature
        exp_terms = torch.exp(-distances / temperature)
        
        # Create mask for same-class pairs (excluding self)
        same_class = target.unsqueeze(1) == target.unsqueeze(0)
        same_class.fill_diagonal_(False)  # exclude self-comparisons
        
        # Compute numerator and denominator
        numerator = torch.sum(exp_terms * same_class, dim=1)
        denominator = torch.sum(exp_terms, dim=1) - 1  # subtract 1 to exclude self
        
        # Handle cases where numerator might be zero (no same-class neighbors)
        safe_ratio = torch.where(
            numerator > 0,
            numerator / denominator,
            torch.ones_like(numerator) * 1e-8  # small value when no same-class neighbors
        )
        
        # Compute loss and return mean
        losses = -torch.log(safe_ratio)
        return torch.mean(losses)


class customNNModule(nn.Module):
    def __init__(self, loss_type):
        super(customNNModule, self).__init__()
        self.loss_type = loss_type
    
    def compute_loss(self, outputs, targets):
        if self.loss_type == 'contrastive':
            return CustomLosses.contrastive_loss(outputs, targets)
        elif self.loss_type == 'constant_margin':
            return CustomLosses.constant_margin_loss(outputs, targets)
        elif self.loss_type == 'spherical':
            return CustomLosses.spherical_loss(outputs, targets)
        elif self.loss_type == 'soft_nn':
            return CustomLosses.soft_nearest_neighbor_loss(outputs, targets)
        elif self.loss_type == 'harmonic':
            return (-1)*(outputs[torch.arange(outputs.size(0)), targets.squeeze()].mean())
        elif self.loss_type == 'cross_entropy': # Cross-Entropy Loss
            return nn.CrossEntropyLoss()(outputs, targets.squeeze())
    

    def train(self, param_dict: dict):

        num_epochs = param_dict['num_epochs']
        learning_rate = param_dict['learning_rate']
        train_dataloader = param_dict['train_dataloader']
        test_dataloader = param_dict['test_dataloader']
        device = param_dict['device']
        weight_decay = param_dict['weight_decay']
        model_id = param_dict['model_id']
        video = False if 'video' not in param_dict else param_dict['video']

        verbose = True
        if 'verbose' in param_dict:
            verbose = param_dict['verbose']

        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []       

        best_loss = float('inf')
        patience = 100
        min_delta = 1e-4
        counter = 0

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lamb_reg = 0.01 if 'lambda' not in param_dict else param_dict['lambda']
        for epoch in tqdm(range(num_epochs)):
            # if keyboard.is_pressed('ctrl+d'):
            #     print("Manual early stopping occurring.")
            #     break
            if video and epoch%10 == 0: # save every 10 epochs
                if hasattr(self.embedding, 'weight'):
                    embd = self.embedding.weight
                else:
                    embd = self.embedding.data
                visualize_embedding(embd, title=f"Epoch {epoch}", save_path=f"../video_imgs/{epoch}.png", dict_level = None, color_dict = True, adjust_overlapping_text = False)

            train_loss = 0
            train_correct = 0
            train_total = 0
            for batch_inputs, batch_targets in train_dataloader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.type(torch.LongTensor).to(device)
                optimizer.zero_grad()
                outputs = self.forward(batch_inputs)

#               class_counts = torch.bincount(batch_targets.squeeze(), minlength=self.vocab_size).double() + 1e-8
#               class_weights = 1 / class_counts.cuda()

                # criterion = nn.CrossEntropyLoss()#weight=class_weights)
                # if 'H_' in model_id: # Harmonic Model
                #     loss = (-1)*(outputs[torch.arange(outputs.size(0)), batch_targets.squeeze()].mean())
                # else:
                # loss = criterion(outputs, batch_targets.squeeze())

                loss = self.compute_loss(outputs, batch_targets)
                
                if hasattr(self.embedding, 'weight'):
                    total_loss = loss + lamb_reg * torch.mean(torch.sqrt(torch.mean(self.embedding.weight**2, dim=0)))
                else:
                    total_loss = loss + lamb_reg * torch.mean(torch.sqrt(torch.mean(self.embedding.data**2, dim=0)))
                
                total_loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Compute training accuracy
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_targets).sum().item()
                train_total += batch_targets.size(0)

            test_loss = 0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for batch_inputs, batch_targets in test_dataloader:
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.type(torch.LongTensor).to(device)
                    logits = self.forward(batch_inputs)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(logits, batch_targets.squeeze())
                    test_loss += loss.item()

                    # Compute test accuracy
                    _, predicted = torch.max(logits, 1)
                    test_correct += (predicted == batch_targets).sum().item()
                    test_total += batch_targets.size(0)

            if (epoch + 1) % 50 == 0 and verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Train Acc: {train_correct / train_total:.4f}, Test Loss: {test_loss / len(test_dataloader):.4f}, Test Acc: {test_correct / test_total:.4f}")
                sys.stdout.flush()
            
            train_losses.append(train_loss / len(train_dataloader))
            test_losses.append(test_loss / len(test_dataloader))
            train_accuracies.append(train_correct / train_total)
            test_accuracies.append(test_correct / test_total)

            epoch_loss = train_loss / len(train_dataloader)
            # Check for convergence
            if best_loss - epoch_loss > min_delta:
                best_loss = epoch_loss
                counter = 0  # Reset counter if there's an improvement
            else:
                counter += 1  # Increment counter if no improvement

            '''
            if counter >= patience:
                print("Early stopping triggered!")
                break
            '''

        ret_dic = {}
        ret_dic['train_losses'] = train_losses
        ret_dic['test_losses'] = test_losses
        ret_dic['train_accuracies'] = train_accuracies
        ret_dic['test_accuracies'] = test_accuracies

        return ret_dic

    

class MLP(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., unembd=False, weight_tied=False, seed=0, loss_type='cross_entropy'):
        super(MLP, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.depth = len(shp) - 1
            
        linear_list = []
        for i in range(self.depth):
            linear_list.append(nn.Linear(shp[i], shp[i+1]))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim))
#        self.embedding = torch.nn.Parameter(torch.normal(0,1/torch.tensor(embd_dim),size=(vocab_size, embd_dim))*init_scale)
        #self.embedding = torch.nn.Parameter(torch.normal(0,1,size=(vocab_size, embd_dim))*init_scale)
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.unembd = unembd
        
        if unembd:
            assert shp[-2] == embd_dim
            if weight_tied:
                #self.linears[-1].weight = self.embedding
                self.embedding = self.linears[-1].weight

    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch,-1)
    
    def forward(self, x):
        x = self.id2embd(x)
#        print(torch.sqrt(torch.mean(x**2)))
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2 or not self.unembd:
                x = f(x)
        x = self.linears[-1](x)
        return x
    
    def pred_logit(self, x):
        return self.forward(x)


class MLP_HS_Euclidean(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., weight_tied=True, n=1., seed=0, loss_type="harmonic"):
        super(MLP_HS_Euclidean, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.depth = len(shp) - 1
            
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                linear_list.append(EuclideanDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        #self.embedding = torch.nn.Parameter(torch.normal(0,1/torch.tensor(embd_dim),size=(vocab_size, embd_dim))*init_scale)
#        self.embedding = torch.nn.Parameter(torch.normal(0,1,size=(vocab_size, embd_dim))*init_scale)
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
            
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch,-1)

    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)

        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)
    
class MLP_HS_Manhattan(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic"):
        super(MLP_HS_Manhattan, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.depth = len(shp) - 1
            
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                # Use Manhattan distance layer instead of Euclidean
                linear_list.append(ManhattanDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
            
    def id2embd(self, data_id):
        """Convert token IDs to embeddings"""
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch,-1)

    def forward(self, x):
        """CRITICAL: This is the missing forward method!"""
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)

        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        """Prediction method"""
        return self.forward(x)


# Corresponding MLP class for cosine distance
class MLP_HS_Cosine(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic"):
        super(MLP_HS_Cosine, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.depth = len(shp) - 1
            
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                # Use original CosineDistLayer
                linear_list.append(CosineDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
            
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch,-1)

    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)

        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits

class MLP_HS_CosineStable(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic"):
        super(MLP_HS_CosineStable, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.depth = len(shp) - 1
            
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                # Use stable CosineDistLayerStable
                linear_list.append(CosineDistLayerStable(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
            
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch,-1)

    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)

        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits


class MLP_HS_Minkowski(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, p=1.5, n=1., seed=0, loss_type="harmonic"):
        super(MLP_HS_Minkowski, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                linear_list.append(MinkowskiDistLayer(shp[i], shp[i+1], p=p, n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)


class MLP_HS_Hamming_Regular(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", threshold=0.5):
        super(MLP_HS_Hamming_Regular, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                linear_list.append(HammingDistLayer(shp[i], shp[i+1], n=n, 
                                                  eps=1e-4, bias=False, threshold=threshold))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)


class MLP_HS_Hamming_Soft(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", temperature=1.0):
        super(MLP_HS_Hamming_Soft, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                linear_list.append(HammingDistLayerSoft(shp[i], shp[i+1], n=n, 
                                                       eps=1e-4, bias=False, temperature=temperature))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)


class MLP_HS_Hamming_Gumbel(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", temperature=1.0):
        super(MLP_HS_Hamming_Gumbel, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                linear_list.append(HammingDistLayerGumbel(shp[i], shp[i+1], n=n, 
                                                        eps=1e-4, bias=False, temperature=temperature))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)

class MLP_HS_Chebyshev(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 smooth_chebyshev=False, alpha=10.0):
        super(MLP_HS_Chebyshev, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if smooth_chebyshev:
                    linear_list.append(ChebyshevDistLayerSmooth(shp[i], shp[i+1], n=n, alpha=alpha))
                else:
                    linear_list.append(ChebyshevDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)

class MLP_HS_Chebyshev_Soft(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 smooth_chebyshev=True, alpha=10.0):
        super(MLP_HS_Chebyshev_Soft, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if smooth_chebyshev:
                    linear_list.append(ChebyshevDistLayerSmooth(shp[i], shp[i+1], n=n, alpha=alpha))
                else:
                    linear_list.append(ChebyshevDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)



class MLP_HS_Canberra(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="standard", min_denom=1e-3, weight_power=1.0, normalize_weights=True):
        super(MLP_HS_Canberra, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "robust":
                    linear_list.append(CanberraDistLayerRobust(shp[i], shp[i+1], n=n, min_denom=min_denom))
                elif variant == "weighted":
                    linear_list.append(CanberraDistLayerWeighted(shp[i], shp[i+1], n=n, 
                                                               weight_power=weight_power, 
                                                               normalize_weights=normalize_weights))
                else:  # standard
                    linear_list.append(CanberraDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)


class MLP_HS_Canberra_Weighted(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="weighted", min_denom=1e-3, weight_power=1.0, normalize_weights=True):
        super(MLP_HS_Canberra_Weighted, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "robust":
                    linear_list.append(CanberraDistLayerRobust(shp[i], shp[i+1], n=n, min_denom=min_denom))
                elif variant == "weighted":
                    linear_list.append(CanberraDistLayerWeighted(shp[i], shp[i+1], n=n, 
                                                               weight_power=weight_power, 
                                                               normalize_weights=normalize_weights))
                else:  # standard
                    linear_list.append(CanberraDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)

class MLP_HS_Canberra_Robust(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="robust", min_denom=1e-3, weight_power=1.0, normalize_weights=True):
        super(MLP_HS_Canberra_Robust, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "robust":
                    linear_list.append(CanberraDistLayerRobust(shp[i], shp[i+1], n=n, min_denom=min_denom))
                elif variant == "weighted":
                    linear_list.append(CanberraDistLayerWeighted(shp[i], shp[i+1], n=n, 
                                                               weight_power=weight_power, 
                                                               normalize_weights=normalize_weights))
                else:  # standard
                    linear_list.append(CanberraDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)



class MLP_HS_BrayCurtis_Absolute(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="absolute", normalize_inputs=True, min_sum=1e-6):
        super(MLP_HS_BrayCurtis_Absolute, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "absolute":
                    linear_list.append(BrayCurtisDistLayerAbs(shp[i], shp[i+1], n=n))
                elif variant == "normalized":
                    linear_list.append(BrayCurtisDistLayerNormalized(shp[i], shp[i+1], n=n, 
                                                                   normalize_inputs=normalize_inputs, 
                                                                   min_sum=min_sum))
                else:  # standard
                    linear_list.append(BrayCurtisDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)

class MLP_HS_BrayCurtis_Normalized(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="normalized", normalize_inputs=True, min_sum=1e-6):
        super(MLP_HS_BrayCurtis_Normalized, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "absolute":
                    linear_list.append(BrayCurtisDistLayerAbs(shp[i], shp[i+1], n=n))
                elif variant == "normalized":
                    linear_list.append(BrayCurtisDistLayerNormalized(shp[i], shp[i+1], n=n, 
                                                                   normalize_inputs=normalize_inputs, 
                                                                   min_sum=min_sum))
                else:  # standard
                    linear_list.append(BrayCurtisDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)

class MLP_HS_BrayCurtis(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="standard", normalize_inputs=True, min_sum=1e-6):
        super(MLP_HS_BrayCurtis, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "absolute":
                    linear_list.append(BrayCurtisDistLayerAbs(shp[i], shp[i+1], n=n))
                elif variant == "normalized":
                    linear_list.append(BrayCurtisDistLayerNormalized(shp[i], shp[i+1], n=n, 
                                                                   normalize_inputs=normalize_inputs, 
                                                                   min_sum=min_sum))
                else:  # standard
                    linear_list.append(BrayCurtisDistLayer(shp[i], shp[i+1], n=n))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)


class MLP_HS_Mahalanobis(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="standard", learn_cov=True, init_identity=True, 
                 regularize_cov=True, reg_lambda=1e-3):
        super(MLP_HS_Mahalanobis, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "cholesky":
                    linear_list.append(MahalanobisDistLayerCholesky(shp[i], shp[i+1], n=n))
                elif variant == "diagonal":
                    linear_list.append(MahalanobisDistLayerDiagonal(shp[i], shp[i+1], n=n))
                else:  # standard
                    linear_list.append(MahalanobisDistLayer(shp[i], shp[i+1], n=n, 
                                                          learn_cov=learn_cov, 
                                                          init_identity=init_identity,
                                                          regularize_cov=regularize_cov, 
                                                          reg_lambda=reg_lambda))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)

class MLP_HS_Mahalanobis_Diagonal(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="diagonal", learn_cov=True, init_identity=True, 
                 regularize_cov=True, reg_lambda=1e-3):
        super(MLP_HS_Mahalanobis_Diagonal, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "cholesky":
                    linear_list.append(MahalanobisDistLayerCholesky(shp[i], shp[i+1], n=n))
                elif variant == "diagonal":
                    linear_list.append(MahalanobisDistLayerDiagonal(shp[i], shp[i+1], n=n))
                else:  # standard
                    linear_list.append(MahalanobisDistLayer(shp[i], shp[i+1], n=n, 
                                                          learn_cov=learn_cov, 
                                                          init_identity=init_identity,
                                                          regularize_cov=regularize_cov, 
                                                          reg_lambda=reg_lambda))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)


class MLP_HS_Mahalanobis_Cholesky(customNNModule):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., 
                 weight_tied=True, n=1., seed=0, loss_type="harmonic", 
                 variant="cholesky", learn_cov=True, init_identity=True, 
                 regularize_cov=True, reg_lambda=1e-3):
        super(MLP_HS_Mahalanobis_Cholesky, self).__init__(loss_type)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.depth = len(shp) - 1
        
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                if variant == "cholesky":
                    linear_list.append(MahalanobisDistLayerCholesky(shp[i], shp[i+1], n=n))
                elif variant == "diagonal":
                    linear_list.append(MahalanobisDistLayerDiagonal(shp[i], shp[i+1], n=n))
                else:  # standard
                    linear_list.append(MahalanobisDistLayer(shp[i], shp[i+1], n=n, 
                                                          learn_cov=learn_cov, 
                                                          init_identity=init_identity,
                                                          regularize_cov=regularize_cov, 
                                                          reg_lambda=reg_lambda))
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(embd_dim)*init_scale)
        
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
    
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch, -1)
    
    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        
        prob_unnorm = x
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    
    def pred_logit(self, x):
        return self.forward(x)

class ToyTransformer(customNNModule):
    def __init__(self, vocab_size, d_model, nhead, num_layers, seq_len=16, init_scale=1., 
                 use_dist_layer=False, seed=0, n_dist=1., loss_type='cross_entropy', 
                 distance_type='euclidean'):
        super(ToyTransformer, self).__init__(loss_type)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(d_model)*init_scale)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        # Define transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.use_dist_layer = use_dist_layer
        self.distance_type = distance_type
        
        if use_dist_layer:
            # Choose distance layer based on distance_type
            if distance_type == 'manhattan':
                self.dist = ManhattanDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'euclidean':
                self.dist = EuclideanDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'cosine':
                self.dist = CosineDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'cosine_stable':
                self.dist = CosineDistLayerStable(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'minkowski':
                # Default to p=2 (Euclidean-like behavior)
                self.dist = MinkowskiDistLayer(d_model, vocab_size, p=1.5, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'hamming_soft':
                # Use soft version for better gradients in transformer training
                self.dist = HammingDistLayerSoft(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, temperature=1.0)
            elif distance_type == 'hamming':
                # Hard version if explicitly requested
                self.dist = HammingDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, threshold=0.5)
            elif distance_type == 'hamming_gumbel':
                # Hard version if explicitly requested
                self.dist = HammingDistLayerGumbel(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, temperature=0.5)
            elif distance_type == 'chebyshev_soft':
                # Use smooth version for better gradients
                self.dist = ChebyshevDistLayerSmooth(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, alpha=10.0)
            elif distance_type == 'chebyshev':
                # Standard version if explicitly requested
                self.dist = ChebyshevDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'canberra_robust':
                # Use robust version for stability
                self.dist = CanberraDistLayerRobust(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, min_denom=1e-3)
            elif distance_type == 'canberra_standard':
                self.dist = CanberraDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'canberra_weighted':
                self.dist = CanberraDistLayerWeighted(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, 
                                                    weight_power=1.0, normalize_weights=True)
            elif distance_type == 'bray_curtis_norm':
                # Use normalized version for bounded [0,1] distances
                self.dist = BrayCurtisDistLayerNormalized(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, 
                                                        normalize_inputs=True, min_sum=1e-6)
            elif distance_type == 'bray_curtis_standard':
                self.dist = BrayCurtisDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'bray_curtis_abs':
                self.dist = BrayCurtisDistLayerAbs(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'mahalanobis_cholesky':
                # Use Cholesky version for stability
                self.dist = MahalanobisDistLayerCholesky(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'mahalanobis_diagonal':
                # Diagonal version for efficiency
                self.dist = MahalanobisDistLayerDiagonal(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'mahalanobis':
                # Full version with regularization
                self.dist = MahalanobisDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, 
                                               learn_cov=True, init_identity=True, regularize_cov=True, reg_lambda=1e-3)
            
        self.fc = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding

        # Pass through transformer layers with residual connections
        x = embedded
        for layer in self.layers:
            x = layer(x) + x  # Explicit residual connection
            
        if self.use_dist_layer:
            x = x[:, -1]
            x = self.dist(x)
            prob = x/torch.sum(x, dim=1, keepdim=True)
            logits = torch.log(prob)
        else:
            logits = torch.einsum('bh,vh->bv', x[:, -1], self.embedding.weight)
        return logits



class ToyTransformer(customNNModule):
    def __init__(self, vocab_size, d_model, nhead, num_layers, seq_len=16, init_scale=1., 
                 use_dist_layer=False, seed=0, n_dist=1., loss_type='cross_entropy', 
                 distance_type='euclidean'):
        super(ToyTransformer, self).__init__(loss_type)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=1/np.sqrt(d_model)*init_scale)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        # Define transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.use_dist_layer = use_dist_layer
        self.distance_type = distance_type
        
        if use_dist_layer:
            # Choose distance layer based on distance_type with sensible defaults
            if distance_type == 'euclidean':
                self.dist = EuclideanDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'manhattan':
                self.dist = ManhattanDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'cosine':
                self.dist = CosineDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'cosine_stable':
                self.dist = CosineDistLayerStable(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            
            # NEW DISTANCES WITH SENSIBLE DEFAULTS
            elif distance_type == 'minkowski':
                # Default to p=2 (Euclidean-like behavior)
                self.dist = MinkowskiDistLayer(d_model, vocab_size, p=1.5, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'hamming_soft':
                # Use soft version for better gradients in transformer training
                self.dist = HammingDistLayerSoft(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, temperature=1.0)
            elif distance_type == 'hamming':
                # Hard version if explicitly requested
                self.dist = HammingDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, threshold=0.5)
            elif distance_type == 'hamming_gumbel':
                # Gumbel sigmoid version
                self.dist = HammingDistLayerGumbel(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, temperature=0.5)
            elif distance_type == 'chebyshev_soft':
                # Use smooth version for better gradients
                self.dist = ChebyshevDistLayerSmooth(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, alpha=10.0)
            elif distance_type == 'chebyshev':
                # Standard version if explicitly requested
                self.dist = ChebyshevDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            
            elif distance_type == 'canberra_robust':
                # Use robust version for stability
                self.dist = CanberraDistLayerRobust(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, min_denom=1e-3)
            elif distance_type == 'canberra_standard':
                self.dist = CanberraDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'canberra_weighted':
                self.dist = CanberraDistLayerWeighted(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, 
                                                    weight_power=1.0, normalize_weights=True)
            
            elif distance_type == 'bray_curtis_norm':
                # Use normalized version for bounded [0,1] distances
                self.dist = BrayCurtisDistLayerNormalized(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, 
                                                        normalize_inputs=True, min_sum=1e-6)
            elif distance_type == 'bray_curtis_standard':
                self.dist = BrayCurtisDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'bray_curtis_abs':
                self.dist = BrayCurtisDistLayerAbs(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            
            elif distance_type == 'mahalanobis_cholesky':
                # Use Cholesky version for stability
                self.dist = MahalanobisDistLayerCholesky(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'mahalanobis_diagonal':
                # Diagonal version for efficiency
                self.dist = MahalanobisDistLayerDiagonal(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False)
            elif distance_type == 'mahalanobis':
                # Full version with regularization
                self.dist = MahalanobisDistLayer(d_model, vocab_size, n=n_dist, eps=1e-4, bias=False, 
                                               learn_cov=True, init_identity=True, regularize_cov=True, reg_lambda=1e-3)
            
            else:
                raise ValueError(f"Unknown distance_type: {distance_type}. "
                               f"Supported types: euclidean, manhattan, cosine, cosine_stable, "
                               f"minkowski, minkowski_manhattan, minkowski_robust, "
                               f"hamming, hamming_hard, chebyshev, chebyshev_soft, "
                               f"canberra_robust, canberra_standard, canberra_weighted, "
                               f"bray_curtis, bray_curtis_standard, bray_curtis_abs, "
                               f"mahalanobis, mahalanobis_diagonal, mahalanobis_full")
            
        self.fc = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding

        # Pass through transformer layers with residual connections
        x = embedded
        for layer in self.layers:
            x = layer(x) + x  # Explicit residual connection
            
        if self.use_dist_layer:
            x = x[:, -1]
            x = self.dist(x)
            prob = x/torch.sum(x, dim=1, keepdim=True)
            logits = torch.log(prob)
        else:
            logits = torch.einsum('bh,vh->bv', x[:, -1], self.embedding.weight)
        return logits