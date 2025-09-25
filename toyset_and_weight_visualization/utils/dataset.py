import numpy as np
import torch
import math
import itertools

import sys
sys.path.append("..")

def modular_addition_dataset(p, num, seed=0, device='cpu'):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x = np.arange(p)
    y = np.arange(p)
    XX, YY = np.meshgrid(x, y)
    data_id = np.transpose([XX.reshape(-1,), YY.reshape(-1,)])

    sample_id = np.random.choice(len(data_id), size=num, replace=True)
    data_id = data_id[sample_id]
    labels = (data_id[:,0] + data_id[:,1]) % p
    labels = torch.tensor(labels, dtype=torch.long)

    
    vocab_size = p
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size

    return dataset

def permutation_group_dataset(p, num, seed=0, device='cpu'): 
    torch.manual_seed(seed)
    np.random.seed(seed)

    perms = list(itertools.permutations(range(p)))
    num_perms = len(perms)

    perm_dict = dict(enumerate(perms))
    swapped_dict = {v:k for k,v in perm_dict.items()}

    idx = torch.arange(num_perms)

    data_id = [[perms[int(i)], perms[int(j)]] for i, j in torch.cartesian_prod(idx, idx)]
    keyed_data_id = np.array([[swapped_dict[data_id[i][0]], swapped_dict[data_id[i][1]]] for i in range(len(data_id))])

    labels = [tuple(np.array(perms[int(i)])[np.array(perms[int(j)])]) for i, j in torch.cartesian_prod(idx, idx)]
    keyed_labels = np.array([swapped_dict[labels[i]] for i in range(len(labels))])
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    perm_vals = ["".join(np.array(perm_dict[i]).astype(str)) for i in range(len(perm_dict))]
    new_perm_dict = dict(zip(perm_dict.keys(), perm_vals)) # are these indices correct?
    
    dataset = {}

    dataset['data_id'] = keyed_data_id
    dataset['label'] = keyed_labels
    dataset['vocab_size'] = num_perms
    dataset['dict_level'] = new_perm_dict

    return dataset


def split_dataset(dataset, train_ratio, seed=0):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset2 = {}
    
    num = dataset['data_id'].shape[0]
    
    train_num = int(num*train_ratio)
    test_num = num - train_num

    train_id = np.random.choice(num,train_num,replace=False)
    test_id = np.array(list(set(np.arange(num)) - set(train_id)))
    
    dataset2['train_data_id'] = dataset['data_id'][train_id]
    dataset2['test_data_id'] = dataset['data_id'][test_id]
    dataset2['train_label'] = dataset['label'][train_id]
    dataset2['test_label'] = dataset['label'][test_id]
    dataset2['vocab_size'] = dataset['vocab_size']
    if 'dict_level' in dataset:
        dataset2['dict_level'] = dataset['dict_level']
    return dataset2

def repeat_dataset(dataset):
    
    dataset2 = {}
    
    dataset2['train_data_id'] = dataset['data_id']
    dataset2['test_data_id'] = dataset['data_id']
    dataset2['train_label'] = dataset['label']
    dataset2['test_label'] = dataset['label']
    dataset2['vocab_size'] = dataset['vocab_size']

    if 'dict_level' in dataset:
        dataset2['dict_level'] = dataset['dict_level']
    
    return dataset2


def combine_dataset(train_dataset, test_dataset):
    
    dataset_c = {}
    
    dataset_c['train_data_id'] = train_dataset['data_id']
    dataset_c['test_data_id'] = test_dataset['data_id']
    dataset_c['train_label'] = train_dataset['label']
    dataset_c['test_label'] = test_dataset['label']
    
    assert train_dataset['vocab_size'] == test_dataset['vocab_size']
    dataset_c['vocab_size'] = train_dataset['vocab_size']
    
    return dataset_c


# Dataset and DataLoader
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
def descendant_dataset(p, num, seed=0, device='cpu'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N_sample = num
    x = np.random.choice(range(2,p), N_sample*2).reshape(N_sample, 2)

    # Check if b is a descendant of a
    # In a complete binary tree where two children of x is 2x and 2x+1
    def is_desc(a, b):
        while b > 1:
            if b == a:
                return True
            b //= 2  # Move up to the parent node
        return b == a
    target = np.array([1 if is_desc(x[i,0]-1, x[i,1]-1) else 0 for i in range(N_sample)])
    
    data_id = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(target).to(device)
    
    vocab_size = p
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size
    
    return dataset

def descendant_dataset_2(p, num, seed=0, device='cpu'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N_sample = num*4
    x = np.random.choice(range(1,(p-1)//2), num*2).reshape(num, 2)

    data = np.zeros((N_sample, 4), dtype=np.int32)
    data[:num,0] = x[:,0]
    data[:num,1] = 2*x[:,0]
    data[:num,2] = x[:,1]
    data[:num,3] = 2*x[:,1]

    data[num:(2*num),0] = x[:,0]
    data[num:(2*num),1] = 2*x[:,0] + 1
    data[num:(2*num),2] = x[:,1]
    data[num:(2*num),3] = 2*x[:,1] + 1

    data[2*num:(3*num),0] = 2*x[:,0] + 1
    data[2*num:(3*num),1] = x[:,0]
    data[2*num:(3*num),2] = 2*x[:,1] + 1
    data[2*num:(3*num),3] = x[:,1]

    data[3*num:(4*num),0] = 2*x[:,0] + 1
    data[3*num:(4*num),1] = x[:,0]
    data[3*num:(4*num),2] = 2*x[:,1] + 1
    data[3*num:(4*num),3] = x[:,1]
    
    np.random.shuffle(data)
    
    data_id = torch.from_numpy(data[:, :3]).to(device)
    labels = torch.from_numpy(data[:, 3]).to(device)
    
    vocab_size = p+1
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size
    
    return dataset


def greater_than_dataset(p, num, seed=0, device='cpu'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N_sample = num
    x = np.random.choice(range(p), N_sample*2).reshape(N_sample, 2)

    target = np.array([p+1 if x[i,0] > x[i,1] else p for i in range(N_sample)])
    
    data_id = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(target).to(device)
    
    vocab_size = p+2
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size
    
    return dataset


def xor_dataset(p, num, seed=0, device='cpu'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N_sample = num
    x = np.random.choice(range(p), N_sample*2).reshape(N_sample, 2)

    target = np.array([x[i,0]^x[i,1] for i in range(N_sample)])
    
    data_id = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(target).to(device)
    
    vocab_size = p+2
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size
    
    return dataset

def multi_step_dataset(p, num, seed=0, device='cpu'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N_sample = num
    x = np.random.choice(range(p), N_sample*3).reshape(N_sample, 3)

    target = np.array([(x[i,0]*x[i,1]+x[i,2])%p for i in range(N_sample)])
    
    data_id = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(target).to(device)
    
    vocab_size = p
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size
    
    return dataset


def mod_classification_dataset(p, num, seed=0, device='cpu'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N_sample = num
    x = np.random.choice(range(p), N_sample).reshape(N_sample, 1)

    target = np.array([x[i,0]%5 for i in range(N_sample)])
    
    data_id = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(target).to(device)
    
    vocab_size = p
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size
    
    return dataset


def mod_equiv_dataset(p, num, seed=0, device='cpu'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N_sample = num
    x = np.random.choice(range(p), N_sample*2).reshape(N_sample, 2)

    target = np.array([p if (x[i,0]-x[i,1])%5 == 0 else p+1 for i in range(N_sample)])
    
    data_id = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(target).to(device)
    
    vocab_size = p+2
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size
    
    return dataset