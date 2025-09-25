import sys
sys.path.append("..")
from dataset import *
from model import *
from distlayers import *
import os

import numpy as np

def set_seed(seed: int) -> None:
    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_single_model(param_dict: dict):

    if "seed" not in param_dict:
        raise ValueError("seed must be provided in param_dict")
    if "data_id" not in param_dict:
        raise ValueError("data_id must be provided in param_dict")
    if "data_size" not in param_dict:
        raise ValueError("data_size must be provided in param_dict")
    if "train_ratio" not in param_dict:
        raise ValueError("train_ratio must be provided in param_dict")
    if "model_id" not in param_dict:
        raise ValueError("model_id must be provided in param_dict")
    if "device" not in param_dict:
        raise ValueError("device must be provided in param_dict")
    if "embd_dim" not in param_dict:
        raise ValueError("embd_dim must be provided in param_dict")
    if "n_exp" not in param_dict: 
        raise ValueError("n_exp must be provided in param_dict")
    if "distance_type" not in param_dict: 
        raise ValueError("distance_type must be provided in param_dict")

    
    seed = param_dict['seed']
    data_id = param_dict['data_id']
    data_size = param_dict['data_size']
    train_ratio = param_dict['train_ratio']
    model_id = param_dict['model_id']
    device = param_dict['device']
    embd_dim = param_dict['embd_dim']
    n_exp = param_dict['n_exp']

    video = False if 'video' not in param_dict else param_dict['video']
    lr = 0.002 if 'lr' not in param_dict else param_dict['lr']
    weight_decay = 0.01 if 'weight_decay' not in param_dict else param_dict['weight_decay']
    verbose = False if 'verbose' not in param_dict else param_dict['verbose']
    lamb_reg = 0.01 if 'lamb_reg' not in param_dict else param_dict['lamb_reg']
    custom_loss = None if 'custom_loss' not in param_dict else param_dict['custom_loss']
    use_custom_loss = False if custom_loss is None else True

    distance_type = param_dict['distance_type']

    # Minkowski parameters
    minkowski_p = 1.5 if 'minkowski_p' not in param_dict else param_dict['minkowski_p']
    
    # Hamming parameters
    hamming_threshold = 0.5 if 'hamming_threshold' not in param_dict else param_dict['hamming_threshold']
    hamming_temperature = 1.0 if 'hamming_temperature' not in param_dict else param_dict['hamming_temperature']
    
    # Chebyshev parameters
    chebyshev_alpha = 10.0 if 'chebyshev_alpha' not in param_dict else param_dict['chebyshev_alpha']
    
    # Canberra parameters
    canberra_min_denom = 1e-3 if 'canberra_min_denom' not in param_dict else param_dict['canberra_min_denom']
    canberra_weight_power = 1.0 if 'canberra_weight_power' not in param_dict else param_dict['canberra_weight_power']
    canberra_normalize_weights = True if 'canberra_normalize_weights' not in param_dict else param_dict['canberra_normalize_weights']
    
    # Bray-Curtis parameters
    bray_curtis_normalize_inputs = True if 'bray_curtis_normalize_inputs' not in param_dict else param_dict['bray_curtis_normalize_inputs']
    bray_curtis_min_sum = 1e-6 if 'bray_curtis_min_sum' not in param_dict else param_dict['bray_curtis_min_sum']
    
    # Mahalanobis parameters
    mahalanobis_learn_cov = True if 'mahalanobis_learn_cov' not in param_dict else param_dict['mahalanobis_learn_cov']
    mahalanobis_init_identity = True if 'mahalanobis_init_identity' not in param_dict else param_dict['mahalanobis_init_identity']
    mahalanobis_regularize_cov = True if 'mahalanobis_regularize_cov' not in param_dict else param_dict['mahalanobis_regularize_cov']
    mahalanobis_reg_lambda = 1e-3 if 'mahalanobis_reg_lambda' not in param_dict else param_dict['mahalanobis_reg_lambda']


    set_seed(seed)

    # define dataset
    input_token = 2
    num_epochs = 7000 if 'num_epochs' not in param_dict else param_dict['num_epochs']
    dataset = modular_addition_dataset(p=31, num=data_size, seed=seed, device=device)
    
    dataset = split_dataset(dataset, train_ratio=train_ratio, seed=seed)
    vocab_size = dataset['vocab_size']

    # define model
    if model_id == "H_MLP":
        weight_tied = True
        hidden_size = 100
        shp = [input_token * embd_dim, hidden_size, embd_dim, vocab_size]
        if use_custom_loss:
            loss_type = custom_loss
        else:
            loss_type = 'harmonic'
        print(f"Loss type: {loss_type}, Distance type: {distance_type}")

        # Choose between Euclidean and Manhattan distance implementations
        if distance_type == 'manhattan':
            model = MLP_HS_Manhattan(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                   input_token=input_token, weight_tied=weight_tied, 
                                   seed=seed, n=n_exp, init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'euclidean':
            model = MLP_HS_Euclidean(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                          input_token=input_token, weight_tied=weight_tied, 
                          seed=seed, n=n_exp, init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'cosine':
            model = MLP_HS_Cosine(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                input_token=input_token, weight_tied=weight_tied, 
                                seed=seed, n=n_exp, init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'cosine_stable':
            model = MLP_HS_CosineStable(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                input_token=input_token, weight_tied=weight_tied, 
                                seed=seed, n=n_exp, init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'minkowski':
            model = MLP_HS_Minkowski(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                   input_token=input_token, weight_tied=weight_tied, 
                                   seed=seed, n=n_exp, p=minkowski_p, init_scale=1, loss_type=loss_type).to(device)       
        # In your train_single_model function:
        elif distance_type == 'hamming':
            model = MLP_HS_Hamming_Regular(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                        input_token=input_token, weight_tied=weight_tied, 
                                        seed=seed, n=n_exp, threshold=0.5,  # No hamming_type!
                                        init_scale=1, loss_type=loss_type).to(device)

        elif distance_type == 'hamming_soft':
            model = MLP_HS_Hamming_Soft(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                    input_token=input_token, weight_tied=weight_tied, 
                                    seed=seed, n=n_exp, temperature=1.0,  # No hamming_type!
                                    init_scale=1, loss_type=loss_type).to(device)

        elif distance_type == 'hamming_gumbel':
            model = MLP_HS_Hamming_Gumbel(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                        input_token=input_token, weight_tied=weight_tied, 
                                        seed=seed, n=n_exp, temperature=0.5,  # No hamming_type!
                                        init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'chebyshev':
            model = MLP_HS_Chebyshev(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                   input_token=input_token, weight_tied=weight_tied, 
                                   seed=seed, n=n_exp, smooth_chebyshev=False,
                                   alpha=chebyshev_alpha, init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'chebyshev_soft':
            model = MLP_HS_Chebyshev_Soft(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                   input_token=input_token, weight_tied=weight_tied, 
                                   seed=seed, n=n_exp, smooth_chebyshev=True,
                                   alpha=chebyshev_alpha, init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'canberra_standard':
            model = MLP_HS_Canberra(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                  input_token=input_token, weight_tied=weight_tied, 
                                  seed=seed, n=n_exp, variant="standard",
                                  min_denom=canberra_min_denom, weight_power=canberra_weight_power,
                                  normalize_weights=canberra_normalize_weights,
                                  init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'canberra_robust':
            model = MLP_HS_Canberra_Robust(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                  input_token=input_token, weight_tied=weight_tied, 
                                  seed=seed, n=n_exp, variant="robust",
                                  min_denom=canberra_min_denom, weight_power=canberra_weight_power,
                                  normalize_weights=canberra_normalize_weights,
                                  init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'canberra_weighted':
            model = MLP_HS_Canberra_Weighted(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                  input_token=input_token, weight_tied=weight_tied, 
                                  seed=seed, n=n_exp, variant="weighted",
                                  min_denom=canberra_min_denom, weight_power=canberra_weight_power,
                                  normalize_weights=canberra_normalize_weights,
                                  init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'bray_curtis_standard':
            model = MLP_HS_BrayCurtis(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                    input_token=input_token, weight_tied=weight_tied, 
                                    seed=seed, n=n_exp, variant="standard",
                                    normalize_inputs=bray_curtis_normalize_inputs,
                                    min_sum=bray_curtis_min_sum,
                                    init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'bray_curtis_norm':
            model = MLP_HS_BrayCurtis_Normalized(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                    input_token=input_token, weight_tied=weight_tied, 
                                    seed=seed, n=n_exp, variant="normalized",
                                    normalize_inputs=bray_curtis_normalize_inputs,
                                    min_sum=bray_curtis_min_sum,
                                    init_scale=1, loss_type=loss_type).to(device)                                  
        elif distance_type == 'bray_curtis_abs':
            model = MLP_HS_BrayCurtis_Absolute(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                    input_token=input_token, weight_tied=weight_tied, 
                                    seed=seed, n=n_exp, variant="absolute",
                                    normalize_inputs=bray_curtis_normalize_inputs,
                                    min_sum=bray_curtis_min_sum,
                                    init_scale=1, loss_type=loss_type).to(device) 
        elif distance_type == 'mahalanobis':
            model = MLP_HS_Mahalanobis(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                     input_token=input_token, weight_tied=weight_tied, 
                                     seed=seed, n=n_exp, variant="standard",
                                     learn_cov=mahalanobis_learn_cov, 
                                     init_identity=mahalanobis_init_identity,
                                     regularize_cov=mahalanobis_regularize_cov, 
                                     reg_lambda=mahalanobis_reg_lambda,
                                     init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'mahalanobis_cholesky':
            model = MLP_HS_Mahalanobis_Cholesky(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                     input_token=input_token, weight_tied=weight_tied, 
                                     seed=seed, n=n_exp, variant="cholesky",
                                     learn_cov=mahalanobis_learn_cov, 
                                     init_identity=mahalanobis_init_identity,
                                     regularize_cov=mahalanobis_regularize_cov, 
                                     reg_lambda=mahalanobis_reg_lambda,
                                     init_scale=1, loss_type=loss_type).to(device)
        elif distance_type == 'mahalanobis_diagonal':
            model = MLP_HS_Mahalanobis_Diagonal(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, 
                                     input_token=input_token, weight_tied=weight_tied, 
                                     seed=seed, n=n_exp, variant="diagonal",
                                     learn_cov=mahalanobis_learn_cov, 
                                     init_identity=mahalanobis_init_identity,
                                     regularize_cov=mahalanobis_regularize_cov, 
                                     reg_lambda=mahalanobis_reg_lambda,
                                     init_scale=1, loss_type=loss_type).to(device)
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}. "
                           f"Supported types: euclidean, manhattan, cosine, cosine_stable, "
                           f"minkowski, hamming, chebyshev, canberra, bray_curtis, mahalanobis")

        

        # Former model defintion #
        # model = MLP_HS(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, weight_tied=weight_tied, seed=seed, n=n_exp, init_scale=1, loss_type=loss_type).to(device)

    elif model_id == "standard_MLP":
        unembd = True
        weight_tied = True
        hidden_size = 100
        shp = [input_token * embd_dim, hidden_size, embd_dim, vocab_size]
        model = MLP(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, unembd=unembd, weight_tied=weight_tied, seed=seed, init_scale=1, loss_type = "cross_entropy").to(device)

    elif model_id == "H_transformer":
        if use_custom_loss:
            loss_type = custom_loss
        else:
            loss_type = 'harmonic'
        print(f"Loss type: {loss_type}, Distance type: {distance_type}")

        model = ToyTransformer(vocab_size=vocab_size, d_model=embd_dim, nhead=2, 
                             num_layers=2, n_dist=n_exp, seq_len=input_token, 
                             seed=seed, use_dist_layer=True, init_scale=1, 
                             loss_type=loss_type, distance_type=distance_type).to(device)
        
        # Former model defintion #                         
        #model = ToyTransformer(vocab_size=vocab_size, d_model=embd_dim, nhead=2, num_layers=2, n_dist=n_exp,seq_len=input_token, seed=seed, use_dist_layer=True, init_scale=1, loss_type=loss_type).to(device)

    elif model_id == "standard_transformer":
        model = ToyTransformer(vocab_size=vocab_size, d_model=embd_dim, nhead=2, num_layers=2, seq_len=input_token, seed=seed, use_dist_layer=False, init_scale=1, loss_type = "cross_entropy").to(device)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    
    # define dataloader
    batch_size = 32
    train_dataset = ToyDataset(dataset['train_data_id'], dataset['train_label'])
    test_dataset = ToyDataset(dataset['test_data_id'], dataset['test_label'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ret_dic = {}
    ret_dic["results"] = model.train(param_dict={'model_id':model_id,'num_epochs': num_epochs, 'learning_rate': lr, 'weight_decay':weight_decay, 'train_dataloader': train_dataloader, 'test_dataloader': test_dataloader, 'device': device, 'video': video, 'verbose': verbose, 'lambda': lamb_reg})
    ret_dic["model"] = model
    ret_dic["dataset"] = dataset

    return ret_dic