# Rethinking Harmonic Loss

## Table of Contents

- [Why Harmonic Loss](#why-harmonic-loss)   
- [Repository Structure](#repository-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Downloading Data & Models](#downloading-data--models)  
  - [Running Experiments](#running-experiments)  
- [Using the Harmonic Variant](#using-the-harmonic-variant)  
- [Circle Toy Experiment](#Circle-toy-experiment)  
- [Citation](#citation)
- [Key Contributions](#key-contributions)   
- [License](#license)

This repository provides code for training and evaluating models with **harmonic loss**, a distance-based alternative to cross-entropy.  
We extend harmonic loss beyond Euclidean distance, benchmarking multiple metrics across **vision backbones** and **large language models (LLMs)**.  

The codebase supports reproducible experiments on:
- **Language models** (GPT-style architectures, trained on OpenWebText)
- **Vision models** (e.g., MNIST, CIFAR)
- **Toy datasets** for exploring representation geometry and weight dynamics

## Why Harmonic Loss?

Cross-entropy has dominated deep learning but comes with drawbacks:
- Poor interpretability of weight dynamics  
- Unbounded weight growth  
- Training inefficiencies that can cause instability (e.g., grokking)  

Harmonic loss reframes training as a **distance minimization problem**.  
This repo generalizes it to multiple distance metrics (e.g., cosine, Bray–Curtis, Mahalanobis) and evaluates them for **accuracy, interpretability, and efficiency**.

## Repository Structure

├── llms/ # Language model experiments
│ └── train_adam_distances_loss.py # Runner file for GPT/BERT/Qwen with harmonic loss
│
├── vision/ # Vision experiments
│ └── run_with_configs.py # Runner file for CIFAR, MNIST, etc.
│
├── toyset_and_weight_visualization/ # Toy datasets & weight analysis
│ └── run/
│ ├── mlp_visualization.py # Runner for MLP visualization experiments
│ └── toy_circle.py # Runner for synthetic circle dataset
│
├── results/figures/ # Generated figures and plots
├── config/
│ └── config.py # Central configuration file for runs
├── requirements.txt # Python dependencies
└── LICENSE # CC0 license

Tested with Python ≥ 3.8 and PyTorch ≥ 1.12.

## Getting Started

### Prerequisites

1. Python ≥ 3.8
2. Clone the repository locally with your preferred method

  HTTPS:
    git clone https://github.com/maxwellmg/rethinking-harmonic-loss.git

3. Install dependencies:

  pip install -r requirements.txt

### Data Setup

#### Language Models (LLMs)

Requires OpenWebText in .bin format (train.bin, val.bin).
1. Create a folder data/openwebtext
2. Download the binary files for train.bin and val.bin. Accessible [here](https://www.dropbox.com/scl/fo/v24k2eltevgiszdfvean6/AF0j1Pu9ladYpDZbqSVKHGI?rlkey=jwa73nxrwt5bj13a6c9q0z20w&st=090g6v8w&dl=0).
3. Put both binary files train.bin and val.bin into your data/openwebtext folder. 

#### Vision Models

Datasets (MNIST, CIFAR10, CIFAR100) are downloaded automatically via torchvision when running the code the first time and then accessed locally with each successive run.

### Running Experiments

#### LLMs

Train a GPT/BERT/Qwen model with harmonic loss:

1. (Highly recommended) Run 
python llms/train_adam_distances_loss_gpt_bert_qwen.py --config config/config.py

Vision Models
python vision/run_with_configs.py --dataset cifar10 --loss harmonic_cosine


Results and plots are saved in results/figures/.

Toy Experiments & Visualization

Explore representation geometry and weight dynamics with toy data:

python toyset_and_weight_visualization/run/mlp_visualization.py
python toyset_and_weight_visualization/run/toy_circle.py

Customization

Loss function: add or modify metrics in the model code.

Configs: adjust dataset, optimizer, scheduler, and training hyperparams in config/config.py.

Plots: generated automatically, saved in results/figures/.

License

Released under CC0-1.0 (public domain dedication). Free for all use.

Acknowledgments

Built on nanoGPT
 foundations.

Includes optimizer logic inspired by the [Sophia repo].

Extended with harmonic loss variants for systematic evaluation across tasks.
















# rethinking-harmonic-loss
 We extend harmonic loss beyond Euclidean distance by testing diverse metrics on vision and language models, showing that tailored variants can outperform cross-entropy and Euclidean, improving accuracy, interpretability, and sustainability

In many cases, I run the runners with a separate py/lsf file combination, most of which set configs.

Runners

Vision: main_run_function.py (also test_seed_runs.py to run multiple seeds at once)

MNIST weight visualization: notebooks -> run_data_agnostic_numbers_codecarbon.py

ModAdd Toy Set: notebooks -> run_circle_codecarbon_new_dist.py

LLM: train_adam_distances_loss_gpt_bert_qwen.py


# Start

* Step 1: create a folder data/openwebtext, and put both binary files train.bin and val.bin into the folder. Binary files can be downloaded from [here](https://www.dropbox.com/scl/fo/v24k2eltevgiszdfvean6/AF0j1Pu9ladYpDZbqSVKHGI?rlkey=jwa73nxrwt5bj13a6c9q0z20w&st=090g6v8w&dl=0).

* Step 2: in terminal type `sbatch train_adam_l2loss.sh`. That's it! This should immediately work on supercloud (except that perhaps you need to pip install wandb etc., I don't remember exactly). If you don't want to train the model by yourself, pre-trained models can be found [here](https://www.dropbox.com/scl/fo/v24k2eltevgiszdfvean6/AF0j1Pu9ladYpDZbqSVKHGI?rlkey=jwa73nxrwt5bj13a6c9q0z20w&st=090g6v8w&dl=0) in folders `out_small_adam` (standard) and `out_small_adam_hm` (harmonic). Place both folders in the current folder.

# Notice
* The code is based on [sophia repo](https://github.com/Liuhong99/Sophia/tree/main), which in turn is based on [nanogpt](https://github.com/karpathy/nanoGPT/). The training pipeline might be unnecessarily complicated for our purposes (a lot of parallelization etc.).
* My major changes (relevant to harmonic losses) are in `model_l2loss.py` and highlighted with comments "Ziming's note". The standard transformer is in `model.py`. The line in `train_adam_l2loss.py`, which is `from model_l2loss import GPT, GPTConfig`, specifies that we're using GPT with harmonic similarity. To use standard GPT, change the line to `from model import GPT, GPTConfig`.
* To change configurations, e.g., the size of the network, go to  `config/train_gpt2_small_adam_l2loss.py`. Although there are some hyperparameters being set up at the beginning of `train_adam_l2loss.py`, these hyperparameters are later overwritten by `config/train_gpt2_small_adam_l2loss.py`.
* Given the complexity of the training code, I suspect a faster way to kickstart is playing with the `GPT` model in `model_l2loss.py` and `model.py`, writing training loops by oneself without caring to read other files.
