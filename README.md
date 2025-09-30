# Rethinking Harmonic Loss

## Table of Contents

- [Why Harmonic Loss](#why-harmonic-loss)   
- [Repository Structure](#repository-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Downloading Data & Models](#downloading-data--models)  
    - [Setup LLMs](#setup-llms)
    - [Setup Vision](#setup-vision)
- [Running Experiments](#running-experiments)  
  - [Run LLMs](#run-llms)
  - [Run Vision](#run-vision)
  - [Run Toy Experiment & Visualization](#run-toy-experiments--visualization)  
- [Acknowledgements](#acknowledgements)
- [License](#license)

This repository provides code for training and evaluating models with **harmonic loss**, a distance-based alternative to cross-entropy.  
We extend harmonic loss beyond Euclidean distance, benchmarking multiple metrics across **vision backbones** and **large language models (LLMs)**.  

The codebase supports reproducible experiments on:
- **Language models** (GPT, BERT, and QWEN architectures, trained on OpenWebText)
- **Vision models** (MLP, CNN, ResNet50, and PVT backbones tested on MNIST, CIFAR10, and CIFAR100)
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

├── vision/ # Vision experiments

│ └── run_with_configs.py # Runner file for CIFAR, MNIST, etc.

├── toyset_and_weight_visualization/ # Toy datasets & weight analysis

│ └── run/

│ ├── mlp_visualization.py # Runner for MLP visualization experiments

│ └── toy_circle.py # Runner for synthetic circle dataset

├── results/figures/ # Generated figures and plots

├── config/

│ └── config.py # Central configuration file for runs

├── requirements.txt # Python dependencies

└── LICENSE # CC0 license

Tested with Python ≥ 3.8 and PyTorch ≥ 1.12.

## Getting Started

### Prerequisites

1. Python ≥ 3.8
2. Recommended: Set up a Python virtual environment

  python -m venv .venv
  source .venv/bin/activate   # Linux/macOS

3. Clone the repository locally with your preferred method

  HTTPS:
    git clone https://github.com/maxwellmg/rethinking-harmonic-loss.git
    cd rethinking-harmonic-loss

4. Install dependencies:

  pip install -r requirements.txt

### Data Setup

#### Setup LLMs

Requires OpenWebText in .bin format (train.bin, val.bin).
1. Create a folder data/openwebtext
2. Download the binary files for train.bin and val.bin. Accessible [here](https://www.dropbox.com/scl/fo/v24k2eltevgiszdfvean6/AF0j1Pu9ladYpDZbqSVKHGI?rlkey=jwa73nxrwt5bj13a6c9q0z20w&st=090g6v8w&dl=0).
3. Put both binary files train.bin and val.bin into your data/openwebtext folder. 

#### Setup Vision

Datasets (MNIST, CIFAR10, CIFAR100) are downloaded automatically via torchvision when running the code the first time and then accessed locally with each successive run.

### Running Experiments

#### Run LLMs

Train a GPT/BERT/Qwen model with harmonic loss:

1. (Highly recommended) Run the model_distance_agnostic_memory_test.py (with or without the helper lsf file) to determine an ideal batch size and gradient accumulation given your system's constraints

  Note: if you don't use the given lsf file, the test will automatically run for GPT, BERT, and QWEN on a cross-entropy loss baseline.

2. (Highly recommended) Review config/config.py to specify specific parameters for your run. This includes, but is not limited to:
  a. adding in the results of the memory test (to avoid cuda OOM errors)
  b. choosing between gpt, bert, and qwen backbones
  c. setting length of experiment with "max_iters", etc.

3. (Recommended) Sign up for an account with WandB to see the training data in real time

4. Run:
  python llms/train_adam_distances_loss_gpt_bert_qwen.py

#### Run Vision

1. (Recommended) Set your configs in run_with_configs.py. The program will loop over lists of inputs depending on the test you wish to run. Thus, you should set:
  a. What hardware to run on based on personal constraints
  b. The number of desired epochs (default values pulled from config/regular_config.py)
  c. Dataset/s
  d. Model type/s
  e. Seed/s
  f. DistLayer distances (under 'distance_layer_types')
  g. Regularizer distances and lambdas (under 'distance_types'. Highly recommended to keep the default of baseline and 0.0)

2. Run:
  python vision/run_with_configs.py

#### Run Toy Experiments & Visualization

1. (Optional) Change inline configurations in mlp_visualization.py and toy_circle.py. There, you can adjust the dataset, optimizer, scheduler, and training hyperparams.

2. Run:
  a. For MLP weight visualization on MNIST, CIFAR10, or CIFAR1OO:
    python toyset_and_weight_visualization/run/mlp_visualization.py
  b. For the toy circle experiment and visualization:
    python toyset_and_weight_visualization/run/toy_circle.py

## Acknowledgments

Inspiration for much of the code found in this repo can be sourced from two repositories. Due to the necessity of anonymity due to current conference submission/review, these will be shared at a future date.

## License

Released under CC0-1.0 (public domain dedication). Free for all use.