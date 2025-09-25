In many cases, I run the runners with a separate py/lsf file combination, most of which set configs.

Runners

Vision: main_run_function.py (also test_seed_runs.py to run multiple seeds at once)

MNIST weight visualization: notebooks -> run_data_agnostic_numbers_codecarbon.py

ModAdd Toy Set: notebooks -> run_circle_codecarbon_new_dist.py

LLM: train_adam_distances_loss_gpt_bert_qwen.py


I think in all cases you can ignore what's in the "old" file.

# Start

* Step 1: create a folder data/openwebtext, and put both binary files train.bin and val.bin into the folder. Binary files can be downloaded from [here](https://www.dropbox.com/scl/fo/v24k2eltevgiszdfvean6/AF0j1Pu9ladYpDZbqSVKHGI?rlkey=jwa73nxrwt5bj13a6c9q0z20w&st=090g6v8w&dl=0).

* Step 2: in terminal type `sbatch train_adam_l2loss.sh`. That's it! This should immediately work on supercloud (except that perhaps you need to pip install wandb etc., I don't remember exactly). If you don't want to train the model by yourself, pre-trained models can be found [here](https://www.dropbox.com/scl/fo/v24k2eltevgiszdfvean6/AF0j1Pu9ladYpDZbqSVKHGI?rlkey=jwa73nxrwt5bj13a6c9q0z20w&st=090g6v8w&dl=0) in folders `out_small_adam` (standard) and `out_small_adam_hm` (harmonic). Place both folders in the current folder.

# Notice
* The code is based on [sophia repo](https://github.com/Liuhong99/Sophia/tree/main), which in turn is based on [nanogpt](https://github.com/karpathy/nanoGPT/). The training pipeline might be unnecessarily complicated for our purposes (a lot of parallelization etc.).
* My major changes (relevant to harmonic losses) are in `model_l2loss.py` and highlighted with comments "Ziming's note". The standard transformer is in `model.py`. The line in `train_adam_l2loss.py`, which is `from model_l2loss import GPT, GPTConfig`, specifies that we're using GPT with harmonic similarity. To use standard GPT, change the line to `from model import GPT, GPTConfig`.
* To change configurations, e.g., the size of the network, go to  `config/train_gpt2_small_adam_l2loss.py`. Although there are some hyperparameters being set up at the beginning of `train_adam_l2loss.py`, these hyperparameters are later overwritten by `config/train_gpt2_small_adam_l2loss.py`.
* Given the complexity of the training code, I suspect a faster way to kickstart is playing with the `GPT` model in `model_l2loss.py` and `model.py`, writing training loops by oneself without caring to read other files.

# Parallelogram experiments
* Function vector data are taken from [function vectors](https://github.com/ericwtodd/function_vectors/tree/main/dataset_files/abstractive) and are stored in ./abstractive. We only keep datasets that are consistent with our experimental setup. 
* function_vectors.ipynb: parallelogram loss
* function_vectors_parallelogram.ipynb: parallelogram
* How to generate text output: prediction.ipynb
