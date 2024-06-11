# Tensor Decompositions in Autoencoders for Electrocardiography Classification
Implementation in python of my thesis work in unsupervised pre-training using tensor decomposed autoencoders for ECG classification.


### Installation
1. Clone the repository to your local machine.
2. Create a virtual environment (recommended, not necessary).
2. Install the required packages with: `pip3 install -r requirements.txt`
3. Create a directory for the raw ECG files.
4. Download the raw ECG files as a zip from physionet.org (recommended) or execute the following commands:.
    - MIT-BIH: `wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/`.
    - INCART: `wget -r -N -c -np https://physionet.org/files/incartdb/1.0.0/`.
    - PTB-XL: `wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/`.
    - CPSC: `wget -r -N -c -np https://physionet.org/files/cpsc2021/1.0.0/`.
    - G12EC has to be downloaded directly at: https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database?resource=download.
5. Run each of the data generation files under `generation/`.

### Usage
To run the experiment execute the four corresponding files: `exp1.py`, `exp2_1.py`, `exp2_2.py` and `exp2_3.py`.

#### Arguments
##### Model parameters
- --model: Name of model to train (default: default)
- --batch_size_mae: Per GPU batch size (default: 256)
- --epochs_mae: Number of epochs to train the MAE (default: 50)
- --warmup_epochs_mae: Number of epochs to warmup LR for MAE (default: 0)
- --num_runs: Number of repeat runs (default: 3)
- --batch_size_class: Per GPU batch size for classifier (default: 256)
- --epochs_class: Number of epochs to train the classifier (default: 20)
- --warmup_epochs_class: Number of epochs to warmup LR for classifier (default: 0)

##### Optimizer parameters
- --weight_decay_mae: Weight decay for MAE (default: 1e-4)
- --lr_mae: Learning rate for MAE (default: 5e-4)
- --min_lr_mae: Lower LR bound for cyclic schedulers for MAE (default: 1e-5)
- --weight_decay_class: Weight decay for classifier (default: 1e-4)
- --lr_class: Learning rate for classifier (default: 5e-4)
- --min_lr_class: Lower LR bound for cyclic schedulers for classifier (default: 1e-5)

##### Other parameters
- --gpu: Specify to use single GPU or all GPUs (default: all)

##### Save parameters
- --contrun: Flag to continue from last run
- --no_train_mae: Do not train MAE
- --no_train_class: Do not train classifier
- --no_save_mae: Do not save MAE model
- --no_save_class: Do not save classifier model


### Results




### Abstract
