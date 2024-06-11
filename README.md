# Tensor Decompositions in Autoencoders for Electrocardiography Classification
Implementation in python of my thesis work in unsupervised pre-training using tensor decomposed autoencoders for ECG classification.

Thesis can be found at: repository.tudelft.nl
### Getting started
1. Clone the repository to your local machine.
2. Create a virtual environment (recommended, not necessary).
2. Install the required packages with: `pip3 install -r requirements.txt`
3. Create directories for the raw ECG files. The data generation scripts look for the raw data in this structure:
    ```
    ../
    └───physionet
    │   └───cpsc/
    │   └───incart/
    │   └───ptbxl/
    │   └───mitbih/
    │   └───g12ec/
    ```
4. Download the raw ECG files as a zip from physionet.org (recommended) or execute the following commands:.
    - MIT-BIH: `wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/`.
    - INCART: `wget -r -N -c -np https://physionet.org/files/incartdb/1.0.0/`.
    - PTB-XL: `wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/`.
    - CPSC: `wget -r -N -c -np https://physionet.org/files/cpsc2021/1.0.0/`.
    - G12EC has to be downloaded directly at: https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database?resource=download.
5. Create a `data/` directory at the root of the project.
5. Run each of the data generation files under `generation/`. Ensure the correct directories are set at the top of each of the files.

### Models
This thesis contains five different models:
- ResNet autoencoder
- ConvNeXt autoencoder
- U-Net
- Basic autoencoder
- CPD Basic autoencoder

### Running the experiments
To run the experiment execute the four corresponding files: `exp1.py`, `exp2_1.py`, `exp2_2.py` and `exp2_3.py`.
Each of the experiments can be run with the following arguments:

| Parameter                 | Description                                          | Default | type |
|---------------------------|------------------------------------------------------|---------|------|
| `--model`                 | Name of model to train                               | default | str |
| `--num_runs`              | Number of repeat runs                                | 3       |int|
| `--batch_size_mae`        | Per GPU batch size                                   | 256     |int|
| `--epochs_mae`            | Number of epochs to train the autoencoder            | 50      |int|
| `--warmup_epochs_mae`     | Number of epochs to warmup LR for autoencoder        | 0       |int|
| `--batch_size_class`      | Per GPU batch size for classifier                    | 256     |int|
| `--epochs_class`          | Number of epochs to train the classifier             | 20      |int|
| `--warmup_epochs_class`   | Number of epochs to warmup LR for classifier         | 0       |int|
| `--weight_decay_mae`      | Weight decay for autoencoder                         | 1e-4    |int|
| `--lr_mae`                | Learning rate for autoencoder                        | 5e-4    |int|
| `--min_lr_mae`            | Lower LR bound for cyclic schedulers for autoencoder | 1e-5    |int|
| `--weight_decay_class`    | Weight decay for classifier                          | 1e-4    |int|
| `--lr_class`              | Learning rate for classifier                         | 5e-4    |int|
| `--min_lr_class`          | Lower LR bound for cyclic schedulers for classifier  | 1e-5    |int|
| `--gpu`                   | Specify to use single GPU or all GPUs                | all     |str|

#### Flags

| Parameter          | Description                           |
|--------------------|---------------------------------------|
| `--contrun`        | Flag to continue from last run        |
| `--no_train_mae`   | Do not train autoencoder              |
| `--no_train_class` | Do not train classifier               |
| `--no_save_mae`    | Do not save autoencoder model         |
| `--no_save_class`  | Do not save classifier model          |


To reconstruct images use the function `reconstruct_img` in `nn_funcs.py`.
