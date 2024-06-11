import argparse
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import tqdm
from datetime import datetime

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from models.basic import Basic, Classifier_Basic
from models.basic_CPD import Basic_CPD
from models.resnet import ResNet
from models.convnext import ConvNext
from models.unet import UNet, ClassifierUnet

from print_funs import plot_losses, plotimg, plot_single_img, count_parameters
from nn_funcs import CosineAnnealingwithWarmUp, EarlyStopper, train_mae, train_classifier, eval_mae, eval_classifier



def get_args_parser():
    parser = argparse.ArgumentParser('training', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='default', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--batch_size_mae', default=256, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs_mae', default=50, type=int)
    parser.add_argument('--warmup_epochs_mae', type=int, default=0,
                        help='epochs to warmup LR')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of repeat runs')

    parser.add_argument('--batch_size_class', default=256, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs_class', default=20, type=int)
    parser.add_argument('--warmup_epochs_class', type=int, default=0,
                        help='epochs to warmup LR') 
    
    # Optimizer parameters
    parser.add_argument('--weight_decay_mae', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_mae', type=float, default=5e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr_mae', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay_class', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_class', type=float, default=5e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr_class', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    

    parser.add_argument('--gpu', default='all', type=str,
                        help='single or all')

    # save parameters
    parser.add_argument('--contrun', action='store_true', help='flag continue from last run')
    parser.add_argument('--no_train_mae', action='store_false',  help='Train MAE', dest='train_mae')
    parser.add_argument('--no_train_class', action='store_false', help='Train Classifier', dest='train_class')
    parser.add_argument('--no_save_mae', action='store_false',  help='Save MAE model', dest='save_mae')
    parser.add_argument('--no_save_class', action='store_false', help='Save Classifier model', dest='save_class')

    return parser



if __name__ == "__main__":
    torch.manual_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()

    dtype = torch.float32
    device_ids = [0, 2, 3]
    main_device = device_ids[0]
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{main_device}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'SELECTED DEVICE: {device}')


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),         
        ])


    # UNLABELED
    ptbxl_dir = 'data/physionet/ptbxl/'
    ptbxl_dataset = datasets.ImageFolder(root=ptbxl_dir, transform=transform)
    georgia_dir = 'data/physionet/georgia/'
    georgia_dataset = datasets.ImageFolder(root=georgia_dir, transform=transform)
    china_dir = 'data/physionet/china/'
    china_dataset = datasets.ImageFolder(root=china_dir, transform=transform)
    combined_unsupervised_train = torch.utils.data.ConcatDataset([ptbxl_dataset, georgia_dataset, china_dataset])
    trainset_un, testset_un, valset_un = torch.utils.data.random_split(combined_unsupervised_train, [190000, 25000, 17077])

    # LABELED
    mitbih_ds11_dir = 'data/physionet/mitbih/DS11/'
    mitbih_ds12_dir = 'data/physionet/mitbih/DS12/'
    mitbih_ds2_dir = 'data/physionet/mitbih/DS2/'
    mitbih_dataset_train = datasets.ImageFolder(root=mitbih_ds11_dir, transform=transform)
    mitbih_dataset_val = datasets.ImageFolder(root=mitbih_ds12_dir, transform=transform)
    mitbih_dataset_test = datasets.ImageFolder(root=mitbih_ds2_dir, transform=transform) 

    incartdb_dir = 'data/physionet/incartdb/'
    incartdb_dataset = datasets.ImageFolder(root=incartdb_dir, transform=transform)

    trainset_sup = torch.utils.data.ConcatDataset([mitbih_dataset_train, incartdb_dataset])
    valset_sup = mitbih_dataset_val
    testset_sup = mitbih_dataset_test

    mega_mses = []
    accuracies = []

    # MAE
    num_warmup_epochs_mae = args.warmup_epochs_mae
    num_epochs_mae = args.epochs_mae + num_warmup_epochs_mae

    # CLASSIFIER
    num_warmup_epochs_classifier = args.warmup_epochs_class
    num_epochs_classifier = args.epochs_class + num_warmup_epochs_classifier

    mae_losses_run = np.zeros((4, num_epochs_mae))
    mae_val_losses_run = np.zeros((4, num_epochs_mae))
    class_losses_run = np.zeros((4, num_epochs_classifier))
    class_val_losses_run = np.zeros((4, num_epochs_classifier))

    models = [Basic(channels=[32, 64, 128, 256]), ConvNext() ,UNet(), ResNet()]

    model_strs = ['basic', 'unet', 'resnet', 'convnext'] 
    lr = [5e-5, 1e-4, 1e-4, 1e-4]

    CLASSIFY = True
    NUM_RUNS = args.num_runs

    now = datetime.now()
    run_dir = f'trained_models/model_comparison/RUN_{now.day}_{now.month}_{now.hour}_{now.minute}_Basic'
    for i, model in enumerate(models):
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
        mses = []
        current_pams = count_parameters(model)
        print(f'num params: {current_pams}')
        for j in range(NUM_RUNS):
            os.makedirs(f'{run_dir}/{model_strs[i]}/{j}', exist_ok=True)
            mae, mae_losses, mae_val_losses, epoch_time = train_mae(model=model, 
                                                        trainset=trainset_un,
                                                        valset=valset_un,
                                                        learning_rate=lr[i],
                                                        min_lr = args.min_lr_mae,
                                                        weight_decay = args.weight_decay_mae,
                                                        num_epochs=num_epochs_mae,
                                                        n_warmup_epochs=num_warmup_epochs_mae,
                                                        TRAIN_MAE=args.train_mae,
                                                        SAVE_MODEL_MAE=args.save_mae,
                                                        R=0,
                                                        batch_size=args.batch_size_mae,
                                                        fact=model_strs[i],
                                                        run_dir = run_dir,
                                                        contrun = args.contrun,
                                                        device = device,
                                                        step_size=15)
            
            mae_losses_run[i,:] = mae_losses
            mae_val_losses_run[i,:] = mae_val_losses
            train_save_folder = f'{run_dir}/{model_strs[i]}/{j}/MAE_losses_train.npy'
            val_save_folder = f'{run_dir}/{model_strs[i]}/{j}/MAE_losses_{model_strs[i]}_val.npy'
            np.save(train_save_folder, mae_losses)
            np.save(val_save_folder, mae_val_losses)

            mses.append(eval_mae(mae, testset_un, R=0, device=device))

            if CLASSIFY:
                num_classes = 5
                if args.model == 'default':
                    classifier = Classifier_Basic(autoencoder=mae.module, in_features=2048, out_features=num_classes)
                    # classifier = ClassifierUnet(autoencoder=mae.module, in_features=2048, out_features=num_classes)
                if args.gpu == 'all':
                    classifier = nn.DataParallel(classifier, device_ids=device_ids).to(device) 

                classifier, class_losses, class_val_losses = train_classifier(classifier=classifier, 
                                            trainset=trainset_sup, 
                                            valset=valset_sup, 
                                            num_epochs=num_epochs_classifier, 
                                            n_warmup_epochs=num_warmup_epochs_classifier, 
                                            learning_rate=args.lr_class,
                                            min_lr = args.min_lr_class,
                                            weight_decay = args.weight_decay_class,
                                            batch_size=args.batch_size_class, 
                                            TRAIN_CLASSIFIER=args.train_class, 
                                            SAVE_MODEL_CLASSIFIER=args.save_class,
                                            R=0,
                                            fact=model_strs[i],
                                            run_dir = run_dir,
                                            device = device,
                                            testset=testset_sup)
                                            
                accuracy = eval_classifier(classifier, testset_sup, device=device)
        mega_mses.append(np.mean(mses))


    print(mega_mses)

    






