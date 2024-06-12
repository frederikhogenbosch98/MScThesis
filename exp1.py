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
from models.resnet import ResNet, ClassifierResNet
from models.convnext import ConvNext
from models.unet import UNet, ClassifierUnet

from utils.print_funcs import plot_losses, plotimg, plot_single_img, count_parameters
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
    ptbxl_dir = 'data/ptbxl/'
    ptbxl_dataset = datasets.ImageFolder(root=ptbxl_dir, transform=transform)
    g12ec_dir = 'data/g12ec/'
    g12ec_dataset = datasets.ImageFolder(root=g12ec_dir, transform=transform)
    cpsc_dir = 'data/cpsc/'
    cpsc_dataset = datasets.ImageFolder(root=cpsc_dir, transform=transform)
    combined_unsupervised_train = torch.utils.data.ConcatDataset([ptbxl_dataset, g12ec_dataset, cpsc_dataset])
    trainset_un, testset_un, valset_un = torch.utils.data.random_split(combined_unsupervised_train, [190000, 25000, 17077])

    # LABELED
    mitbih_ds11_dir = 'data/mitbih/DS11/'
    mitbih_ds12_dir = 'data/mitbih/DS12/'
    mitbih_ds2_dir = 'data/mitbih/DS2/'
    mitbih_dataset_train = datasets.ImageFolder(root=mitbih_ds11_dir, transform=transform)
    mitbih_dataset_val = datasets.ImageFolder(root=mitbih_ds12_dir, transform=transform)
    mitbih_dataset_test = datasets.ImageFolder(root=mitbih_ds2_dir, transform=transform) 

    incartdb_dir = 'data/incartdb/'
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


    models = [Basic(channels=[32, 64, 128, 256]), ConvNext(), UNet(), ResNet()]

    model_strs = ['basic', 'unet', 'resnet', 'convnext'] 
    lr = [5e-5, 1e-4, 1e-4, 1e-4]

    CLASSIFY = True
    NUM_RUNS = args.num_runs
    NUM_CLASSES = 5

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
            

            eval_mae(mae, testset_un, R=0, device=device)

            if CLASSIFY:
                if args.model == 'default':
                    if model_strs[i] == 'basic' or model_strs[i] == 'convnext':
                        classifier = Classifier_Basic(autoencoder=mae.module, out_features=NUM_CLASSES)
                    elif model_strs[i] == 'resnet':
                        classifier = ClassifierResNet(autoencoder=mae.module, out_features=NUM_CLASSES)
                    elif model_strs[i] == 'unet':
                        classifier = ClassifierUnet(autoencoder=mae.module, out_features=NUM_CLASSES)
                    else:
                        raise ValueError('Model not recognized')

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
                                            
                eval_classifier(classifier, testset_sup, device=device)



    






