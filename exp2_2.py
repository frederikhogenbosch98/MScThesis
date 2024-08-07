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

from print_funcs import plot_losses, plotimg, plot_single_img, count_parameters
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

    incartdb_dir = 'data/incart/'
    incartdb_dataset = datasets.ImageFolder(root=incartdb_dir, transform=transform)

    trainset_sup = torch.utils.data.ConcatDataset([mitbih_dataset_train, incartdb_dataset])
    valset_sup = mitbih_dataset_val
    testset_sup = mitbih_dataset_test

    training_supset = trainset_sup

    total_unsupervised = len(combined_unsupervised_train)
    total_supervised = len(training_supset)

    # MAE
    num_warmup_epochs_mae = args.warmup_epochs_mae
    num_epochs_mae = args.epochs_mae + num_warmup_epochs_mae

    # CLASSIFIER
    num_warmup_epochs_classifier = args.warmup_epochs_class
    num_epochs_classifier = args.epochs_class + num_warmup_epochs_classifier

    NUM_PARAMS_UNCOMPRESSED = 9411649

    CLASSIFY = True
    fact = 'cp'
    R_LIST = [0, 100]
    ratios = [0.025, 0.05, 0.10, 0.2]

    now = datetime.now()
    run_dir = f'trained_models/compressed/RUN_{now.day}_{now.month}_{now.hour}_{now.minute}_high_rank_run'
    os.makedirs(f'{run_dir}/', exist_ok=True)
    for i, R in enumerate(R_LIST):
        print(f'R: {R}')
        # for r in ratios:
        #     print(f'ratio: {r}')
        #     un_vals = int(r*190000)
        #     un_vals_other = 190000 - un_vals 
        #     sup_vals = int(r*200000)
        #     sup_vals_other = int(18192 + 200000 - sup_vals)
        #     print(len(training_supset))
        #     print(sup_vals + sup_vals_other)
        #     trainset_un, testset_un, valset_un, _ = torch.utils.data.random_split(combined_unsupervised_train, [un_vals, 25000, 17077, un_vals_other])
        #     trainset_sup, _ = torch.utils.data.random_split(training_supset, [sup_vals, sup_vals_other])
        for r in ratios:
            print(f'ratio: {r}')
            
            un_train = int(r * total_unsupervised)
            un_test = int((1 - r) * 0.6 * total_unsupervised)
            un_val = int((1 - r) * 0.3 * total_unsupervised)
            un_other = total_unsupervised - un_train - un_test - un_val
            
            sup_train = int(r * total_supervised)
            sup_other = total_supervised - sup_train
            
            trainset_un, testset_un, valset_un, remaining_un = torch.utils.data.random_split(
                combined_unsupervised_train, 
                [un_train, un_test, un_val, un_other]
            )
            
            trainset_sup, remaining_sup = torch.utils.data.random_split(
                training_supset, 
                [sup_train, sup_other]
            )
            if R == 0:
                model = Basic(channels=[64,128,256,512])
            else:
                model = Basic_CPD(R=R, factorization='cp')

            model = nn.DataParallel(model, device_ids=device_ids).to(device)
            mses = []
            current_pams = count_parameters(model)
            print(f'NUM PARAMS: {current_pams}')
            comp_ratio = NUM_PARAMS_UNCOMPRESSED/current_pams
            print(f'COMPRESSION RATIO: {comp_ratio}')

            for j in range(args.num_runs):
                os.makedirs(f'{run_dir}/R_{R}/{j}', exist_ok=True)
                mae, mae_losses, mae_val_losses, epoch_time = train_mae(model=model, 
                                                            trainset=trainset_un,
                                                            valset=valset_un,
                                                            learning_rate=args.lr_mae,
                                                            min_lr = args.min_lr_mae,
                                                            weight_decay = args.weight_decay_mae,
                                                            num_epochs=num_epochs_mae,
                                                            n_warmup_epochs=num_warmup_epochs_mae,
                                                            TRAIN_MAE=args.train_mae,
                                                            SAVE_MODEL_MAE=args.save_mae,
                                                            R=R,
                                                            batch_size=args.batch_size_mae,
                                                            fact=fact,
                                                            run_dir = run_dir,
                                                            contrun = args.contrun,
                                                            device = device,
                                                            step_size=15)
                
                train_save_folder = f'{run_dir}/R_{R}/{j}/MAE_losses_train.npy'
                val_save_folder = f'{run_dir}/R_{R}/{j}/MAE_losses_val.npy'
                np.save(train_save_folder, mae_losses)
                np.save(val_save_folder, mae_val_losses)

                mses.append(eval_mae(mae, testset_un,R,device=device))


                if CLASSIFY:
                    num_classes = 5
                    if args.model == 'default':
                        classifier = Classifier_Basic(autoencoder=mae.module,  out_features=num_classes)
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
                                                R=R,
                                                fact='cp',
                                                run_dir = run_dir,
                                                device = device,
                                                testset=testset_sup)

                    eval_classifier(classifier, testset_sup, device=device)
                
