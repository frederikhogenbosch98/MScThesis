
from torchvision import datasets, transforms
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder 
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt
import time
import numpy as np
import os
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime
import tqdm

class CosineAnnealingwithWarmUp():

    def __init__(self, optimizer, n_warmup_epochs, warmup_lr, start_lr, lower_lr, alpha, epoch_int, num_epochs):
        self._optimizer = optimizer
        self.n_steps = 0
        self.n_warmup_steps = n_warmup_epochs
        self.warmup_lr = warmup_lr
        self.start_lr = start_lr
        self.epoch_int = epoch_int
        self.num_epochs = num_epochs - n_warmup_epochs
        self.current_epoch = 0
        self.lower_lr = lower_lr

        self.warmup = np.linspace(self.warmup_lr, self.start_lr, self.n_warmup_steps)
        assert epoch_int % num_epochs  != 0, "num_epochs should be a multiple of epoch interval"

        self.alpha = np.power(alpha, np.arange(num_epochs // epoch_int))
        self.lrs = self.get_cosine_epoch()
        

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()

    def normalize(self,array):
        return np.interp(array, (array.min(), array.max()), (self.lower_lr, self.start_lr))

    def zero_grad(self):
        self._optimizer.zero_grad()

    def print_seq(self):
        plt.plot(np.concatenate((self.warmup, self.get_cosine_epoch())))
        plt.title('Learning Rate Custom Scheduler')
        plt.xlabel('epochs')
        plt.ylabel('learning rate')
        plt.show()

    def get_cosine_epoch(self):
        full_ls = np.zeros(self.num_epochs)
        for i in range(self.num_epochs // self.epoch_int):
            full_ls[i*self.epoch_int:(i+1)*self.epoch_int] = self.start_lr * self.alpha[i] * np.cos(np.linspace(0, np.pi/2, self.epoch_int))

        return np.array(self.normalize(full_ls))
    
    def _update_learning_rate(self):

        if self.n_steps < self.n_warmup_steps:
            lr = self.warmup[self.n_steps]
            self.current_epoch = 0
        else:
            lr = self.lrs[self.n_steps-self.n_warmup_steps]
            
        self.n_steps += 1

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EarlyStopperClassifier:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def conf_matrix(y_true, y_pred):
    classes = ('F', 'N', 'Q', "S", 'V')

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])

    # plt.figure(figsize = (10,6))
    # sn.heatmap(df_cm, annot=True)
    # plt.savefig('imgs/conf_matrix.png')
    return df_cm



def train_mae(model,
              trainset,
              run_dir,
              device,
              min_lr=1e-5,
              valset=None,
              weight_decay=1e-4,
              num_epochs=50, 
              n_warmup_epochs=5, 
              batch_size=128, 
              learning_rate=5e-4, 
              TRAIN_MAE=True, 
              SAVE_MODEL_MAE=True, 
              R=None, 
              fact=None, 
              contrun=False, 
              step_size=15,
              optim='Adam'):
    

    now = datetime.now()

    if TRAIN_MAE:

        if contrun:
            model.load_state_dict(torch.load('trained_models/last/last_run.pth'))

        criterion = nn.MSELoss() # mean square error loss

        if optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate, 
                                    weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)

        train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True)#, num_workers=4)
        num_iters = len(train_loader)
        if valset:
            val_loader = torch.utils.data.DataLoader(valset, 
                                        batch_size=batch_size, 
                                        shuffle=False)#, num_workers=2)

        # scheduler = CosineAnnealingwithWarmUp(optimizer, 
        #                                     n_warmup_epochs=n_warmup_epochs,
        #                                     warmup_lr=5e-5,
        #                                     start_lr=5e-5,
        #                                     lower_lr=5e-6,
        #                                     alpha=0.1,
        #                                     epoch_int=20,
        #                                     num_epochs=num_epochs)


        # early_stopper = EarlyStopper(patience=6)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=0.25)
        outputs = []
        losses = []
        val_losses = []

        scaler = GradScaler()

        print(f"Start MAE training for {n_warmup_epochs} warm-up epochs and {num_epochs-n_warmup_epochs} training epochs")
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            t_epoch_start = time.time()
            model.train()
            with tqdm.tqdm(train_loader, unit="batch", leave=False) as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"epoch {epoch+1}")
                    img, _ = data
                    img = img.to(device)
                    unmasked_img = img
                    optimizer.zero_grad()
                    with autocast():
                        recon = model(img)
                        loss = criterion(recon, unmasked_img)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item()
                    tepoch.set_postfix(lr=optimizer.param_groups[0]['lr'])
                scheduler.step()

            if (epoch + 1) % 10 == 0 and epoch != 0 and SAVE_MODEL_MAE:
                torch.save(model.state_dict(), f'{run_dir}/MAE_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), 'trained_models/last/last_run.pth')

            if valset:
                model.eval()
                validation_loss = 0.0

                with torch.no_grad():
                    for data in val_loader:
                        imgs, _  = data
                        imgs = imgs.to(device)
                        with autocast():
                            outputs = model(imgs)
                            loss = criterion(outputs, imgs)
                        validation_loss += loss.item() * imgs.size(0)

                validation_loss /= len(val_loader.dataset)
                
                # if early_stopper.early_stop(validation_loss):             
                #     print(f"EARLY STOPPING AT EPOCH: {epoch}")
                #     break
            else:
                validation_loss = 0

            epoch_loss = running_loss / len(train_loader)
            losses.append(epoch_loss)
            val_losses.append(validation_loss)
            t_epoch_end = time.time()

            print('epoch {}: training loss: {:.7f}, val loss: {:.7f}, duration: {:.2f}s, lr: {:.2e}'.format(epoch+1, epoch_loss, validation_loss, t_epoch_end - t_epoch_start, optimizer.param_groups[0]['lr']))

           
        t_end = time.time()
        print(f"End of MAE training. Training duration: {np.round((t_end-t_start)/60.0,2)}m.")

        if SAVE_MODEL_MAE:
            save_folder = f'{run_dir}/MAE_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}.pth'
            torch.save(model.state_dict(), save_folder)
            torch.save(model.state_dict(), 'trained_models/last/last_run.pth')
            print(f'MAE model saved to {save_folder}')

        # plot_losses(epoch+1, losses)        
        print("\n")


    else:
        model.load_state_dict(torch.load('trained_models/last/last_run.pth'))
        print(f'dataset loaded')
        losses = np.zeros(num_epochs)
        val_losses = np.zeros(num_epochs)
        t_epoch_end = 0
        t_epoch_start = 0


    return model, losses, val_losses, (t_epoch_end - t_epoch_start )



def eval_mae(model,
            testset, 
            R, 
            device=torch.device('cuda:0'), 
            batch_size=128):
    
    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    criterion = nn.MSELoss(reduction='mean')
    total_loss = 0.0
    count = 0

    with torch.no_grad():  
        for inputs, _ in test_loader:
            inputs = inputs.to(device) 
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
            total_loss += loss.item() 
            count += 1

    average_loss = np.round(total_loss / count, 6)

    print(f'Average MSE Loss on Test Set: {average_loss}')



    return average_loss


def train_classifier(classifier, 
                     trainset,
                    run_dir, 
                    device, 
                    testset, 
                    weight_decay = 1e-4, 
                    min_lr=1e-6, 
                    valset=None, 
                    num_epochs=25, 
                    n_warmup_epochs=5, 
                    learning_rate=1e-4, 
                    batch_size=128, 
                    TRAIN_CLASSIFIER=True, 
                    SAVE_MODEL_CLASSIFIER=True, 
                    R=None, 
                    fact=None,
                    model_str='basic',
                    optim='Adam'):

    now = datetime.now()
    classifier.to(device)
    if TRAIN_CLASSIFIER:

        if model_str == 'basic' or model_str == 'convnext':
            for param in classifier.module.encoder.parameters():
                param.requires_grad = False
        elif model_str == 'unet':
            for param in classifier.module.enc1.parameters():
                param.requires_grad = False
            for param in classifier.module.enc2.parameters():
                param.requires_grad = False
            for param in classifier.module.enc3.parameters():
                param.requires_grad = False
        elif model_str == 'resnet':
            for param in classifier.module.conv1.parameters():
                param.requires_grad = False
            for param in classifier.module.downlayer0.parameters():
                param.requires_grad = False
            for param in classifier.module.downlayer1.parameters():
                param.requires_grad = False
            for param in classifier.module.downlayer2.parameters():
                param.requires_grad = False
            for param in classifier.module.downlayer3.parameters():
                param.requires_grad = False

        train_loader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=2)
        

        if optim == 'Adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()),
                                    lr=learning_rate, 
                                    weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()),
                            lr=learning_rate,
                            weight_decay=weight_decay)

        scheduler = StepLR(optimizer, step_size=10, gamma=0.5) 
        if valset:
            val_loader = torch.utils.data.DataLoader(valset, 
                                batch_size=batch_size, 
                                shuffle=False, num_workers=2)    

        # scheduler = CosineAnnealingwithWarmUp(optimizer, 
        #                                         n_warmup_epochs=n_warmup_epochs, 
        #                                         warmup_lr=1e-4, 
        #                                         start_lr=learning_rate, 
        #                                         lower_lr=min_lr,
        #                                         alpha=0.75, 
        #                                         epoch_int=20, 
        #                                         num_epochs=num_epochs)

        criterion =  nn.CrossEntropyLoss().to(device)

        # early_stopper = EarlyStopper(patience=10, min_delta=0.0001)
        scaler = GradScaler()
        losses = []
        val_losses = []
        run_accs = []
        print(f"Start CLASSIFIER training for {n_warmup_epochs} warm-up epochs and {num_epochs-n_warmup_epochs} training epochs")        
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            classifier.train()
            t_epoch_start = time.time()
            with tqdm.tqdm(train_loader, unit="batch", leave=False) as tepoch:
                for inputs, labels in tepoch: 
                    tepoch.set_description(f"epoch {epoch+1}")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with autocast():
                        outputs = classifier(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss / (batch_size*(epoch+1)))
            scheduler.step()

            if epoch % 10 == 0 and epoch != 0:
                torch.save(classifier.state_dict(), f'{run_dir}/CLASSIFIER_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}_epoch_{epoch}.pth')

            if valset:
                classifier.eval()  
                validation_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():  
                    for data, target in val_loader:
                        data,  target = data.to(device), target.to(device)
                        with autocast():
                            output = classifier(data)
                            loss = criterion(output, target)

                        validation_loss += loss.item() * data.size(0)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                validation_loss /= len(val_loader.dataset)
                accuracy = correct / total * 100


                # if early_stopper.early_stop(validation_loss):             
                #     print(f"EARLY STOPPING AT EPOCH: {epoch}")
                #     break
            else:
                validation_loss = 0


            epoch_loss = running_loss / len(train_loader)
            t_epoch_end = time.time()
            epoch_loss = running_loss / len(train_loader)

            print('epoch {}: training loss: {:.7f}, val loss: {:.7f}, accuracy: {:.2f}, duration: {:.2f}s, lr: {:.2e}'.format(epoch+1, epoch_loss, validation_loss, accuracy, t_epoch_end - t_epoch_start, optimizer.param_groups[0]['lr']))

            losses.append(epoch_loss)
            val_losses.append(validation_loss)


        t_end = time.time()
        print(f"End of CLASSIFIER training. Training duration: {np.round((t_end-t_start)/60.0,2)}m. final loss: {loss}.")

        if SAVE_MODEL_CLASSIFIER:
            save_folder = f'{run_dir}/CLASSIFIER_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}.pth'
            torch.save(classifier.state_dict(), save_folder)
            print(f'classifier model saved to {save_folder}')

        print("\n")

    else:
        print('classifier model loaded')
        classifier.load_state_dict(torch.load('trained_models/'))
        losses = np.zeros(num_epochs)
        val_losses = np.zeros(num_epochs)


    return classifier, losses, val_losses


def eval_classifier(model,
                    testset, 
                    device, 
                    batch_size=128):

    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    y_pred = []
    y_true = []
    correct = 0
    total = 0
    test_accuracy = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                output = model(images)
            _, predicted = torch.max(output.data, 1)
            for i in range(len(labels)):
                labels_cpu = labels.cpu()
                preds_cpu = predicted.cpu()
                y_true.append(labels_cpu[i].item())
                y_pred.append(preds_cpu[i].item())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
        
    print(f'acc: {np.mean(test_accuracy)}')

    accuracy = 100 * correct / total
    print(f'Accuracy: {np.round(accuracy,3)}%')
    conf_matrix(y_true, y_pred)

    return np.mean(test_accuracy)



def reconstruct_img(model, testset, R, run_dir, device):
    data_list = []

    for data, _ in testset:
        data_list.append(data.unsqueeze(0))

    test_data_tensor = torch.cat(data_list, dim=0)

    test_data_tensor = test_data_tensor.to(device)

    recon = model(test_data_tensor[0:64,:,:,:])
    for i in range(10):
        recon_cpu = recon[i,:,:,:]#.detach().numpy()
        recon_cpu = recon_cpu.cpu()
        plotimg(test_data_tensor[i,:,:,:], recon_cpu, R, run_dir, i)
    print(f'10 reconstructed images saved to {run_dir}')