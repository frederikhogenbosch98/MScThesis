# encoder based on the work in https://github.com/JayPatwardhan/ResNet-PyTorch/

import torch, torchvision
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, layers=[1, 1, 3, 1], num_classes = 7):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.outplanes = 256
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(32),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.downlayer0 = self._make_layer(ResidualBlock, 32, layers[0], stride = 1)
        self.downlayer1 = self._make_layer(ResidualBlock, 64, layers[1], stride = 1)
        self.downlayer2 = self._make_layer(ResidualBlock, 128, layers[2], stride = 1)
        self.downlayer3 = self._make_layer(ResidualBlock, 256, layers[3], stride = 1)

        self.uplayer1 = self._make_uplayer(ResidualBlock, 256, layers[3], stride = 1) 
        self.uplayer2 = self._make_uplayer(ResidualBlock, 128, layers[2], stride = 1)
        self.uplayer3 = self._make_uplayer(ResidualBlock, 64, layers[1], stride = 1)
        self.uplayer4 = self._make_uplayer(ResidualBlock, 32, layers[0], stride = 1) 

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upconv1 =  nn.Sequential(
                        nn.Conv2d(32, 1, kernel_size = 7, stride = 1, padding = 3),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Sigmoid()) 
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _make_uplayer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.outplanes != planes:
            
            downsample = nn.Sequential(
                nn.ConvTranspose2d(self.outplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.outplanes, planes, stride, downsample))
        self.outplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.outplanes, planes))

        return nn.Sequential(*layers) 

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.downlayer0(x)
        x = self.downlayer1(x)
        x = self.maxpool(x)
        x = self.downlayer2(x)
        x = self.downlayer3(x)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.upsample(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.upsample(x)
        x = self.upconv1(x)
        return x
    

class ClassifierResNet(nn.Module):
    def __init__(self, autoencoder, out_features):
        super(ClassifierResNet, self).__init__()
        self.conv1 = autoencoder.conv1
        self.pool1 = autoencoder.maxpool
        self.downlayer0 = autoencoder.downlayer0
        self.downlayer1 = autoencoder.downlayer1
        self.downlayer2 = autoencoder.downlayer2
        self.downlayer3 = autoencoder.downlayer3

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
                nn.Linear(32768, 256),
                nn.GELU(),
                nn.BatchNorm1d(num_features=256),
                nn.Dropout(0.5),
                nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.downlayer0(x)
        x = self.downlayer1(x)
        x = self.pool1(x)
        x = self.downlayer2(x)
        x = self.downlayer3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
