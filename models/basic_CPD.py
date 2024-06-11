import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

class Basic_CPD(nn.Module):
    def __init__(self, R=20, factorization='cp', in_channels=1, channels=[64, 128, 256, 512]):
        super(Basic_CPD, self).__init__()
        print(channels)
        self.encoder = nn.Sequential(
            # LAYER 1
            tltorch.FactorizedConv.from_conv(nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),


            # LAYER 2
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),


            # LAYER 3
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),


            # LAYER 4
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2)
            
        )

        self.decoder = nn.Sequential(

            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),


            # Corresponds to LAYER 3 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),


            # Corresponds to LAYER 2 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),


            # Corresponds to LAYER 1 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),            
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x