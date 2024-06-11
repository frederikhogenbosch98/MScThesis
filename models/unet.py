import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, channels=[32, 64, 128, 256], depths=[1, 1, 1]):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)


        self.enc4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
        )
        self.pool4 = nn.MaxPool2d(2, stride=2)


        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = nn.Sequential(
            nn.Conv2d(channels[3] + channels[3], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(), 
            
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(channels[2] + channels[2], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels[1] + channels[1], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels[0] + channels[0], in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        
        # Decoder
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        return x


class ClassifierUnet(nn.Module):
    def __init__(self, autoencoder, in_features, out_features):
        super(ClassifierUnet, self).__init__()
        self.enc1 = autoencoder.enc1
        self.pool1 = autoencoder.pool1
        self.enc2 = autoencoder.enc2
        self.pool2 = autoencoder.pool2
        self.enc3 = autoencoder.enc3
        self.pool3 = autoencoder.pool3

        self.flatten = nn.Flatten()
        self.norm = nn.LayerNorm(in_features, eps=1e-6) 

        self.classifier = nn.Sequential(
                nn.Linear(32768, 256),
                nn.GELU(),
                nn.BatchNorm1d(num_features=256),
                nn.Dropout(0.5),
                nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.classifier(x)
        return x
