# encoder based on original paper implementation: https://github.com/facebookresearch/ConvNeXt

import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch
from torchvision.ops import stochastic_depth


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)
    

class ConvNextBlock(nn.Module):
    def __init__(self, filter_dim, layer_scale=1e-6):
        super().__init__()
        self.block = nn.Sequential(*[
            LayerNorm(filter_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(filter_dim, filter_dim, kernel_size=7, padding=3, groups=filter_dim),
            nn.GELU(),
            nn.Conv2d(filter_dim, filter_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(filter_dim * 4, filter_dim, kernel_size=1),
        ])
        self.gamma = nn.Parameter(layer_scale * torch.ones((filter_dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        return x + self.block(x) * self.gamma


class ConvNextLayer(nn.Module):

    def __init__(self, filter_dim, depth, drop_rates):
        super().__init__()
        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(ConvNextBlock(filter_dim=filter_dim))

        self.drop_rates = drop_rates

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = x + stochastic_depth(block(x),
                                     self.drop_rates[idx],
                                     mode="batch",
                                     training=self.training)
        return x


class ConvNextEncoder(nn.Module):

    def __init__(self,
                 num_channels=1,
                 patch_size=4,
                 layer_dims=[32, 64, 128, 256],
                 depths=[1, 1, 3, 1],
                 drop_rate=0.):
        super(ConvNextEncoder, self).__init__()

        # init downsample layers with stem
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(num_channels, layer_dims[0], kernel_size=1, stride=1),
                LayerNorm(layer_dims[0],
                              eps=1e-6,
                              data_format="channels_first")
            )])
        for idx in range(len(layer_dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(layer_dims[idx],
                              eps=1e-6,
                              data_format="channels_first"),
                    nn.Conv2d(layer_dims[idx],
                              layer_dims[idx + 1],
                              kernel_size=2,
                              stride=2),
                ))

        drop_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(depths))] 
        self.stage_layers = nn.ModuleList([])
        for idx, layer_dim in enumerate(layer_dims):
            layer_dr = drop_rates[sum(depths[:idx]): sum(depths[:idx]) + depths[idx]]
            self.stage_layers.append(
                ConvNextLayer(filter_dim=layer_dim, depth=depths[idx], drop_rates=layer_dr))


    def forward(self, x):
        all_layers = list(zip(self.downsample_layers, self.stage_layers))
        for downsample_layer, stage_layer in all_layers:
            x = downsample_layer(x)
            x = stage_layer(x)
        
        return x
    


class ConvNextDecoder(nn.Module):

    def __init__(self,
                 num_channels=1,
                 num_classes=10,
                 patch_size=4,
                 layer_dims=[32, 64, 128, 256],
                 depths=[1, 1, 3, 1],
                 drop_rate=0.):
        super(ConvNextDecoder, self).__init__()
        layer_dims = list(reversed(layer_dims))
        self.upsample_layers = nn.ModuleList([])

        for idx in range(len(layer_dims) - 1):
            print(layer_dims[idx])
            self.upsample_layers.append(nn.Conv2d(layer_dims[idx],
                              layer_dims[idx + 1],
                              kernel_size=1,
                              stride=1))

        drop_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(depths))] 
        self.stage_layers = nn.ModuleList([])
        for idx, layer_dim in enumerate(layer_dims):
            layer_dr = drop_rates[sum(depths[:idx]): sum(depths[:idx]) + depths[idx]]
            self.stage_layers.append(
                ConvNextLayer(filter_dim=layer_dim, depth=depths[idx], drop_rates=layer_dr))

                
        self.upsample_layers.append(
            nn.Conv2d(layer_dims[-1], num_channels, kernel_size=1, stride=1)
        )
        self.upsamples = nn.ModuleList([])
        self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.upsamples.append(nn.Upsample(scale_factor=1, mode='bilinear'))

    def forward(self, x):
        all_layers = list(zip(self.upsamples, self.upsample_layers, self.stage_layers))
        for upsample, upsample_layers, stage_layer in all_layers:
            x = upsample(x)
            x = stage_layer(x)
            x = upsample_layers(x)

        return x




class ConvNext(nn.Module):
    def __init__(self,
                 layer_dims=[32, 64, 128, 256],
                 depths=[1, 1, 3, 1]
                 ):

        super(ConvNext, self).__init__()
        self.encoder = ConvNextEncoder(layer_dims=layer_dims, depths=depths)
        self.decoder = ConvNextDecoder(layer_dims=layer_dims, depths=depths)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
