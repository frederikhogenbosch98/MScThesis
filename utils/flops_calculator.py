import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import sys, os
from pprint import pprint
from tabulate import tabulate
from collections import Counter

sys.path.append(os.path.abspath(os.path.join('..', 'models')))
sys.path.append(os.path.abspath(os.path.join('..')))

from print_funs import count_parameters
from resnet import ResNet
from unet import UNet
from convnext import ConvNext
from basic import Basic
from basic_CPD import Basic_CPD

x = torch.randn(1, 1, 128, 128)
num_params_uncompressed = 9411649

flops = FlopCountAnalysis(Basic(), x)
current_pams = count_parameters(Basic())
print(f'NUM PARAMS: {current_pams}')
print(f"FLOPs: {flops.total()}")

flops_by_operator = flops.by_operator()
print("\nFLOPs by Operator:")
print(tabulate(flops_by_operator.items(), headers=["Operator", "FLOPs"], tablefmt="pretty"))


for r in [5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175, 200]:

    print(f'R: {r}')
    flops = FlopCountAnalysis(Basic_CPD(R=r), x)

    current_pams = count_parameters(Basic_CPD(R=r))
    print(f'NUM PARAMS: {current_pams}')
    comp_ratio = num_params_uncompressed/current_pams
    print(f'COMPRESSION RATIO: {comp_ratio}')
    
    total_flops = flops.total()
    flops_by_operator = flops.by_operator()

    print("Total FLOPs:", total_flops)
    print("\nFLOPs by Operator:")
    print(tabulate(flops_by_operator.items(), headers=["Operator", "FLOPs"], tablefmt="pretty"))


