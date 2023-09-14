from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import ResNet18_Weights, ResNet34_Weights
import torch.nn as nn
from models.mlp import *
from models.cnn import *
from models.featureExtractor import *
import numpy as np
import random

def getModel(args):

    local_models = []
    if args.models == 'cnn':
        net_temp1 = CNN2().to(args.device)
        net_temp2 = CNN3().to(args.device)
        net_temp2b = CNN3b().to(args.device)
        net_temp3 = CNN3c().to(args.device)
        net_temp3b = CNN4().to(args.device)
        net_temp3c = CNN4b().to(args.device)
        net_temp4 = CNN4c().to(args.device)
        net_temp4b = CNN5().to(args.device)
        net_temp4c = CNN5b().to(args.device)
        net_temp5 = CNN5c().to(args.device)
        
        common_net = FE_CNN().to(args.device)

        # if args.pretrained:
        #     net_temp1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        #     net_temp2 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # else:
        #     net_temp1 = resnet18(weights=None)
        #     net_temp2 = resnet34(weights=None)

        # net_temp1.fc = nn.Linear(512 * 1, args.num_classes)
        # net_temp1.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # net_temp2.fc = nn.Linear(512 * 1, args.num_classes)
        # net_temp2.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # common_net = FE_ResNet()
        # common_net.to(args.device)
        
    elif args.models == 'mlp':
        net_temp1 = MLP2().to(args.device)
        net_temp2 = MLP3().to(args.device)
        net_temp2b = MLP3b().to(args.device)
        net_temp3 = MLP4().to(args.device)
        net_temp3b = MLP4b().to(args.device)
        net_temp3c = MLP4c().to(args.device)
        net_temp4 = MLP5().to(args.device)
        net_temp4b = MLP5b().to(args.device)
        net_temp4c = MLP5c().to(args.device)
        net_temp5 = MLP6().to(args.device)
        
        common_net = FE_MLP().to(args.device)

    local_models.append(net_temp1)
    local_models.append(net_temp2)
    local_models.append(net_temp2b)
    local_models.append(net_temp3)
    local_models.append(net_temp3b)
    local_models.append(net_temp3c)
    local_models.append(net_temp4)
    local_models.append(net_temp4b)
    local_models.append(net_temp4c)
    local_models.append(net_temp5)

    args.num_models = len(local_models)
    return local_models, common_net

def getModelwoC(args): # without common net

    local_models = []
    if args.dataset == 'cifar10':
        if args.pretrained:
            net_temp1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            net_temp2 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            net_temp1 = resnet18(weights=None)
            net_temp2 = resnet34(weights=None)

        net_temp1.fc = nn.Linear(512 * 1, args.num_classes)
        net_temp1.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net_temp2.fc = nn.Linear(512 * 1, args.num_classes)
        net_temp2.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        common_net = FE_ResNet()
        common_net.to(args.device)
        
    elif args.dataset == 'mnist':
        net_temp1 = MLP2().to(args.device)
        net_temp2 = MLP3().to(args.device)
        net_temp2b = MLP3b().to(args.device)
        net_temp3 = MLP4().to(args.device)
        net_temp3b = MLP4b().to(args.device)
        net_temp3c = MLP4c().to(args.device)
        net_temp4 = MLP5().to(args.device)
        net_temp4b = MLP5b().to(args.device)
        net_temp4c = MLP5c().to(args.device)
        net_temp5 = MLP6().to(args.device)

        
    local_models.append(net_temp1)
    local_models.append(net_temp2)
    local_models.append(net_temp2b)
    local_models.append(net_temp3)
    local_models.append(net_temp3b)
    local_models.append(net_temp3c)
    local_models.append(net_temp4)
    local_models.append(net_temp4b)
    local_models.append(net_temp4c)
    local_models.append(net_temp5)

    args.num_models = len(local_models)
    return local_models