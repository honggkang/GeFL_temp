from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import ResNet18_Weights, ResNet34_Weights
import torch.nn as nn

# from models.featureExtractor import *
import numpy as np
import random
import torch

def getModel(args):
    torch.manual_seed(args.rs)
    torch.cuda.manual_seed(args.rs)
    torch.cuda.manual_seed_all(args.rs) # if use multi-GPU

    local_models = []
    if args.models == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        from mainNetModels.cnn import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c, FE_CNN
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


    elif args.models == 'cnnbn' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        from mainNetModels.cnnbn import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c, FE_CNN
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


    elif args.models == 'cnn2' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        from mainNetModels.cnn_sameSize import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c, FE_CNN
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


    elif args.models == 'cnn3' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        from mainNetModels.cnn_sameSize2 import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c, FE_CNN
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
        
        
    elif args.models == 'cnn3' and args.dataset == 'svhn':
        from mainNetModels.cnnbn import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c, FE_CNN
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
                
    elif args.models == 'mlp':
        from mainNetModels.mlp import MLP2, MLP3, MLP3b, MLP4, MLP4b, MLP4c, MLP5, MLP5b, MLP5c, MLP6, FE_MLP
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

    elif args.models == 'mlp2':
        from mainNetModels.mlp2 import MLP2, MLP3, MLP3b, MLP4, MLP4b, MLP4c, MLP5, MLP5b, MLP5c, MLP6, FE_MLP
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
    if args.dataset == 'mnist':
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