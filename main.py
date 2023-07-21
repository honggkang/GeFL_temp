from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import ResNet18_Weights, ResNet34_Weights

import argparse
import os
import torchsummary
import wandb
from datetime import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import copy

from utils.fed import *
from utils.getData import *
from models.featureExtractor import *
from utils.util import test_img, get_logger
# from models import *
# from utils.NeFedAvg import NeFedAvg
# from AutoAugment.autoaugment import ImageNetPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=100)
parser.add_argument('--noniid', action='store_true') # default: false2
parser.add_argument('--class_per_each_client', type=int, default=10)

parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--wu_epochs', type=int, default=20) # warm-up epochs
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal') # normal worst
parser.add_argument('--rs', type=int, default=0)

parser.add_argument('--min_flex_num', type=int, default=0, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=0, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--model_name', type=str, default='resnet56') # wide_resnet101_2
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--learnable_step', type=bool, default=True) # False: FjORD / HeteroFL / DepthFL
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=False)

parser.add_argument('--dataset', type=str, default='cifar10') # stl10, cifar10, svhn
parser.add_argument('--method', type=str, default='DD') # DD, W, WD / fjord, depthfl

parser.add_argument('--name', type=str, default='[cifar10][NeFLADD2][R56]') # L-A: bad character
   
args = parser.parse_args()
args.device = 'cuda:' + args.device_id

dataset_train, dataset_test = getDataset(args)

if args.noniid:
    dict_users = cifar_noniid(args, dataset_train)
else:
    dict_users = cifar_iid(dataset_train, args.num_users, args.rs)
# img_size = dataset_train[0][0].shape

def main():

    local_models = []

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
    
    local_models.append(net_temp1)
    local_models.append(net_temp2)
    args.num_models = 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = './output/nefl/'+ timestamp + str(args.name) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        run = wandb.init(dir=filename, project='GeFL-0721', name= str(args.name)+ str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)
    # logger = get_logger(logpath=os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))
    
    loss_train = []
    lr = args.lr
    # mlist = [_ for _ in range(args.num_models)]
    '''
    Clients train main networks with local dataset
    Clients share common feature extractor and FedAvg
    LOOP:
        1. Clients create intermittent features of local dataset
        2. Clients train generator by local feature dataset and share it - FedAvg
        3. Clients train main networks with local dataset and features sampled by generator
    '''
    # ws_glob = [[] for _ in range(args.num_models)]
    ws_glob = [net_temp1.state_dict(), net_temp2.state_dict()]
    w_comm = common_net.state_dict()
    '''
    Each client: get parameter from ws_glob and local update. Save updated parameter to ws_local
    FedAvg ws_local and save it ws_glob
    '''
    for iter in range(1,args.wu_epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        loss_locals = []
        # w_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            # model_idx = random.choice(mlist[max(0,dev_spec_idx-args.min_flex_num):min(len(args.ps),dev_spec_idx+1+args.max_flex_num)])
            model_idx = dev_spec_idx
            model = local_models[model_idx]
            model.load_state_dict(ws_glob[model_idx])
                        
            local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
            weight, loss = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=lr)

            ws_local[model_idx].append(copy.deepcopy(weight))
            loss_locals.append(loss)
        
        ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm)
        
        # common_keys = ['conv1.weight', 'bn1.weight', bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked']
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        loss_train.append(loss_avg)
        if iter % 10 == 0 or iter == args.wu_epochs:
            for i in range(args.num_models):
                model_e = local_models[i]
                model_e.load_state_dict(ws_glob[i])
                model_e.eval()
                acc_test, loss_test = test_img(model_e, dataset_test, args)
                print("Testing accuracy " + str(i) + ": {:.2f}".format(acc_test))
                if args.wandb:
                    wandb.log({
                        "Communication round": iter,
                        "Local model " + str(i) + " test accuracy": acc_test
                    })
                    
    common_net.load_state_dict(w_comm)

#########
    for iter in range(1,args.wu_epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        loss_locals = []
        # w_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            # model_idx = random.choice(mlist[max(0,dev_spec_idx-args.min_flex_num):min(len(args.ps),dev_spec_idx+1+args.max_flex_num)])
            model_idx = dev_spec_idx
            model = local_models[model_idx]
            model.load_state_dict(ws_glob[model_idx])

            local_gen = LocalUpdate_Gen(args, common_net, dataset=dataset_train, idxs=dict_users[idx])
            local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
            weight, loss = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=lr)

            ws_local[model_idx].append(copy.deepcopy(weight))
            loss_locals.append(loss)
        
        ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm)
        
        # common_keys = ['conv1.weight', 'bn1.weight', bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked']
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        loss_train.append(loss_avg)
        if iter % 10 == 0 or iter == args.wu_epochs:
            for i in range(args.num_models):
                model_e = local_models[i]
                model_e.load_state_dict(ws_glob[i])
                model_e.eval()
                acc_test, loss_test = test_img(model_e, dataset_test, args)
                print("Testing accuracy " + str(i) + ": {:.2f}".format(acc_test))
                if args.wandb:
                    wandb.log({
                        "Communication round": iter,
                        "Local model " + str(i) + " test accuracy": acc_test
                    })
#########

    filename = '/home/hong/GeFL/output/nefl/'+ timestamp + str(args.name) + str(args.rs) + '/models'
    if not os.path.exists(filename):
        os.makedirs(filename)
    for ind in range(ti):
        p = args.ps[ind]
        model_e = copy.deepcopy(local_models[ind])       
        f = extract_submodel_weight_from_globalM(net = copy.deepcopy(net_glob), BN_layer=BN_layers, Step_layer=Steps, p=p, model_i=ind)
        torch.save(f, os.path.join(filename, 'model' + str(ind) + '.pt'))

    # testing
    net_glob.eval()

    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
  
    if args.wandb:
        run.finish()


if __name__ == "__main__":
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs) # if use multi-GPU
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(args.rs)
        random.seed(args.rs)
        main()
        args.rs = args.rs+1