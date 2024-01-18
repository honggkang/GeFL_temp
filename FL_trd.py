'''
GeFL trade-off experiment
As feature extractor size gets large, what happens?
- privacy
- robustness to scalability
- performance (e.g., accuracy)
'''

import argparse, wandb
from datetime import datetime
import numpy as np
import os, random, copy

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.localUpdate import *
from utils.average import *
from utils.getData import *
from utils.getModels import *
from utils.util import test_img

from generators.CCVAE import *

parser = argparse.ArgumentParser()
### clients
parser.add_argument('--num_users', type=int, default=50)
parser.add_argument('--frac', type=float, default=1)
parser.add_argument('--partial_data', type=float, default=0.1)
### model & feature size
parser.add_argument('--models', type=str, default='cnn') # cnn, mlp / cnn3
parser.add_argument('--output_channel', type=int, default=3) # local epochs for training generator
parser.add_argument('--img_size', type=int, default=16) # local epochs for training generator
parser.add_argument('--orig_img_size', type=int, default=32) # local epochs for training generator
### dataset
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, emnist
parser.add_argument('--noniid', action='store_true') # default: false
parser.add_argument('--dir_param', type=float, default=0.3)
parser.add_argument('--num_classes', type=int, default=10)
### optimizer
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--local_bs', type=int, default=64)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
### reproducibility
parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--device_id', type=str, default='0')
### warming-up
parser.add_argument('--wu_epochs', type=int, default=70) # warm-up epochs for main networks
parser.add_argument('--gen_wu_epochs', type=int, default=100) # warm-up epochs for generator

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--local_ep_gen', type=int, default=1) # local epochs for training target nets by generated samples
parser.add_argument('--gen_local_ep', type=int, default=5) # local epochs for training generator

parser.add_argument('--aid_by_gen', type=bool, default=False)
parser.add_argument('--freeze_FE', type=bool, default=True)
parser.add_argument('--freeze_gen', type=bool, default=True)
parser.add_argument('--only_gen', type=bool, default=False)
parser.add_argument('--load_trained_FE', type=bool, default=False)
parser.add_argument('--avg_FE', type=bool, default=True)
### logging
parser.add_argument('--sample_test', type=int, default=10)
parser.add_argument('--save_imgs', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--name', type=str, default='HomoFL') # L-A: bad character
### VAE parameters
parser.add_argument('--latent_size', type=int, default=16) # local epochs for training generator
### target nets
parser.add_argument('--lr', type=float, default=1e-1)
#### Feature size varying
parser.add_argument('--feature_size', type=int, default=5) # N/A warm-up epochs for main networks

args = parser.parse_args()
args.img_shape = (args.output_channel, args.img_size, args.img_size)
args.device = 'cuda:' + args.device_id
# cuda = True if torch.cuda.is_available() else False
kwargs = {'num_workers': 4, 'pin_memory': True}

dataset_train, dataset_test = getDataset(args)

print(args)
def main():

    dict_users = cifar_iid(dataset_train, int(1/args.partial_data*args.num_users), args.rs)

    if not args.aid_by_gen:
        args.gen_wu_epochs = 0
        args.local_ep_gen = 0
        args.gen_local_ep = 0
 
    local_models = []
    from mainNetModels.cnn_feat_exp import CNN5_5,FE_CNN_5

    temp_model = CNN5_5().to(args.device)
    for _ in range(10):
        local_models.append(copy.deepcopy(temp_model))
    common_net = FE_CNN_5().to(args.device)
    args.aid_by_gen = False

    w_comm = common_net.state_dict()
    ws_glob = []
    args.num_models = 10
    for _ in range(args.num_models):
        ws_glob.append(local_models[_].state_dict())

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = './output/gefl/'+ timestamp + str(args.name) + str(args.feature_size) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)
    if args.wandb:
        run = wandb.init(dir=filename, project='GeFL-VFeat-CVAEF-0115', name= str(args.num_users)+str(args.name)+ str(args.feature_size)+  str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)

    loss_train = []
    lr = 1e-1

    w_comm = common_net.state_dict()

    for iter in range(1, args.wu_epochs+1):
        ''' ------------------------
        Warming up for main networks
        ------------------------ '''
        ws_local = [[] for _ in range(args.num_models)]
        loss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            model_idx = dev_spec_idx
            model = local_models[model_idx]
            model.load_state_dict(ws_glob[model_idx])
                        
            local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
            weight, loss, _ = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=lr)

            ws_local[model_idx].append(copy.deepcopy(weight))
            loss_locals.append(loss)
        
        if args.avg_FE: # LG-FedAvg
            ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        else: # FedAvG
            ws_glob = FedAvg_FE_raw(args, ws_glob, ws_local)
        
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Warm-up TargetNet Round {:3d}, Avg loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        if iter % 5 == 0 or iter == args.wu_epochs:
            acc_test_tot = []
            
            for i in range(args.num_models):
                model_e = local_models[i]
                model_e.load_state_dict(ws_glob[i])
                model_e.eval()
                acc_test, loss_test = test_img(model_e, dataset_test, args)
                acc_test_tot.append(acc_test)                
                print("Testing accuracy " + str(i) + ": {:.2f}".format(acc_test))
                if args.wandb:
                    wandb.log({
                        "Communication round": iter,
                        "Local model " + str(i) + " test accuracy": acc_test
                    })
            if args.wandb:
                wandb.log({
                    "Communication round": iter,
                    "Mean test accuracy": sum(acc_test_tot) / len(acc_test_tot)
                })

    # if args.aid_by_gen:
    #     sample_num = 50
    #     samples = gen_glob.sample_image_4visualization(sample_num)
    #     save_image(samples.view(sample_num, args.output_channel, args.img_size, args.img_size),
    #                 'imgFedCVAE/' + 'sample_' + str(args.dataset) + '.png', nrow=10)
    # torch.save(gen_w_glob, 'checkpoint/FedCVAEF' + str(args.rs) + '.pt')

    if args.wandb:
        run.finish()



if __name__ == "__main__":
    
    results = []
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs) # if use multi-GPU
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(args.rs)
        random.seed(args.rs)
        results.append(main())
        args.rs = args.rs+1
        print(results)