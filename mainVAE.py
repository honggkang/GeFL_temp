'''
nn.embedding for one_hot (label)
'''
from torchvision.utils import save_image

import argparse
import os
import wandb
from datetime import datetime
import numpy as np
import random
import torch
import copy

from utils.localUpdate import *
from utils.average import *
from utils.getData import *
from utils.getModels import *

from mlp_generators.VAE import *
from utils.util import test_img, get_logger
# from models import *
# from utils.NeFedAvg import NeFedAvg
# from AutoAugment.autoaugment import ImageNetPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=10)
parser.add_argument('--partial_data', type=float, default=0.1)

parser.add_argument('--noniid', action='store_true') # default: false
parser.add_argument('--dir_param', type=float, default=0.3)

parser.add_argument('--frac', type=float, default=1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")

parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=10)

parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--device_id', type=str, default='1')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=False)

parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, emnist
parser.add_argument('--name', type=str, default='under_dev') # L-A: bad character

parser.add_argument('--wu_epochs', type=int, default=5) # warm-up epochs for main networks
parser.add_argument('--gen_wu_epochs', type=int, default=10) # warm-up epochs for generator

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--local_ep_gen', type=int, default=1) # local epochs for training main nets by generated samples
parser.add_argument('--gen_local_ep', type=int, default=10) # local epochs for training generator
parser.add_argument('--aid_by_gen', type=bool, default=True)

parser.add_argument('--sample_test', type=int, default=5) # local epochs for training generator
parser.add_argument('--lr', type=float, default=1e-1)

args = parser.parse_args()
args.device = 'cuda:' + args.device_id

dataset_train, dataset_test = getDataset(args)
latent_size = 20
args.feature_size = 14 # 14

if args.noniid:
    dict_users = noniid_dir(args, args.dir_param, dataset_train)
else:
    dict_users = cifar_iid(dataset_train, int(1/args.partial_data*args.num_users), args.rs)
# img_size = dataset_train[0][0].shape

if not args.aid_by_gen:
    args.gen_wu_epochs = 0
    args.local_ep_gen = 0
    args.gen_local_ep = 0

def main():

    local_models, common_net = getModel(args)
    ws_glob = []
    for _ in range(args.num_models):
        ws_glob.append(local_models[_].state_dict())

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = './output/gefl/'+ timestamp + str(args.name) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        run = wandb.init(dir=filename, project='GeFL-0906', name= str(args.name)+ str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)
    # logger = get_logger(logpath=os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))
    
    loss_train = []
    lr = args.lr

    w_comm = common_net.state_dict()

    for iter in range(1, args.wu_epochs+1):
        '''
        Warming up for main networks
        '''
        ws_local = [[] for _ in range(args.num_models)]
        loss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            # model_idx = random.choice(mlist[max(0,dev_spec_idx-args.min_flex_num):min(len(args.ps),dev_spec_idx+1+args.max_flex_num)])
            model_idx = dev_spec_idx
            model = local_models[model_idx]
            model.load_state_dict(ws_glob[model_idx])
                        
            local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
            weight, loss, _ = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=lr)

            ws_local[model_idx].append(copy.deepcopy(weight))
            loss_locals.append(loss)
        
        ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm)
        
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Avg loss {:.3f}'.format(iter, loss_avg))
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

    common_net.load_state_dict(w_comm)    
    gen_glob = CVAE(args.feature_size*args.feature_size, latent_size, args.num_classes).to(args.device)
    gen_w_glob = gen_glob.state_dict()

    for iter in range(1, args.gen_wu_epochs+1):
        '''
        Warming up for generative model
        '''
        gen_w_local = []
        loss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        gen_glob.load_state_dict(gen_w_glob)
        for idx in idxs_users:
                        
            local = LocalUpdate_VAE(args, common_net, dataset=dataset_train, idxs=dict_users[idx])
            gen_weight, loss = local.train(net=copy.deepcopy(gen_glob))

            gen_w_local.append(copy.deepcopy(gen_weight))
            loss_locals.append(loss)
        
        gen_w_glob = FedAvg(gen_w_local)
        
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    '''
    - 1. Main networks (including a common feature extrator (FE)) are first trained by warming up stages
    - 2. Generator is trained to generate feature (outpu by a common FE) by warming up stages
    
    Clients train main networks with local dataset
    Clients share common feature extractor and a server FedAvg feature extrator
    LOOP:
        === Training Generator ===
        1. Clients create intermittent features of local dataset
        2. Clients train generator by local feature dataset and share it - FedAvg

        === Training Main Net ===
        3. Clients train main networks with local dataset and features sampled by generator
    '''
    '''
    Each client: get parameter from ws_glob and local update. Save updated parameter to ws_local
    FedAvg ws_local and save it ws_glob
    '''
    best_perf = [0 for _ in range(args.num_models)]

    for iter in range(1,args.epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        gen_w_local = []
        loss_locals = []
        gen_loss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        gen_glob.load_state_dict(gen_w_glob)
        
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            # model_idx = random.choice(mlist[max(0,dev_spec_idx-args.min_flex_num):min(len(args.ps),dev_spec_idx+1+args.max_flex_num)])
            model_idx = dev_spec_idx
            model = local_models[model_idx]
            model.load_state_dict(ws_glob[model_idx])
            local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
            weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=lr)
            ws_local[model_idx].append(weight)
            loss_locals.append(loss)
            gen_loss_locals.append(gen_loss)

            local_gen = LocalUpdate_VAE(args, common_net, dataset=dataset_train, idxs=dict_users[idx])
            gen_weight, loss = local_gen.train(net=copy.deepcopy(gen_glob))
            gen_w_local.append(gen_weight)
        
        ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        gen_w_glob = FedAvg(gen_w_local)
                
        loss_avg = sum(loss_locals) / len(loss_locals)
        try:
            gen_loss_avg = sum(gen_loss_locals) / len(gen_loss_locals)
        except:
            gen_loss_avg = -1
        print('Round {:3d}, Average loss {:.3f}, Average loss by Gen {:.3f}'.format(iter, loss_avg, gen_loss_avg))

        loss_train.append(loss_avg)
        if iter % args.sample_test == 0 or iter == args.epochs:
            acc_test_tot = []

            for i in range(args.num_models):
                model_e = local_models[i]
                model_e.load_state_dict(ws_glob[i])
                model_e.eval()
                acc_test, loss_test = test_img(model_e, dataset_test, args)
                if acc_test > best_perf[i]:
                    best_perf[i] = float(acc_test)
                
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
                                    
    print(best_perf, 'AVG'+str(args.rs), sum(best_perf)/len(best_perf))
    sample_num = 10
    c = torch.eye(sample_num, 10).to(args.device)
    # one_c = one_hot(c, args.num_classes).to(args.device)
    sample = torch.randn(sample_num, latent_size).to(args.device)
    sample = gen_glob.decode(sample, c)
    # sample.shape = [10, 256]
    save_image(sample.view(sample_num, 1, args.feature_size, args.feature_size), 
                'imgFedCVAE/' + 'sample_' + '.png', nrow=10)

    if args.wandb:
        run.finish()
    return sum(best_perf)/len(best_perf)


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