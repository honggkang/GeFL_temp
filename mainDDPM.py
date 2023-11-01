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

from DDPM.ddpm14 import *
from utils.util import test_img, get_logger
# from models import *
# from utils.NeFedAvg import NeFedAvg
# from AutoAugment.autoaugment import ImageNetPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=10)
parser.add_argument('--partial_data', type=float, default=0.1)

parser.add_argument('--noniid', action='store_true') # default: false
parser.add_argument('--dir_param', type=float, default=0.3)

parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")

parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--local_ep', type=int, default=5)

parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal') # normal worst
parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=10)

parser.add_argument('--min_flex_num', type=int, default=0, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=0, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=False)

parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist
parser.add_argument('--name', type=str, default='under_dev') # L-A: bad character
parser.add_argument('--wu_epochs', type=int, default=2) # warm-up epochs

parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--lr', type=float, default=0.0002) # GAN lr

args = parser.parse_args()
args.device = 'cuda:' + args.device_id

dataset_train, dataset_test = getDataset(args)
args.latent_dim = 100
args.feature_size = 14 # 14

if args.noniid:
    dict_users = noniid_dir(args, args.dir_param, dataset_train)
else:
    dict_users = cifar_iid(dataset_train, args.num_users, args.rs)
# img_size = dataset_train[0][0].shape

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
        run = wandb.init(dir=filename, project='GeFL-0725', name= str(args.name)+ str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)
    # logger = get_logger(logpath=os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))
    
    loss_train = []
    lr = 1e-1 # MLP

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
        print('Round {:3d}, Avg loss {:.3f}, Model 1 {:2d}, Model 2 {:2d}, Model 3 {:2d}'.format
              (iter, loss_avg, len(ws_local[0]), len(ws_local[1]), len(ws_local[2])))
        loss_train.append(loss_avg)
        if iter % 5 == 0 or iter == args.wu_epochs:
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

    n_feat = 128 # 128 ok, 256 better (but slower)
    n_T = 200 # 400, 500
    gen_glob = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=args.num_classes),
                    betas=(1e-4, 0.02), n_T=n_T, device=args.device, drop_prob=0.1)

    gen_glob.to(args.device)
    gen_w_glob = gen_glob.state_dict()
    
    for iter in range(1, args.wu_epochs+1):
        ''' ---------------------------
        Warming up for generative model
        --------------------------- '''
        gen_w_local = []
        gloss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        gen_glob.load_state_dict(gen_w_glob)
        
        for idx in idxs_users:
                        
            local = LocalUpdate_DDPM(args, common_net, dataset=dataset_train, idxs=dict_users[idx])
            g_weight, gloss = local.train(net=copy.deepcopy(gen_glob), round=iter)

            gen_w_local.append(copy.deepcopy(g_weight))
            gloss_locals.append(gloss)
        
        gen_w_glob = FedAvg(gen_w_local)
        gloss_avg = sum(gloss_locals) / len(gloss_locals)

        print('Round {:3d}, G Average loss {:.3f}'.format(iter, gloss_avg))

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
        
        gloss_locals = []
        
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

            local_gen = LocalUpdate_DDPM(args, common_net, dataset=dataset_train, idxs=dict_users[idx])
            g_weight, gloss = local_gen.train(net=copy.deepcopy(gen_glob), round=iter+args.wu_epochs)

            gen_w_local.append(copy.deepcopy(g_weight))
            gloss_locals.append(gloss)

        gloss_avg = sum(gloss_locals) / len(gloss_locals)
        
        ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        gen_w_glob = FedAvg(gen_w_local)

        loss_avg = sum(loss_locals) / len(loss_locals)
        gen_loss_avg = sum(gen_loss_locals) / len(gen_loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Average loss by Gen samples {:.3f}, G Average loss {:.3f}'.format(iter, loss_avg, gen_loss_avg, gloss_avg))

        loss_train.append(loss_avg)
        if iter % 10 == 0 or iter == args.epochs:
            for i in range(args.num_models):
                model_e = local_models[i]
                model_e.load_state_dict(ws_glob[i])
                model_e.eval()
                acc_test, loss_test = test_img(model_e, dataset_test, args)
                if acc_test > best_perf[i]:
                    best_perf[i] = float(acc_test)
                
                print("Testing accuracy " + str(i) + ": {:.2f}".format(acc_test))
                if args.wandb:
                    wandb.log({
                        "Communication round": iter,
                        "Local model " + str(i) + " test accuracy": acc_test
                    })
    print(best_perf, sum(best_perf)/len(best_perf))
    sample_num = 10
    gen_glob.eval()
    # ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    ws_test = [2.0] # strength of generative guidance
    save_dir = './imgFedCDDPM/'

    with torch.no_grad():
        n_sample = args.num_classes
        for w_i, w in enumerate(ws_test):
            x_gen, x_gen_store = gen_glob.sample(n_sample, (1, 14, 14), args.device, guide_w=w)

            grid = make_grid(x_gen, nrow=10)
            save_image(grid, save_dir + f"sample_.png")
                    

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