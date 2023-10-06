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

from FeatureExtractor.mlp import *
from utils.util import test_img, get_logger
# from models import *
# from utils.NeFedAvg import NeFedAvg
# from AutoAugment.autoaugment import ImageNetPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=100)
parser.add_argument('--noniid', action='store_true') # default: false
parser.add_argument('--dir_param', type=float, default=0.3)

parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-1)

parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal') # normal worst
parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=10)

parser.add_argument('--min_flex_num', type=int, default=0, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=0, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--device_id', type=str, default='1')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=False)

parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist
parser.add_argument('--name', type=str, default='under_dev') # L-A: bad character
parser.add_argument('--wu_epochs', type=int, default=5) # warm-up epochs

args = parser.parse_args()
args.device = 'cuda:' + args.device_id

dataset_train, dataset_test = getDataset(args)
latent_size = 20
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
    lr = args.lr

    w_comm = common_net.state_dict()
    best_perf = [0 for _ in range(args.num_models)]

    for iter in range(1, args.wu_epochs+args.epochs+1):
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
        
        Nm = []
        for i in range(args.num_models):
            Nm.append(len(ws_local[i]))
            
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Avg loss {:.3f}, Model # {}'.format
              (iter-args.wu_epochs, loss_avg, Nm))

        loss_train.append(loss_avg)
        if (iter+args.wu_epochs) % 10 == 0 or iter == args.wu_epochs or iter == args.epochs:
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
    print(best_perf, 'AVG'+str(args.rs), sum(best_perf)/len(best_perf))
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