import argparse
import os
import wandb
from datetime import datetime
import numpy as np
import random
import torch
import copy

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.localUpdate import *
from utils.average import *
from utils.getData import *
from utils.getModels import *

from mlp_generators.VAE import *
from utils.util import test_img

parser = argparse.ArgumentParser()
### clients
parser.add_argument("--num_users", type=int, default=10, help="interval between image sampling")
parser.add_argument('--frac', type=float, default=1)
parser.add_argument('--partial_data', type=float, default=0.1)
### model & feature size
parser.add_argument('--models', type=str, default='mlp') # cnn, mlp
parser.add_argument("--output_channel", type=int, default=1, help="number of image channels")
parser.add_argument("--img_size", type=int, default=14, help="size of each image dimension")
### dataset
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--num_classes', type=int, default=10)
### optimizer
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--local_bs', type=int, default=128) # 128
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
### reproducibility
parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--device_id', type=str, default='0')
### warming-up
parser.add_argument('--wu_epochs', type=int, default=20) # warm-up epochs for main networks
parser.add_argument('--gen_wu_epochs', type=int, default=50) # warm-up epochs for generator

parser.add_argument('--epochs', type=int, default=50) # total communication round (train main nets by (local samples and gen) + train gen)
parser.add_argument('--local_ep', type=int, default=5) # local epochs for training main nets by local samples
parser.add_argument('--local_ep_gen', type=int, default=1) # local epochs for training main nets by generated samples
parser.add_argument('--gen_local_ep', type=int, default=5) # local epochs for training generator

parser.add_argument('--aid_by_gen', type=bool, default=False)
parser.add_argument('--freeze_FE', type=bool, default=False) # N/A
parser.add_argument('--freeze_gen', type=bool, default=False)
parser.add_argument('--only_gen', type=bool, default=False)
parser.add_argument('--load_trained_FE', type=bool, default=False)

### logging
parser.add_argument("--sample_test", type=int, default=5, help="interval between image sampling")
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--name', type=str, default='under_dev') # L-A: bad character
### VAE parameters
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")

args = parser.parse_args()
args.img_shape = (args.output_channel, args.img_size, args.img_size)
args.device = 'cuda:' + args.device_id
# cuda = True if torch.cuda.is_available() else False
kwargs = {'num_workers': 4, 'pin_memory': True} 
dataset_train, dataset_test = getDataset(args)

'''
no normalized dataset needed for VAE
'''
print(args)
def main():
    
    dict_users = cifar_iid(dataset_train, int(1/args.partial_data*args.num_users), args.rs)
    
    if not args.aid_by_gen:
        args.gen_wu_epochs = 0
        args.local_ep_gen = 0
        args.gen_local_ep = 0

    local_models, common_net = getModel(args)
    ws_glob = []
    for _ in range(args.num_models):
        ws_glob.append(local_models[_].state_dict())

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = './output/gefl/'+ timestamp + str(args.name) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)
    if args.wandb:
        run = wandb.init(dir=filename, project='GeFL-VAEF14-1016', name= str(args.name)+ str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)

    loss_train = []
    lr = 1e-1

    if not args.load_trained_FE:
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
            
            ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm)
            
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
        torch.save(w_comm, 'models/save/Fed' + '_'
                    + str(args.models) + '_common_net.pt') # _common_net_sameSize.pt
    else:
        w_comm = torch.load('models/save/Fed_mlp2_common_net_sameSize.pt') # Fed_mlp2_common_net_sameSize common_net = FE_MLP.to(args.device)
            
# .to(args.device)
    common_net.load_state_dict(w_comm)
    w_comm = common_net.state_dict()
    
    gen_glob = CVAE(args).to(args.device)
    gen_w_glob = gen_glob.state_dict()
    # optim = None
    for iter in range(1, args.gen_wu_epochs+1):
        ''' ---------------------------
        Warming up for generative model
        --------------------------- '''
        gen_w_local = []
        gloss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        gen_glob.load_state_dict(gen_w_glob)
        
        for idx in idxs_users:
                        
            local = LocalUpdate_VAE(args, net_com=common_net, dataset=dataset_train, idxs=dict_users[idx])
            g_weight, gloss = local.train(net=copy.deepcopy(gen_glob))

            gen_w_local.append(copy.deepcopy(g_weight))
            
            gloss_locals.append(gloss)
        
        gen_w_glob = FedAvg(gen_w_local)
        
        gloss_avg = sum(gloss_locals) / len(gloss_locals)

        if iter % args.sample_test == 0 or iter == args.gen_wu_epochs:
            sample_num = 50
            samples = gen_glob.sample_image_4visualization(sample_num)
            save_image(samples.view(sample_num, args.output_channel, args.img_size, args.img_size),
                        'imgFedVAEF/' + 'SynFeat28_' + str(iter) + '.png', nrow=10)
        print('Warm-up GEN Round {:3d}, G Avg loss {:.3f}'.format(iter, gloss_avg))

    best_perf = [0 for _ in range(args.num_models)]
    
    for iter in range(1, args.epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        gen_w_local = []
        
        loss_locals = []
        gen_loss_locals = []
        
        gloss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if args.aid_by_gen:
            gen_glob.load_state_dict(gen_w_glob)
        
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            model_idx = dev_spec_idx
            model = local_models[model_idx]
            model.load_state_dict(ws_glob[model_idx])    
            if args.freeze_FE:
                if args.only_gen:
                    # update (header; layers after FE) only by synthetic data
                    local = LocalUpdate_onlyGen(args, dataset=dataset_train, idxs=dict_users[idx])
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), feature_start=True, gennet=copy.deepcopy(gen_glob), learning_rate=lr)
                else:
                    # update header by syn+real data
                    local = LocalUpdate_header(args, dataset=dataset_train, idxs=dict_users[idx])
                    if args.aid_by_gen:
                        weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), feature_extractor=common_net, gennet=copy.deepcopy(gen_glob), learning_rate=lr)
                    else:
                        weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), feature_extractor=common_net, learning_rate=lr) # weights of models
            else:
                local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
                # synthetic data updates header & real data updates whole target network
                if args.aid_by_gen:
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=lr)
                else:
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=lr)                

            ws_local[model_idx].append(weight)
            loss_locals.append(loss)
            gen_loss_locals.append(gen_loss)
            
            if args.aid_by_gen and not args.freeze_gen: # update GEN
                local_gen = LocalUpdate_VAE(args, net_com=common_net, dataset=dataset_train, idxs=dict_users[idx])
                g_weight, gloss = local_gen.train(net=copy.deepcopy(gen_glob))

                gen_w_local.append(copy.deepcopy(g_weight))

                gloss_locals.append(gloss)
                
        if args.aid_by_gen and not args.freeze_gen:
            gloss_avg = sum(gloss_locals) / len(gloss_locals)
            
            gen_w_glob = FedAvg(gen_w_local)
        else:
            gloss_avg = -1
            dloss_avg = -1

        ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update

        loss_avg = sum(loss_locals) / len(loss_locals)
        try:
            gen_loss_avg = sum(gen_loss_locals) / len(gen_loss_locals)
        except:
            gen_loss_avg = -1
        print('Round {:3d}, Avg loss {:.3f}, Avg loss by Gen samples {:.3f}, G Avg loss {:.3f}'.format(iter, loss_avg, gen_loss_avg, gloss_avg))

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
                        "Communication round": args.wu_epochs+iter,
                        "Local model " + str(i) + " test accuracy": acc_test
                    })
            if args.wandb:
                wandb.log({
                    "Communication round": args.wu_epochs+iter,
                    "Mean test accuracy": sum(acc_test_tot) / len(acc_test_tot)
                })

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