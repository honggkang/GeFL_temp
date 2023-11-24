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

from torchvision.utils import save_image

import argparse, os
import wandb
from datetime import datetime
import numpy as np
import random
import torch, copy

from utils.localUpdate import *
from utils.average import *
from utils.getData import *
from utils.getModels import *

from DDPM.ddpm16_2 import *
'''
DDPM.ddpm14 features  / DDPM.ddpm16 features
DDPM.ddpm28 orig/feat / DDPM.ddpm32 orig
'''
from utils.util import test_img, get_logger
# from models import *
# from utils.NeFedAvg import NeFedAvg
# from AutoAugment.autoaugment import ImageNetPolicy

parser = argparse.ArgumentParser()
### clients
parser.add_argument('--num_users', type=int, default=10)
parser.add_argument('--frac', type=float, default=1)
parser.add_argument('--partial_data', type=float, default=0.1)
### model & feature size
parser.add_argument('--models', type=str, default='cnn') # cnn, mlp 
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
# parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
### reproducibility
parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--device_id', type=str, default='0')
### warming-up
parser.add_argument('--wu_epochs', type=int, default=20) # warm-up epochs
parser.add_argument('--gen_wu_epochs', type=int, default=100) # warm-up epochs

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--local_ep_gen', type=int, default=1)
parser.add_argument('--gen_local_ep', type=int, default=5)

parser.add_argument('--aid_by_gen', type=bool, default=True)
parser.add_argument('--freeze_FE', type=bool, default=True)
parser.add_argument('--freeze_gen', type=bool, default=True)
parser.add_argument('--only_gen', type=bool, default=False)
parser.add_argument('--load_trained_FE', type=bool, default=False)
parser.add_argument('--avg_FE', type=bool, default=True)
### logging
parser.add_argument('--sample_test', type=int, default=10) # local epochs for training generator
parser.add_argument('--save_imgs', type=bool, default=True) # local epochs for training generator
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--name', type=str, default='under_dev') # L-A: bad character
### DDPM parameters
parser.add_argument('--n_feat', type=int, default=128) # 128 ok, 256 better (but slower)
parser.add_argument('--n_T', type=int, default=400) # 400, 500
parser.add_argument('--guide_w', type=float, default=0.0) # 0, 0.5, 2

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.img_shape = (args.output_channel, args.img_size, args.img_size)

def main():
    dataset_train, dataset_test = getDataset(args)

    if args.noniid:
        dict_users = noniid_dir(args, args.dir_param, dataset_train)
    else:
        dict_users = cifar_iid(dataset_train, int(1/args.partial_data*args.num_users), args.rs)
    # img_size = dataset_train[0][0].shape

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
        run = wandb.init(dir=filename, project='GeFL-DDPMF16-1122', name= str(args.name)+ str(args.rs) +'w'+str(args.guide_w), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)
    # logger = get_logger(logpath=os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))
    
    loss_train = []
    lr = 1e-1 # MLP

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
            
            if args.avg_FE:
                ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm)
            else:
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
        torch.save(w_comm, 'models/save/FedDDPM' + str(args.guide_w) + '_'
                + str(args.models) + '16_common_net' + str(args.rs) + '.pt')
    else:
        w_comm = torch.load('models/save/Fed_cnn_common_net.pt') # common_net = FE_MLP.to(args.device)

    common_net.load_state_dict(w_comm)

    if args.freeze_FE:
        common_net.eval()

    gen_glob = DDPM(args, nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
                    betas=(1e-4, 0.02), drop_prob=0.1).to(args.device)
    opt = torch.optim.Adam(gen_glob.parameters(), lr=1e-4).state_dict()
    opts = [copy.deepcopy(opt) for _ in range(args.num_users)]
    
    gen_w_glob = gen_glob.state_dict()
    
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
                        
            local = LocalUpdate_DDPM(args, common_net, dataset=dataset_train, idxs=dict_users[idx])
            if args.freeze_gen:
                lr_rate = 1-iter/args.gen_wu_epochs
            else:
                lr_rate = 1-iter/(args.gen_wu_epochs+args.epochs)
            g_weight, gloss, opts[idx] = local.train(net=copy.deepcopy(gen_glob), lr_decay_rate=lr_rate, opt=opts[idx])

            gen_w_local.append(copy.deepcopy(g_weight))
            gloss_locals.append(gloss)
        
        gen_w_glob = FedAvg(gen_w_local)
        gloss_avg = sum(gloss_locals) / len(gloss_locals)
        if args.save_imgs and (iter % args.sample_test == 0 or iter == args.gen_wu_epochs):
            gen_glob.eval()
            sample_num = 40
            samples = gen_glob.sample_image_4visualization(sample_num, guide_w=args.guide_w)
            save_image(samples.view(sample_num, args.output_channel, args.img_size, args.img_size),
                        'imgs/imgFedDDPMF/' + str(args.name)+ str(args.rs) + 'w_' + str(args.guide_w) + str(iter) + '.png', nrow=10, normalize=True)
            gen_glob.train()
        print('Warm-up Gen Round {:3d}, Average loss {:.3f}'.format(iter, gloss_avg))

    best_perf = [0 for _ in range(args.num_models)]

    for iter in range(1,args.epochs+1):
        ''' ----------------------------------------
        Train main networks by local sample
        and generated samples, then update generator
        ---------------------------------------- '''        
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
                    # update header only by synthetic data
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

            if args.aid_by_gen and not args.freeze_gen:
                local_gen = LocalUpdate_DDPM(args, common_net, dataset=dataset_train, idxs=dict_users[idx])
                lr_rate = 1-(iter+args.gen_wu_epochs)/(args.gen_wu_epochs+args.epochs)
                g_weight, gloss, opts[idx] = local_gen.train(net=copy.deepcopy(gen_glob), lr_decay_rate=lr_rate, opt=opts[idx])

                gen_w_local.append(copy.deepcopy(g_weight))
                gloss_locals.append(gloss)
           
                if args.save_imgs and (iter % args.sample_test == 0 or iter == args.epochs):
                    gen_glob.eval()
                    sample_num = 40
                    samples = gen_glob.sample_image_4visualization(sample_num, guide_w=args.guide_w)
                    save_image(samples.view(sample_num, args.output_channel, args.img_size, args.img_size),
                        'imgs/imgFedDDPMF/' + str(args.name)+ str(args.rs) + 'w_' + str(args.guide_w) + str(iter) + '.png', nrow=10, normalize=True)
                    gen_glob.train()
        if args.aid_by_gen and not args.freeze_gen:
            gloss_avg = sum(gloss_locals) / len(gloss_locals)

            gen_w_glob = FedAvg(gen_w_local)
        else:
            gloss_avg = -1

        if args.freeze_FE:
            ws_glob, w_comm = FedAvg_frozen_FE(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        else:
            if args.avg_FE:
                ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update            
            else:
                ws_glob = FedAvg_FE_raw(args, ws_glob, ws_local) # main net, feature extractor weight update            
    
        loss_avg = sum(loss_locals) / len(loss_locals)
        try:
            gen_loss_avg = sum(gen_loss_locals) / len(gen_loss_locals)
        except:
            gen_loss_avg = -1
        print('Round {:3d}, Average loss {:.3f}, Average loss by Gen {:.3f}, DDPM Average loss {:.3f}'.format(iter, loss_avg, gen_loss_avg, gloss_avg))

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

    print(best_perf, 'AVG'+str(args.rs), sum(best_perf)/len(best_perf))
    torch.save(gen_w_glob, 'checkpoint/FedDDPMF' + str(args.rs) + '.pt')

    # sample_num = 10
    # gen_glob.eval()
    # # ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    # ws_test = [2.0] # strength of generative guidance
    # save_dir = './imgFedCDDPM/'

    # with torch.no_grad():
    #     n_sample = args.num_classes
    #     for w_i, w in enumerate(ws_test):
    #         x_gen, x_gen_store = gen_glob.sample(n_sample, (1, 14, 14), args.device, guide_w=w)

    #         grid = make_grid(x_gen, nrow=10)
    #         save_image(grid, save_dir + f"sample_.png")

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