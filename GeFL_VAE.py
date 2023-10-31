import argparse
import os
import wandb
from datetime import datetime
import numpy as np
import random
import torch
import copy

import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.localUpdateRaw import *
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
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
### dataset
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--num_classes', type=int, default=10)
### optimizer
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--local_bs', type=int, default=64) # 128
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
### reproducibility
parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--device_id', type=str, default='0')
### warming-up
parser.add_argument('--gen_wu_epochs', type=int, default=100) # warm-up epochs for generator

parser.add_argument('--epochs', type=int, default=50) # total communication round (train main nets by (local samples and gen) + train gen)
parser.add_argument('--local_ep', type=int, default=5) # local epochs for training main nets by local samples
parser.add_argument('--local_ep_gen', type=int, default=1) # local epochs for training main nets by generated samples
parser.add_argument('--gen_local_ep', type=int, default=5) # local epochs for training generator

parser.add_argument('--aid_by_gen', type=bool, default=True)
parser.add_argument('--freeze_gen', type=bool, default=False)
parser.add_argument('--only_gen', type=bool, default=False)
parser.add_argument('--avg_FE', type=bool, default=True)
### logging
parser.add_argument("--sample_test", type=int, default=10, help="interval between image sampling")
parser.add_argument('--save_imgs', type=bool, default=True) # local epochs for training generator
parser.add_argument('--wandb', type=bool, default=False)
parser.add_argument('--name', type=str, default='dev') # L-A: bad character
### VAE parameters
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
### N/A
parser.add_argument('--wu_epochs', type=int, default=0) # N/A warm-up epochs for main networks
parser.add_argument('--freeze_FE', type=bool, default=False)

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
        
    tf = transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize([0.5], [0.5])
                            ]) # mnist is already normalised 0 to 1
    train_data = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=True, transform=tf, download=True) # VAE training data
    dict_users = cifar_iid(train_data, int(1/args.partial_data*args.num_users), args.rs)
    
    if not args.aid_by_gen:
        args.gen_wu_epochs = 0
        args.local_ep_gen = 0
        args.gen_local_ep = 0

    local_models, common_net = getModel(args)
    w_comm = common_net.state_dict()
    ws_glob = []
    for _ in range(args.num_models):
        ws_glob.append(local_models[_].state_dict())

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = './output/gefl/'+ timestamp + str(args.name) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)
    if args.wandb:
        run = wandb.init(dir=filename, project='GeFL-VAE-onlySyn-1024', name= str(args.name)+ str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)

    loss_train = []
    lr = 1e-1
    
    gen_glob = CVAE(args).to(args.device)
    gen_w_glob = gen_glob.state_dict()
    
    opt = torch.optim.Adam(gen_glob.parameters(), lr=1e-3).state_dict()
    opts = [copy.deepcopy(opt) for _ in range(args.num_users)]

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

            local = LocalUpdate_VAE_raw(args, dataset=train_data, idxs=dict_users[idx])
            g_weight, gloss, opts[idx] = local.train(net=copy.deepcopy(gen_glob), opt=opts[idx])

            gen_w_local.append(copy.deepcopy(g_weight))
            gloss_locals.append(gloss)
        
        gen_w_glob = FedAvg(gen_w_local)
        gloss_avg = sum(gloss_locals) / len(gloss_locals)

        if args.save_imgs and (iter % args.sample_test == 0 or iter == args.gen_wu_epochs):
            sample_num = 40
            samples = gen_glob.sample_image_4visualization(sample_num)
            save_image(samples.view(sample_num, args.output_channel, args.img_size, args.img_size),
                        'imgFedVAE/' + str(args.name)+ str(args.rs) +'SynOrig_' + str(iter) + '.png', nrow=10, normalize=True)
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
    
            if args.only_gen:
                local = LocalUpdate_onlyGen(args, dataset=dataset_train, idxs=dict_users[idx])
                weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=lr)
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
                local_gen = LocalUpdate_VAE_raw(args, dataset=train_data, idxs=dict_users[idx])
                # g_weight, gloss = local_gen.train(net=copy.deepcopy(gen_glob))
                g_weight, gloss, opts[idx] = local_gen.train(net=copy.deepcopy(gen_glob), opt=opts[idx])

                gen_w_local.append(copy.deepcopy(g_weight))
                gloss_locals.append(gloss)
                
        if args.aid_by_gen and not args.freeze_gen:
            gloss_avg = sum(gloss_locals) / len(gloss_locals)
            gen_w_glob = FedAvg(gen_w_local)
            if args.save_imgs and (iter % args.sample_test == 0 or iter == args.epochs):
                sample_num = 40
                samples = gen_glob.sample_image_4visualization(sample_num)
                save_image(samples.view(sample_num, args.output_channel, args.img_size, args.img_size),
                            'imgFedVAE/' + str(args.name)+ str(args.rs) +'SynOrig_' + str(args.gen_wu_epochs +iter) + '.png', nrow=10, normalize=True)
            print('GEN Round {:3d}, G Avg loss {:.3f}'.format(args.gen_wu_epochs +iter, gloss_avg))
                    
        else:
            gloss_avg = -1
            dloss_avg = -1

        if args.avg_FE:
            ws_glob, w_comm = FedAvg_FE(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        else:
            ws_glob = FedAvg_FE_raw(args, ws_glob, ws_local) # main net, feature extractor weight update
    
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

    torch.save(gen_w_glob, 'checkpoint/FedVAE' + str(args.name) + str(args.rs) + '.pt')
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



# dataloaders = []
# for i in range(args.num_users):
#     dataloaders.append(torch.utils.data.DataLoader(
#         DatasetSplit(train_data,dict_users[i]), batch_size=args.batch_size, shuffle=True, **kwargs))
    
# # train_loader = torch.utils.data.DataLoader(
# #     datasets.MNIST('./data', train=True, download=True,
# #                    transform=transforms.ToTensor()),
# #     batch_size=batch_size, shuffle=True, **kwargs)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train=False,
#                             transform=tf),
#     batch_size=args.batch_size, shuffle=False, **kwargs)


# def one_hot(labels, class_size):
#     targets = torch.zeros(labels.size(0), class_size)
#     for i, label in enumerate(labels):
#         targets[i, label] = 1
#     return targets.to(args.device)


# # create a CVAE model
# gmodel = CVAE(args).to(args.device)
# x = torch.zeros(1,1,28,28, device='cuda') # 
# c = torch.cuda.LongTensor((1,))
# # summary(gmodel, x, c)

# models = []
# for i in range(args.num_users):
#     models.append(CVAE(args).to(args.device))


# optimizers = []
# for i in range(args.num_users):
#     optimizers.append(optim.Adam(models[i].parameters(), lr=1e-3))
# # optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # logvar: log(sigma^2)
#     return BCE + KLD


# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# def train(epoch):
#     for di in range(args.num_users):
#         models[di].train()
#         train_loss = 0
#         for batch_idx, (data, labels) in enumerate(dataloaders[di]):
#             data, labels = data.to(args.device), labels.to(args.device)
#             # labels = one_hot(labels, 10)
#             labels = Variable(labels.type(LongTensor))
#             recon_batch, mu, logvar = models[di](data, labels)
#             optimizers[di].zero_grad()
#             loss = loss_function(recon_batch, data, mu, logvar)
#             loss.backward()
#             train_loss += loss.detach().cpu().numpy()
#             optimizers[di].step()
#             # if batch_idx % 200 == 0:
#             #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#             #         epoch, batch_idx * len(data), len(dataloaders[di].dataset),
#             #         100. * batch_idx / len(dataloaders[di]),
#             #         loss.item() / len(data)))

#         print('====> Epoch: {} Average loss: {:.4f}'.format(
#             epoch, train_loss / len(dataloaders[di].dataset)))

#     # update global weights
#     ws = [models[i].state_dict() for i in range(args.num_users)]
#     wg = FedAvg(ws)
    
#     gmodel.load_state_dict(wg)
#     for i in range(args.num_users):
#         models[i].load_state_dict(wg)


# def test(epoch):
#     gmodel.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, labels) in enumerate(test_loader):
#             data, labels = data.to(args.device), labels.to(args.device)
#             # labels = one_hot(labels, 10)
#             labels = Variable(labels.type(LongTensor))
#             recon_batch, mu, logvar = gmodel(data, labels)
#             test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
#             if i == 0:
#                 n = min(data.size(0), 10)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(-1, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'imgFedCVAE/' + 'recon' + str(args.num_users) + '_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))


# for epoch in range(1, args.n_epochs + 1):
#         train(epoch)
#         if epoch % args.sample_interval == 0 or epoch == args.n_epochs:
#             test(epoch)
#             n_row = 10
#             with torch.no_grad():
#                 # c = torch.eye(n_row**2, 10).to(device)
#                 c = np.array([num for _ in range(n_row) for num in range(n_row)])
#                 c = Variable(LongTensor(c)).to(args.device)
#                 sample = torch.randn(n_row**2, args.latent_size).to(args.device)
#                 sample = gmodel.sample_decode(sample, c).cpu()
#                 save_image(sample.view(n_row**2, 1, 28, 28),
#                         'RAWimgFedCVAE/' + str(args.num_users) +'_' + str(epoch) + '.png', nrow=n_row, normalize=True)

# torch.save(gmodel.state_dict(), 'models/save/' + str(args.num_users)+ '_'+ str(args.n_epochs)+ 'cvae.pt')