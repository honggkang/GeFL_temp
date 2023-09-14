import argparse

import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
# from torchsummary import summary
from torchsummaryX import summary
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import copy
from modelsMNIST.DDPM import *

from utils.getData import *
from utils.average import *

os.makedirs("RAWimgFedCDDPM", exist_ok=True)

def train_mnist(args):
    save_dir = './RAWimgFedCDDPM/'
    # hardcoding these here
    n_epoch = args.n_epochs # 20
    batch_size = args.batch_size # 256
    n_T = args.n_T # 500

    n_classes = args.n_classes
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4

    save_model = True
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    # device = "cpu"
    device = args.device

    gddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    gddpm.to(device)

    ddpms = []
    for i in range(args.num_users):
        ddpms.append(DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).to(device))
    
    # x = torch.zeros(1,1,28,28) # , device='cuda'
    # c = torch.cuda.LongTensor((1,))
    # x = torch.zeros(1,1,14,14) # , device='cuda'
    # c = torch.LongTensor((1,))
    # summary(ddpm, x, c)
    # summary(ddpm, (1, 28,28), (1)) # [torch.Tensor((3, 64, 64)),torch.Tensor((1,))]
    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))


    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    train_data = MNIST("./data", train=True, download=True, transform=tf)
    dict_users = dict_iid(train_data, int(1/opt.partial_data*opt.num_users))

    dataloaders = []
    for i in range(args.num_users):
        dataloaders.append(torch.utils.data.DataLoader(
            DatasetSplit(train_data, dict_users[i]), batch_size=args.batch_size, shuffle=True))

    # Optimizers
    optimizers = []
    for i in range(args.num_users):
        optimizers.append(torch.optim.Adam(ddpms[i].parameters(), lr=1e-3))


    for ep in range(1, 1+n_epoch):
        print(f'epoch {ep}')
        for di in range(opt.num_users):
            ddpms[di].train()

            # linear lrate decay
            optimizers[di].param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

            pbar = tqdm(dataloaders[di])
            loss_ema = None
            for x, c in pbar:
                optimizers[di].zero_grad()
                x = x.to(device)
                c = c.to(device)
                loss = ddpms[di](x, c)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
                optimizers[di].step()

        # update global weights
        ws = [ddpms[i].state_dict() for i in range(args.num_users)]
        wg = FedAvg(ws)
        
        gddpm.load_state_dict(wg)
        for i in range(args.num_users):
            ddpms[i].load_state_dict(wg)
        
        if ep % args.sample_interval == 0:
            # for eval, save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)        
            gddpm.eval()
            with torch.no_grad():
                n_sample = 10*n_classes
                for w_i, w in enumerate(ws_test):
                    x_gen, x_gen_store = gddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                    # # append some real images at bottom, order by class also
                    # x_real = torch.Tensor(x_gen.shape).to(device)
                    # for k in range(n_classes):
                    #     for j in range(int(n_sample/n_classes)):
                    #         try: 
                    #             idx = torch.squeeze((c == k).nonzero())[j]
                    #         except:
                    #             idx = 0
                    #         x_real[k+(j*n_classes)] = x[idx]

                    # x_all = torch.cat([x_gen, x_real])
                    # grid = make_grid(x_all*-1 + 1, nrow=10)
                    grid = make_grid(x_gen, nrow=10)
                    save_image(grid, save_dir + f"{args.num_users}_nT{n_T}_ep{ep}_w{w}.png")
                    print('saved image at ' + save_dir + f"{args.num_users}_nT{n_T}_ep{ep}_w{w}.png")

                    # if ep%20==0 or ep == int(n_epoch-1):
                    #     # create gif of images evolving over time, based on x_gen_store
                    #     fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    #     def animate_diff(i, x_gen_store):
                    #         print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                    #         plots = []
                    #         for row in range(int(n_sample/n_classes)):
                    #             for col in range(n_classes):
                    #                 axs[row, col].clear()
                    #                 axs[row, col].set_xticks([])
                    #                 axs[row, col].set_yticks([])
                    #                 # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                    #                 plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                    #         return plots
                    #     ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])
                    #     ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    #     print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch):
            save_dir = './models/save/'
            torch.save(gddpm.state_dict(), save_dir + f"{args.num_users}_{ep}_nT{n_T}_ddpm.pt")
            print('saved model at ' + save_dir + f"{args.num_users}_{ep}_nT{n_T}_ddpm.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=40, help="interval between image sampling")
    parser.add_argument('--device_id', type=str, default='2')
    parser.add_argument("--n_T", type=int, default=60)

    parser.add_argument("--num_users", type=int, default=10, help="interval between image sampling")
    parser.add_argument('--partial_data', type=float, default=0.1)
    opt = parser.parse_args()
    opt.device = 'cuda:' + opt.device_id

    print(opt)    
    train_mnist(opt)