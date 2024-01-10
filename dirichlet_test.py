import argparse
import os
import wandb
from datetime import datetime
import numpy as np
import random
import torch
import copy

from utils.localUpdateRaw import *
from utils.average import *
from utils.getData import *
from utils.getModels import *

from generators32.DCGAN import *
from utils.util import test_img, get_logger
from torchsummary import summary

parser = argparse.ArgumentParser()
### clients
parser.add_argument('--num_users', type=int, default=10)
parser.add_argument('--frac', type=float, default=1)
parser.add_argument('--partial_data', type=float, default=0.1)
### model & feature size
parser.add_argument('--models', type=str, default='cnn') # cnn, mlp
parser.add_argument('--output_channel', type=int, default=1, help='channel size of image generator generates') # local epochs for training main nets by generated samples
parser.add_argument('--img_size', type=int, default=32) # local epochs for training generator
### dataset
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--noniid', action='store_false') # default: true
parser.add_argument('--dir_param', type=float, default=0.3)
parser.add_argument('--num_classes', type=int, default=10)
### optimizer
parser.add_argument('--bs', type=int, default=64, help="batch size to load testset")
parser.add_argument('--local_bs', type=int, default=64, help='bs of real/syn training img and training GENs')
# parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
### reproducibility
parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--device_id', type=str, default='3')
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
parser.add_argument('--sample_test', type=int, default=10) # local epochs for training generator
parser.add_argument('--save_imgs', type=bool, default=True) # local epochs for training generator
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--name', type=str, default='dev') # L-A: bad character
### GAN parameters
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--lr', type=float, default=0.0002) # GAN lr
parser.add_argument('--latent_dim', type=int, default=100)

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.img_shape = (args.output_channel, args.img_size, args.img_size)
# args.feature_size = 32
print(args)

# def normal_init(m, mean, std):
#     if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#         m.weight.data.normal_(mean, std)
#         m.bias.data.zero_()

tf = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),transforms.Normalize([0.5], [0.5])]) # mnist is already normalised 0 to 1
# tf = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),]) # mnist is already normalised 0 to 1
train_data = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=True, transform=tf, download=True) # VAE training data

def main():
    dataset_train, dataset_test = getDataset(args)

    if args.noniid:
        dict_users = noniid_dir(args, args.dir_param, dataset_train)
    else:
        dict_users = cifar_iid(dataset_train, int(1/args.partial_data*args.num_users), args.rs)
        
if __name__ == "__main__":
    main()