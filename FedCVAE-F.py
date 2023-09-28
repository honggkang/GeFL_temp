import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

# from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

import copy
from generators16.CCVAE import * # creates 3 x 16 x 16
from models.mlp import *
from models.cnn import *
from utils.getData import *
from utils.util import *

os.makedirs("imgFedCVAE", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument('--latent_size', type=int, default=16) # local epochs for training generator

parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--models', type=str, default='cnn') # cnn, mlp

parser.add_argument('--num_users', type=int, default=10)
parser.add_argument('--partial_data', type=float, default=0.1)
parser.add_argument("--img_size", type=int, default=16, help="size of each image dimension")
parser.add_argument("--output_channel", type=int, default=3, help="number of image channels")
parser.add_argument('--rs', type=int, default=0)

args = parser.parse_args()
print(args)

args.img_shape = (args.output_channel, args.img_size, args.img_size)
cuda = True if torch.cuda.is_available() else False
device = 'cuda:' + args.device_id
args.device = device
# Loss functions
# adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
ggenerator = CCVAE(args).to(args.device)

generators = []
for i in range(args.num_users):
    generators.append(CCVAE(args).to(args.device))       
    
# adversarial_loss.to(device) # .cuda()
# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)

dataset_train, dataset_test = getDataset(args)
# dict_users = dict_iid(dataset_train, args.num_users)
dict_users = cifar_iid(dataset_train, int(1/args.partial_data*args.num_users), args.rs)

dataloaders = []
for i in range(args.num_users):
    dataloaders.append(torch.utils.data.DataLoader(
        DatasetSplit(dataset_train,dict_users[i]), batch_size=args.batch_size, shuffle=True, drop_last=True))

# optimizers
optimizers = []
for i in range(args.num_users):
    optimizers.append(torch.optim.Adam(generators[i].parameters(), lr=1e-3, weight_decay=0.001))


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

feature_extractor = FE_CNN().to(device) # .cuda()
feature_extractor.load_state_dict(torch.load('models/save/0.1_100CNN_common_net.pt'))
feature_extractor.eval()


def sample_image(n_row, batches_done):
    sample_num = 50
    samples = ggenerator.sample_image_4visualization(sample_num)
    # sample.shape = [10, 256]
    save_image(samples.view(sample_num, 3, args.img_size, args.img_size), 
                'imgFedCVAE/F' + 'sample_' + "_%d.png" % batches_done + '.png', nrow=10)
    # save_image(gen_imgs.data, "imgFedCGAN/F" + str(args.num_users) + "_%d.png" % batches_done, nrow=n_row, normalize=True)

# ----------
#  Training
# ----------
def loss_function_ccvae(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld

epoch_loss = []
for epoch in range(1, 1+args.n_epochs):
    G_losses = []
    
    for di in range(args.num_users):
        batch_loss = []
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloaders[di]):
            images = images.to(args.device) # images.shape: torch.Size([batch_size, 1, 28, 28])
            images = feature_extractor(images) # x.view(-1, self.feature_size*self.feature_size) in CVAE.forward 
            label = np.zeros((images.shape[0], 10))
            label[np.arange(images.shape[0]), labels] = 1
            label = torch.tensor(label)
        
            optimizers[di].zero_grad()
            pred, mu, logvar = generators[di](images, label.to(args.device))
            
            recon_loss, kld = loss_function_ccvae(images, pred, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optimizers[di].step()
            train_loss += loss.detach().cpu().numpy()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))   
                        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [CVAE loss: %f]"
            % (epoch, args.n_epochs, batch_idx, len(dataloaders[di]), epoch_loss[-1])
        )

    gws = [generators[i].state_dict() for i in range(args.num_users)]

    gw = FedAvg(gws)

    ggenerator.load_state_dict(gw)
    for i in range(args.num_users):
        generators[i].load_state_dict(gw)

    # epochs_done = epoch * len(dataloader)
    if epoch % args.sample_interval == 0:
        sample_image(n_row=10, batches_done=epoch)

torch.save(gw, 'models/save/F' + str(args.num_users)+ '_'+ str(args.n_epochs)+ 'cvae.pt')