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

from generators16.DCGAN import * # creates 3 x 16 x 16
from FeatureExtractor.mlp import *
from mainNetModels.cnn import *
from utils.getData import *
from utils.util import *

os.makedirs("imgFedCGAN", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate") # 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
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
ggenerator = generator(args)
gdiscriminator = discriminator(args)

generators, discriminators = [], []
for i in range(args.num_users):
    generators.append(generator(args))
    discriminators.append(discriminator(args))

if cuda:
    for i in range(args.num_users):
        generators[i].to(device)
        discriminators[i].to(device)
        
    ggenerator.to(device)
    gdiscriminator.to(device)
    
    # adversarial_loss.to(device) # .cuda()


# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)

dataset_train, dataset_test = getDataset(args)
dict_users = cifar_iid(dataset_train, int(1/args.partial_data*args.num_users), args.rs)

dataloaders = []
for i in range(args.num_users):
    dataloaders.append(torch.utils.data.DataLoader(
        DatasetSplit(dataset_train, dict_users[i]), batch_size=args.batch_size, shuffle=True, drop_last=True))

# optimizers
argsimizer_Gs , argsimizer_Ds = [], []
for i in range(args.num_users):
    argsimizer_Gs.append(torch.optim.Adam(generators[i].parameters(), lr=args.lr, betas=(args.b1, args.b2)))
    argsimizer_Ds.append(torch.optim.Adam(discriminators[i].parameters(), lr=args.lr, betas=(args.b1, args.b2)))
    

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
                'imgFedDCGAN/F' + 'sample_' + "_%d.png" % batches_done + '.png', nrow=10)
    # save_image(gen_imgs.data, "imgFedCGAN/F" + str(args.num_users) + "_%d.png" % batches_done, nrow=n_row, normalize=True)


# label preprocess
onehot = torch.zeros(10, 10)
img_size = args.img_shape[1]
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1) # 10 x 10 eye matrix
fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1

y_real_ = torch.ones(args.batch_size)
y_fake_ = torch.zeros(args.batch_size)
y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
# ----------
#  Training
# ----------
BCE_loss = nn.BCELoss().to(device)

g_epoch_loss = []
d_epoch_loss = []

for epoch in range(1, 1+args.n_epochs):
    D_losses = []
    D_real_losses = []
    D_fake_losses = []
    G_losses = []
    
    for di in range(args.num_users):
        for i, (x_, y_) in enumerate(dataloaders[di]):
            '''
            Train Discriminator
            '''
            discriminators[di].zero_grad()
        
            y_fill_ = fill[y_]
            x_, y_fill_ = Variable(x_.cuda()), Variable(y_fill_.cuda())
            x_ = feature_extractor(x_)

            D_result = discriminators[di](x_, y_fill_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((args.batch_size, 100)).view(-1, 100, 1, 1)
            y_ = (torch.rand(args.batch_size, 1) * 10).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            y_fill_ = fill[y_]
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())

            G_result = generators[di](z_, y_label_)
            D_result = discriminators[di](G_result, y_fill_).squeeze()

            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_real_losses.append(D_real_loss)
            D_fake_losses.append(D_fake_loss)
            
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            argsimizer_Ds[di].step()

            D_losses.append(D_train_loss.data)

            # -----------------
            #  Train Generator
            # -----------------

            generators[di].zero_grad()

            z_ = torch.randn((args.batch_size, 100)).view(-1, 100, 1, 1)
            y_ = (torch.rand(args.batch_size, 1) * 10).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            y_fill_ = fill[y_]
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())

            G_result = generators[di](z_, y_label_)
            D_result = discriminators[di](G_result, y_fill_).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            argsimizer_Gs[di].step()

            G_losses.append(G_train_loss.data)
    
        g_epoch_loss.append(sum(G_losses)/len(G_losses))
        d_epoch_loss.append(sum(D_losses)/len(D_losses))
                        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.n_epochs, i, len(dataloaders[di]), d_epoch_loss[-1], g_epoch_loss[-1])
        )

    gws = [generators[i].state_dict() for i in range(args.num_users)]
    dws = [discriminators[i].state_dict() for i in range(args.num_users)]

    gw = FedAvg(gws)
    dw = FedAvg(dws)

    ggenerator.load_state_dict(gw)
    gdiscriminator.load_state_dict(dw)
    for i in range(args.num_users):
        generators[i].load_state_dict(gw)
        discriminators[i].load_state_dict(dw)

    # epochs_done = epoch * len(dataloader)
    if epoch % args.sample_interval == 0:
        sample_image(n_row=10, batches_done=epoch)

torch.save(gw, 'models/save/F' + str(args.num_users)+ '_'+ str(args.n_epochs)+ 'dcgan_generator.pt')
torch.save(dw, 'models/save/F' + str(args.num_users)+ '_'+ str(args.n_epochs)+ 'dcgan_discriminator.pt')