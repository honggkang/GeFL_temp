'''
Train CGAN by federated learning
'''
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
import random

import copy

from modelsMNIST.GAN import *
from utils.getData import *
from utils.util import *
from torchsummaryX import summary

os.makedirs("RAWimgFedCGAN", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate") # 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")

parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=20, help="interval between image sampling")
parser.add_argument('--rs', type=int, default=0)

parser.add_argument("--num_users", type=int, default=10, help="interval between image sampling")
parser.add_argument('--partial_data', type=float, default=0.1)

opt = parser.parse_args()
print(opt)

opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
torch.manual_seed(opt.rs)
torch.cuda.manual_seed(opt.rs)
torch.cuda.manual_seed_all(opt.rs) # if use multi-GPU
np.random.seed(opt.rs)
random.seed(opt.rs)
# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
ggenerator = Generator(opt)
gdiscriminator = Discriminator(opt)

generators, discriminators = [], []
for i in range(opt.num_users):
    generators.append(Generator(opt))
    discriminators.append(Discriminator(opt))

if cuda:
    for i in range(opt.num_users):
        generators[i].cuda()
        discriminators[i].cuda()
    ggenerator.cuda()
    gdiscriminator.cuda()
    
    adversarial_loss.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

bs=2
z = Variable(FloatTensor(np.random.normal(0, 1, (bs, opt.latent_dim))))
c = Variable(LongTensor(np.random.randint(0, opt.n_classes, bs)))
x = torch.zeros(bs,1,28,28, device='cuda') # 
summary(ggenerator, z, c) # torchsummaryX
summary(gdiscriminator, x, c)

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
train_data = datasets.MNIST(root='./data/', train=True,
                            transform=transforms.Compose(
                                [transforms.ToTensor(),
                                #  transforms.Resize(opt.img_size),
                                 transforms.Normalize([0.5], [0.5])
                                ]
                            ), download=True)

dict_users = dict_iid(train_data, int(1/opt.partial_data*opt.num_users))


dataloaders = []
for i in range(opt.num_users):
    dataloaders.append(torch.utils.data.DataLoader(
        DatasetSplit(train_data, dict_users[i]), batch_size=opt.batch_size, shuffle=True))

# Optimizers
optimizer_Gs , optimizer_Ds = [], []
for i in range(opt.num_users):
    optimizer_Gs.append(torch.optim.Adam(generators[i].parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)))
    optimizer_Ds.append(torch.optim.Adam(discriminators[i].parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)))
    

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = ggenerator(z, labels)
    save_image(gen_imgs.data, "RAWimgFedCGAN/" + str(opt.num_users) + "_%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(1,1+opt.n_epochs):
    for di in range(opt.num_users):
        for i, (imgs, labels) in enumerate(dataloaders[di]):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_Gs[di].zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generators[di](z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminators[di](gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_Gs[di].step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_Ds[di].zero_grad()

            # Loss for real images
            validity_real = discriminators[di](real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminators[di](gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_Ds[di].step()

            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            # )

            # batches_done = epoch * len(dataloader) + i
            # if batches_done % opt.sample_interval == 0:
            #     sample_image(n_row=10, batches_done=batches_done)
            
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloaders[di]), d_loss.item(), g_loss.item())
        )
            
    gws = [generators[i].state_dict() for i in range(opt.num_users)]
    dws = [discriminators[i].state_dict() for i in range(opt.num_users)]
    
    gw = FedAvg(gws)
    dw = FedAvg(dws)
    
    ggenerator.load_state_dict(gw)    
    gdiscriminator.load_state_dict(dw)
    for i in range(opt.num_users):
        generators[i].load_state_dict(gw)
        discriminators[i].load_state_dict(dw)
        
    # epochs_done = epoch * len(dataloader)
    if epoch % opt.sample_interval == 0:
        sample_image(n_row=10, batches_done=epoch)

torch.save(gw, 'models/save/' + str(opt.num_users)+ '_'+ str(opt.n_epochs)+ 'gan_generator.pt')
torch.save(dw, 'models/save/' + str(opt.num_users)+ '_'+ str(opt.n_epochs)+ 'gan_discriminator.pt')