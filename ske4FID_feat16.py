'''
skeleton code for evaluating FID-score, MACs
'''
import argparse
from torchvision import datasets, transforms
from torchvision.utils import save_image

import torch
from torchsummaryX import summary
import numpy as np
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument("--output_channel", type=int, default=3, help="number of image channels")
parser.add_argument("--img_size", type=int, default=16, help="size of each image dimension")
### DCGAN
parser.add_argument('--latent_dim', type=int, default=100)
### VAE parameters
parser.add_argument('--latent_size', type=int, default=16) # local epochs for training generator
### DDPM parameters
parser.add_argument('--n_feat', type=int, default=128)
parser.add_argument('--n_T', type=int, default=400)
parser.add_argument('--guide_w', type=float, default=2.0) # 0, 0.5, 2
### CUDA
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.img_shape = (args.output_channel, args.img_size, args.img_size)

# tf = transforms.Compose([
#                         transforms.ToTensor(),
#                         # transforms.Normalize([0.5], [0.5])
#                         ]) # mnist is already normalised 0 to 1
# train_data = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=True, transform=tf, download=True)

#########
bs=2
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
z = Variable(FloatTensor(np.random.normal(0, 1, (bs, args.latent_dim)))).view(-1, args.latent_dim, 1, 1) # generator
x = torch.zeros(bs, args.output_channel, args.img_size, args.img_size, device='cuda') # discriminator , VAE, DDPM
c = Variable(LongTensor(np.random.randint(0, args.num_classes, bs)))


######### DCGAN #########
# from generators16.DCGAN import *
# fedgen = generator(args, d=128).to(args.device)
# add = discriminator(args, d=128).to(args.device)
# # fedgen.load_state_dict(torch.load('checkpoint/FedGAN1000.pt'))
# # fedgen.load_state_dict(torch.load('checkpoint/FedGAN1001.pt'))
# # fedgen.load_state_dict(torch.load('checkpoint/FedGAN1002.pt'))
# onehot = torch.zeros(10, 10)
# onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1) # 10 x 10 eye matrix
# y_ = (torch.rand(bs, 1) * 10).type(torch.LongTensor).squeeze()
# y_label_ = onehot[y_]
# y_label_ = Variable(y_label_.cuda())
# fill = torch.zeros([10, 10, args.img_size, args.img_size])
# for i in range(10):
#     fill[i, i, :, :] = 1
# y_fill_ = fill[y_]
# y_fill_ = Variable(y_fill_.cuda())
# summary(fedgen, z, y_label_) # torchsummaryX
# summary(add, x, y_fill_)

######### CVAE #########
# from generators16.CCVAE import *
# fedgen = CCVAE(args).to(args.device) # [transforms.ToTensor(),]
# # fedgen.load_state_dict(torch.load('checkpoint/FedVAE1000.pt'))
# # fedgen.load_state_dict(torch.load('checkpoint/FedVAE1001.pt'))
# # fedgen.load_state_dict(torch.load('checkpoint/FedVAE1002.pt'))
# y = (torch.rand(bs, 1) * 10).type(torch.LongTensor).squeeze()
# label = np.zeros((bs, 10))
# label[np.arange(bs), y] = 1
# label = torch.tensor(label).to(args.device)
# summary(fedgen, x, label) # torchsummaryX
# encode: fc1, fc21, fc22 / decode: fc3, fc4

######### DDPM #########
from DDPM.ddpm16 import *
fedgen = DDPM(args, nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
                betas=(1e-4, 0.02), drop_prob=0.1).to(args.device) # [transforms.ToTensor(),]
fedgen.load_state_dict(torch.load('checkpoint/FedDDPMF1.pt')) # evaluate over args.guide_w = 0, 2
# fedgen.load_state_dict(torch.load('checkpoint/FedDDPM1001.pt')) # evaluate over args.guide_w = 0, 2
# fedgen.load_state_dict(torch.load('checkpoint/FedDDPM1002.pt')) # evaluate over args.guide_w = 0, 2
# summary(fedgen, x, c) # torchsummaryX

fedgen.eval()
with torch.no_grad():
    img_batch, _ = fedgen.sample_image(args, sample_num=40) # outputs imgs of size (sample_num, 1*28*28)
img_batch = img_batch.view(-1, args.output_channel, args.img_size, args.img_size) # (sample_num, 1, 28, 28)

save_image(img_batch, 'imgs/imgFedGEN/Feat_50w2' + '.png', nrow=10) # , normalize=True
