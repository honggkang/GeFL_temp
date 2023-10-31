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
parser.add_argument("--output_channel", type=int, default=1, help="number of image channels")
parser.add_argument("--img_size", type=int, default=14, help="size of each image dimension")
### GAN
parser.add_argument('--latent_dim', type=int, default=100)
### VAE parameters
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
### DDPM parameters
parser.add_argument('--n_feat', type=int, default=128)
parser.add_argument('--n_T', type=int, default=100)
parser.add_argument('--guide_w', type=float, default=0.0) # 0, 0.5, 2
### CUDA
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.img_shape = (args.output_channel, args.img_size, args.img_size)

tf = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize([0.5], [0.5])
                        ]) # mnist is already normalised 0 to 1
train_data = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=True, transform=tf, download=True)

#########
bs=2
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
z = Variable(FloatTensor(np.random.normal(0, 1, (bs, args.latent_dim)))) # generator
x = torch.zeros(bs, args.output_channel, args.img_size, args.img_size, device='cuda') # discriminator , VAE, DDPM
c = Variable(LongTensor(np.random.randint(0, args.num_classes, bs)))

######### GAN #########
from mlp_generators.GAN import *
fedgen = Generator(args).to(args.device) # [transforms.ToTensor(),transforms.Normalize([0.5], [0.5])]
# fedgen.load_state_dict(torch.load('checkpoint/FedGAN1000.pt'))
# fedgen.load_state_dict(torch.load('checkpoint/FedGAN1001.pt'))
# fedgen.load_state_dict(torch.load('checkpoint/FedGAN1002.pt'))
add = Discriminator(args).to(args.device)
summary(fedgen, z, c) # torchsummaryX
summary(add, x, c)

######### VAE #########
# from mlp_generators.VAE import *
# fedgen = CVAE(args).to(args.device) # [transforms.ToTensor(),]
# fedgen.load_state_dict(torch.load('checkpoint/FedVAE1000.pt'))
# fedgen.load_state_dict(torch.load('checkpoint/FedVAE1001.pt'))
# fedgen.load_state_dict(torch.load('checkpoint/FedVAE1002.pt'))
# summary(fedgen, x, c) # torchsummaryX
# encode: fc1, fc21, fc22 / decode: fc3, fc4

######### DDPM #########
# from DDPM.ddpm14 import *
# fedgen = DDPM(args, nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
#                 betas=(1e-4, 0.02), drop_prob=0.1).to(args.device) # [transforms.ToTensor(),]
# fedgen.load_state_dict(torch.load('checkpoint/FedDDPM1000.pt')) # evaluate over args.guide_w = 0, 2
# fedgen.load_state_dict(torch.load('checkpoint/FedDDPM1001.pt')) # evaluate over args.guide_w = 0, 2
# fedgen.load_state_dict(torch.load('checkpoint/FedDDPM1002.pt')) # evaluate over args.guide_w = 0, 2
# summary(fedgen, x, c) # torchsummaryX

# fedgen.eval()
# with torch.no_grad():
#     img_batch, _ = fedgen.sample_image(args, sample_num=10) # outputs imgs of size (sample_num, 1*28*28)
# img_batch = img_batch.view(-1, args.output_channel, args.img_size, args.img_size) # (sample_num, 1, 28, 28)

# save_image(img_batch, 'imgFedGEN/SynOrig_Ex' + '.png', nrow=10, normalize=True)
