'''
skeleton code for evaluating FID-score, MACs
'''
import os
import argparse
from torchvision import datasets, transforms

import torch
from torchsummaryX import summary
import numpy as np
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument("--output_channel", type=int, default=1, help="number of image channels")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
### GAN
parser.add_argument('--latent_dim', type=int, default=100)
### VAE parameters
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
### DDPM parameters
parser.add_argument('--n_feat', type=int, default=128)
parser.add_argument('--n_T', type=int, default=100)
parser.add_argument('--guide_w', type=float, default=0) # 0, 0.5, 2
### CUDA
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.img_shape = (args.output_channel, args.img_size, args.img_size)

# tf = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize([0.5], [0.5])
# ]) # mnist is already normalised 0 to 1
# train_data = datasets.MNIST(root='./home/hong/NeFL/.data/mnist', train=True, transform=tf, download=True)
# test_data = datasets.MNIST(root='./home/hong/NeFL/.data/mnist', train=False, transform=tf, download=True)

# # Define a transform to convert the image to a tensor and back to PIL
to_pil = transforms.Compose([
    transforms.ToPILImage()
])

# #########
bs=2
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
z = Variable(FloatTensor(np.random.normal(0, 1, (bs, args.latent_dim)))) # generator
x = torch.zeros(bs,1,28,28, device='cuda') # discriminator , VAE, DDPM
c = Variable(LongTensor(np.random.randint(0, args.num_classes, bs)))

# ######### source image (train)#########
# save_dir = 'imgs/mnist/src_train/'
# os.makedirs(save_dir, exist_ok=True)
# print(f'Saving {len(train_data)} MNIST images in {save_dir}......')

# for i, (image, label) in enumerate(train_data):
#     image = to_pil(image)
#     file_path = os.path.join(save_dir, f'mnist_train_{i}.png')

#     if not os.path.exists(file_path):
#         image.save(file_path)
#         # Skipped saving as it already exists

# print(f'Saved {len(train_data)} MNIST images in {save_dir}.')

# ######### source image (test)#########
# save_dir = 'imgs/mnist/src_test/'
# os.makedirs(save_dir, exist_ok=True)
# print(f'Saving {len(test_data)} MNIST images in {save_dir}......')

# for i, (image, label) in enumerate(test_data):
#     image = to_pil(image)
#     file_path = os.path.join(save_dir, f'mnist_train_{i}.png')

#     if not os.path.exists(file_path):
#         image.save(file_path)
#         # Skipped saving as it already exists

# print(f'Saved {len(test_data)} MNIST images in the directory {save_dir}.')

# ######## GAN #########
# from mlp_generators.GAN import *
# fedgan = Generator(args).to(args.device) # [transforms.ToTensor(),transforms.Normalize([0.5], [0.5])]
# #fedgan.load_state_dict(torch.load('checkpoint/FedGAN1000.pt'))
# #fedgan.load_state_dict(torch.load('checkpoint/FedGAN1001.pt'))
# fedgan.load_state_dict(torch.load('checkpoint/FedGAN1002.pt'))
# add = Discriminator(args).to(args.device)
# fedgan.eval()
# #summary(fedgen, z, c) # torchsummaryX
# #summary(add, x, c)

# # Create a directory to save the generated images
# save_dir = 'imgs/mnist/imgFedGAN1002/'
# os.makedirs(save_dir, exist_ok=True)

# # Generate and save 50,000 images with random labels
# num_images = 500
# sample_num = 100
# print(f'Saving {num_images * sample_num} MNIST images in {save_dir}......')

# with torch.no_grad():
#     for i in range(num_images):
#         generated_images, _ = fedgan.sample_image(args, sample_num=sample_num)
#         generated_images = generated_images.view(-1, args.output_channel, args.img_size, args.img_size)
        
#         # Save each image in the batch as a PNG file
#         for j in range(sample_num):
#             image = generated_images[j].squeeze(0).cpu()  # Reshape to 28x28
#             pil_image = to_pil(image)
#             file_path = os.path.join(save_dir, f'FedGAN_image_{i * sample_num + j}.png')
#             if not os.path.exists(file_path):
#                 pil_image.save(file_path)
# print(f'Saved {num_images * sample_num} generated images in the directory {save_dir}.')



# ######### VAE #########
# from mlp_generators.VAE import *
# fedvae = CVAE(args).to(args.device) # [transforms.ToTensor(),]
# #fedvae.load_state_dict(torch.load('checkpoint/FedVAE1000.pt'))
# #fedvae.load_state_dict(torch.load('checkpoint/FedVAE1001.pt'))
# fedvae.load_state_dict(torch.load('checkpoint/FedVAE1002.pt'))
# # summary(fedvae, x, c) # torchsummaryX
# fedvae.eval()

# # Create a directory to save the generated images
# save_dir = 'imgs/mnist/imgFedVAE1002/'
# os.makedirs(save_dir, exist_ok=True)

# # Generate and save 50,000 images with random labels
# num_images = 500
# sample_num = 100
# print(f'Saving {num_images * sample_num} MNIST images in {save_dir}......')

# with torch.no_grad():
#     for i in range(num_images):
#         generated_images, _ = fedvae.sample_image(args, sample_num=sample_num)
#         generated_images = generated_images.view(-1, args.output_channel, args.img_size, args.img_size)
        
#         # Save each image in the batch as a PNG file
#         for j in range(sample_num):
#             image = generated_images[j].squeeze(0).cpu()  # Reshape to 28x28
#             pil_image = to_pil(image)
#             file_path = os.path.join(save_dir, f'FedVAE_image_{i * sample_num + j}.png')
#             if not os.path.exists(file_path):
#                 pil_image.save(file_path)
# print(f'Saved {num_images * sample_num} generated images in the directory {save_dir}.')


# ######### DDPM #########
from DDPM.ddpm28 import *
fedddpm = DDPM(args, nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
              betas=(1e-4, 0.02), drop_prob=0.1).to(args.device) # [transforms.ToTensor(),]
fedddpm.load_state_dict(torch.load('checkpoint/FedDDPM01000.pt')) # evaluate over args.guide_w = 0, 2
# #fedddpm.load_state_dict(torch.load('checkpoint/FedDDPM1001.pt')) # evaluate over args.guide_w = 0, 2
# #fedddpm.load_state_dict(torch.load('checkpoint/FedDDPM1002.pt')) # evaluate over args.guide_w = 0, 2
# summary(fedddpm, x, c) # torchsummaryX

fedddpm.eval()

# # Create a directory to save the generated images
save_dir = 'imgs/mnist/imgFedDDPM1000_w2/'
os.makedirs(save_dir, exist_ok=True)

# # Generate and save 50,000 images with random labels
num_images = 50
sample_num = 10
print(f'Saving {num_images * sample_num} MNIST images in {save_dir}......')

with torch.no_grad():
    for i in range(num_images):
        generated_images, _ = fedddpm.sample_image(args, sample_num=sample_num)
        generated_images = generated_images.view(-1, args.output_channel, args.img_size, args.img_size)
        
        # Save each image in the batch as a PNG file
        for j in range(sample_num):
            image = generated_images[j].squeeze(0).cpu()  # Reshape to 28x28
            pil_image = to_pil(image)
            file_path = os.path.join(save_dir, f'FedDDPM_image_{i * sample_num + j}.png')
            if not os.path.exists(file_path):
                pil_image.save(file_path)
print(f'Saved {num_images * sample_num} generated images in the directory {save_dir}.')

# with torch.no_grad():
#     img_batch, _ = fedgen.sample_image(args, sample_num=10) # outputs imgs of size (sample_num, 1*28*28)
# img_batch = img_batch.view(-1, args.output_channel, args.img_size, args.img_size) # (sample_num, 1, 28, 28)
# 
# save_image(img_batch, 'img/imgFedGEN/SynOrig_Ex' + '.png', nrow=10, normalize=True)