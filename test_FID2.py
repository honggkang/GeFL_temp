'''
skeleton code for evaluating FID-score, MACs
'''
import os
import argparse
from torchvision import datasets, transforms
from torchvision.utils import save_image

import torch
from torchsummaryX import summary
import numpy as np
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')  # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument("--output_channel", type=int, default=1, help="number of image channels")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--bs", type=int, default=100, help="size of the batches")

### GAN
parser.add_argument('--latent_dim', type=int, default=100)
### VAE parameters
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
### DDPM parameters
parser.add_argument('--n_feat', type=int, default=128)
parser.add_argument('--n_T', type=int, default=100)
parser.add_argument('--guide_w', type=float, default=0)  # 0, 0.5, 2
### CUDA
parser.add_argument('--device_id', type=str, default='3')
### Parameters for FID
parser.add_argument('--gen', type=str, default='dcgan')  # gan(GAN), vae(VAE), ddpm(DDPM)
parser.add_argument('--gen_dir', type=str, default='checkpoint/FedDCGANupdateGEN2.pt')
parser.add_argument('--img_dir', type=str, default='imgs/')

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.img_shape = (args.output_channel, args.img_size, args.img_size)

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(args.img_size),
    transforms.Normalize([0.5], [0.5])
])  # mnist is already normalised 0 to 1

train_data = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=True, transform=tf, download=True)
test_data = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=False, transform=tf, download=True)

# Check if the provided generator value is in the list of valid generators
valid_generators = ['gan', 'GAN', 'vae', 'VAE', 'ddpm', 'DDPM', 'dcgan', 'DCGAN']
if args.gen.lower() not in valid_generators:
    print("Warning: The provided --generator value is not one of 'gan', 'vae', or 'ddpm'.")
    print("Please provide a valid generator value.")
if not os.path.exists(args.gen_dir):
    print("Please provide a valid generator model path.")

bs = 2
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
z = Variable(FloatTensor(np.random.normal(0, 1, (bs, args.latent_dim))))  # generator
x = torch.zeros(bs, 1, 28, 28, device=args.device)  # discriminator , VAE, DDPM
c = Variable(LongTensor(np.random.randint(0, args.num_classes, bs)))


######### source image (train)#########
save_dir = args.img_dir + args.dataset + '/src_train/'
os.makedirs(save_dir, exist_ok=True)
print(f'Saving {len(train_data)} MNIST train images in {save_dir}......')

for i, (image, label) in enumerate(train_data):
    file_path = os.path.join(save_dir, f'mnist_train_{i}.png')
    # Skipped saving as it already exists
    if not os.path.exists(file_path):
        save_image(image, file_path, nrow=1)

print(f'Saved {len(train_data)} MNIST images in {save_dir}.')

######### source image (test)#########
save_dir = args.img_dir + args.dataset + '/src_test/'
os.makedirs(save_dir, exist_ok=True)
print(f'Saving {len(test_data)} MNIST test images in {save_dir}......')

for i, (image, label) in enumerate(test_data):
    file_path = os.path.join(save_dir, f'mnist_train_{i}.png')
    # Skipped saving as it already exists
    if not os.path.exists(file_path):
        save_image(image, file_path, nrow=1)

print(f'Saved {len(test_data)} MNIST images in the directory {save_dir}.')

######## GAN #########
if args.gen == "gan" or args.gen == "GAN":
    from mlp_generators.GAN import *

    fedgen = Generator(args).to(args.device)  # [transforms.ToTensor(),transforms.Normalize([0.5], [0.5])]
    fedgen.load_state_dict(torch.load(args.gen_dir))
    add = Discriminator(args).to(args.device)
    fedgen.eval()
    # summary(fedgen, z, c) # torchsummaryX
    # summary(add, x, c)

    # Create a directory to save the generated images
    save_dir = args.img_dir + args.dataset + '/' + args.gen_dir[:-3] + '/'
    os.makedirs(save_dir, exist_ok=True)

    # Generate and save 60,000 images with random labels
    num_images = len(train_data) // args.bs
    print(f'Saving {num_images * args.bs} MNIST images in {save_dir}......')

    with torch.no_grad():
        for i in range(num_images):
            generated_images, _ = fedgen.sample_image(args, sample_num=args.bs)
            generated_images = generated_images.view(-1, args.output_channel, args.img_size, args.img_size)

            # Save each image in the batch as a PNG file
            for j in range(args.bs):
                image = generated_images[j].squeeze(0)  # Reshape to 28x28
                file_path = os.path.join(save_dir, f'FedGAN_image_{i * args.bs + j}.png')
                if not os.path.exists(file_path):
                    save_image(image, file_path, nrow=1)
    print(f'Saved {num_images * args.bs} generated images in the directory {save_dir}.')


######## DCGAN ########
elif args.gen == "dcgan" or args.gen == "DCGAN":
    from generators32.DCGAN import *

    fedgen = generator(args, d=128).to(args.device)
    fedgen.load_state_dict(torch.load(args.gen_dir))
    add = discriminator(args, d=128).to(args.device)
    fedgen.eval()
    # summary(fedgen, z, c) # torchsummaryX
    # summary(add, x, c)

    # Create a directory to save the generated images
    save_dir = args.img_dir + args.dataset + '/' + args.gen_dir[:-3] + '/'
    os.makedirs(save_dir, exist_ok=True)

    # Generate and save 60,000 images with random labels
    num_images = len(train_data) // args.bs
    print(f'Saving {num_images * args.bs} MNIST images in {save_dir}......')

    with torch.no_grad():
        for i in range(num_images):
            generated_images, _ = fedgen.sample_image(args, sample_num=args.bs)
            generated_images = generated_images.view(-1, args.output_channel, args.img_size, args.img_size)

            # Save each image in the batch as a PNG file
            for j in range(args.bs):
                image = generated_images[j].squeeze(0)  # Reshape to 28x28
                file_path = os.path.join(save_dir, f'FedGAN_image_{i * args.bs + j}.png')
                if not os.path.exists(file_path):
                    save_image(image, file_path, nrow=1)
    print(f'Saved {num_images * args.bs} generated images in the directory {save_dir}.')

######### VAE #########
elif args.gen == "vae" or args.gen == "VAE":
    from mlp_generators.VAE import *

    fedgen = CVAE(args).to(args.device)  # [transforms.ToTensor(),]
    fedgen.load_state_dict(torch.load(args.gen_dir))
    # summary(fedgen, x, c) # torchsummaryX
    fedgen.eval()

    # Create a directory to save the generated images
    save_dir = args.img_dir + args.dataset + '/' + args.gen_dir[:-3] + '/'
    os.makedirs(save_dir, exist_ok=True)

    # Generate and save 60,000 images with random labels
    num_images = len(train_data) // args.bs
    print(f'Saving {num_images * args.bs} MNIST images in {save_dir}......')

    with torch.no_grad():
        for i in range(num_images):
            generated_images, _ = fedgen.sample_image(args, sample_num=args.bs)
            generated_images = generated_images.view(-1, args.output_channel, args.img_size, args.img_size)

            # Save each image in the batch as a PNG file
            for j in range(args.bs):
                image = generated_images[j].squeeze(0)  # Reshape to 28x28
                file_path = os.path.join(save_dir, f'FedVZE_image_{i * args.bs + j}.png')
                if not os.path.exists(file_path):
                    save_image(image, file_path, nrow=1)
    print(f'Saved {num_images * args.bs} generated images in the directory {save_dir}.')


######### DDPM #########
elif args.gen == "ddpm" or args.gen == "DDPM":
    from DDPM.ddpm28 import *

    fedgen = DDPM(args,
                  nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
                  betas=(1e-4, 0.02), drop_prob=0.1).to(args.device)  # [transforms.ToTensor(),]
    fedgen.load_state_dict(torch.load(args.gen_dir))
    fedgen.eval()

    # Create a directory to save the generated images
    save_dir = args.img_dir + args.dataset + '/' + args.gen_dir[:-3] + '_w' + str(args.guide_w) + '/'
    os.makedirs(save_dir, exist_ok=True)

    # Generate and save 60,000 images with random labels
    args.bs = 60
    num_images = len(train_data) // args.bs
    print(f'Saving {num_images * args.bs} MNIST images in {save_dir}......')

    with torch.no_grad():
        for i in range(num_images):
            generated_images = fedgen.sample_image_4visualization(args.bs, guide_w=args.guide_w)
            # generated_images, _ = fedddpm.sample_image(args, args.bs=args.bs)
            generated_images = generated_images.view(-1, args.output_channel, args.img_size, args.img_size)

            # Save each image in the batch as a PNG file
            for j in range(args.bs):
                image = generated_images[j].squeeze(0)  # Reshape to 28x28
                # pil_image = to_pil(image)
                file_path = os.path.join(save_dir, f'FedDDPM_image_{i * args.bs + j}.png')
                if not os.path.exists(file_path):
                    save_image(image, file_path, nrow=1)
                    # pil_image.save(file_path)
    print(f'Saved {num_images * args.bs} generated images in the directory {save_dir}.')


## Calculate FID
save_dir_1 = args.img_dir + args.dataset + '/src_train'
if args.gen == "ddpm" or args.gen == "DDPM":
    save_dir_2 = args.img_dir + args.dataset + '/' + args.gen_dir[:-3] + '_w' + str(args.guide_w)
else:
    save_dir_2 = args.img_dir + args.dataset + '/' + args.gen_dir[:-3]

os.system("python -m pytorch_fid " + save_dir_1 + ' ' + save_dir_2) #python -m pytorch_fid path/to/dataset1 path/to/dataset2
