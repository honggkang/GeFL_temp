from modelsMNIST.DDPM import *
import argparse
from torchvision.utils import save_image
import os 

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--bs", type=int, default=128, help="size of the batches")
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist
parser.add_argument('--gen', type=str, default='ddpm') # gan, vae, ddpm

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # GAN
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space") # VAE
parser.add_argument("--n_feat", type=int, default=128) # DDPM
parser.add_argument("--n_T", type=int, default=100) # DDPM
parser.add_argument("--w", type=int, default=1) # DDPM guidance

parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")

parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)

args = parser.parse_args()
args.device = 'cuda:' + args.device_id

gennet = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes), betas=(1e-4, 0.02), n_T=100, device=args.device, drop_prob=0.1).to(args.device)
gennet.load_state_dict(torch.load('models/save/10_400_nT100_ddpm.pt'))
gennet.eval()

p_dir = './generatedImages/mnist/10ddpm' + str(args.w) +'_nTt100nTg' + str(args.n_T) + '/'
if not os.path.exists(p_dir):
    os.makedirs(p_dir)

# 32*188 = 6016
iter = 6000//args.batch_size

with torch.no_grad():
    for l in range(10):
        print('label:', l)
        save_dir = p_dir + str(l) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(iter+1):
            imgs = gennet.save_image(args, l, guide_w=args.w)
            for b in range(args.batch_size):
                file_dir = save_dir + str(i) + '_' + str(b) + '.png'
                save_image(imgs[b], file_dir)