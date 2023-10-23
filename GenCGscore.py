# from generators16.CCVAE import *
import argparse
from torchvision.utils import save_image
from utils.getModels import *
from utils.getData import *
from torch.utils.data import DataLoader, Dataset
# import ipdb
from utils.CGScore import *
from DDPM.ddpm28 import *
from mlp_generators.VAE import *
from mlp_generators.GAN import *

class FeatDataset(Dataset):
    def __init__(self, images, labels, transform=False):
        super(FeatDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        image = self.images[idx]
        if torch.is_tensor(self.labels[idx]):
            label = self.labels[idx].item()
        else:
            label = self.labels[idx]
        
        return image, label

    
parser = argparse.ArgumentParser()
### model & feature size
parser.add_argument('--models', type=str, default='mlp') # cnn, mlp 
parser.add_argument('--output_channel', type=int, default=1) # local epochs for training generator
parser.add_argument('--img_size', type=int, default=28) # local epochs for training generator
### dataset
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, emnist
parser.add_argument('--bs', type=int, default=600)
parser.add_argument('--local_bs', type=int, default=600)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--device_id', type=str, default='0')

### Pruning based on CG-score
parser.add_argument('--synthetic_pruning', action='store_true', default=False)
parser.add_argument('--real_pruning', action='store_true', default=False)
parser.add_argument('--syn_from_total', action='store_true', default=True)
parser.add_argument('--pruning_ratio', type=float, default=0.1)
parser.add_argument('--from_low', default=True)
parser.add_argument('--rep_num', default=5)
### CVAE parameters
# parser.add_argument('--latent_size', type=int, default=16) # local epochs for training generator
### VAE parameters
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
### GAN parameters
parser.add_argument('--latent_dim', type=int, default=100)
### DDPM parameters
parser.add_argument('--n_feat', type=int, default=128) # 128 ok, 256 better (but slower)
parser.add_argument('--n_T', type=int, default=200) # 400, 500
parser.add_argument('--guide_w', type=float, default=2.0) # 0, 0.5, 2

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.rs=41
args.img_shape = (args.output_channel, args.img_size, args.img_size)
# gennet = CCVAE(args).to(args.device)
# gennet.load_state_dict(torch.load('checkpoint/FedCVAEF.pt'))


### GAN
# gennet = Generator(args).to(args.device)
# gennet.load_state_dict(torch.load('checkpoint/FedGAN.pt'))
### VAE
gennet = CVAE(args).to(args.device)
gennet.load_state_dict(torch.load('checkpoint/FedVAE.pt'))
### DDPM
# gennet = DDPM(args, nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
#                 betas=(1e-4, 0.02), drop_prob=0.1).to(args.device)
# gennet.load_state_dict(torch.load('checkpoint/FedDDPM.pt'))

gennet.eval()

local_models, common_net = getModel(args)

# w_comm = torch.load('checkpoint/Fed_cnn_common_net.pt') # common_net = FE_MLP.to(args.device)
# common_net.load_state_dict(w_comm) # feature extractor

dataset_train, dataset_test = getDataset(args)


# """Synthetic feature pruning"""
# if args.synthetic_pruning == True:
    
#     # Generated features
#     with torch.no_grad():
#         syn_images, one_hot_labels = gennet.sample_image(args, sample_num=args.local_bs) # images.shape (bs, feature^2)

#     syn_labels = torch.argmax(one_hot_labels, dim=1)
#     print(syn_images.shape) # (local_bs, output_channel, img_size, img_size)

#     # Generate real features  
#     syn_feat_dataset = FeatDataset(syn_images, syn_labels)
#     # syn_feat_dataset, measure, data_idx = pruning(syn_feat_dataset, args.pruning_ratio, args.rep_num, args.device, args.from_low)
#     # mean_dict, std_dict = mean_cg_score(measure, data_idx, save_img=True, img_path='./figs/')

#     sampler = torch.utils.data.RandomSampler(syn_feat_dataset, replacement=True)
#     syn_feat_loader = torch.utils.data.DataLoader(syn_feat_dataset,
#                                                     batch_size=args.bs,
#                                                     sampler=sampler,
#                                                     num_workers=16,
#                                                     drop_last=True)

# """Real feature pruning"""
# if args.real_pruning == True:
#     # Generate real features 
#     ldr_train = DataLoader(dataset_train, batch_size=args.bs, shuffle=True)
#     for batch_idx, (images, labels) in enumerate(ldr_train):
#         images = images.to(args.device)
#         # images = common_net(images)
#         feat_images = torch.cat([feat_images, images], dim=0) if batch_idx!=0 else images
#         feat_labels = torch.cat([feat_labels, labels], dim=0) if batch_idx!=0 else labels
    
#     real_feat_dataset = FeatDataset(images, labels) # for 1 batch dataset
#     # real_feat_dataset, measure, data_idx = pruning(real_feat_dataset, args.pruning_ratio, args.rep_num, args.device, args.from_low)
#     # mean_dict, std_dict = mean_cg_score(measure, data_idx, save_img=True, img_path='./figs/')
#     sampler = torch.utils.data.RandomSampler(real_feat_dataset, replacement=True)
#     real_feat_loader = torch.utils.data.DataLoader(real_feat_dataset,
#                                                     batch_size=args.bs,
#                                                     sampler=sampler,
#                                                     num_workers=16,
#                                                     drop_last=True)

""" Concatenate synthetic and real features """
if args.syn_from_total == True:
    
    # Prepare synthetic feature dataset
    with torch.no_grad():
        syn_images, one_hot_labels = gennet.sample_image(args, sample_num=args.local_bs) # images.shape (bs, feature^2)

    if len(syn_images.shape)==2:
        syn_images = syn_images.view(-1, args.output_channel, args.img_size, args.img_size)

    if len(one_hot_labels.shape)>1:
        syn_labels = torch.argmax(one_hot_labels, dim=1)
    else:
        syn_labels = one_hot_labels
        
    print(syn_images.shape) # (local_bs, output_channel, img_size, img_size)
    syn_feat_dataset = FeatDataset(syn_images, syn_labels)
    
    # Prepare real feature dataset
    ldr_train = DataLoader(dataset_train, batch_size=args.bs, shuffle=True)
    for batch_idx, (images, labels) in enumerate(ldr_train):
        images = images.to(args.device) 
        # images = common_net(images)
    
    syn_images, syn_labels = syn_images.to(args.device), syn_labels.to(args.device)
    images, labels = images.to(args.device), labels.to(args.device)

    # Concatenate feature + real dataset 
    images = torch.cat([syn_images, images], dim=0) 
    labels = torch.cat([syn_labels, labels], dim=0)
    feat_dataset = FeatDataset(images, labels)

    # Prune synthetic dataset based on CG-score
    pruned_syn_dataset, syn_measure, data_idx = pruning_only_syn(feat_dataset, syn_feat_dataset, args.pruning_ratio, args.rep_num, args.device, len(syn_images))
    mean_dict, std_dict = mean_cg_score(syn_measure, data_idx, save_img=True)
    print('Mean CG-Score of synthetic feature: ',mean_dict['all'])