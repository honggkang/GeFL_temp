'''
args.img_shape makes last linear layer to generate wished img size 
original/feature
Conditional GAN
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
'''
import torch.nn as nn
import torch
import numpy as np
from models.utilSample import *

# FloatTensor = torch.cuda.FloatTensor
# LongTensor = torch.cuda.LongTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(args.num_classes, args.num_classes)
        self.args = args
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim + args.num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(args.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        # img = img.view(img.size(0), *img_shape)
        return img
    
    def sample_image(self, args, sample_num=0):
        with torch.no_grad():
            z = Variable(FloatTensor(np.random.normal(0, 1, (sample_num, self.args.latent_dim)))).to(self.args.device)
            # z = torch.randn((sample_num, args.latent_dim)).to(args.device)
            c = torch.randint(10, (sample_num, )).to(args.device) # MAX_NUM, (SIZE, )
            input = torch.cat((self.label_emb(c), z), -1)
            gen_imgs = self.model(input)
            one_c = one_hot(c, args.num_classes).to(args.device)
        return gen_imgs, one_c


    def sample_image_4visualization(self, sample_num):
        with torch.no_grad():
            # z = torch.randn((sample_num, self.args.latent_dim)).to(self.args.device)
            # z = Variable(FloatTensor(np.random.normal(0, 1, (sample_num, self.args.latent_dim)))).to(self.args.device)
            # labels = torch.arange(0,10).to(self.args.device) # context for us just cycles throught the mnist labels
            # labels = labels.repeat(int(sample_num/labels.shape[0]))
            # input = torch.cat((self.label_emb(labels), z), -1)
            # gen_imgs = self.model(input)

            z = Variable(FloatTensor(np.random.normal(0, 1, (sample_num, self.args.latent_dim)))).to(self.args.device)
            labels = np.array([num for _ in range(int(sample_num/10)) for num in range(10)])
            labels = Variable(LongTensor(labels)).to(self.args.device)
            input = torch.cat((self.label_emb(labels), z), -1)
            gen_imgs = self.model(input)
            # labels = torch.Tensor(labels).to(self.args.device)            
        return gen_imgs
    
    
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(args.num_classes, args.num_classes)

        self.model = nn.Sequential(
            nn.Linear(args.num_classes + int(np.prod(args.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

''' =======================================================================
==================== 3-channel dataset ====================================
========================================================================'''