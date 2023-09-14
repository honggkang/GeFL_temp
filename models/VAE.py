import torch
from torch import nn


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


class CVAE(nn.Module):
    def __init__(self, img_feat_1d_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = img_feat_1d_size # feature_size^2, i.e., 256
        self.class_size = class_size
        self.latent_size = latent_size

        # encode
        self.fc1  = nn.Linear(img_feat_1d_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, img_feat_1d_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var # bs, latent_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # logvar: log(sigma^2)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
    def sample_image(self, args):
        with torch.no_grad():
            z = torch.randn((args.batch_size, self.latent_size)).to(args.device)
            # c = 0 # 0 ~ 9 randint
            c = torch.randint(10, (args.local_bs, )) # MAX_NUM, (SIZE, )
            one_c = one_hot(c, args.num_classes).to(args.device)
            return self.decode(z, one_c), one_c

    # def make_batch(self, args):    
    #     with torch.no_grad():
    #         # c = torch.eye(10, 10).to(args.device)
    #         c = torch.randint(10, (args.local_bs, ))
    #         one_c = one_hot(c, args.num_classes).to(args.device)
    #         sample = torch.randn(args.local_bs, self.latent_size).to(args.device)
    #         sample = self.decode(sample, one_c)
    #         # save_image(sample.view(10, 1, 14, 14),
    #         #             'imgFedCVAE/' + 'sample_' + '.png')
    
''' =======================================================================
==================== 3-channel dataset ====================================
========================================================================'''