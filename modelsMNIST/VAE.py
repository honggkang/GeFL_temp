import torch
from torch import nn
from .utilSample import *


class CVAE(nn.Module):
    def __init__(self, opt):
        super(CVAE, self).__init__()
        # self.feature_size = opt.img_size**2
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.img_shape = opt.img_shape
        self.img_size = opt.img_size

        # encode
        self.fc1  = nn.Linear(opt.img_size**2 + opt.n_classes, 400)
        self.fc21 = nn.Linear(400, opt.latent_size)
        self.fc22 = nn.Linear(400, opt.latent_size)

        # decode
        self.fc3 = nn.Linear(opt.latent_size + opt.n_classes, 400)
        self.fc4 = nn.Linear(400, opt.img_size**2)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, n_classes)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+n_classes)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, n_classes)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+n_classes)
        h3 = self.elu(self.fc3(inputs))
        h4 = self.sigmoid(self.fc4(h3))
        return h4.view(h4.size(0), *self.img_shape)

    def forward(self, x, c):
        c = self.label_emb(c)
        mu, logvar = self.encode(x.view(-1, self.img_size**2), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
    def sample_decode(self, z, c):
        c = self.label_emb(c)
        inputs = torch.cat([z, c], 1) # (bs, latent_size+n_classes)
        h3 = self.elu(self.fc3(inputs))
        h4 = self.sigmoid(self.fc4(h3))
        return h4.view(h4.size(0), *self.img_shape)

    # def sample_image(self, args):
    #     with torch.no_grad():
    #         # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    #         z = torch.randn((args.local_bs, args.latent_size)).to(args.device)
    #         c = torch.randint(10, (args.local_bs, )).to(args.device) # MAX_NUM, (SIZE, )
    #         input = torch.cat((self.label_emb(c), z), -1) # Wrong !
    #         h3 = self.elu(self.fc3(input))
    #         gen_imgs = self.sigmoid(self.fc4(h3))
    #         one_c = one_hot(c, args.n_classes).to(args.device)
    #         return gen_imgs, one_c