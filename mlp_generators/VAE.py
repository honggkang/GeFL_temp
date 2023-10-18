import torch
from torch import nn
from models.utilSample import *

class CVAE(nn.Module):
    def __init__(self, args):
        super(CVAE, self).__init__()
        self.args = args
        self.label_emb = nn.Embedding(args.num_classes, args.num_classes)
        self.feature_size = args.img_size*args.img_size # args.img_size / feature_size^2, i.e., 16*16 256
        self.latent_size = args.latent_size
        self.img_shape = args.img_shape

        # encode
        self.fc1  = nn.Linear(self.feature_size + args.num_classes, 400)
        self.fc21 = nn.Linear(400, args.latent_size)
        self.fc22 = nn.Linear(400, args.latent_size)

        # decode
        self.fc3 = nn.Linear(args.latent_size + args.num_classes, 400)
        self.fc4 = nn.Linear(400, self.feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, args.num_classes)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+args.num_classes)
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
        c: (bs, args.num_classes)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+args.num_classes)
        h3 = self.elu(self.fc3(inputs))
        h4 = self.sigmoid(self.fc4(h3))
        out = h4.view(h4.size(0), *self.img_shape) # ex) bs, 1, 28, 28
        return out

    def forward(self, x, c):
        c = self.label_emb(c)
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
    def sample_image(self, args):
        with torch.no_grad():
            z = torch.randn((args.local_bs, self.latent_size)).to(args.device)
            # c = 0 # 0 ~ 9 randint
            c = torch.randint(10, (args.local_bs, )).to(args.device) # MAX_NUM, (SIZE, )
            input = torch.cat((self.label_emb(c), z), -1)
            h3 = self.elu(self.fc3(input))
            gen_imgs = self.sigmoid(self.fc4(h3))
            one_c = one_hot(c, args.num_classes).to(args.device)
            return gen_imgs, one_c


    def sample_image_4visualization(self, sample_num):
        with torch.no_grad():
            # c = np.array([num for _ in range(n_row) for num in range(n_row)])
            # c = Variable(LongTensor(c)).to(device)
            labels = torch.arange(0,10).to(self.args.device) # context for us just cycles throught the mnist labels
            labels = labels.repeat(int(sample_num/labels.shape[0]))
            z = torch.randn(sample_num, self.args.latent_size).to(self.args.device)        
            input = torch.cat((self.label_emb(labels), z), -1)
            h3 = self.elu(self.fc3(input))
            gen_imgs = self.sigmoid(self.fc4(h3))
            return gen_imgs
        
    # def make_batch(self, args):    
    #     with torch.no_grad():
    #         # c = torch.eye(10, 10).to(args.device)
    #         c = torch.randint(10, (args.local_bs, ))
    #         one_c = one_hot(c, args.num_classes).to(args.device)
    #         sample = torch.randn(args.local_bs, self.latent_size).to(args.device)
    #         sample = self.decode(sample, one_c)
    #         # save_image(sample.view(10, 1, 14, 14),
    #         #             'imgFedCVAE/' + 'sample_' + '.png')
