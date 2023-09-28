import argparse

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import copy
import numpy as np

from modelsMNIST.VAE import *
from mainNetModels.mlp import *
from utils.getData import *
from utils.util import *


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")

parser.add_argument("--img_size", type=int, default=14, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument('--device_id', type=str, default='0')

parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")

parser.add_argument("--num_users", type=int, default=1, help="interval between image sampling")

opt = parser.parse_args()
print(opt)
opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

# cuda setup
# device = torch.device("cuda:1")
device = 'cuda:' + opt.device_id
kwargs = {'num_workers': 4, 'pin_memory': True} 

cuda = True if torch.cuda.is_available() else False

tf = transforms.Compose(
                        [transforms.ToTensor(),
                        # transforms.Normalize([0.5], [0.5])
                        ]
                        ) # mnist is already normalised 0 to 1

train_data = datasets.MNIST(root='./data/', train=True,
                            transform=tf, download=True)

dict_users = dict_iid(train_data, opt.num_users)

dataloaders = []
for i in range(opt.num_users):
    dataloaders.append(torch.utils.data.DataLoader(
        DatasetSplit(train_data,dict_users[i]), batch_size=opt.batch_size, shuffle=True, **kwargs))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                            transform=tf),
    batch_size=opt.batch_size, shuffle=False, **kwargs)

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

gmodel = CVAE(opt).to(device)
models = []
for i in range(opt.num_users):
    models.append(CVAE(opt).to(device))

optimizers = []
for i in range(opt.num_users):
    optimizers.append(optim.Adam(models[i].parameters(), lr=1e-3))

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, opt.img_size**2), x.view(-1, opt.img_size**2), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # logvar: log(sigma^2)
    return BCE + KLD

LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
feature_extractor = FE_MLP().to(device)
feature_extractor.load_state_dict(torch.load('models/save/common_net.pt'))
feature_extractor.eval()


def train(epoch):
    for di in range(opt.num_users):
        models[di].train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(dataloaders[di]):
            data, labels = data.to(device), labels.to(device)
            # labels = one_hot(labels, 10)
            labels = Variable(labels.type(LongTensor))
            data = feature_extractor(data) # x.view(-1, self.feature_size*self.feature_size) in CVAE.forward 
            
            recon_batch, mu, logvar = models[di](data, labels)
            optimizers[di].zero_grad()
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.detach().cpu().numpy()
            optimizers[di].step()
            # if batch_idx % 200 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(dataloaders[di].dataset),
            #         100. * batch_idx / len(dataloaders[di]),
            #         loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloaders[di].dataset)))

    # update global weights
    ws = [models[i].state_dict() for i in range(opt.num_users)]
    wg = FedAvg(ws)
    
    gmodel.load_state_dict(wg)
    for i in range(opt.num_users):
        models[i].load_state_dict(wg)


def test(epoch):
    gmodel.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            # labels = one_hot(labels, 10)
            labels = Variable(labels.type(LongTensor))
            data = feature_extractor(data) # x.view(-1, self.feature_size*self.feature_size) in CVAE.forward             
            recon_batch, mu, logvar = gmodel(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
            if i == 0:
                n = min(data.size(0), 10)
                comparison = torch.cat([data.view(-1,1,opt.img_size,opt.img_size)[:n],
                                      recon_batch.view(-1, 1, opt.img_size, opt.img_size)[:n]])
                save_image(comparison.cpu(),
                         'imgFedCVAE/' + 'Frecon' + str(opt.num_users) + '_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, opt.n_epochs + 1):
        train(epoch)
        if epoch % opt.sample_interval == 0 or epoch == opt.n_epochs:
            test(epoch)
            n_row = 10
            with torch.no_grad():
                # c = torch.eye(n_row**2, 10).to(device)
                c = np.array([num for _ in range(n_row) for num in range(n_row)])
                c = Variable(LongTensor(c)).to(device)
                sample = torch.randn(n_row**2, opt.latent_size).to(device)
                sample = gmodel.sample_decode(sample, c).cpu()
                save_image(sample.view(n_row**2, 1, opt.img_size, opt.img_size),
                        'imgFedCVAE/F' + str(opt.num_users) +'_' + str(epoch) + '.png', nrow=n_row, normalize=True)

torch.save(gmodel.state_dict(), 'models/save/F' + str(opt.num_users)+ '_'+ str(opt.n_epochs)+ 'cvae.pt')