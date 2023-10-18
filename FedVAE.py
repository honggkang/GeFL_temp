import argparse
import os

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
from utils.getData import *
from utils.util import *
# from torchsummaryX import summary

os.makedirs("RAWimgFedCVAE", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=20, help="interval between image sampling")
parser.add_argument('--device_id', type=str, default='0')

parser.add_argument("--num_users", type=int, default=10, help="interval between image sampling")
parser.add_argument('--partial_data', type=float, default=0.1)

opt = parser.parse_args()
print(opt)
opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

# cuda setup
# device = torch.device("cuda:1")
device = 'cuda:' + opt.device_id
kwargs = {'num_workers': 4, 'pin_memory': True} 

cuda = True if torch.cuda.is_available() else False

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)

# train_data = datasets.MNIST(root='./data/', train=True,
#                             transform=transforms.Compose(
#                                 [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#                             ), download=True)

tf = transforms.Compose(
                        [transforms.ToTensor(),
                        # transforms.Normalize([0.5], [0.5])
                        ]
                        ) # mnist is already normalised 0 to 1

train_data = datasets.MNIST(root='./data/', train=True,
                            transform=tf, download=True)

# dict_users = dict_iid(train_data, int(1/opt.partial_data*opt.num_users))
dict_users = cifar_iid(train_data, int(1/opt.partial_data*opt.num_users), 0)

dataloaders = []
for i in range(opt.num_users):
    dataloaders.append(torch.utils.data.DataLoader(
        DatasetSplit(train_data,dict_users[i]), batch_size=opt.batch_size, shuffle=True, **kwargs))
    
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                            transform=tf),
    batch_size=opt.batch_size, shuffle=False, **kwargs)


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


# create a CVAE model
gmodel = CVAE(opt).to(device)
# x = torch.zeros(1,1,28,28, device='cuda') # 
# c = torch.cuda.LongTensor((1,))
# summary(gmodel, x, c)

models = []
for i in range(opt.num_users):
    models.append(CVAE(opt).to(device))


optimizers = []
for i in range(opt.num_users):
    optimizers.append(optim.Adam(models[i].parameters(), lr=1e-3))
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # logvar: log(sigma^2)
    return BCE + KLD


LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train(epoch):
    for di in range(opt.num_users):
        models[di].train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(dataloaders[di]):
            data, labels = data.to(device), labels.to(device)
            # labels = one_hot(labels, 10)
            labels = Variable(labels.type(LongTensor))
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
            recon_batch, mu, logvar = gmodel(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
            if i == 0:
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'imgFedCVAE/' + 'recon' + str(opt.num_users) + '_' + str(epoch) + '.png', nrow=n)

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
                save_image(sample.view(n_row**2, 1, 28, 28),
                        'RAWimgFedCVAE/' + str(opt.num_users) +'_' + str(epoch) + '.png', nrow=n_row, normalize=True)

torch.save(gmodel.state_dict(), 'models/save/' + str(opt.num_users)+ '_'+ str(opt.n_epochs)+ 'cvae.pt')