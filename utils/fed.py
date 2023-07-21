from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, learning_rate):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits, log_probs = net(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdate_Gen(object):
    def __init__(self, args, net_com, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(FeatureDataset(net_com, dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, learning_rate):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits, log_probs = net(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class FeatureDataset(Dataset):    
    def __init__(self, model, dataset, idxs):
        self.dataset = model(dataset)
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
'''
objective:
1. outputs features of images a client have
2. trains a generative model with features
'''    
    
def FedAvg_FE(args, wg, ws, wc):
    '''
    wg: global (previous) weights (ws_glob)
    ws: local weights (ws_local)
    wc: common weights (w_comm)
    '''
    w_com = copy.deepcopy(wc)
    num = 0
    for j in range(args.num_models):
        num += len(ws[j])
    
    for k in wc.keys():
        for j in range(args.num_models):
            if ws[j]:
                for i in range(len(ws[j])):
                    w_com[k] += ws[j][i][k]
        w_com[k] = torch.div(w_com[k], num)    

    w_avg = [None for _ in range(args.num_models)]
    for j in range(args.num_models):
        if ws[j]:
            w_avg[j] = copy.deepcopy(ws[j][0])
            for k in w_avg[j].keys():
                if k not in wc.keys():
                    for i in range(1, len(ws[j])):
                        w_avg[j][k] += ws[j][i][k]
                    w_avg[j][k] = torch.div(w_avg[j][k], len(ws[j]))
                else:
                    w_avg[j][k] = w_com[k]
        else:
            w_avg[j] = copy.deepcopy(wg[j])
            for k in wc.keys():
                w_avg[j][k] = w_com[k]
     
    return w_avg, w_com 