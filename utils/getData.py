import numpy as np
from torchvision import datasets, transforms
from math import sqrt

# import lasagne
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def mnist_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users, seed):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(seed)

    num_shards, num_imgs = 200, 300 # 2 (class) x 100 (users), 2 x 300 (imgs) for each client
    # {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy() # targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # label-wise sort
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return

def cifar_iid(dataset, num_users, seed):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(seed)
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(args, dataset):
    """
    Sample non-I.I.D client data from CIFAR dataset 50000
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(args.rs)

    num_shards, num_imgs = args.num_users * args.class_per_each_client, int(50000/args.num_users/args.class_per_each_client)
    # {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 5000, 6: 5000, 7: 5000, 8: 5000, 9: 5000}
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # label-wise sort
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(args.num_users):
        rand_set = set(np.random.choice(idx_shard, args.class_per_each_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
    

def getDataset(args):
    if args.dataset =='cifar10':
        ## CIFAR
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # transforms.Resize(256), transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('/home/hong/NeFL/.data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('/home/hong/NeFL/.data/cifar', train=False, download=True, transform=transform_test)

    elif args.dataset == 'svhn':
        ### SVHN
        transform_train = transforms.Compose([
                transforms.Pad(padding=2),
                transforms.RandomCrop(size=(32, 32)),
                transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
        dataset_train = datasets.SVHN('/home/hong/NeFL/.data/svhn', split='train', download=True, transform=transform_train)
        dataset_test = datasets.SVHN('/home/hong/NeFL/.data/svhn', split='test', download=True, transform=transform_test)

    elif args.dataset == 'stl10':
        ### STL10
        transform_train = transforms.Compose([
                        transforms.RandomCrop(96, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2471, 0.2435, 0.2616])
                    ])
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2471, 0.2435, 0.2616])
                ])
        dataset_train = datasets.STL10('/home/hong/NeFL/.data/stl10', split='train', download=True, transform=transform_train)
        dataset_test = datasets.STL10('/home/hong/NeFL/.data/stl10', split='test', download=True, transform=transform_test)

    ### downsampled ImageNet
    # imagenet_data = datasets.ImageNet('/home')
        
    ### Flowers
    # tranform_train = transforms.Compose([
    #                                     #   transforms.RandomRotation(30),
    #                                     #   transforms.RandomResizedCrop(224),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(), 
    #                                       transforms.Normalize([0.485, 0.456, 0.406], 
    #                                                            [0.229, 0.224, 0.225])
    #                                      ])
        
    # tranform_test = transforms.Compose([
    #                                     #   transforms.Resize(256),
    #                                     #   transforms.CenterCrop(224),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406], 
    #                                                            [0.229, 0.224, 0.225])
    #                                      ])

    # dataset_train = datasets.Flowers102('/home/hong/NeFL/.data/flowers102', download=True, transform=tranform_train)
    # dataset_test = datasets.Flowers102('/home/hong/NeFL/.data/flowers102', split='test', download=True, transform=tranform_test)
    # split='train',

    ### Food 101
    # tranform_train = transforms.Compose([transforms.RandomRotation(30),
    #                                        transforms.RandomResizedCrop(224),
    #                                        transforms.RandomHorizontalFlip(),ImageNetPolicy(),
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize([0.485, 0.456, 0.406],
    #                                                             [0.229, 0.224, 0.225])])

    # tranform_test = transforms.Compose([transforms.Resize(255),
    #                                       transforms.CenterCrop(224),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406],
    #                                                            [0.229, 0.224, 0.225])])
    # dataset_train = datasets.Food101('/home/hong/NeFL/.data/food101', split='train', download=True, transform=tranform_train)
    # dataset_test = datasets.Food101('/home/hong/NeFL/.data/food101', split='test', download=True, transform=tranform_test)
    return dataset_train, dataset_test