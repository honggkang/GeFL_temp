import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from collections import defaultdict
from functools import reduce
# import ipdb
import time
import matplotlib.pyplot as plt
import os 

def calc_cg_score(
    _dataset, device, *args, 
    rep_num=1, unbalance_ratio=1, 
    sub_term=False
):

    vi_base = { # a = first term of Equation 6, b = second term of Equation 6
        "vi": np.zeros((len(_dataset))),
        "ab": np.zeros((len(_dataset))), 
        "a2": np.zeros((len(_dataset))),
        "b2": np.zeros((len(_dataset))),
        "times": np.zeros((len(_dataset)))
    }

    with torch.no_grad():
        # Repeting calculation
        for _ in range(rep_num):
            dataset = defaultdict(list)
            data_idx = defaultdict(list)

            # Load and normalize data
            for j in range(len(_dataset)):
                data, label = _dataset[j]
                data_unnormed = torch.flatten(data).unsqueeze(0).type(torch.DoubleTensor)
                data_normed = data_unnormed/torch.norm(data_unnormed)
                label = label.item() if torch.is_tensor(label) else label

                # dataset : defaultdict({label: data_list}) ex. {{1: [data tensor]}}
                # data_idx : defaultdict({label: index of data sample})
                dataset[label].append(torch.flatten(data_normed).unsqueeze(0)) # [1, 784]
                data_idx[label].append(j) 

            
            new_dataset = {}
            # key : label, data_list : 각 label에 속하는 data sample들의 list, list of tensors
            # data_list : len 6000 (for FMNIST)
            for key, data_list in dataset.items():
                new_dataset[key] = np.array(data_list, dtype=object) # (6000, ) and each one is tensor with size [1, 784]
                data_idx[key] = np.array(data_idx[key]) # 각 class에 해당하는 sample의 index 
            dataset = new_dataset

            # Calculate CG-score in each class
            for curr_label, data_list in dataset.items():
                curr_num = len(data_list)

                # shuffle data index in each class 
                chosen_curr_idx = np.random.choice(range(len(data_list)), curr_num, replace=False)
                chosen_curr_list = data_list[chosen_curr_idx]

                # Sub-sample another class examples
                another_labels = [label for label in dataset if label != curr_label]
                # reduce 함수 : numpy array of data samples of another labels 
                another_list = reduce(
                    lambda acc, idx: np.concatenate((acc, dataset[idx])), another_labels, np.array([])) # 54000
                another_num = min(int(curr_num * unbalance_ratio), len(another_list)) # 6000
                chosen_another_list = another_list[
                    np.random.choice(range(len(another_list)), another_num, replace=False)] # pick 6000 samples of another classes

                # Make gram matrix H^\infty
                a = torch.cat(
                        list(np.concatenate((chosen_curr_list, chosen_another_list))), 0
                    ).type(torch.DoubleTensor).to(device) # concatenate curr class data & same size another class data => [12000, 784]
                y = torch.Tensor(
                        [1 for _ in range(curr_num)] + [-1 for _ in range(another_num)]
                    ).type(torch.DoubleTensor).to(device) # assign label 1 for curr_num and label -1 for another_num 

                H_inner = torch.matmul(a, a.transpose(0, 1)) # 12000 x 12000
                del a
                H = H_inner*(np.pi-torch.acos(H_inner))/(2*np.pi) # equation (1)
                del H_inner
                H.fill_diagonal_(1/2)

                invH = torch.inverse(H) # takes long time 
                del H
                
                # complexity score of whole dataset 
                original_error = y@(invH@y) 

                vi_class = defaultdict(list)

                # Calculate CG-score at each example 
                for k in range(curr_num):
                    A_with_row = torch.cat((invH[:, :k], invH[:, (k+1):]), axis=1) # exclude column of corresponding sample (앞에 sample들이 curr_label에 속함)
                    A = torch.cat((A_with_row[:k, :], A_with_row[(k+1):, :]), axis=0) # explude row as well
                    B = A_with_row[k, :].unsqueeze(0) 
                    del A_with_row
                    D = invH[k, k]

                    invH_mi = A - (B.T@B)/D # equation (4)
                    y_mi = torch.cat((y[:k], y[(k+1):])) # label vector except one sample 

                    vi_class['vi'].append((original_error - y_mi@(invH_mi@y_mi)).item())
                    y_i = y[k]
                    if sub_term:
                        vi_class['ab'].append((y_i*B@y_mi).item())
                        vi_class['a2'].append((((B@y_mi)**2)/D).item())
                        vi_class['b2'].append((((y_i)**2)*D).item())

                    del A, B, invH_mi, y_mi

                for keys, values in vi_class.items():
                    vi_base[keys][data_idx[curr_label][chosen_curr_idx]] += np.array(values) if (keys == "vi" or sub_term) else 0
                vi_base["times"][data_idx[curr_label][chosen_curr_idx]] += 1

                del invH
            
                torch.cuda.empty_cache()

    vi = {
        "vi": np.zeros((len(_dataset))),
        "ab": np.zeros((len(_dataset))),
        "a2": np.zeros((len(_dataset))),
        "b2": np.zeros((len(_dataset))),
    }
    
    # average over multiple runs
    for keys, values in vi_base.items():
        if keys == "times":
            continue
        vi[keys] = values / np.where(vi_base["times"] > 0, vi_base["times"], 1)

    return vi if sub_term else vi["vi"], data_idx


def pruning(dataset, pruning_ratio, rep_num, device, from_low=True):
    
    # Compute CG Score for synthetic features
    data_val_start = time.time()
    measure, data_idx = calc_cg_score(dataset, device, rep_num=rep_num)
    data_val_end = time.time()
    print("Time for calculating CG score: ", data_val_end - data_val_start)

    # Data pruning
    pruning_idx = []
    
    if from_low == True:
        measure = - measure # CG-score 낮은 애들부터 없애기 위함 
       
    for _, idx in data_idx.items():
        label_measure = measure[idx] 
        label_measure_rank = label_measure.argsort()
        thres_idx = int(len(label_measure)*(1-pruning_ratio))
        pruning_idx += list(np.array(idx)[label_measure_rank[:thres_idx]]) # pruning_idx : 남는 애들 
    print('Pruning index: ',len(pruning_idx))
    
    # Generate dataset after pruning
    dataset = torch.utils.data.Subset(dataset, pruning_idx)
   
    return dataset, measure, data_idx


def pruning_only_syn(dataset, syn_dataset, pruning_ratio, rep_num, device, syn_size, from_low=True):
    
    # Compute CG Score for synthetic features
    data_val_start = time.time()
    measure, data_idx = calc_cg_score(dataset, device, rep_num=rep_num)
    print('Repeated for '+ str())
    data_val_end = time.time()
    print("Time for calculating CG score: ", data_val_end - data_val_start)
    
    # Data pruning
    pruning_idx = []
    
    syn_measure = measure[:syn_size]
    real_measure = measure[syn_size:]
    
    plt.figure()
    ax = plt.gca()
    ax.set_xlim([0, 5])
    plt.hist(syn_measure, 50, density=True, label='synthetic', alpha=0.6)
    plt.hist(real_measure, 50, density=True, label='real', alpha=0.6)
    plt.title('histogram of CG-score')
    plt.xlabel('CG-score')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/real and synthetic.png')
    
    # extract data idx of synthetic features
    syn_data_idx = {}
    for class_idx, idx in data_idx.items():
        idx = np.array([id for id in idx if id < syn_size])
        syn_data_idx[class_idx] = idx
    
    # prune synthetic feature dataset 
    for _, idx in syn_data_idx.items():
        if from_low == True:
            label_measure = - syn_measure[idx] 
        else:
            label_measure = syn_measure[idx]
        label_measure_rank = label_measure.argsort()
        thres_idx = int(len(label_measure)*(1-pruning_ratio))
        pruning_idx += list(np.array(idx)[label_measure_rank[:thres_idx]]) # pruning_idx : 남는 애들 
    print('Pruning index: ',len(pruning_idx))
    
    # Generate dataset after pruning
    syn_dataset = torch.utils.data.Subset(syn_dataset, pruning_idx)
    return syn_dataset, syn_measure, syn_data_idx

def mean_cg_score(measure, data_idx, save_img=True, img_path='./figs/'):
    
    mean_dict = {}
    std_dict = {}
    
    mean_dict['all'] = np.mean(measure, axis=0)
    std_dict['all'] = np.std(measure, axis=0)

    if save_img == True:  
        
        # plot histogram of CG-score
        if not os.path.exists(img_path):
            os.makedirs(img_path)  
        
        plt.figure()
        plt.hist(measure, 50, density=True, label='All classes')
        plt.title('histogram of CG-score')
        plt.xlabel('CG-score')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.legend()
        plt.savefig(img_path + 'histogram.png')

    # plot histogram of each class
    for class_idx, idx in data_idx.items():
        label_measure = measure[idx]
        mean_dict[class_idx] = np.mean(label_measure, axis=0)
        std_dict[class_idx] = np.std(label_measure, axis=0)

        if save_img == True:
            plt.figure()
            plt.hist(label_measure, 50, density=True, label='Class '+str(class_idx))
            plt.title('histogram of CG-score')
            plt.xlabel('CG-score')
            plt.ylabel('Probability')
            plt.grid(True)
            plt.legend()
            plt.savefig(img_path + 'histogram_class_'+str(class_idx)+'.png')
    
    if save_img == True:
        plt.figure()
        for class_idx, _ in data_idx.items():
            x = mean_dict[class_idx]
            y = std_dict[class_idx]
            plt.scatter(x, y)
            plt.text(x, y, str(class_idx), va='bottom', ha='center')
        plt.xlabel('Mean of CG-score')
        plt.ylabel('Std of CG-score')
        plt.grid(True)
        plt.savefig(img_path + 'Distribution of CG-score.png')
    
    # mean and std dictionary 
    return mean_dict, std_dict
