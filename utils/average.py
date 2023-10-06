import copy
from collections import OrderedDict
import torch

def FedAvg_frozen_FE(args, wg, ws, wc):
    '''
    wg: global (previous) weights (ws_glob)
    ws: local weights (ws_local)
    wc: common weights (w_comm)
    '''
    w_com = wc
    
    '''
    Averaging non-common weights
    '''
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
            w_avg[j] = copy.deepcopy(wg[j]) # get weights from previous
            for k in wc.keys():
                w_avg[j][k] = w_com[k]
     
    return w_avg, w_com


def FedAvg_FE(args, wg, ws, wc):
    '''
    wg: global (previous) weights (ws_glob)
    ws: local weights (ws_local)
    wc: common weights (w_comm)
    '''
    w_com = OrderedDict()
    for k in wc.keys(): # initialization - debugged
        w_com[k] = 0*copy.deepcopy(wc[k])

    num = 0
    for j in range(args.num_models):
        num += len(ws[j])
    
    '''
    Averaging common feature extractor
    '''
    for k in wc.keys():
        for j in range(args.num_models):
            if ws[j]:
                for i in range(len(ws[j])):
                    w_com[k] += ws[j][i][k]
        w_com[k] = torch.div(w_com[k], num)

    '''
    Averaging non-common weights
    '''
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
            w_avg[j] = copy.deepcopy(wg[j]) # get weights from previous
            for k in wc.keys():
                w_avg[j][k] = w_com[k]
     
    return w_avg, w_com


def FedAvg_FE_raw(args, wg, ws):
    '''
    wg: global (previous) weights (ws_glob)
    ws: local weights (ws_local)
    '''

    '''
    Averaging non-common weights
    '''
    w_avg = [None for _ in range(args.num_models)]
    for j in range(args.num_models):
        if ws[j]:
            w_avg[j] = copy.deepcopy(ws[j][0])
            for k in w_avg[j].keys():
                for i in range(1, len(ws[j])):
                    w_avg[j][k] += ws[j][i][k]
                w_avg[j][k] = torch.div(w_avg[j][k], len(ws[j]))
        else:
            w_avg[j] = copy.deepcopy(wg[j]) # get weights from previous
     
    return w_avg


def FedAvg(ws):
    w_avg = copy.deepcopy(ws[0])
    for k in w_avg.keys():
        for i in range(1, len(ws)):
            w_avg[k] += ws[i][k]
        w_avg[k] = torch.div(w_avg[k], len(ws))
    return w_avg