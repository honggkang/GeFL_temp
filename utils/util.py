
# from math import ceil as up
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import logging


def test_img(net_g, datatest, args):
    net_g.eval()
    net_g.to(args.device)
    # testing
    test_loss = 0
    correct = 0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    with torch.no_grad():
      for idx, (data, target) in enumerate(data_loader):
          if 'cuda' in args.device:
              data, target = data.to(args.device), target.to(args.device)
          logits, log_probs = net_g(data)
          test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
          y_pred = log_probs.data.max(1, keepdim=True)[1]

          correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    net_g.to('cpu')
    return accuracy, test_loss


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        # info_file_handler.terminator = ""
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # console_handler.terminator = ""
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger