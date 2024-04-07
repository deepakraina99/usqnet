import numpy as np
import matplotlib.pyplot as plt

import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import argparse

from models.usqnet import USQNET
from dataloader import load_data, USQDataset, collate_fn
from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', dest="data_dir", default="dataset/")
    parser.add_argument('--height', dest="height", default=224, type=int)
    parser.add_argument('--width', dest="width", default=224, type=int)
    parser.add_argument('--load_model', dest="load_model", default='outputs/model.pth')
    
    args = parser.parse_args()
    return args

def main(args):
    # load data
    cls_labels = ['1', '2', '3', '4', '5'] #[P, U, M, A, E]
    test_images, test_labels = load_data(img_dir=args.data_dir, cls_labels=cls_labels)

    # transforms
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Resize((args.width,args.height))])
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Resize((args.width,args.height))])

    # datasets
    test_ds = USQDataset(test_images, test_labels, transform=transform_val)

    # dataloaders
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # intialize model
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('device:', device)
    net = USQNET()
    net.load_state_dict(torch.load(args.load_model))
    net = net.to(device)

    # test model
    test_model(net, test_dl, cls_labels, device, save_dir=args.save_dir)        

if __name__ == "__main__":
    args = parse()
    main(args)

