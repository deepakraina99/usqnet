import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn.model_selection import StratifiedShuffleSplit
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
    parser.add_argument('--num_epochs', dest="num_epochs", default=100, type=int)
    parser.add_argument('--folds', dest="folds", default=10, type=int)
    parser.add_argument('--lr', dest="lr", default=5e-3, type=float)
    parser.add_argument('--height', dest="height", default=224, type=int)
    parser.add_argument('--width', dest="width", default=224, type=int)
    parser.add_argument('--save_dir', dest="save_dir", default="outputs/")
    parser.add_argument('--optimizer', dest="optimizer", default="sgd")
    parser.add_argument('--batch_size', dest="batch_size", default=16, type=int)

    args = parser.parse_args()
    return args

def main(args):
    # load data
    cls_labels = ['1', '2', '3', '4', '5'] #[P, U, M, A, E]
    all_images, all_labels = load_data(img_dir=args.data_dir, cls_labels=cls_labels)
    
    # transforms
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Resize((args.width,args.height))])
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Resize((args.width,args.height))])

    # Intialize model
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('device:', device)
    net = USQNET() 
    print(net)
    net = net.to(device)

    # hyper-parameters
    if args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    criterion = nn.CrossEntropyLoss()

    # cross-validation
    sss = StratifiedShuffleSplit(n_splits=args.folds, test_size=0.1, random_state=SEED)
    sss.get_n_splits(all_images, all_labels)
    for fold, (train_data, val_data) in enumerate(sss.split(all_images, all_labels)):
        print('-*'*25)
        print('fold:{}/{}'.format(fold+1,args.folds))
        train_images, val_images = [all_images[i] for i in train_data], [all_images[i] for i in val_data]
        train_labels, val_labels = [all_labels[i] for i in train_data], [all_labels[i] for i in val_data]
        
        # datasets
        train_ds = USQDataset(train_images, train_labels, transform=transform_train)
        test_ds = USQDataset(val_images, val_labels, transform=transform_val)

        # dataloaders
        weighted_sampler = get_weighted_sampler(train_labels, cls_labels)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, sampler=weighted_sampler, collate_fn=collate_fn)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
        dataloaders_dict = {'train': train_dl, 'val': test_dl}

        # training loop
        net, net_hist = train_model(net, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=args.num_epochs, fold=fold, device=device, model_dir=args.save_dir)
            
if __name__ == "__main__":
    args = parse()
    main(args)

