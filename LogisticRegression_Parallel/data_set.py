import torchvision
from torchvision import transforms
import torch
from torch.utils import data

import numpy as np
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset

class libsvmdata(Dataset):
    def __init__(self, args, train=True):
        if args.dataset == 'a8a':
            if train == True:
                svmdata = load_svmlight_file('SVMData/a8a.txt', n_features=123)
            else:
                svmdata = load_svmlight_file('SVMData/a8a_test.txt', n_features=123)
        elif args.dataset == 'a9a':
            if train == True:
                svmdata = load_svmlight_file('SVMData/a9a.txt', n_features=123)
            else:
                svmdata = load_svmlight_file('SVMData/a9a_test.txt', n_features=123)
        elif args.dataset == 'dna':
            if train == True:
                svmdata = load_svmlight_file('SVMData/dna.txt', n_features=180)
            else:
                svmdata = load_svmlight_file('SVMData/dna_test.txt', n_features=180)
        elif args.dataset == 'usps':
            if train == True:
                svmdata = load_svmlight_file('SVMData/usps.txt', n_features=256)
            else:
                svmdata = load_svmlight_file('SVMData/usps_test.txt', n_features=256)
        elif args.dataset == 'pendigits':
            if train == True:
                svmdata = load_svmlight_file('SVMData/pendigits.txt', n_features=16)
            else:
                svmdata = load_svmlight_file('SVMData/pendigits_test.txt', n_features=16)
        
        X, y = svmdata[0].toarray(), svmdata[1]
        y[y==-1] = 0
        if args.dataset == 'dna' or args.dataset == 'usps':
            y -= 1
        self.data = X
        self.targets = y
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
    
def set_data(args):
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(root='Dataset/', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='Dataset/', train=False, download=True, transform=transform_test)

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=1)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=True, num_workers=1)
    
    if args.dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root='Dataset/', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='Dataset/', train=False, download=True, transform=transform)

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=60000, shuffle=False, num_workers=1)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=1)
    
    if args.dataset == 'a8a' or args.dataset == 'a9a' or args.dataset == 'dna' or args.dataset == 'usps' or args.dataset == 'pendigits':
        train_set = libsvmdata(args, train=True)
        test_set = libsvmdata(args, train=False)

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False, num_workers=1)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
 
    return train_dataloader, test_dataloader