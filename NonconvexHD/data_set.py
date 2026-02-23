import torchvision
from torchvision import transforms
import torch
from torch.utils import data

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

    
    return train_dataloader, test_dataloader