import torch
import torch.nn as nn
import torch.nn.functional as F

class model_net(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

        if self.dataset == 'EMNIST':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, self.n_cls)
            self.weight_keys = [['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias']]


        if  self.dataset == 'FMNIST':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=1,  out_channels=4, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=4,  out_channels=12, kernel_size=5)
            self.fc1 = nn.Linear(12 * 4 * 4, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, self.n_cls)

            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias']]


        if self.dataset == 'CIFAR10':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, self.n_cls)
            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias']]

        if self.dataset == 'CIFAR100':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, self.n_cls)
            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias']]


    def forward(self, x):

        if self.dataset == 'EMNIST':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.dataset == 'FMNIST':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 12 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)


        if self.dataset == 'CIFAR10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.dataset == 'CIFAR100':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)


        return x