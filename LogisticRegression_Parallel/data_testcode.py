import numpy as np
# import os.path as osp
from sklearn.datasets import load_svmlight_file
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

class libsvmdata(Dataset):
    def __init__(self, name, train=True):
        if name == 'a8a':
            if train == True:
                svmdata = load_svmlight_file('SVMData/a8a.txt')
            else:
                svmdata = load_svmlight_file('SVMData/a8a_test.txt')
        elif name == 'a9a':
            if train == True:
                svmdata = load_svmlight_file('SVMData/a9a.txt')
            else:
                svmdata = load_svmlight_file('SVMData/a9a_test.txt')
        
        X, y = svmdata[0].toarray(), svmdata[1]
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
    
train_set = libsvmdata(name='a8a', train=True)
test_set = libsvmdata(name='a8a', train=False)

print(len(test_set))

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False, num_workers=1)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)

test_iter = iter(test_dataloader)

inputs, targets = next(test_iter)

print(inputs.shape())