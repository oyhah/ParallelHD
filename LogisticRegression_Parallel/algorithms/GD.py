import torch
import torch.nn.functional as F
import torch.nn as nn

import functions

def Gradient_Descent(x_net, optimizer, train_dataloader, args):
    loss_func = nn.CrossEntropyLoss()

    x_net.train()

    for i, (inputs, targets) in enumerate(train_dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        optimizer.zero_grad()
        outputs = x_net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

    return x_net