import torch
import torch.nn.functional as F
import torch.nn as nn

import functions

def Leap_Frog(x_net, optimizer, train_dataloader, args):

    loss_func = nn.CrossEntropyLoss()
    x_net.train()

    num_iter = args.K
    theta = args.theta

    train_iter = train_dataloader.__iter__()

    num_inner = int(len(train_dataloader) / num_iter)

    for iter_inner in range(num_inner):

        v = {}
        for name, param in x_net.named_parameters():
            v[name] = torch.zeros_like(param).to(args.device)

        # v = v.to(args.device)
    
        for iter in range(num_iter):
            inputs, targets = train_iter.__next__()
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            optimizer.zero_grad()
            outputs = x_net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            
            for name, param in x_net.named_parameters():
                v[name] = v[name] - theta / 2 * param.grad

                if args.norm == 'l2':
                    param.data = param.data + theta * v[name]
                elif args.norm == 'l1':
                    param.data = param.data + theta * torch.sign(v[name])
                elif args.norm == 'normalized':
                    param.data = param.data + theta * v[name] / torch.linalg.norm(v[name])
                elif args.norm == 'coordinate':
                    index = torch.argmax(torch.abs(v[name]))
                    v_index = torch.zeros_like(v[name])
                    v_index[index] = torch.sign(v[name][index])
                    param.data = param.data + theta * v_index

            optimizer.zero_grad()
            outputs = x_net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()

            for name, param in x_net.named_parameters():
                v[name] = v[name] - theta / 2 * param.grad

    return x_net