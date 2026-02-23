import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

import functions

def Stratified_Integrator(x_net, optimizer, train_dataloader, args):

    loss_func = nn.CrossEntropyLoss()
    x_net.train()

    num_iter = args.K
    theta = args.theta

    train_iter = train_dataloader.__iter__()


    num_inner = int(len(train_dataloader) / num_iter)

    for iter_inner in range(num_inner):

        v = {}
        v_half = {}
        for name, param in x_net.named_parameters():
            v[name] = torch.zeros_like(param).to(args.device)
            v_half[name] = torch.zeros_like(param).to(args.device)
    
        for iter in range(num_iter):
            inputs, targets = train_iter.__next__()
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            u = torch.rand(1) * theta
            u = u.to(args.device)

            x_net_original = copy.deepcopy(x_net)


            for name, param in x_net.named_parameters():
                if args.norm == 'l2':
                    param.data = param.data + u * v[name]
                elif args.norm == 'l1':
                    param.data = param.data + u * torch.sign(v[name])
                elif args.norm == 'normalized':
                    param.data = param.data + u * v[name] / torch.linalg.norm(v[name])
                elif args.norm == 'coordinate':
                    index = torch.argmax(torch.abs(v[name]))
                    v_index = torch.zeros_like(v[name])
                    v_index[index] = torch.sign(v[name][index])
                    param.data = param.data + u * v_index
        
            optimizer.zero_grad()
            outputs = x_net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()

            params_x = {name: param for name, param in x_net.named_parameters()}
            params_x_original = {name: param for name, param in x_net_original.named_parameters()}

            for name in params_x:    
                v_half[name] = v[name] - theta / 2 * params_x[name].grad
                v[name] = v[name] - theta * params_x[name].grad
            
                if args.norm == 'l2':
                    params_x[name].data = params_x_original[name].data + theta * v_half[name]
                elif args.norm == 'l1':
                    params_x[name].data = params_x_original[name].data + theta * torch.sign(v_half[name])
                elif args.norm == 'normalized':
                    params_x[name].data = params_x_original[name].data + theta * v_half[name] / torch.linalg.norm(v_half[name])
                elif args.norm == 'coordinate':
                    index = torch.argmax(torch.abs(v_half[name]))
                    v_half_index = torch.zeros_like(v_half[name])
                    v_half_index[index] = torch.sign(v_half[name][index])
                    params_x[name].data = params_x_original[name].data + theta * v_half_index

    return x_net