import random
import torch
import numpy as np
import time, os
import copy

import torch.linalg

from setting import args_parser
from logg import get_logger
import functions
from algorithms.GD import Gradient_Descent
from algorithms.LeapFrog import Leap_Frog
from algorithms.Stratified import Stratified_Integrator
from data_set import set_data
from models import model_net
from model_LR import LogisticRegressionModel
from resnet import ResNet18
from test import test

import torch.multiprocessing as mp
import torch.nn as nn



def ParallelInnerLoop(args, x_net, train_dataloader, input_dim, output_dim):

    mp.set_start_method('forkserver', force=True)
    manager = mp.Manager()
    queues = mp.Queue(), mp.Queue()
    queue_x = mp.Queue()

    processes = []
    num_processes = args.num_processes #torch.cuda.device_count()

    for rank in range(-1, num_processes):
        p = mp.Process(target=run, args=(args, rank, num_processes, queues, queue_x, x_net, train_dataloader, input_dim, output_dim))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()    # wait for all subprocesses to finish

    # x_net = queue_x.get()
    # x_net.load_state_dict(x_net_rec.state_dict())
    
    # return x_net


def run(args, rank, num_processes, queues, queue_x, x_net, train_dataloader, input_dim, output_dim):

    loss_func = nn.CrossEntropyLoss()
    lr = (args.theta)**2
    theta = args.theta
    threshold = 0.1

    if rank == -1:
        device = f'cuda:0'
        # Model
        nets = [LogisticRegressionModel(input_dim, output_dim).to(device) for i in range(num_processes + 1)]
        vs = []
        # x_net_send = copy.deepcopy(x_net)
        for i in range(num_processes):
            nets[i].load_state_dict(x_net.state_dict())
        
        for i in range(num_processes + 1):
            v = {}
            for name, param in x_net.named_parameters():
                v[name] = torch.zeros_like(param.data)
            vs.append(v)

        # Parallel training        
        window_start = 0
        window_size = torch.tensor(num_processes)
        ite = 0
        # train_iter = train_dataloader.__iter__()
        while window_start < args.K:
            # print('This is the main process', os.getpid())

            w_0 = {name: param for name, param in nets[0].named_parameters()}
            v_0 = vs[0]

            for i in range(window_size):
                # inputs, targets = train_iter.__next__()
                queues[0].put( (nets[i].state_dict(), i) )
            
            grad_all = [None for _ in range(window_size)]

            for i in range(window_size):
                info = queues[1].get()
                grad_rec, id = info
                grad_all[id] = {name: value.to(device) for name, value in grad_rec.items()}
                del info
                # print(f'Grad id is: {id}')
            
            torch.cuda.synchronize()

            for i in range(1, window_size + 1):
                for name, param in nets[i].named_parameters():
                    grad = 0
                    for j in range(i):
                        grad += grad_all[j][name]
                    vs[i][name] = v_0[name] - theta * grad
            
            error_window = torch.zeros(window_size)
            for i in range(1, window_size + 1):
                for name, param in nets[i].named_parameters():

                    # grad = 0
                    # for j in range(i):
                    #     grad += (i - j) * grad_all[j][name]

                    param_last = param.data.clone()

                    update = 0
                    for j in range(1, i + 1):

                        if args.norm == 'l2':
                            v_update = vs[j][name]
                        elif args.norm == 'l1':
                            v_update = torch.sign(vs[j][name])
                        elif args.norm == 'normalized':
                            v_update = vs[j][name] / torch.linalg.norm(vs[j][name])
                        elif args.norm == 'coordinate':
                            v_update_size = vs[j][name].size()
                            if len(v_update_size) == 2:
                                size1, size2 = v_update_size[0], v_update_size[1]
                                v_update_flat = vs[j][name].flatten()
                                index = torch.argmax(torch.abs(v_update_flat))
                                v_update_index = torch.zeros_like(v_update_flat)
                                v_update_index[index] = torch.sign(v_update_flat[index])
                                v_update_index = v_update_index.view(size1, size2)
                            elif len(v_update_size) == 1:
                                index = torch.argmax(torch.abs(vs[j][name]))
                                v_update_index = torch.zeros_like(vs[j][name])
                                v_update_index[index] = torch.sign(vs[j][name][index])
                            v_update = v_update_index

                        update += v_update
                    
                    param.data = w_0[name].data + theta * update

                    # if args.norm == 'l2':
                    #     param.data = w_0[name].data - lr * grad
                    # elif args.norm == 'l1':
                    #     grad = torch.sign(grad)
                    #     param.data = w_0[name].data - theta * grad
                    # elif args.norm == 'normalized':
                    #     grad = grad / torch.linalg.norm(grad)
                    #     param.data = w_0[name].data - theta * grad
                    # elif args.norm == 'coordinate':
                    #     grad_size = grad.size()
                    #     if len(grad_size) == 2:
                    #         size1, size2 = grad_size[0], grad_size[1]
                    #         grad_flat = grad.flatten()
                    #         index = torch.argmax(torch.abs(grad_flat))
                    #         grad_index = torch.zeros_like(grad_flat)
                    #         grad_index[index] = torch.sign(grad_flat[index])
                    #         grad_index = grad_index.view(size1, size2)
                    #     elif len(grad_size) == 1:
                    #         index = torch.argmax(torch.abs(grad))
                    #         grad_index = torch.zeros_like(grad)
                    #         grad_index[index] = torch.sign(grad[index])
                        
                    #     grad = grad_index
                    #     param.data = w_0[name].data - theta * grad

                    error_window[i - 1] = torch.max( torch.norm(torch.abs(param.data - param_last)) / torch.norm(param_last) , error_window[i - 1] ) #torch.numel(param_last)
            
            print(error_window)
            
            (id_error,) = torch.where(error_window > threshold)

            if id_error.numel() == 0:
                stride = window_size
            else:
                stride = torch.min(id_error)

            # stride = 1

            for  i in range(window_size - stride + 1):
                nets[i].load_state_dict(nets[i + stride].state_dict())
                vs[i] = vs[i + stride]
            
            net_state = nets[window_size].state_dict()
            for i in range(window_size - stride + 1, window_size + 1):
                nets[i].load_state_dict(net_state)

            window_start += stride
            window_size = torch.minimum(window_size, args.K - window_start)
            ite += 1

        
        print('Internal iteration is:', ite)
        
        # Finish the inner loop training
        for _ in range(num_processes):
            queues[0].put(None)
        
        x_net_state = nets[window_size].state_dict().copy()
        x_net.load_state_dict(x_net_state)
        # queue_x.put(x_net_send)
            
    else:
        device = f'cuda:{rank}'
        net_local = LogisticRegressionModel(input_dim, output_dim).to(device)
        optimizer_local = torch.optim.SGD(net_local.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weigh_delay)

        # print(f'This is initial process: {os.getpid()}')

        while True:
            info = queues[0].get()
            if info is None:
                del info
                return
            
            net_rec_state, id = info
            del info
            # print(f'This is process: {os.getpid()}, received id: {id}')

            net_local.load_state_dict(net_rec_state)

            train_iter = train_dataloader.__iter__()

            inputs, targets = train_iter.__next__()
            inputs = inputs.view(-1, input_dim).requires_grad_()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer_local.zero_grad()
            outputs = net_local(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()

            grad_local = {name: param.grad for name, param in net_local.named_parameters()}

            queues[1].put( (grad_local, id) )
            # time.sleep(3)


if __name__ == '__main__':

    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Dataset
    train_dataloader, test_dataloader = set_data(args)

    # Model
    if args.dataset == 'MNIST':
        input_dim = 28 * 28
        output_dim = 10
    elif args.dataset == 'CIFAR10':
        input_dim = 3 * 32 * 32
        output_dim = 10
    elif args.dataset == 'a8a' or args.dataset == 'a9a':
        input_dim = 123
        output_dim = 2
    elif args.dataset == 'dna':
        input_dim = 180
        output_dim = 3
    elif args.dataset == 'usps':
        input_dim = 256
        output_dim = 10
    elif args.dataset == 'pendigits':
        input_dim = 16
        output_dim = 10
    x_net = LogisticRegressionModel(input_dim, output_dim)
    x_net = x_net.to(args.device)


    # Training
    logger = get_logger(args.filepath)
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')

    logger.info('start training!')

    loss_results = []
    acc_results = []
    time_results = []

    time_start = time.time()

    for iter in range(args.epochs + 1):

        # Training

        ParallelInnerLoop(args, x_net, train_dataloader, input_dim, output_dim)

        # Test
        
        test_loss, test_acc = test(x_net, test_dataloader, args, input_dim, output_dim)

        time_now = time.time()
        time_ite = time_now - time_start

        logger.info('Epoch:[{}]\tlr =\t{:.5f}\ttest_loss=\t{:.5f}\ttest_acc=\t{:.5f}\ttime=\t{:.5f}'.
                            format(iter, args.lr, test_loss, test_acc, time_ite))

        loss_results.append(test_loss)
        acc_results.append(test_acc)
        time_results.append(time_ite)
    

    logger.info('finish training!')

    loss_results = np.array(loss_results)
    acc_results = np.array(acc_results)
    time_results = np.array(time_results)

    np.save('result/loss_%s_%s_%s_theta%.5f_K%d_momentum%.3f_gpu%d_up.npy' % (args.dataset, args.method, args.norm, args.theta, args.K, args.momentum, args.num_processes), loss_results)
    np.save('result/acc_%s_%s_%s_theta%.5f_K%d_momentum%.3f_gpu%d_up.npy' % (args.dataset, args.method, args.norm, args.theta, args.K, args.momentum, args.num_processes), acc_results)
    np.save('result/time_%s_%s_%s_theta%.5f_K%d_momentum%.3f_gpu%d_up.npy' % (args.dataset, args.method, args.norm, args.theta, args.K, args.momentum, args.num_processes), time_results)