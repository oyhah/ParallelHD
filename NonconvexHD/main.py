import random
import torch
import torch.nn as nn
import numpy as np

from setting import args_parser
from logg import get_logger
from data_set import set_data
from resnet import ResNet18
from test import test
from AdaHD_Optimizer import HD, AdaHD

if __name__ == '__main__':

    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Dataset
    train_dataloader, test_dataloader = set_data(args)

    # Model
    x_net = ResNet18()
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

    loss_func = nn.CrossEntropyLoss()

    if args.method == 'GD':
        optimizer = torch.optim.SGD(x_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weigh_delay)
    elif args.method == 'Adam':
        optimizer = torch.optim.Adam(x_net.parameters(), lr=args.lr, betas=(args.momentum, args.second_moment), weight_decay=args.weigh_delay)
    elif args.method == 'HD':
        optimizer = HD(x_net.parameters(), lr=args.lr, weight_decay=args.weigh_delay, mean_duration=args.mean_duration, mu=args.mu)
    elif args.method == 'AdaHD':
        optimizer = AdaHD(x_net.parameters(), lr=args.lr, betas=(args.momentum, args.second_moment), weight_decay=args.weigh_delay, mean_duration=args.mean_duration, mu=args.mu)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for iter in range(args.epochs + 1):

        # Training

        for i, (inputs, targets) in enumerate(train_dataloader):

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            outputs = x_net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            
            optimizer.step()
            # scheduler.step()

        # Test
        
        test_loss, test_acc = test(x_net, test_dataloader, args)

        logger.info('Epoch:[{}]\tlr =\t{:.5f}\ttest_loss=\t{:.5f}\ttest_acc=\t{:.5f}'.
                            format(iter, args.lr, test_loss, test_acc))

        loss_results.append(test_loss)
        acc_results.append(test_acc)
    

    logger.info('finish training!')

    loss_results = np.array(loss_results)
    acc_results = np.array(acc_results)

    np.save('results/loss_%s_%s_lr%.5f_duration%.2f_momentum%.3f_mu%.2f.npy' % (args.dataset, args.method, args.lr, args.mean_duration, args.momentum, args.mu), loss_results)
    np.save('results/acc_%s_%s_lr%.5f_duration%.2f_momentum%.3f_mu%.2f.npy' % (args.dataset, args.method, args.lr, args.mean_duration, args.momentum, args.mu), acc_results)

