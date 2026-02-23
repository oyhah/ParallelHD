import random
import torch
import numpy as np

from setting import args_parser
from logg import get_logger
from algorithms.GD import Gradient_Descent
from algorithms.LeapFrog import Leap_Frog
from algorithms.Stratified import Stratified_Integrator
from data_set import set_data
from model_LR import LogisticRegressionModel
from test import test

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

    optimizer = torch.optim.SGD(x_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weigh_delay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for iter in range(args.epochs + 1):

        # Training

        if args.method == 'GD':
            x = Gradient_Descent(x_net, optimizer, train_dataloader, args)
        elif args.method == 'LF':
            x = Leap_Frog(x_net, optimizer, train_dataloader, args)
        elif args.method == 'SI':
            x = Stratified_Integrator(x_net, optimizer, train_dataloader, args)

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

    np.save('result/loss_%s_%s_%s_theta%.5f_K%d_momentum%.3f.npy' % (args.dataset, args.method, args.norm, args.theta, args.K, args.momentum), loss_results)
    np.save('result/acc_%s_%s_%s_theta%.5f_K%d_momentum%.3f.npy' % (args.dataset, args.method, args.norm, args.theta, args.K, args.momentum), acc_results)

