import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=80, help="rounds of training")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="parameter of momentum")
    parser.add_argument('--weigh_delay', type=float, default=0, help="weigh_delay in GD")

    parser.add_argument('--method', type=str, default='GD', help='method name')
    parser.add_argument('--num_processes', type=int, default='4', help='Number of parallel processes')

    # Parameter in Integrators
    parser.add_argument('--theta', type=float, default=0.1, help='step size in integrator')
    parser.add_argument('--K', type=int, default=20, help='Number of inner iterations in integrator')
    parser.add_argument('--norm', type=str, default='l2', help='Form of function of v')

    # other arguments
    parser.add_argument('--dataset', type=str, default='MNIST', help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=23, help='random seed (default: 23)')
    parser.add_argument('--filepath', type=str, default='filepath', help='whether error accumulation or not')
    args = parser.parse_args()

    return args