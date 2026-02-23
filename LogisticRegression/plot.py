import matplotlib.pyplot as plt
import numpy as np

acc_GD = np.load('result/acc_MNIST_GD_l2_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_GDM = np.load('result/acc_MNIST_GD_l2_lr0.01000_theta0.10000_K10_momentum0.900.npy', allow_pickle=True)
acc_GDN = np.load('result/acc_MNIST_GD_l2_lr0.01000_theta0.10000_K10_momentum0.900_nesterov.npy', allow_pickle=True)
acc_GD_l1 = np.load('result/acc_MNIST_GD_l1_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_GDM_l1 = np.load('result/acc_MNIST_GD_l1_lr0.01000_theta0.10000_K10_momentum0.900.npy', allow_pickle=True)
acc_GD_nor = np.load('result/acc_MNIST_GD_normalized_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_GDM_nor = np.load('result/acc_MNIST_GD_normalized_lr0.01000_theta0.10000_K10_momentum0.900.npy', allow_pickle=True)
acc_GD_coo = np.load('result/acc_MNIST_GD_coordinate_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_GDM_coo = np.load('result/acc_MNIST_GD_coordinate_lr0.01000_theta0.10000_K10_momentum0.900.npy', allow_pickle=True)

acc_LF = np.load('result/acc_MNIST_LF_l2_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_LF_l1 = np.load('result/acc_MNIST_LF_l1_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_LF_nor = np.load('result/acc_MNIST_LF_normalized_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_LF_coo = np.load('result/acc_MNIST_LF_coordinate_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)

acc_SI = np.load('result/acc_MNIST_SI_l2_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_SI_l1 = np.load('result/acc_MNIST_SI_l1_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_SI_nor = np.load('result/acc_MNIST_SI_normalized_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
acc_SI_coo = np.load('result/acc_MNIST_SI_coordinate_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)


loss_GD = np.load('result/loss_MNIST_GD_l2_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_GDM = np.load('result/loss_MNIST_GD_l2_lr0.01000_theta0.10000_K10_momentum0.900.npy', allow_pickle=True)
loss_GDN = np.load('result/loss_MNIST_GD_l2_lr0.01000_theta0.10000_K10_momentum0.900_nesterov.npy', allow_pickle=True)
loss_GD_l1 = np.load('result/loss_MNIST_GD_l1_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_GDM_l1 = np.load('result/loss_MNIST_GD_l1_lr0.01000_theta0.10000_K10_momentum0.900.npy', allow_pickle=True)
loss_GD_nor = np.load('result/loss_MNIST_GD_normalized_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_GDM_nor = np.load('result/loss_MNIST_GD_normalized_lr0.01000_theta0.10000_K10_momentum0.900.npy', allow_pickle=True)
loss_GD_coo = np.load('result/loss_MNIST_GD_coordinate_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_GDM_coo = np.load('result/loss_MNIST_GD_coordinate_lr0.01000_theta0.10000_K10_momentum0.900.npy', allow_pickle=True)

loss_LF = np.load('result/loss_MNIST_LF_l2_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_LF_l1 = np.load('result/loss_MNIST_LF_l1_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_LF_nor = np.load('result/loss_MNIST_LF_normalized_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_LF_coo = np.load('result/loss_MNIST_LF_coordinate_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)

loss_SI = np.load('result/loss_MNIST_SI_l2_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_SI_l1 = np.load('result/loss_MNIST_SI_l1_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_SI_nor = np.load('result/loss_MNIST_SI_normalized_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)
loss_SI_coo = np.load('result/loss_MNIST_SI_coordinate_lr0.01000_theta0.10000_K10_momentum0.000.npy', allow_pickle=True)

x_axis_GD = np.arange(len(acc_GD)) + 1
x_axis_LF = (np.arange(len(acc_LF)) + 1) * 10


# fig1 = plt.figure()
# plt.plot(x_axis_GD, acc_GD, linewidth=2, marker='o', markersize=8, markevery=10, color='firebrick', label='GD')
# plt.plot(x_axis_GD, acc_GDM, linewidth=2, marker='v', markersize=8, markevery=10, color='deepskyblue', label='GDM')
# plt.plot(x_axis_GD, acc_GDN, linewidth=2, marker='>', markersize=8, markevery=10, color='blue', label='GDN')
# plt.plot(x_axis_LF, acc_LF, linewidth=2, marker='^', markersize=8, markevery=2, color='red', label='HD-LF')
# plt.plot(x_axis_LF, acc_SI, linewidth=2, marker='*', markersize=8, markevery=2, color='darkorange', label='HD-SI')


# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Number of Rounds (K)', fontsize=15)
# plt.ylabel('Test accuracy', fontsize=15)
# plt.legend(prop = {'size': 12})
# fig1.suptitle('Loss', fontsize=20)
# fig1.savefig('picture/acc_algorithm_MNIST.eps')

# fig2 = plt.figure()
# plt.plot(x_axis_GD, loss_GD, linewidth=2, marker='o', markersize=8, markevery=10, color='firebrick', label='GD')
# plt.plot(x_axis_GD, loss_GDM, linewidth=2, marker='v', markersize=8, markevery=10, color='deepskyblue', label='GDM')
# plt.plot(x_axis_GD, loss_GDN, linewidth=2, marker='>', markersize=8, markevery=10, color='blue', label='GDN')
# plt.plot(x_axis_LF, loss_LF, linewidth=2, marker='^', markersize=8, markevery=2, color='red', label='HD-LF')
# plt.plot(x_axis_LF, loss_SI, linewidth=2, marker='*', markersize=8, markevery=2, color='darkorange', label='HD-SI')


# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Number of Rounds (K)', fontsize=15)
# plt.ylabel('Training Loss', fontsize=15)
# plt.legend(prop = {'size': 12})
# fig2.suptitle('Loss', fontsize=20)
# fig2.savefig('picture/loss_algorithm_MNIST.eps')

# plt.show()


fig1 = plt.figure()
plt.plot(x_axis_GD, acc_GD, linewidth=2, marker='v', markersize=8, markevery=20, color='deepskyblue', label='GD')
plt.plot(x_axis_GD, acc_GDM, linewidth=2, marker='>', markersize=8, markevery=20, color='blue', label='HB')
plt.plot(x_axis_LF, acc_LF, linewidth=2, marker='*', markersize=8, markevery=2, color='darkorange', label='HD-LF')
plt.plot(x_axis_LF, acc_SI, linewidth=2, marker='x', markersize=8, markevery=2, color='firebrick', label='HD-ST')


plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Number of Gradient Computations', fontsize=15)
plt.ylabel('Test accuracy (%)', fontsize=15)
plt.legend(prop = {'size': 12})
plt.grid()
fig1.suptitle('Test Accuracy', fontsize=20)
fig1.savefig('picture/acc_l2.eps')

fig2 = plt.figure()
plt.plot(x_axis_GD, loss_GD, linewidth=2, marker='v', markersize=8, markevery=20, color='deepskyblue', label='GD')
plt.plot(x_axis_GD, loss_GDM, linewidth=2, marker='>', markersize=8, markevery=20, color='blue', label='HB')
plt.plot(x_axis_LF, loss_LF, linewidth=2, marker='*', markersize=8, markevery=2, color='darkorange', label='HD-LF')
plt.plot(x_axis_LF, loss_SI, linewidth=2, marker='x', markersize=8, markevery=2, color='firebrick', label='HD-ST')


plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Number of Gradient Computations', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(prop = {'size': 12})
plt.grid()
fig2.suptitle('Training Loss', fontsize=20)
fig2.savefig('picture/loss_l2.eps')

plt.show()