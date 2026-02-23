import matplotlib.pyplot as plt
import numpy as np


loss_gpu1 = np.load('results/loss_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu1_up.npy')
acc_gpu1 = np.load('results/acc_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu1_up.npy')
time_gpu1 = np.load('results/time_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu1_up.npy') / 1000

loss_gpu2 = np.load('results/loss_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu2_up.npy')
acc_gpu2 = np.load('results/acc_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu2_up.npy')
time_gpu2 = np.load('results/time_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu2_up.npy') / 1000

loss_gpu4 = np.load('results/loss_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu4_up.npy')
acc_gpu4 = np.load('results/acc_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu4_up.npy')
time_gpu4 = np.load('results/time_usps_GD_l2_theta0.10000_K20_momentum0.000_gpu4_up.npy') / 1000

print(time_gpu1)
print(time_gpu2)
print(time_gpu4)

print(acc_gpu1)
print(acc_gpu2)
print(acc_gpu4)

time_gpu4 = time_gpu4[0:31]
acc_gpu4 = acc_gpu4[0:31]
loss_gpu4 = loss_gpu4[0:31]
time_gpu2 = time_gpu2[0:24]
acc_gpu2 = acc_gpu2[0:24]
loss_gpu2 = loss_gpu2[0:24]
time_gpu1 = time_gpu1[0:15]
acc_gpu1 = acc_gpu1[0:15]
loss_gpu1 = loss_gpu1[0:15]

fig1 = plt.figure()
plt.plot(time_gpu1, acc_gpu1, linewidth=2, marker='o', markersize=10, markevery=1, color='firebrick', label='1 GPU')
plt.plot(time_gpu2, acc_gpu2, linewidth=2, marker='>', markersize=10, markevery=1, color='black', label='2 GPU')
plt.plot(time_gpu4, acc_gpu4, linewidth=2, marker='v', markersize=10, markevery=1, color='royalblue', label='4 GPU')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r'Running Time ($\times$1000 s)', fontsize=15)
plt.ylabel('Test Accuracy (%)', fontsize=15)
plt.legend(prop = {'size': 15})
fig1.suptitle('Test Accuracy', fontsize=20)
fig1.savefig('pictures/accuracy_time_usps_up.eps')

fig2 = plt.figure()
plt.plot(time_gpu1, loss_gpu1, linewidth=2, marker='o', markersize=10, markevery=1, color='firebrick', label='1 GPU')
plt.plot(time_gpu2, loss_gpu2, linewidth=2, marker='>', markersize=10, markevery=1, color='black', label='2 GPU')
plt.plot(time_gpu4, loss_gpu4, linewidth=2, marker='v', markersize=10, markevery=1, color='royalblue', label='4 GPU')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r'Running Time ($\times$1000 s)', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(prop = {'size': 15})
fig2.suptitle('Training Loss', fontsize=20)
fig2.savefig('pictures/loss_time_usps_up.eps')
plt.show()