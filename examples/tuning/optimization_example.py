import numpy as np
import matplotlib.pyplot as plt
import time

from Optimization._optimization import optimization
from Optimization._interferometer import Interferometer
from Learning.Network import *
from Optimization._optimization import func_fidelity, derivative_func_fidelity, func_frobenius, derivative_func_frobenius
from Optimization._optimization import func_weak, derivative_func_weak, func_sst, derivative_func_sst
from funcs_for_matrices.funcs_for_matrices import get_random_phase, create_fourier_matrix

# plt.style.use('classic')

start_time = time.time()

N = 4
counts_of_epochs = 300
# file_name = '/home/sergey/QLearn_save/goal_matrices.txt'
file_name = None

func, grad_func = func_frobenius, derivative_func_frobenius
label = r'$J_{FR}$'
if label == r'$J_{INF}$':
    func, grad_func = func_fidelity, derivative_func_fidelity
if label == r'$J_{FR}$':
    func, grad_func = func_frobenius, derivative_func_frobenius
if label == r'$J_{W}$':
    func, grad_func = func_weak, derivative_func_weak
if label == r'$J_{SST}$':
    func, grad_func = func_sst, derivative_func_sst

# F_target = get_random_phase(N)
# Fl = create_list_fl(F_target, N)
# Ul = inter.list_U
#
# target = interferometer(Fl, Ul, N)
# target = create_fourier_matrix(N)
# target = generator_unitary_matrix(N)
# inter.set_target(target)

m = 10

mean_bfgs = np.zeros(counts_of_epochs)
std_bfgs = np.zeros(counts_of_epochs)
mean_nm = np.zeros(counts_of_epochs)
std_nm = np.zeros(counts_of_epochs)

list_steps = []
list_bfgs = []
list_nm = []

epochs = []

# basis = 'fourier'
basis = 'stoch'

for i in range(m):
    inter = Interferometer(N, basis, file_name)
    F_target = get_random_phase(N)
    Fl = create_list_fl(F_target, N)
    Ul = inter.list_U
    # target = interferometer(Fl, Ul, N)
    target = generator_unitary_matrix(N)
    inter.set_target(target)
    steps, results = optimization(inter, counts_of_epochs, func, grad_func, 'L-BFGS-B')
    list_steps.append(steps)
    list_bfgs.append(results)

    mean_bfgs += results
    epochs = steps

for i in range(m):
    inter = Interferometer(N, basis, file_name)
    F_target = get_random_phase(N)
    Fl = create_list_fl(F_target, N)
    Ul = inter.list_U
    # target = interferometer(Fl, Ul, N)
    target = generator_unitary_matrix(N)
    inter.set_target(target)
    steps, results = optimization(inter, counts_of_epochs, func, grad_func,
                                                    'Nelder-Mead')
    list_steps.append(steps)
    list_nm.append(results)

    mean_nm += results
    epochs = steps

mean_bfgs = mean_bfgs / m
mean_nm = mean_nm / m

for i in range(m):
    std_bfgs += (list_bfgs[i] - mean_bfgs) ** 2
    std_nm += (list_nm[i] - mean_nm) ** 2

std_bfgs = (std_bfgs / (m - 1)) ** 0.5
std_nm = (std_nm / (m - 1)) ** 0.5

delta_time = time.time() - start_time
print('--- %s seconds ---' % delta_time)
print('--- %s seconds ---' % (delta_time / m))

fig, ax = plt.subplots()
plt.plot(epochs, mean_bfgs, color='green', lw=2, label='BFGS')
# plt.fill_between(epochs, mean_bfgs - std_bfgs, mean_bfgs + std_bfgs, color='#CCCCCC')
plt.plot(epochs, mean_nm, color='red', lw=2, label='Nelder-Mead')
# plt.fill_between(epochs, mean_nm - std_nm,
#                   mean_nm + std_nm, color='#CCCCCC')
plt.tick_params(which='major', direction='in')
plt.tick_params(which='minor', direction='in')
# lower left
# lower right
# upper left
# upper right
plt.legend(loc="upper right")
ax.grid()
ax.minorticks_off()
plt.xlim(0, counts_of_epochs - 1)
plt.yscale('log')
# plt.ylim(0.00001, 3.0)
plt.xlabel('Итерации алгоритма оптимизации', fontsize=11)
plt.ylabel(label, fontsize=15)
plt.title('N = '+str(N))
plt.show()
