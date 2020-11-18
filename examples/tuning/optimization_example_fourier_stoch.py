import numpy as np
import matplotlib.pyplot as plt
import time

from nnoptic.tuning import Interferometer
from nnoptic.tuning import func_frobenius, func_fidelity, func_sst, func_weak
from nnoptic.tuning import derivative_func_frobenius, derivative_func_fidelity
from nnoptic.tuning import derivative_func_sst, derivative_func_weak
from nnoptic.tuning import optimizer
from nnoptic import create_list_fl, generator_unitary_matrix, get_random_phase

start_time = time.time()

N = 20
counts_of_epochs = 150
file_name = None

func, grad_func = func_frobenius, derivative_func_frobenius
label = r'$J_{W}$'
if label == r'$J_{INF}$':
    func, grad_func = func_fidelity, derivative_func_fidelity
if label == r'$J_{FR}$':
    func, grad_func = func_frobenius, derivative_func_frobenius
if label == r'$J_{W}$':
    func, grad_func = func_weak, derivative_func_weak
if label == r'$J_{SST}$':
    func, grad_func = func_sst, derivative_func_sst

m = 5

mean_fourier = np.zeros(counts_of_epochs)
mean_stoch = np.zeros(counts_of_epochs)

list_steps = []
list_fourier = []
list_stoch = []

epochs = []

basis = 'fourier'

for i in range(m):
    inter = Interferometer(N, basis, file_name)
    F_target = get_random_phase(N)
    Fl = create_list_fl(F_target, N)
    Ul = inter.list_U
    target = generator_unitary_matrix(N)
    inter.set_target(target)
    steps, results = optimizer(inter, counts_of_epochs, func, grad_func, 'L-BFGS-B')
    list_steps.append(steps)
    list_fourier.append(results)

    mean_fourier += results
    epochs = steps

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
    list_stoch.append(results)

    mean_stoch += results
    epochs = steps

mean_fourier = mean_fourier / m
mean_stoch = mean_stoch / m

delta_time = time.time() - start_time
print('--- %s seconds ---' % delta_time)
print('--- %s seconds ---' % (delta_time / m))

fig, ax = plt.subplots()
plt.plot(epochs, mean_fourier, color='green', lw=2, label='Базисные матрицы - Фурье')
# plt.fill_between(epochs, mean_bfgs - std_bfgs, mean_bfgs + std_bfgs, color='#CCCCCC')
plt.plot(epochs, mean_stoch, color='blue', lw=2, label='Базисные матрицы - случайные')
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
