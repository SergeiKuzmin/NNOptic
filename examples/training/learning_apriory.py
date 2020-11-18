import numpy as np
import matplotlib.pyplot as plt
import time

from load_data.generator_unitary_matrix import save_base_unitary_matrices
from load_data.generator_sample_of_unitaries_matrices import save_sample_unitary_matrices
from Learning.learning import learning, func_frobenius, derivative_func_frobenius
from Learning.learning import func_weak, derivative_func_weak
from Learning.learning import func_sst, derivative_func_sst
from functionals.functionals import frobenius_reduced, weak_reduced, sst
from Learning.Network import Network

# plt.style.use('classic')

start_time = time.time()

N = 20  # Dimension of unitary matrices
M = 300  # Sample size of unitary matrices
mini_batch_size = 5  # Mini-packet size for one step of the Optimization algorithm
counts_of_epochs = 200  # Number of Learning steps
coeff = 0.11  # Correctness coefficient of initialization of basis matrices
noisy_f = 0.00
noisy_u = 0.00
method = 'L-BFGS-B'
# method = 'SGD'
func, grad_func, functional = func_frobenius, derivative_func_frobenius, frobenius_reduced
label = r'$J_{FR}$'
if label == r'$J_{FR}$':
    func, grad_func, functional = func_frobenius, derivative_func_frobenius, frobenius_reduced
if label == r'$J_{W}$':
    func, grad_func, functional = func_weak, derivative_func_weak, weak_reduced
if label == r'$J_{SST}$':
    func, grad_func, functional = func_sst, derivative_func_sst, sst

m = 1

file_name1 = '../../Save/goal_matrices.txt'
file_name2 = '../../Save/sample_of_unitaries_matrices.txt'
file_name3 = '../../Save/save_network.txt'

save_base_unitary_matrices(N, file_name1)
save_sample_unitary_matrices(N, M, file_name1, file_name2)

mean_results = np.zeros(counts_of_epochs)
mean_cross_validation = np.zeros(counts_of_epochs)

list_steps = []
list_results = []
list_cross_validation = []

epochs = []

for i in range(m):
    network = Network(N, M, mini_batch_size, False)
    steps, results, cross_validation, norma = learning(file_name1, file_name2, False, N, M, mini_batch_size, counts_of_epochs,
                                                func, grad_func, functional,
                                                coeff, noisy_f, noisy_u, network, method)
    list_steps.append(steps)
    list_results.append(results)
    list_cross_validation.append(cross_validation)

    mean_results += results
    mean_cross_validation += cross_validation

    epochs = steps

mean_results = mean_results / m
mean_cross_validation = mean_cross_validation / m

delta_time = time.time() - start_time
print('--- %s seconds ---' % delta_time)
print('--- %s seconds ---' % (delta_time / m))
# 99ff99
#d0e3f7
# CCCCCC
fig, ax = plt.subplots()
plt.plot(epochs, mean_cross_validation, color='green', lw=2, label=label + ' тестового набора')
plt.plot(epochs, mean_results, color='blue', lw=2, label=label + ' тренировочного набора')
plt.tick_params(which='major', direction='in')
plt.tick_params(which='minor', direction='in')
# lower left
# lower right
# upper left
# upper right
plt.legend(loc="upper right")
ax.grid()
ax.minorticks_off()
plt.xlim(0, 200)
plt.yscale('log')
# plt.ylim(0.00001, 1.5)
plt.xlabel('Эпохи обучения', fontsize=11)
plt.ylabel(label, fontsize=15)
plt.title('N = '+str(N)+', M = '+str(M)+', '+r'$\alpha$'+' ='+str(coeff))
plt.show()
