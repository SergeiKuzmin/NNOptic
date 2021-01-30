import numpy as np
import matplotlib.pyplot as plt
import time

from nnoptic import save_base_unitary_matrices, save_sample_unitary_matrices
from nnoptic import trainer
from nnoptic.training import func_frobenius, derivative_func_frobenius
from nnoptic.training import func_sst, derivative_func_sst
from nnoptic.training import func_weak, derivative_func_weak
from nnoptic import frobenius_reduced, sst, weak_reduced
from nnoptic import Network

start_time = time.time()

N = 5  # Dimension of matrices and number of basis matrices
M = 200  # Sample size of matrices
mini_batch_size = 5  # Mini-packet size for one step of the Training algorithm
counts_of_epochs = 1000  # Number of Training steps
coeff = None
noisy_u = 0.0
noisy_f = 0.0
method = 'L-BFGS-B'
# method = 'SGD'
func, grad_func, functional = func_frobenius, derivative_func_frobenius, frobenius_reduced
label = r'$J_{FR}$'  # Choice of functional
if label == r'$J_{FR}$':
    func, grad_func, functional = func_frobenius, derivative_func_frobenius, frobenius_reduced
if label == r'$J_{W}$':
    func, grad_func, functional = func_weak, derivative_func_weak, weak_reduced
if label == r'$J_{SST}$':
    func, grad_func, functional = func_sst, derivative_func_sst, sst

m = 1  # Number of realizations

file_name1 = 'goal_matrices.txt'
file_name2 = 'sample_of_unitaries_matrices.txt'
file_name3 = 'save_network.txt'

save_base_unitary_matrices(N, file_name1)
save_sample_unitary_matrices(N, M, file_name1, file_name2)

mean_results = np.zeros(counts_of_epochs)
std_results = np.zeros(counts_of_epochs)
mean_cross_validation = np.zeros(counts_of_epochs)
std_cross_validation = np.zeros(counts_of_epochs)

list_steps = []
list_results = []
list_cross_validation = []

epochs = []

error_average = 0.0

for i in range(m):
    network = Network(N, M, mini_batch_size, False)
    steps, results, cross_validation, norma, error = trainer(file_name1, file_name2, False, N, M, mini_batch_size,
                                                             counts_of_epochs, func, grad_func, functional, coeff,
                                                             noisy_f, noisy_u, network, method)
    list_steps.append(steps)
    list_results.append(results)
    list_cross_validation.append(cross_validation)

    mean_results += results
    mean_cross_validation += cross_validation

    epochs = steps

    print(error)
    error_average += error

mean_results = mean_results / m
mean_cross_validation = mean_cross_validation / m

for i in range(m):
    std_results += (list_results[i] - mean_results) ** 2
    std_cross_validation += (list_cross_validation[i] - mean_cross_validation) ** 2

std_results = (std_results / m * (m - 1)) ** 0.5
std_cross_validation = (std_cross_validation / m * (m - 1)) ** 0.5

delta_time = time.time() - start_time
print('--- %s seconds ---' % delta_time)

fig, ax = plt.subplots()
plt.plot(epochs, mean_cross_validation, color='green', lw=2, label=label + ' Test set')
plt.plot(epochs, mean_results, color='blue', lw=2, label=label + ' Train set')
# plt.fill_between(epochs, mean_results - std_results, mean_results + std_results, color='#CCCCCC')
plt.tick_params(which='major', direction='in')
plt.tick_params(which='minor', direction='in')
plt.legend(loc="upper right")
ax.grid()
ax.minorticks_off()
plt.xlim(0, 999)
plt.yscale('log')
plt.xlabel('Epochs', fontsize=11)
plt.ylabel(label, fontsize=15)
plt.title('N = '+str(N)+', M = '+str(M))
plt.show()
