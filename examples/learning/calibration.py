import numpy as np
import matplotlib.pyplot as plt

from funcs_for_matrices.funcs_for_matrices import generator_unitary_matrix
from funcs_for_matrices.funcs_for_matrices import fidelity
from funcs_for_matrices.funcs_for_matrices import get_noisy
from functionals.functionals import infidelity

# plt.style.use('classic')

list_coeff = np.linspace(0.0, 1.0, 50)
list_n = [7, 13, 20]
m = 1000

fig, ax = plt.subplots()
for n in list_n:
    mean_results = []
    for coeff in list_coeff:
        fid_mean = 0.0
        list_u = []

        for i in range(m):
            list_u.append(generator_unitary_matrix(n))

        list_a = get_noisy(list_u, coeff)

        for i in range(m):
            fid_mean += infidelity(list_a[i], list_u[i])

        fid_mean = fid_mean / m
        mean_results.append(fid_mean)

    mean_results = np.array(mean_results)
    plt.plot(list_coeff, mean_results, lw=2, label='N = ' + str(n))
    # plt.plot(list_coeff, mean_results, lw=2, label='N = 2')
    # plt.fill_between(list_coeff, mean_results - std_results, mean_results + std_results, color='#CCCCCC')

plt.tick_params(which='major', direction='in')
plt.tick_params(which='minor', direction='in')
plt.legend(loc="lower right")
ax.grid()
ax.minorticks_off()
plt.xlim(0.0, 1.0)
plt.ylim(0.0000000001, 1.0)
plt.xlabel(r'$\alpha$', fontsize=15)
plt.ylabel(r'$1 - F$', fontsize=15)
plt.show()
# print(list_coeff)
# print(mean_results)
