import numpy as np
import matplotlib.pyplot as plt

N = np.array([1, 2, 3, 4, 5, 6])
M = np.array([1, 3, 4, 7, 30, 230])
M_theory = N ** 3

M_std = np.array([1, 1, 1, 1, 5, 50])

fig, ax = plt.subplots()
plt.plot(N, M, color='blue', lw=2, label='Минимальный размер выборки')
plt.fill_between(N, M - M_std, M + M_std, color='#CCCCCC')
# plt.plot(N, M_theory, color='red', lw=2, label=r'$N^3$')
plt.tick_params(which='major', direction='in')
plt.tick_params(which='minor', direction='in')
# lower left
# lower right
# upper left
# upper right
plt.legend(loc="upper left")
ax.grid()
ax.minorticks_off()
plt.xlim(1, 6)
plt.yscale('log')
plt.ylim(1.0001, 250)
plt.xlabel('Размер матриц N', fontsize=11)
plt.ylabel('Размер обучающей выборки M', fontsize=11)
plt.show()
