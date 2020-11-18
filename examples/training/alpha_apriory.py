import numpy as np
import matplotlib.pyplot as plt

list_N = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
list_alpha = np.array([0.39, 0.31, 0.28, 0.24, 0.22, 0.20, 0.18, 0.16, 0.15, 0.14, 0.135, 0.13, 0.12, 0.11])
list_alpha_std = np.array([0.01]*14)

fig, ax = plt.subplots()
plt.plot(list_N, list_alpha, color='blue', lw=2)
plt.fill_between(list_N, list_alpha - list_alpha_std, list_alpha + list_alpha_std, color='#CCCCCC')
plt.tick_params(which='major', direction='in')
plt.tick_params(which='minor', direction='in')
# lower left
# lower right
# upper left
# upper right
# plt.legend(loc="upper left")
ax.grid()
ax.minorticks_off()
plt.xlim(7, 20)
# plt.yscale('log')
# plt.ylim(0.0001, 0.2)
# plt.title('N = 2, M = 3, '+r'$\alpha_{\varphi}$ = 0.0')
plt.xlabel('N', fontsize=11)
plt.ylabel(r'$\alpha$', fontsize=11)
plt.show()
