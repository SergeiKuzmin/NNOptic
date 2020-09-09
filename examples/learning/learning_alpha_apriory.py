import numpy as np
import matplotlib.pyplot as plt

# alpha_u = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
alpha_u = np.array([0.0, 1.49959967e-04, 5.93023082e-04, 1.34096535e-03, 2.41428655e-03,
                    3.70831328e-03, 5.40285011e-03, 7.50880441e-03, 9.44625876e-03, 1.23053916e-02, 1.48866144e-02])
j_inf = np.array([7.57 * 1e-11, 0.0003676, 0.0014663, 0.005764612, 0.00871928, 0.0085194, 0.01141, 0.018429,
                  0.015884, 0.14275, 0.1689185])
# j_inf = np.array([0.02459, 0.027326, 0.05948, 0.059355, 0.0474699, 0.05264412, 0.129213, 0.035296, 0.02767,
#                   0.0407856, 0.07559])

# j_inf_std = np.array([0.0005]*3 + [0.005]*3 + [0.01]*5)
j_inf_std = np.array([0.0005]*3 + [0.005]*3 + [0.01]*5)

fig, ax = plt.subplots()
plt.plot(alpha_u, j_inf, color='blue', lw=2, label='Среднее инфиделити для обученной модели')
plt.fill_between(alpha_u, j_inf - j_inf_std, j_inf + j_inf_std, color='#CCCCCC')
# plt.plot(N, M_theory, color='red', lw=2, label=r'$N^3$')
plt.plot(alpha_u, alpha_u, color='green', lw=2, label='Инфиделити шума (y = x)')
plt.tick_params(which='major', direction='in')
plt.tick_params(which='minor', direction='in')
# lower left
# lower right
# upper left
# upper right
plt.legend(loc="upper left")
ax.grid()
ax.minorticks_off()
plt.xlim(0.0, 0.0147)
# plt.yscale('log')
plt.ylim(0.0001, 0.2)
plt.title('N = 2, M = 3, '+r'$\alpha_{\varphi}$ = 0.0')
plt.xlabel('Параметр шума '+r'$1 - F$', fontsize=11)
plt.ylabel(r'$1 - F$', fontsize=11)
plt.show()
