import numpy as np

# This module generates N unitary matrices of size N by N and writes them to a file called "goal_matrices.txt"


def generator_unitary_matrix(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n))/np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = generator_diagonal_matrix(r, n)
    u = np.dot(q, ph)
    return u


def generator_diagonal_matrix(r, n):
    ph = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    for i in range(n):
        for j in range(n):
            ph[i][j] = 0
    for i in range(n):
        ph[i][i] = r[i][i]/abs(r[i][i])
    return ph


def save_base_unitary_matrices(n, file_name):  # n - size and number of unitary matrices
    list_u = []
    for i in range(n):
        list_u.append(generator_unitary_matrix(n))
    file = open(file_name, 'w')
    for l in range(n):
        for i in range(n):
            for j in range(n):
                file.write(str(list_u[l][i][j].real) + '\n')
                file.write(str(list_u[l][i][j].imag) + '\n')
        file.write('\n')
    file.close()
    print('Base matrices are successfully generated and loaded to file')
