import numpy as np
from scipy import linalg


# Fidelity on the space of unitary matrices
def fidelity(_u, _v):
    return abs((np.trace(np.dot(_u.T.conj(), _v)) * np.trace(np.dot(_v.T.conj(), _u))) / \
               (np.trace(np.dot(_u.T.conj(), _u)) * np.trace(np.dot(_v.T.conj(), _v))))


# A function that returns a list of basic unitary matrices that are close to a list of regular basic unitary matrices
def get_list_noisy(list_goal_u, coeff, n):
    list_u = []
    for l in range(n):
        error = coeff * (np.random.randn(n, n) + 1j * np.random.randn(n, n))
        new_matrix = list_goal_u[l] + error
        _u, _ = linalg.polar(new_matrix)
        new_u = _u
        list_u.append(new_u)
    return list_u


def create_fourier_matrix(n):
    _u = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            _u[i][j] = 1 / (n ** 0.5) * np.exp(2 * np.pi * 1j / n * i * j)
    return _u


# Useful function that allows you to change the format of parameters: list of basis matrices -> list of real numbers
def transform_to_1d_list(list_u, n):
    list_of_parameters = []
    for l in range(n):
        for i in range(n):
            for j in range(n):
                list_of_parameters.append(list_u[l][i][j].real)
                list_of_parameters.append(list_u[l][i][j].imag)
    return list_of_parameters


# Useful function that allows you to change the format of parameters: list of real numbers -> list of basis matrices
def transform_to_matrix(list_of_parameters, n):
    list_u = []
    p = 0  # Счётчик (индекс)
    for l in range(n):
        u = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        for i in range(n):
            for j in range(n):
                u[i][j] = list_of_parameters[p] + 1j * list_of_parameters[p + 1]
                p = p + 2
        list_u.append(u)
    return list_u


# Auxiliary function for generating a random (according to Haar) unitary matrix
def generator_diagonal_matrix(r, n):
    ph = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    for i in range(n):
        for j in range(n):
            ph[i][j] = 0.0
    for i in range(n):
        ph[i][i] = r[i][i] / abs(r[i][i])
    return ph


# Function for generating a random (according to Haar) unitary matrix
def generator_unitary_matrix(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = generator_diagonal_matrix(r, n)
    _u = np.dot(q, ph)
    return _u


# Function that creates a random matrix of phase delays
def get_random_phase(n):
    f = 2 * 3.141592 * np.random.rand(n, n)
    return f


# Introduce the quadratic norm of the matrix
def norma_square(_u, n):
    norma = 0.0
    for i in range(n):
        for j in range(n):
            norma = norma + _u[i][j].real ** 2 + _u[i][j].imag ** 2
    norma = norma / n
    return norma


# Function generating a list of size mini_batch_size random indices from 0 to M
def create_random_list(m, mini_batch_size):
    x = []
    for i in range(mini_batch_size):
        x.append(round(np.random.rand() * (m - 1)))
    return x


# A function that randomly generates a mini-packet for one SGD step (matrices can be repeated in a mini-packet)
def create_mini_batch(n, m, mini_batch_size, fm, um):
    x = create_random_list(m, mini_batch_size)
    mini_batch_f = []  # Список (мини-пакет) для FM
    mini_batch_u = []  # Список (мини-пакет) для UM
    for i in range(mini_batch_size):
        mini_batch_f.append(fm[x[i]])
        mini_batch_u.append(um[x[i]])
    return mini_batch_f, mini_batch_u


# A function that receives an interferometer phase delay matrix input and returns a list of phase delay matrices Fl
def create_list_fl(f, n):
    fl = []
    for l in range(n + 1):
        fl.append(np.zeros((n, n), dtype=complex))
    for l in range(n):
        for i in range(n):
            fl[l][i][i] = np.exp(1j * f[l][i])
    for i in range(n):
        fl[n][i][i] = np.exp(1j * f[i][n - 1])
    return fl


# The function simulating the interferometer,
# lists of matrices Fl and Ul are fed to the input, and the matrix of the interferometer is output
def interferometer(fl, ul, n):
    for l in range(n):
        if l == 0:
            _u = np.eye(n, dtype=complex)
        u_part = np.dot(ul[l], fl[l])
        _u = np.dot(u_part, _u)
    _u = np.dot(fl[n], _u)
    return _u


def kron(i, j):
    if i == j:
        delta = 1
    else:
        delta = 0
    return delta


def transform_sst(_u):
    n = len(_u)
    r_l = np.eye(n, dtype=complex)
    r_r = np.eye(n, dtype=complex)
    for i in range(n):
        r_l[i][i] = (_u[i][0]).conjugate() / abs(_u[i][0])
        if i == 0:
            r_r[i][i] = 1.0
        else:
            r_r[i][i] = (_u[0][i]).conjugate() / abs(_u[0][i])
    u = np.dot(r_l, np.dot(_u, r_r))
    return u


def r_r_r_l(_u):
    n = len(_u)
    r_l = np.eye(n, dtype=complex)
    r_r = np.eye(n, dtype=complex)
    for i in range(n):
        r_l[i][i] = (_u[i][0]).conjugate() / abs(_u[i][0])
        if i == 0:
            r_r[i][i] = 1.0
        else:
            r_r[i][i] = (_u[0][i]).conjugate() / abs(_u[0][i])
    return r_l, r_r


def polar_correct(list_a):
    # We replace the matrices with the closest unitary matrices
    # according to the Frobenius norm using polar decomposition
    list_u = []
    for a in list_a:
        _u, _ = linalg.polar(a)
        list_u.append(_u)
    return list_u


def get_noisy(list_u, coeff):
    list_u_noisy = []
    n = len(list_u[0])
    m = len(list_u)
    for l in range(m):
        error = coeff * (np.random.randn(n, n) + 1j * np.random.randn(n, n))
        new_matrix = list_u[l] + error
        _u, _ = linalg.polar(new_matrix)
        new_u = _u
        list_u_noisy.append(new_u)
    return list_u_noisy


def transform_f_to_list_u(f, n):
    list_f = []
    for j in range(n):
        list_f.append(f[((n ** 2) * j):((n ** 2) * j + (n ** 2))].reshape(n, n))
    ul = [create_fourier_matrix(n)] * n
    list_u = []
    for l in range(n):
        fl = create_list_fl(list_f[l], n)
        list_u.append(interferometer(fl, ul, n))
    return list_u
