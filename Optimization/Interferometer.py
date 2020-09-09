import numpy as np

from funcs_for_matrices.funcs_for_matrices import generator_unitary_matrix, create_list_fl, interferometer
from funcs_for_matrices.funcs_for_matrices import transform_sst, r_r_r_l
from funcs_for_matrices.funcs_for_matrices import create_fourier_matrix


class Interferometer(object):

    def __init__(self, n, basis, file_name):
        # file_name - file to which the neural network is written
        # (trained basis matrices), and, accordingly, is read from it
        self.N = n
        self.file_name = file_name
        self.basis = basis

        # Initialization of a neural network
        self.list_U = []  # A list containing the complete basis of unitary matrices
        for l in range(self.N):
            u = np.zeros((self.N, self.N), dtype=complex)
            self.list_U.append(u)

        self.target = None

        self.A = []
        self.B = []
        self.D = []

        for l in range(self.N + 1):
            a_l = np.zeros((self.N, self.N), dtype=complex)
            b_l = np.zeros((self.N, self.N), dtype=complex)
            self.A.append(a_l)
            self.B.append(b_l)

        for k in range(self.N):
            d_k = np.zeros((self.N, self.N), dtype=complex)
            d_k[k][k] = 1.0
            self.D.append(d_k)

        for l in range(self.N):
            if self.basis == 'fourier':
                u = create_fourier_matrix(self.N)
            else:
                u = generator_unitary_matrix(self.N)
            self.list_U[l] = u  # Initialize the full matrix basis with random unitary matrices

        if self.file_name is not None:  # If passed True, then the full matrix basis is initialized from file_name
            file = open(self.file_name, 'r')
            for l in range(self.N):
                for i in range(self.N):
                    for j in range(self.N):
                        real = float(file.readline())
                        imag = float(file.readline())
                        self.list_U[l][i][j] = real + 1j * imag
                s = file.readline()  # Read the empty line
            file.close()

        self.F = 2 * 3.141592 * np.random.rand(self.N, self.N)  # Random initialization (uniform) of phase delays
        self.grad_F = np.zeros((self.N, self.N), dtype=float)
        # np array containing infidelity gradients for all elements of the phase delay matrix (complex value)

    def set_target(self, u):
        self.target = u

    # The function displays the resulting common unitary matrix with these phase locks
    def forward(self):
        fl = create_list_fl(self.F, self.N)
        _u = interferometer(fl, self.list_U, self.N)
        return _u

    def grad_fidelity(self):
        fl = create_list_fl(self.F, self.N)
        ul = self.list_U
        self.A[self.N] = np.eye(self.N, dtype=complex)
        for l in range(self.N - 1, -1, -1):
            self.A[l] = np.dot(self.A[l + 1], np.dot(fl[l + 1], ul[l]))
        self.B[0] = np.eye(self.N, dtype=complex)
        for l in range(1, self.N + 1, 1):
            self.B[l] = np.dot(np.dot(ul[l - 1], fl[l - 1]), self.B[l - 1])

        u = self.forward()
        z = np.trace(np.dot(self.target.T.conj(), u))

        for l in range(self.N):
            for k in range(self.N - 1):
                grad_u = 1j * np.exp(1j * self.F[l][k]) * np.dot(self.A[l], np.dot(self.D[k], self.B[l]))
                self.grad_F[l][k] = - (2 / (self.N ** 2)) * \
                                    (z.conjugate() * np.trace(np.dot(self.target.T.conj(), grad_u))).real

        for k in range(self.N):
            grad_u = 1j * np.exp(1j * self.F[k][self.N - 1]) * np.dot(self.A[self.N], np.dot(self.D[k], self.B[self.N]))
            self.grad_F[k][self.N - 1] = - (2 / (self.N ** 2)) * \
                                    (z.conjugate() * np.trace(np.dot(self.target.T.conj(), grad_u))).real

    def grad_frobenius(self):
        fl = create_list_fl(self.F, self.N)
        ul = self.list_U
        self.A[self.N] = np.eye(self.N, dtype=complex)
        for l in range(self.N - 1, -1, -1):
            self.A[l] = np.dot(self.A[l + 1], np.dot(fl[l + 1], ul[l]))
        self.B[0] = np.eye(self.N, dtype=complex)
        for l in range(1, self.N + 1, 1):
            self.B[l] = np.dot(np.dot(ul[l - 1], fl[l - 1]), self.B[l - 1])

        u = self.forward()

        for l in range(self.N):
            for k in range(self.N - 1):
                grad_u = 1j * np.exp(1j * self.F[l][k]) * np.dot(self.A[l], np.dot(self.D[k], self.B[l]))
                self.grad_F[l][k] = (2 / self.N) * np.sum((u - self.target).conj() * grad_u).real

        for k in range(self.N):
            grad_u = 1j * np.exp(1j * self.F[k][self.N - 1]) * np.dot(self.A[self.N], np.dot(self.D[k], self.B[self.N]))
            self.grad_F[k][self.N - 1] = (2 / self.N) * np.sum((u - self.target).conj() * grad_u).real

    def grad_weak(self):
        fl = create_list_fl(self.F, self.N)
        ul = self.list_U
        self.A[self.N] = np.eye(self.N, dtype=complex)
        for l in range(self.N - 1, -1, -1):
            self.A[l] = np.dot(self.A[l + 1], np.dot(fl[l + 1], ul[l]))
        self.B[0] = np.eye(self.N, dtype=complex)
        for l in range(1, self.N + 1, 1):
            self.B[l] = np.dot(np.dot(ul[l - 1], fl[l - 1]), self.B[l - 1])

        u = self.forward()

        for l in range(self.N):
            for k in range(self.N - 1):
                grad_u = 1j * np.exp(1j * self.F[l][k]) * np.dot(self.A[l], np.dot(self.D[k], self.B[l]))
                self.grad_F[l][k] = 4 * np.sum((np.abs(u) ** 2 - np.abs(self.target) ** 2) * u.conj() * grad_u).real

        for k in range(self.N):
            grad_u = 1j * np.exp(1j * self.F[k][self.N - 1]) * np.dot(self.A[self.N], np.dot(self.D[k], self.B[self.N]))
            self.grad_F[k][self.N - 1] = 4 * np.sum((np.abs(u) ** 2 - np.abs(self.target) ** 2) * u.conj() * grad_u).real

    def grad_sst(self):
        fl = create_list_fl(self.F, self.N)
        ul = self.list_U
        self.A[self.N] = np.eye(self.N, dtype=complex)
        for l in range(self.N - 1, -1, -1):
            self.A[l] = np.dot(self.A[l + 1], np.dot(fl[l + 1], ul[l]))
        self.B[0] = np.eye(self.N, dtype=complex)
        for l in range(1, self.N + 1, 1):
            self.B[l] = np.dot(np.dot(ul[l - 1], fl[l - 1]), self.B[l - 1])

        u = self.forward()

        r_l, r_r = r_r_r_l(u)

        u_1 = transform_sst(u)
        target_1 = transform_sst(self.target)

        for l in range(self.N):
            for k in range(self.N - 1):
                grad_u = 1j * np.exp(1j * self.F[l][k]) * np.dot(self.A[l], np.dot(self.D[k], self.B[l]))

                grad_r_l = np.eye(self.N, dtype=complex)
                grad_r_r = np.eye(self.N, dtype=complex)

                for i in range(self.N):
                    grad_r_l[i][i] = (1j * ((u[i][0].conjugate() / abs(u[i][0])) * grad_u[i][0]).imag /
                                      u[i][0].conjugate()).conjugate()
                    if i == 0:
                        grad_r_r[i][i] = 0.0
                    else:
                        grad_r_r[i][i] = (1j * ((u[0][i].conjugate() / abs(u[0][i])) * grad_u[0][i]).imag /
                                          u[0][i].conjugate()).conjugate()

                grad_v = np.dot(grad_r_l, np.dot(u, r_r)) + np.dot(r_l, np.dot(grad_u, r_r)) + \
                         np.dot(r_l, np.dot(u, grad_r_r))

                self.grad_F[l][k] = (2 / self.N) * np.sum((u_1 - target_1).conj() * grad_v).real

        for k in range(self.N):
            grad_u = 1j * np.exp(1j * self.F[k][self.N - 1]) * np.dot(self.A[self.N], np.dot(self.D[k], self.B[self.N]))

            grad_r_l = np.eye(self.N, dtype=complex)
            grad_r_r = np.eye(self.N, dtype=complex)

            for i in range(self.N):
                grad_r_l[i][i] = (1j * ((u[i][0].conjugate() / abs(u[i][0])) * grad_u[i][0]).imag /
                                  u[i][0].conjugate()).conjugate()
                if i == 0:
                    grad_r_r[i][i] = 0.0
                else:
                    grad_r_r[i][i] = (1j * ((u[0][i].conjugate() / abs(u[0][i])) * grad_u[0][i]).imag /
                                      u[0][i].conjugate()).conjugate()

            grad_v = np.dot(grad_r_l, np.dot(u, r_r)) + np.dot(r_l, np.dot(grad_u, r_r)) + \
                     np.dot(r_l, np.dot(u, grad_r_r))

            self.grad_F[k][self.N - 1] = (2 / self.N) * np.sum((u_1 - target_1).conj() * grad_v).real
