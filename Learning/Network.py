import numpy as np
from scipy import linalg

from funcs_for_matrices.funcs_for_matrices import generator_unitary_matrix, create_list_fl, interferometer
from funcs_for_matrices.funcs_for_matrices import transform_sst, r_r_r_l


class Network(object):

    def __init__(self, n, m, mini_batch_size, file_name):
        # file_name - file to which the neural network is written
        # (trained basis matrices), and, accordingly, is read from it
        self.N = n
        self.M = m
        self.mini_batch_size = mini_batch_size
        self.file_name = file_name
        self.D = []

        # Initialization of a neural network
        self.list_U = []  # A list containing a complete matrix basis (not necessarily unitary)

        for l in range(self.N):
            u = generator_unitary_matrix(self.N)
            self.list_U.append(u)  # Initialize the full matrix basis with random unitary matrices

        # for l in range(self.N):
        #     u = create_fourier_matrix(self.N)
        #     self.list_U[l] = u  # Initialize the full matrix basis with unitary Fourier matrices

        if file_name:  # If passed True, then the full matrix basis is initialized from file_name
            print('Load data about Network')
            file = open(file_name, 'r')
            for l in range(self.N):
                for i in range(self.N):
                    for j in range(self.N):
                        real = float(file.readline())
                        imag = float(file.readline())
                        self.list_U[l][i][j] = real + 1j * imag
                s = file.readline()  # Read the empty line
            file.close()

        self.grad_U = []
        # A list containing gradients for all elements of the full matrix basis
        # (not necessarily unitary) (complex value)
        for l in range(self.N):
            u = (np.random.rand(self.N, self.N) * 2 - 1) + 1j * (np.random.rand(self.N, self.N) * 2 - 1)
            self.grad_U.append(u)  # Initialized gradients for all elements of the full matrix basis

        self.list_A = []
        # Extra procedure that facilitates understanding, but takes time when creating an object of the Network class
        self.list_B = []
        # Extra procedure that facilitates understanding, but takes time when creating an object of the Network class

        for k in range(mini_batch_size):
            list_of_a = []
            list_of_b = []
            for l in range(self.N):
                x = np.random.rand(self.N, self.N) + 1j * np.random.rand(self.N, self.N)
                y = np.random.rand(self.N, self.N) + 1j * np.random.rand(self.N, self.N)
                list_of_a.append(x)
                list_of_b.append(y)
            self.list_A.append(list_of_a)
            self.list_B.append(list_of_b)

        self.c_derivative_x = []
        # Extra procedure that facilitates understanding, but takes time when creating an object of the Network class
        self.c_derivative_y = []
        # Extra procedure that facilitates understanding, but takes time when creating an object of the Network class
        self.c_derivative_u = []
        # Extra procedure that facilitates understanding, but takes time when creating an object of the Network class

        for k in range(mini_batch_size):
            c_der_x = np.random.rand(self.N, self.N) + 1j * np.random.rand(self.N, self.N)
            c_der_y = np.random.rand(self.N, self.N) + 1j * np.random.rand(self.N, self.N)
            self.c_derivative_x.append(c_der_x)
            self.c_derivative_y.append(c_der_y)

        for p in range(self.N):
            row_d = []
            for t in range(self.N):
                d_pt = np.zeros((self.N, self.N), dtype=complex)
                d_pt[p][t] = 1.0
                row_d.append(d_pt)
            self.D.append(row_d)

    def grad_frobenius(self, mini_batch_f, mini_batch_u):
        # This class method calculates the gradient self.grad_U

        for l in range(self.N):
            self.grad_U[l] = np.zeros((self.N, self.N), dtype=complex)
            # We occupy a gradient for the subsequent calculation of the sum

        # Calculate self.list_A and self.list_B
        for k in range(self.mini_batch_size):
            fl = create_list_fl(mini_batch_f[k], self.N)

            self.list_A[k][self.N - 1] = fl[self.N]
            for l in range(self.N - 2, -1, -1):
                self.list_A[k][l] = np.dot(self.list_A[k][l + 1], np.dot(self.list_U[l + 1], fl[l + 1]))
            self.list_B[k][0] = fl[0]
            for l in range(1, self.N, 1):
                self.list_B[k][l] = np.dot(np.dot(fl[l], self.list_U[l - 1]), self.list_B[k][l - 1])

            u_target = mini_batch_u[k]
            u_result = interferometer(fl, self.list_U, self.N)

            for l in range(self.N):
                for p in range(self.N):
                    for t in range(self.N):
                        a = self.list_A[k][l]
                        b = self.list_B[k][l]
                        d = self.D[p][t]
                        grad_u_x = np.dot(a, np.dot(d, b))
                        grad_u_y = 1j * np.dot(a, np.dot(d, b))
                        self.grad_U[l][p][t] += (2 / self.N) * np.sum((u_result - u_target).conj() * grad_u_x).real + \
                            1j * (2 / self.N) * np.sum((u_result - u_target).conj() * grad_u_y).real

        for l in range(self.N):
            self.grad_U[l] = self.grad_U[l] / self.mini_batch_size  # Average the gradient over the mini-packet

    def grad_weak(self, mini_batch_f, mini_batch_u):
        # This class method calculates the gradient self.grad_U

        for l in range(self.N):
            self.grad_U[l] = np.zeros((self.N, self.N), dtype=complex)
            # We occupy a gradient for the subsequent calculation of the sum

        # Calculate self.list_A and self.list_B
        for k in range(self.mini_batch_size):
            fl = create_list_fl(mini_batch_f[k], self.N)

            self.list_A[k][self.N - 1] = fl[self.N]
            for l in range(self.N - 2, -1, -1):
                self.list_A[k][l] = np.dot(self.list_A[k][l + 1], np.dot(self.list_U[l + 1], fl[l + 1]))
            self.list_B[k][0] = fl[0]
            for l in range(1, self.N, 1):
                self.list_B[k][l] = np.dot(np.dot(fl[l], self.list_U[l - 1]), self.list_B[k][l - 1])

            u_target = mini_batch_u[k]
            u_result = interferometer(fl, self.list_U, self.N)

            for l in range(self.N):
                for p in range(self.N):
                    for t in range(self.N):
                        a = self.list_A[k][l]
                        b = self.list_B[k][l]
                        d = self.D[p][t]
                        grad_u_x = np.dot(a, np.dot(d, b))
                        grad_u_y = 1j * np.dot(a, np.dot(d, b))
                        self.grad_U[l][p][t] += 4 * np.sum((np.abs(u_result) ** 2 - np.abs(u_target) ** 2) *
                                                           u_result.conj() * grad_u_x).real + \
                            1j * 4 * np.sum((np.abs(u_result) ** 2 - np.abs(u_target) ** 2) *
                                            u_result.conj() * grad_u_y).real

        for l in range(self.N):
            self.grad_U[l] = self.grad_U[l] / self.mini_batch_size  # Average the gradient over the mini-packet

    def grad_sst(self, mini_batch_f, mini_batch_u):
        # This class method calculates the gradient self.grad_U

        for l in range(self.N):
            self.grad_U[l] = np.zeros((self.N, self.N), dtype=complex)
            # We occupy a gradient for the subsequent calculation of the sum

        # Calculate self.list_A and self.list_B
        for k in range(self.mini_batch_size):
            fl = create_list_fl(mini_batch_f[k], self.N)

            self.list_A[k][self.N - 1] = fl[self.N]
            for l in range(self.N - 2, -1, -1):
                self.list_A[k][l] = np.dot(self.list_A[k][l + 1], np.dot(self.list_U[l + 1], fl[l + 1]))
            self.list_B[k][0] = fl[0]
            for l in range(1, self.N, 1):
                self.list_B[k][l] = np.dot(np.dot(fl[l], self.list_U[l - 1]), self.list_B[k][l - 1])

            u_target = mini_batch_u[k]
            u_result = interferometer(fl, self.list_U, self.N)

            r_l, r_r = r_r_r_l(u_result)

            u_1 = transform_sst(u_result)
            target_1 = transform_sst(u_target)

            for l in range(self.N):
                for p in range(self.N):
                    for t in range(self.N):
                        a = self.list_A[k][l]
                        b = self.list_B[k][l]
                        d = self.D[p][t]
                        grad_u_x = np.dot(a, np.dot(d, b))
                        grad_u_y = 1j * np.dot(a, np.dot(d, b))

                        grad_r_l_x = np.eye(self.N, dtype=complex)
                        grad_r_r_x = np.eye(self.N, dtype=complex)
                        grad_r_l_y = np.eye(self.N, dtype=complex)
                        grad_r_r_y = np.eye(self.N, dtype=complex)

                        for i in range(self.N):
                            # grad_r_l_x[i][i] = grad_u_x[i][0].conjugate()
                            # grad_r_l_y[i][i] = grad_u_y[i][0].conjugate()
                            grad_r_l_x[i][i] = (1j * ((u_result[i][0].conjugate() /
                                                       abs(u_result[i][0])) * grad_u_x[i][0]).imag /
                                                u_result[i][0].conjugate()).conjugate()
                            grad_r_l_y[i][i] = (1j * ((u_result[i][0].conjugate() /
                                                       abs(u_result[i][0])) * grad_u_y[i][0]).imag /
                                                u_result[i][0].conjugate()).conjugate()
                            if i == 0:
                                grad_r_r_x[i][i] = 0.0
                                grad_r_r_y[i][i] = 0.0
                            else:
                                # grad_r_r_x[i][i] = grad_u_x[0][i].conjugate()
                                # grad_r_r_y[i][i] = grad_u_y[0][i].conjugate()
                                grad_r_r_x[i][i] = (1j * ((u_result[0][i].conjugate() /
                                                           abs(u_result[0][i])) * grad_u_x[0][i]).imag /
                                                    u_result[0][i].conjugate()).conjugate()
                                grad_r_r_y[i][i] = (1j * ((u_result[0][i].conjugate() /
                                                           abs(u_result[0][i])) * grad_u_y[0][i]).imag /
                                                    u_result[0][i].conjugate()).conjugate()

                        grad_v_x = np.dot(grad_r_l_x, np.dot(u_result, r_r)) + np.dot(r_l, np.dot(grad_u_x, r_r)) + \
                                   np.dot(r_l, np.dot(u_result, grad_r_r_x))

                        grad_v_y = np.dot(grad_r_l_y, np.dot(u_result, r_r)) + np.dot(r_l, np.dot(grad_u_y, r_r)) + \
                                   np.dot(r_l, np.dot(u_result, grad_r_r_y))

                        self.grad_U[l][p][t] += (2 / self.N) * np.sum((u_1 - target_1).conj() * grad_v_x).real + \
                            1j * (2 / self.N) * np.sum((u_1 - target_1).conj() * grad_v_y).real

        for l in range(self.N):
            self.grad_U[l] = self.grad_U[l] / self.mini_batch_size  # Average the gradient over the mini-packet

    def sgd(self, rate_learning):
        for l in range(self.N):
            for i in range(self.N):
                for j in range(self.N):
                    self.list_U[l][i][j] = self.list_U[l][i][j] - rate_learning * self.grad_U[l][i][j]

    def polar_correct(self):
        # We replace the basis matrices with the closest unitary matrices
        # according to the Frobenius norm using polar decomposition
        for l in range(self.N):
            _u, _ = linalg.polar(self.list_U[l])
            self.list_U[l] = _u

    def save_network(self, file_name):
        file = open(file_name, 'w')
        for l in range(self.N):
            for i in range(self.N):
                for j in range(self.N):
                    file.write(str(self.list_U[l][i][j].real) + '\n')
                    file.write(str(self.list_U[l][i][j].imag) + '\n')
            file.write('\n')
        file.close()
