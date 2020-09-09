import numpy as np


# The function loads the target unitary matrices from the file 'goal_matrices.txt' and displays them as a list
def load_goal_matrices(n, file_name):
    file = open(file_name, 'r')
    list_goal_u = []
    for l in range(n):
        _u = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        for i in range(n):
            for j in range(n):
                real = float(file.readline())
                imag = float(file.readline())
                _u[i][j] = real + 1j * imag
        s = file.readline()  # Read the empty line
        list_goal_u.append(_u)
    return list_goal_u


# Function loading a sample of phase delay matrices from the file 'sample_of_unitaries_matrices'
# and their corresponding unitary matrices
def load_data(n, m, file_name):
    file1 = open(file_name, 'r')

    fm = []  # Create a list of F1, ..., FM
    um = []  # Create a list of U1, ..., UM

    for k in range(m):
        f = np.random.randn(n, n)
        for i in range(n):
            for j in range(n):
                f[i][j] = float(file1.readline())
        fm.append(f)
        s = file1.readline()  # Read the empty line

    for k in range(m):
        _u = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        for i in range(n):
            for j in range(n):
                real = float(file1.readline())
                imag = float(file1.readline())
                _u[i][j] = real + 1j * imag
        um.append(_u)
        s = file1.readline()  # Read the empty line

    file1.close()

    return fm, um
