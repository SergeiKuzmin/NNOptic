import numpy as np
from nnoptic.funcs_for_matrices import interferometer, create_list_fl

# This module creates a sample of size M from unitary matrices of size N by N, using as base matrices
# matrices from the file 'goal_matrices.txt' and saves this selection to the file 'sample_of_unitaries_matrices'


def save_sample_unitary_matrices(n, m, file_name1, file_name2):
    # Generate random phases for the sample (M different matrices, size N by N)
    fm = []
    for k in range(m):
        fm.append(2 * 3.141592 * np.random.rand(n, n))

    # Create a list from U1, ..., UN
    file1 = open(file_name1, 'r')
    list_u = []
    for l in range(n):
        u = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                real = float(file1.readline())
                imag = float(file1.readline())
                u[i][j] = real + 1j * imag
        s = file1.readline()  # Read the empty line
        list_u.append(u)
    file1.close()

    um = []  # List of resulting unitary matrices of size N by N

    # We pass through the sample
    for k in range(m):
        um.append(interferometer(create_list_fl(fm[k], n), list_u, n))

    file2 = open(file_name2, 'w')

    for k in range(m):  # We go through the selection and write phase matrices to a file
        for i in range(n):
            for j in range(n):
                file2.write(str(fm[k][i][j].real) + '\n')
        file2.write('\n')

    for k in range(m):  # We go through the selection and write the resulting selection of unitary matrices to a file
        for i in range(n):
            for j in range(n):
                file2.write(str(um[k][i][j].real) + '\n')
                file2.write(str(um[k][i][j].imag) + '\n')
        file2.write('\n')
    file2.close()
    print('Training dataset successfully loaded to file')
