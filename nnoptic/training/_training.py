import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from nnoptic.load_data import load_data, load_goal_matrices
from nnoptic.funcs_for_matrices import create_list_fl, transform_to_matrix, transform_to_1d_list, interferometer
from nnoptic.funcs_for_matrices import create_mini_batch, get_random_phase, norma_square, polar_correct, get_list_noisy
from nnoptic.functionals import frobenius_reduced, infidelity, sst, weak_reduced


def func_frobenius(x, network, mini_batch_f, mini_batch_u, n):
    c = 0.0  # The cost function itself, which must be calculated on the mini-package
    mini_batch_size = len(mini_batch_f)
    list_u = transform_to_matrix(x, n)  # Restored our list of trial basis matrices
    for k in range(mini_batch_size):
        fl = create_list_fl(mini_batch_f[k], n)
        c = c + frobenius_reduced(interferometer(fl, list_u, n), mini_batch_u[k])
    c = c / len(mini_batch_f)
    return c  # The function returns a real number


def derivative_func_frobenius(x, network, mini_batch_f, mini_batch_u, n):
    list_u = transform_to_matrix(x, n)  # Restored our list of trial basis matrices
    network.list_U = list_u  # Updated values of trial base matrices (why?)
    network.grad_frobenius(mini_batch_f, mini_batch_u)  # Calculation of stochastic gradient
    list_of_grad = transform_to_1d_list(network.grad_U, n)
    der = np.zeros_like(x)
    for i in range(len(x)):
        der[i] = list_of_grad[i]
    return der


def func_weak(x, network, mini_batch_f, mini_batch_u, n):
    c = 0.0  # The cost function itself, which must be calculated on the mini-package
    mini_batch_size = len(mini_batch_f)
    list_u = transform_to_matrix(x, n)  # Restored our list of trial basis matrices
    for k in range(mini_batch_size):
        fl = create_list_fl(mini_batch_f[k], n)
        c = c + weak_reduced(interferometer(fl, list_u, n), mini_batch_u[k])
    c = c / len(mini_batch_f)
    return c  # The function returns a real number


def derivative_func_weak(x, network, mini_batch_f, mini_batch_u, n):
    list_u = transform_to_matrix(x, n)  # Restored our list of trial basis matrices
    network.list_U = list_u  # Updated values of trial base matrices (why?)
    network.grad_weak(mini_batch_f, mini_batch_u)  # Calculation of stochastic gradient
    list_of_grad = transform_to_1d_list(network.grad_U, n)
    der = np.zeros_like(x)
    for i in range(len(x)):
        der[i] = list_of_grad[i]
    return der


def func_sst(x, network, mini_batch_f, mini_batch_u, n):
    c = 0.0  # The cost function itself, which must be calculated on the mini-package
    mini_batch_size = len(mini_batch_f)
    list_u = transform_to_matrix(x, n)  # Restored our list of trial basis matrices
    for k in range(mini_batch_size):
        fl = create_list_fl(mini_batch_f[k], n)
        c = c + sst(interferometer(fl, list_u, n), mini_batch_u[k])
    c = c / len(mini_batch_f)
    return c  # The function returns a real number


def derivative_func_sst(x, network, mini_batch_f, mini_batch_u, n):
    list_u = transform_to_matrix(x, n)  # Restored our list of trial basis matrices
    network.list_U = list_u  # Updated values of trial base matrices (why?)
    network.grad_sst(mini_batch_f, mini_batch_u)  # Calculation of stochastic gradient
    list_of_grad = transform_to_1d_list(network.grad_U, n)
    der = np.zeros_like(x)
    for i in range(len(x)):
        der[i] = list_of_grad[i]
    return der


def trainer(file_name1, file_name2, file_name3, n, m, mini_batch_size, counts_of_epochs, func, derivative_func,
            functional, coeff, noisy_f, noisy_u, network, method='L-BFGS-B'):
    fm, um = load_data(n, m, file_name2)  # Got the whole sample
    for u in um:
        fm = fm + noisy_f * np.random.randn(n, n)
        um = um + noisy_u * (np.random.randn(n, n) + 1j * np.random.randn(n, n))
    um = polar_correct(um)  # Unitarization of basis matrices

    # network = Network(n, m, mini_batch_size, file_name3)  # Created an object of class Network

    if coeff is not None:
        list_goal_u = load_goal_matrices(n, file_name1)
        # Downloaded the list of correct unitary matrices to facilitate the search
        list_u = get_list_noisy(list_goal_u, coeff, n)
        network.list_U = list_u  # Facilitating the search for a solution with large values of n

    steps = []
    results = []
    cross_validation = []
    norma = []

    x0 = transform_to_1d_list(network.list_U, n)  # Initialized Optimization algorithm
    list_goal_u = load_goal_matrices(n, file_name1)

    if method == 'L-BFGS-B':
        print('Turned on L-BFGS-B')
        for i in range(counts_of_epochs):
            mini_batch_f, mini_batch_u = create_mini_batch(n, m, mini_batch_size, fm, um)
            # Formed a mini-package for Learning at one step
            steps.append(i)
            results.append(func(x0, network, mini_batch_f, mini_batch_u, n))
            f = get_random_phase(n)
            cross_validation.append(functional(interferometer(create_list_fl(f, n), network.list_U, n),
                                    interferometer(create_list_fl(f, n), list_goal_u, n)))
            norma.append(norma_square(interferometer(create_list_fl(mini_batch_f[0], n), network.list_U, n), n))

            res = minimize(func, x0, args=(network, mini_batch_f, mini_batch_u, n), method='L-BFGS-B',
                           jac=derivative_func, options={'disp': False, 'maxiter': 1})  # Optimization step 'BFGS'
            network.list_U = transform_to_matrix(res.x, n)  # Updated the neural network
            network.polar_correct()
            x0 = res.x
            f = get_random_phase(n)
            print('Epoch: ', i + 1, ' Training set: ', results[i], ' Test set: ',
                  functional(interferometer(create_list_fl(f, n), network.list_U, n),
                             interferometer(create_list_fl(f, n), list_goal_u, n)))

    if method == 'SGD':
        print('Turned on SGD')
        # rate_learning = 0.1 # The best
        rate_learning = 0.2
        for i in range(counts_of_epochs):
            mini_batch_f, mini_batch_u = create_mini_batch(n, m, mini_batch_size, fm, um)
            # Formed a mini-package for Learning at one step

            steps.append(i)
            results.append(func(x0, network, mini_batch_f, mini_batch_u, n))
            f = get_random_phase(n)
            cross_validation.append(functional(interferometer(create_list_fl(f, n), network.list_U, n),
                                               interferometer(create_list_fl(f, n), list_goal_u, n)))
            norma.append(norma_square(interferometer(create_list_fl(mini_batch_f[0], n), network.list_U, n), n))

            x0 = x0 - rate_learning * derivative_func(x0, network, mini_batch_f, mini_batch_u, n)
            # Optimization step 'SGD'
            network.list_U = transform_to_matrix(x0, n)  # Updated the neural network
            network.polar_correct()
            # print(norma_square(network.list_U[0], network.N))
            # print(x0, ' ', results[i])
            f = get_random_phase(n)
            print('epoch: ', i + 1, ' Training set: ', results[i], ' Test set: ',
                  functional(interferometer(create_list_fl(f, n), network.list_U, n)))

    # Cross validation
    list_goal_u = load_goal_matrices(n, file_name1)

    for i in range(10):
        f = get_random_phase(n)
        print(frobenius_reduced(interferometer(create_list_fl(f, n), network.list_U, n),
                                interferometer(create_list_fl(f, n), list_goal_u, n)),
              infidelity(interferometer(create_list_fl(f, n), network.list_U, n),
                         interferometer(create_list_fl(f, n), list_goal_u, n)),
              weak_reduced(interferometer(create_list_fl(f, n), network.list_U, n),
                           interferometer(create_list_fl(f, n), list_goal_u, n)),
              sst(interferometer(create_list_fl(f, n), network.list_U, n),
                  interferometer(create_list_fl(f, n), list_goal_u, n)))

    steps = np.array(steps)
    results = np.array(results)
    cross_validation = np.array(cross_validation)
    norma = np.array(norma)

    error = 0.0
    for i in range(1000):
        f = get_random_phase(n)
        error = error + infidelity(interferometer(create_list_fl(f, n), network.list_U, n),
                                   interferometer(create_list_fl(f, n), list_goal_u, n))
    error = error / 1000

    return steps, results, cross_validation, norma, error
