import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import minimize

from functionals.functionals import infidelity, frobenius_reduced, weak_reduced, sst


def func_fidelity(x, inter, n):
    f = x.reshape(inter.N, inter.N)
    inter.F = f  # Updated the value of the matrix of phase delays
    c = infidelity(inter.target, inter.forward())
    return c  # The function returns a real number


def derivative_func_fidelity(x, inter, n):
    f = x.reshape(inter.N, inter.N)
    inter.F = f  # Updated the value of the matrix of phase delays
    inter.grad_fidelity()  # Gradient calculation
    grad = inter.grad_F.reshape(-1)
    return grad


def test_derivative_func(x, inter, n, func):
    epsilon = 1e-8
    f = x.reshape(inter.N, inter.N)
    grad = np.zeros_like(x)
    value = func(x, inter, n)
    for i in range(len(x)):
        new_x = copy.copy(x)
        new_x[i] = x[i] + epsilon
        new_value = func(new_x, inter, n)
        grad[i] = (new_value - value) / epsilon
    return grad


def func_frobenius(x, inter, n):
    f = x.reshape(inter.N, inter.N)
    inter.F = f  # Updated the value of the matrix of phase delays
    c = frobenius_reduced(inter.target, inter.forward())
    return c  # The function returns a real number


def derivative_func_frobenius(x, inter, n):
    f = x.reshape(inter.N, inter.N)
    inter.F = f  # Updated the value of the matrix of phase delays
    inter.grad_frobenius()  # Gradient calculation
    grad = inter.grad_F.reshape(-1)
    return grad


def func_weak(x, inter, n):
    f = x.reshape(inter.N, inter.N)
    inter.F = f  # Updated the value of the matrix of phase delays
    c = weak_reduced(inter.target, inter.forward())
    return c  # The function returns a real number


def derivative_func_weak(x, inter, n):
    f = x.reshape(inter.N, inter.N)
    inter.F = f  # Updated the value of the matrix of phase delays
    inter.grad_weak()  # Gradient calculation
    grad = inter.grad_F.reshape(-1)
    return grad


def func_sst(x, inter, n):
    f = x.reshape(inter.N, inter.N)
    inter.F = f  # Updated the value of the matrix of phase delays
    c = sst(inter.target, inter.forward())
    return c  # The function returns a real number


def derivative_func_sst(x, inter, n):
    f = x.reshape(inter.N, inter.N)
    inter.F = f  # Updated the value of the matrix of phase delays
    inter.grad_sst()  # Gradient calculation
    grad = inter.grad_F.reshape(-1)
    return grad


def optimization(inter, counts_of_epochs, function, der_function, method):
    x0 = inter.F.reshape(-1)  # Initialized Optimization algorithm
    steps = []
    results = []
    # Now we optimize using the BFGS method
    print('Turned on ', method)
    for i in range(counts_of_epochs):
        # Optimization step 'BFGS'
        # jac=der_function, after method
        steps.append(i)
        results.append(function(x0, inter, inter.N))
        res = minimize(function, x0, args=(inter, inter.N), method=method, jac=der_function,
                       options={'disp': False, 'maxiter': 1})
        inter.F = res.x.reshape(inter.N, inter.N)  # Updated the value of the matrix of phase delays
        x0 = res.x
        print('epoch: ', i + 1, ' ', results[i])
    steps = np.array(steps)
    results = np.array(results)
    return steps, results
