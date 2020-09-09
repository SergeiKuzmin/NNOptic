import numpy as np
from funcs_for_matrices.funcs_for_matrices import transform_sst


def frobenius_reduced(_u, _v):
    c = np.sum(np.abs(_u - _v) ** 2)
    c = c / len(_u)  # Bringing to a form in which C weakly depends on N
    return c


# Metric on the space of unitary matrices
def infidelity(_u, _v):
    return 1 - abs((np.trace(np.dot(_u.T.conj(), _v)) * np.trace(np.dot(_v.T.conj(), _u))) / \
                   (np.trace(np.dot(_u.T.conj(), _u)) * np.trace(np.dot(_v.T.conj(), _v))))


def weak_reduced(_u, _v):
    c = np.sum((np.abs(_u) ** 2 - np.abs(_v) ** 2) ** 2)
    return c


def sst(_u, _v):
    u = transform_sst(_u)
    v = transform_sst(_v)
    return frobenius_reduced(u, v)
