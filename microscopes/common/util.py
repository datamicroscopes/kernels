"""
util.py - common utility functions
"""

import numpy as np

def almost_eq(a, b, tol=1e-5):
    return (np.fabs(a - b) <= tol).all()

def rank(A):
    _, S, _ = np.linalg.svd(A)
    return int(np.sum(S >= 1e-10))

def random_orthogonal_matrix(m, n):
    A, _ = np.linalg.qr(np.random.random((m, n)))
    assert rank(A) == min(m, n)
    return A

def random_orthonormal_matrix(n):
    A = random_orthogonal_matrix(n, n)
    assert almost_eq(np.dot(A.T, A), np.eye(n))
    return A
