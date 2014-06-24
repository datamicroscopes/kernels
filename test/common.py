import numpy as np

def KL_discrete(a, b):
    return np.sum([p*np.log(p/q) for p, q in zip(a, b)])

def KL_approx(a, b, dA):
    return np.sum([p*np.log(p/q)*dA for p, q in zip(a, b)])
