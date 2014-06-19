from distributions.dbg.models import bb, gp, nich

from microscopes.models.mixture.dp import DPMM
from microscopes.common.dataset import numpy_dataset
from microscopes.kernels.gibbs import gibbs_assign

import itertools as it
import math
import numpy as np
import scipy as sp
import scipy.misc

def canonical(assignments):
    assignments = np.copy(assignments)
    lowest = 0
    for i in xrange(assignments.shape[0]):
        if assignments[i] < lowest:
            continue
        if assignments[i] == lowest:
            lowest += 1
            continue
        temp = assignments[i]
        idxs = assignments == temp
        assignments[assignments == lowest] = temp
        assignments[idxs] = lowest
        lowest += 1
    return assignments

def permutation_iter(n):
    seen = set()
    for C in it.product(range(n), repeat=n):
        C = tuple(canonical(np.array(C)))
        if C in seen:
            continue
        seen.add(C)
        yield C

def cluster(Y, assignments):
    labels = {}
    for assign in assignments:
        if assign not in labels:
            labels[assign] = len(labels)
    clusters = [[] for _ in xrange(len(labels))]
    for ci, yi in zip(assignments, Y):
        clusters[labels[ci]].append(yi)
    return tuple(np.array(c) for c in clusters)

def kl(a, b):
    return np.sum([p*np.log(p/q) for p, q in zip(a, b)])

def test_convergence():
    N = 4
    D = 5
    dpmm = DPMM(N, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    actual_dpmm = DPMM(N, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    Y_clustered = dpmm.sample(N)
    Y = np.hstack(Y_clustered)
    assert Y.shape[0] == N
    actual_dpmm.fill(Y_clustered)

    idmap = { C : i for i, C in enumerate(permutation_iter(N)) }

    # brute force the posterior of the actual model
    def posterior(assignments):
        actual_dpmm.reset()
        actual_dpmm.fill(cluster(Y, assignments))
        return actual_dpmm.score_joint()
    actual_scores = np.array(map(posterior, permutation_iter(N)))
    actual_scores -= sp.misc.logsumexp(actual_scores)
    actual_scores = np.exp(actual_scores)

    dataset = numpy_dataset(Y)
    dpmm.bootstrap(dataset.data(shuffle=False))

    # burnin
    for _ in xrange(10000):
        gibbs_assign(dpmm, dataset.data(shuffle=True))

    # now grab 1000 samples, every 10 iters
    smoothing = 1e-5
    gibbs_scores = np.zeros(len(actual_scores)) + smoothing
    for _ in xrange(1000):
        for _ in xrange(10):
            gibbs_assign(dpmm, dataset.data(shuffle=True))
        gibbs_scores[idmap[tuple(canonical(dpmm.assignments()))]] += 1
    gibbs_scores /= gibbs_scores.sum()

    assert kl(actual_scores, gibbs_scores) <= 0.1

def test_different_datatypes():
    N = 10
    likelihoods = [bb, gp, nich, bb]
    hyperparams = [
        {'alpha':1.0, 'beta':3.0},
        {'alpha':2.0, 'inv_beta':1.0},
        {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
        {'alpha':2.0, 'beta':1.0}]
    dpmm = DPMM(N, {'alpha':2.0}, likelihoods, hyperparams)
    Y_clustered = dpmm.sample(N)
    Y = np.hstack(Y_clustered)
    assert Y.shape[0] == N
    dataset = numpy_dataset(Y)
    dpmm.bootstrap(dataset.data(shuffle=False))

    # make sure it deals with different types
    for _ in xrange(10):
        gibbs_assign(dpmm, dataset.data(shuffle=True))

    for typ, y in zip(likelihoods, Y[0]):
        assert typ.Value == type(np.asscalar(y))
