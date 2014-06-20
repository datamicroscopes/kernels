from distributions.dbg.models import bb, gp, nich

from microscopes.models.mixture.dp import DirichletProcess
from microscopes.models.mixture.dd import DirichletFixed
from microscopes.common.dataset import numpy_dataset
from microscopes.kernels.gibbs import gibbs_assign, gibbs_assign_fixed

import itertools as it
import math
import numpy as np
import scipy as sp
import scipy.misc

def permutation_canonical(assignments):
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
        C = tuple(permutation_canonical(np.array(C)))
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

def _test_mixture_model_convergence(
        mm,
        actual_mm,
        N,
        all_possible_assignments_fn,
        canonical_fn,
        gibbs_assign_fn,
        burnin_niters=10000,
        nsamples=1000,
        skip=10,
        threshold=0.1):
    Y_clustered = mm.sample(N)
    Y = np.hstack(Y_clustered)
    assert Y.shape[0] == N
    actual_mm.fill(Y_clustered)

    idmap = { C : i for i, C in enumerate(all_possible_assignments_fn(N)) }

    # brute force the posterior of the actual model
    def posterior(assignments):
        actual_mm.reset()
        actual_mm.fill(cluster(Y, assignments))
        return actual_mm.score_joint()
    actual_scores = np.array(map(posterior, all_possible_assignments_fn(N)))
    actual_scores -= sp.misc.logsumexp(actual_scores)
    actual_scores = np.exp(actual_scores)

    dataset = numpy_dataset(Y)
    mm.bootstrap(dataset.data(shuffle=False))

    # burnin
    for _ in xrange(burnin_niters):
        gibbs_assign_fn(mm, dataset.data(shuffle=True))

    # now grab nsamples samples, every skip iters
    smoothing = 1e-5
    gibbs_scores = np.zeros(len(actual_scores)) + smoothing
    for _ in xrange(nsamples):
        for _ in xrange(skip):
            gibbs_assign_fn(mm, dataset.data(shuffle=True))
        gibbs_scores[idmap[tuple(canonical_fn(mm.assignments()))]] += 1
    gibbs_scores /= gibbs_scores.sum()

    kldiv = kl(actual_scores, gibbs_scores)
    print kldiv
    assert kldiv <= threshold

def test_dirichlet_fixed_convergence():
    N = 4
    D = 5
    K = 2
    ddmm = DirichletFixed(N, K, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    # XXX: copy.deepcopy() doesn't work for our models, so we manually create a new one
    actual_ddmm = DirichletFixed(N, K, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    def all_possible_assignments_fn(N):
        return it.product(range(K), repeat=N)
    canonical_fn = lambda x: x
    _test_mixture_model_convergence(
            ddmm, actual_ddmm, N, all_possible_assignments_fn,
            canonical_fn, gibbs_assign_fixed)

def test_dirichlet_process_convergence():
    N = 4
    D = 5
    dpmm = DirichletProcess(N, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    # XXX: copy.deepcopy() doesn't work for our models, so we manually create a new one
    actual_dpmm = DirichletProcess(N, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    _test_mixture_model_convergence(
            dpmm, actual_dpmm, N, permutation_iter,
            permutation_canonical, gibbs_assign)

def test_different_datatypes():
    N = 10
    likelihoods = [bb, gp, nich, bb]
    hyperparams = [
        {'alpha':1.0, 'beta':3.0},
        {'alpha':2.0, 'inv_beta':1.0},
        {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
        {'alpha':2.0, 'beta':1.0}]
    dpmm = DirichletProcess(N, {'alpha':2.0}, likelihoods, hyperparams)
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
