from distributions.dbg.models import bb, gp, nich

from microscopes.models.mixture.dp import DirichletProcess
from microscopes.models.mixture.dd import DirichletFixed
from microscopes.common.dataset import numpy_dataset
from microscopes.kernels.gibbs import \
        gibbs_assign, gibbs_assign_fixed, gibbs_assign_nonconj
from microscopes.kernels.slice import slice_theta
from microscopes.distributions import bbnc

from nose.plugins.attrib import attr

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
        kernel_fn,
        preprocess_data_fn=None,
        burnin_niters=10000,
        nsamples=1000,
        skip=10,
        threshold=0.1):
    Y_clustered, _ = mm.sample(N)
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

    if preprocess_data_fn:
        Yp = preprocess_data_fn(Y)
    else:
        Yp = Y
    dataset = numpy_dataset(Yp)
    mm.bootstrap(dataset.data(shuffle=False))

    # burnin
    for _ in xrange(burnin_niters):
        kernel_fn(mm, dataset.data(shuffle=True))

    print 'finished burnin of', burnin_niters, 'iters'

    smoothing = 1e-5
    gibbs_scores = np.zeros(len(actual_scores)) + smoothing

    outer = 3
    last_kl = None
    while outer > 0:
        # now grab nsamples samples, every skip iters
        for _ in xrange(nsamples):
            for _ in xrange(skip):
                kernel_fn(mm, dataset.data(shuffle=True))
            gibbs_scores[idmap[tuple(canonical_fn(mm.assignments()))]] += 1
        gibbs_scores /= gibbs_scores.sum()
        kldiv = kl(actual_scores, gibbs_scores)
        print 'actual:', actual_scores
        print 'gibbs:', gibbs_scores
        print 'kl:', kldiv
        if kldiv <= threshold:
            return
        if last_kl is not None and kldiv >= last_kl:
            print 'WARNING: KL is not making progress!'
            print 'last KL:', last_kl
            print 'cur KL:', kldiv
        last_kl = kldiv
        outer -= 1
        print 'WARNING: did not converge, trying', outer, 'more times'

    assert False, 'failed to converge!'

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

@attr('slow')
def test_nonconj_inference():
    N = 1000
    D = 5
    dpmm = DirichletProcess(N, {'alpha':0.2}, [bbnc]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    while True:
        Y_clustered, cluster_samplers = dpmm.sample(N)
        if len(Y_clustered) == 2 and max(map(len, Y_clustered)) >= 0.7:
            break
    dominant = np.argmax(map(len, Y_clustered))
    truth = np.array([s.p for s in cluster_samplers[dominant]])
    print 'truth:', truth

    # see if we can learn the p-values for each of the two clusters. we proceed
    # by running gibbs_assign_nonconj, followed by slice sampling on the
    # posterior p(\theta | Y). we'll "cheat" a little by bootstrapping the
    # DP with the correct assignment (but not with the correct p-values)
    dpmm.fill(Y_clustered)
    Y = np.hstack(Y_clustered)
    dataset = numpy_dataset(Y)

    def mkparam():
        return {'thetaw':{'p':0.1}}
    thetaparams = { fi : mkparam() for fi in xrange(D) }
    def kernel():
        gibbs_assign_nonconj(dpmm, dataset.data(shuffle=True), nonempty=2)
        slice_theta(dpmm, thetaparams)

    def inference(niters):
        for _ in xrange(niters):
            kernel()
            groups = dpmm.groups()
            inferred_dominant = groups[np.argmax([dpmm.nentities_in_group(gid) for gid in groups])]
            inferred = np.array([
                [gdata.dump()['p'] for gid, gdata in dpmm.get_suff_stats(d) if gid == inferred_dominant][0] \
                    for d in xrange(D)])
            yield inferred

    posterior = list(inference(100))
    inferred = sum(posterior) / len(posterior)
    diff = np.linalg.norm(truth-inferred)

    print 'inferred:', inferred
    print 'diff:', diff
    assert diff <= 0.2

@attr('slow')
def test_nonconj_inference_kl():
    N = 2
    D = 5
    dpmm = DirichletProcess(N, {'alpha':2.0}, [bbnc]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    actual_dpmm = DirichletProcess(N, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    def mkparam():
        return {'thetaw':{'p':0.1}}
    thetaparams = { fi : mkparam() for fi in xrange(D) }
    def kernel_fn(mm, it):
        gibbs_assign_nonconj(mm, it, nonempty=10)
        slice_theta(mm, thetaparams)
    _test_mixture_model_convergence(
            dpmm, actual_dpmm, N, permutation_iter,
            permutation_canonical, kernel_fn)

def test_missing_data_inference_kl():
    N = 3
    D = 10
    dpmm = DirichletProcess(N, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    # XXX: copy.deepcopy() doesn't work for our models, so we manually create a new one
    actual_dpmm = DirichletProcess(N, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    def preprocess_fn(Y):
        import numpy.ma as ma
        masks = [tuple(j == (i % len(Y)) for j in xrange(D)) for i in xrange(len(Y))]
        return ma.array(Y, mask=masks)
    _test_mixture_model_convergence(
            dpmm, actual_dpmm, N, permutation_iter,
            permutation_canonical, gibbs_assign, preprocess_fn)

def test_different_datatypes():
    N = 10
    likelihoods = [bb, gp, nich, bb]
    hyperparams = [
        {'alpha':1.0, 'beta':3.0},
        {'alpha':2.0, 'inv_beta':1.0},
        {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
        {'alpha':2.0, 'beta':1.0}]
    dpmm = DirichletProcess(N, {'alpha':2.0}, likelihoods, hyperparams)
    Y_clustered, _ = dpmm.sample(N)
    Y = np.hstack(Y_clustered)
    assert Y.shape[0] == N
    dataset = numpy_dataset(Y)
    dpmm.bootstrap(dataset.data(shuffle=False))

    # make sure it deals with different types
    for _ in xrange(10):
        gibbs_assign(dpmm, dataset.data(shuffle=True))

    for typ, y in zip(likelihoods, Y[0]):
        assert typ.Value == type(np.asscalar(y))
