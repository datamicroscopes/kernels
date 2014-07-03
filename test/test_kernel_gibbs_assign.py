from distributions.dbg.models import bb as py_bb, gp as py_gp, nich as py_nich

from microscopes.py.mixture.dp import state as py_state, sample, fill
from microscopes.py.common.dataview import numpy_dataview as py_numpy_dataview
from microscopes.py.kernels.gibbs import \
        gibbs_assign as py_gibbs_assign, \
        gibbs_assign_nonconj as py_gibbs_assign_nonconj
from microscopes.py.kernels.slice import slice_theta as py_slice_theta
from microscopes.py.kernels.bootstrap import likelihood as py_bootstrap_likelihood
from microscopes.py.models import bbnc as py_bbnc

from microscopes.cxx.mixture.model import state as cxx_state
from microscopes.cxx.models import bb as cxx_bb, gp as cxx_gp, nich as cxx_nich, bbnc as cxx_bbnc
from microscopes.cxx.common.dataview import numpy_dataview as cxx_numpy_dataview
from microscopes.cxx.common.rng import rng
from microscopes.cxx.kernels.bootstrap import likelihood as cxx_likelihood
from microscopes.cxx.kernels.gibbs import \
        assign as cxx_gibbs_assign, \
        assign_resample as cxx_gibbs_assign_nonconj
from microscopes.cxx.kernels.slice import theta as cxx_slice_theta
from microscopes.cxx.kernels.bootstrap import likelihood as cxx_bootstrap_likelihood

from microscopes.py.common.util import KL_discrete
from microscopes.py.models import bbnc

import itertools as it
import math
import numpy as np
import numpy.ma as ma
from scipy.misc import logsumexp

from nose.plugins.attrib import attr

def make_dp(ctor, n, models, clusterhp, featurehps):
    mm = ctor(n, models)
    mm.set_cluster_hp(clusterhp)
    for i, hp in enumerate(featurehps):
        mm.set_feature_hp(i, hp)
    return mm

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
    masks = [[] for _ in xrange(len(labels))] if hasattr(Y, 'mask') else None
    for ci, yi in zip(assignments, Y):
        clusters[labels[ci]].append(yi)
        if masks is not None:
            masks[labels[ci]].append(yi.mask)
    if masks is None:
        return tuple(np.array(c) for c in clusters)
    else:
        return tuple(ma.array(np.array(c), mask=m) for c, m in zip(clusters, masks))

def _test_convergence_simple(
    N,
    py_models,
    cxx_models,
    clusterhp,
    featurehps,
    preprocess_data_fn=None,
    burnin_niters=10000,
    nsamples=1000,
    skip=10,
    attempts=3,
    threshold=0.01):

    assert len(py_models) == len(cxx_models)
    assert len(featurehps) == len(py_models)

    # create python version
    py_s = make_dp(py_state, N, py_models, clusterhp, featurehps)

    # create C++ version
    cxx_s = make_dp(cxx_state, N, cxx_models, clusterhp, featurehps)

    # sample from generative process
    Y_clustered, _ = sample(N, py_s)
    Y = np.hstack(Y_clustered)
    assert Y.shape[0] == N

    # preprocess the data (e.g. add masks)
    if preprocess_data_fn:
        Y = preprocess_data_fn(Y)

    # brute force the posterior of the actual model
    py_brute_s = make_dp(py_state, N, py_models, clusterhp, featurehps)

    idmap = { C : i for i, C in enumerate(permutation_iter(N)) }
    # brute force the posterior of the actual model
    def posterior(assignments):
        py_brute_s.reset()
        data = cluster(Y, assignments)
        fill(py_brute_s, data)
        return py_brute_s.score_joint()
    actual_scores = np.array(map(posterior, permutation_iter(N)))
    actual_scores -= logsumexp(actual_scores)
    actual_scores = np.exp(actual_scores)

    # setup python version
    py_view = py_numpy_dataview(Y)
    py_bootstrap_likelihood(py_s, py_view.view(False))
    py_kernel = py_gibbs_assign

    # setup C++ version
    cxx_rng = rng(54389)
    cxx_view = cxx_numpy_dataview(Y)
    cxx_bootstrap_likelihood(cxx_s, cxx_view.view(False, cxx_rng), cxx_rng)
    cxx_kernel = cxx_gibbs_assign

    def test_model(s, dataset, kernel, prng):
        # burnin
        for _ in xrange(burnin_niters):
            kernel(s, dataset.view(True, prng), prng)

        print 'finished burnin of', burnin_niters, 'iters'

        smoothing = 1e-5
        gibbs_scores = np.zeros(len(actual_scores)) + smoothing

        outer = attempts
        last_kl = None
        while outer > 0:
            # now grab nsamples samples, every skip iters
            for _ in xrange(nsamples):
                for _ in xrange(skip):
                    kernel(s, dataset.view(True, prng), prng)
                gibbs_scores[idmap[tuple(permutation_canonical(s.assignments()))]] += 1
            gibbs_scores /= gibbs_scores.sum()
            kldiv = KL_discrete(actual_scores, gibbs_scores)
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

    test_model(py_s, py_view, py_kernel, None)
    test_model(cxx_s, cxx_view, cxx_kernel, cxx_rng)

def test_convergence_bb():
    N = 4
    D = 5
    _test_convergence_simple(
        N=N,
        py_models=[py_bb]*D,
        cxx_models=[cxx_bb]*D,
        clusterhp={'alpha':2.0},
        featurehps=[{'alpha':1.0,'beta':1.0}]*D)

def test_convergence_bb_missing():
    N = 4
    D = 5
    def preprocess_fn(Y):
        masks = [tuple(j == (i % len(Y)) for j in xrange(D)) for i in xrange(len(Y))]
        return ma.array(Y, mask=masks)
    _test_convergence_simple(
        N=N,
        py_models=[py_bb]*D,
        cxx_models=[cxx_bb]*D,
        clusterhp={'alpha':2.0},
        featurehps=[{'alpha':1.0,'beta':1.0}]*D,
        preprocess_data_fn=preprocess_fn)

def _test_nonconj_inference(ctor, bbncmodel, dataview, assign_nonconj_fn, slice_theta_fn, R):
    N = 1000
    D = 5
    dpmm = make_dp(ctor, N, [bbncmodel]*D, {'alpha':0.2}, [{'alpha':1.0, 'beta':1.0}]*D)
    while True:
        Y_clustered, cluster_samplers = sample(N, dpmm, R)
        if len(Y_clustered) == 2 and max(map(len, Y_clustered)) >= 0.7:
            break
    dominant = np.argmax(map(len, Y_clustered))
    truth = np.array([s.p for s in cluster_samplers[dominant]])
    print 'truth:', truth

    # see if we can learn the p-values for each of the two clusters. we proceed
    # by running gibbs_assign_nonconj, followed by slice sampling on the
    # posterior p(\theta | Y). we'll "cheat" a little by bootstrapping the
    # DP with the correct assignment (but not with the correct p-values)
    fill(dpmm, Y_clustered, R)
    Y = np.hstack(Y_clustered)
    view = dataview(Y)

    def mkparam():
        return {'p':0.1}
    thetaparams = { fi : mkparam() for fi in xrange(D) }
    def kernel():
        assign_nonconj_fn(dpmm, view.view(True, R), 2, R)
        slice_theta_fn(dpmm, thetaparams, R)

    def inference(niters):
        for _ in xrange(niters):
            kernel()
            groups = dpmm.groups()
            inferred_dominant = groups[np.argmax([dpmm.groupsize(gid) for gid in groups])]
            inferred = np.array([dpmm.get_suff_stats(inferred_dominant, d)['p'] for d in xrange(D)])
            yield inferred

    posterior = list(inference(100))
    inferred = sum(posterior) / len(posterior)
    diff = np.linalg.norm(truth-inferred)

    print 'inferred:', inferred
    print 'diff:', diff
    assert diff <= 0.2

@attr('wip')
def test_nonconj_inference_cxx():
    _test_nonconj_inference(
            cxx_state, cxx_bbnc, cxx_numpy_dataview,
            cxx_gibbs_assign_nonconj, cxx_slice_theta, rng())

#@attr('slow')
#def test_nonconj_inference_kl():
#    N = 2
#    D = 5
#    dpmm, actual_dpmm = \
#        make_dp(N, [bb]*D, {'alpha':2.0}, [{'alpha':1.0, 'beta':1.0}]*D), \
#        make_dp(N, [bb]*D, {'alpha':2.0}, [{'alpha':1.0, 'beta':1.0}]*D)
#    def mkparam():
#        return {'thetaw':{'p':0.1}}
#    thetaparams = { fi : mkparam() for fi in xrange(D) }
#    def kernel_fn(mm, it):
#        gibbs_assign_nonconj(mm, it, nonempty=10)
#        slice_theta(mm, thetaparams)
#    _test_mixture_model_convergence(
#            dpmm, actual_dpmm, N, permutation_iter,
#            permutation_canonical, kernel_fn)

#def test_different_datatypes():
#    N = 10
#    likelihoods = [bb, gp, nich, bb]
#    hyperparams = [
#        {'alpha':1.0, 'beta':3.0},
#        {'alpha':2.0, 'inv_beta':1.0},
#        {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
#        {'alpha':2.0, 'beta':1.0}]
#    dpmm = make_dp(N, likelihoods, {'alpha':2.0}, hyperparams)
#    Y_clustered, _ = dpmm.sample(N)
#    Y = np.hstack(Y_clustered)
#    assert Y.shape[0] == N
#    dataset = numpy_dataview(Y)
#    dpmm.bootstrap(dataset.data(shuffle=False))
#
#    # make sure it deals with different types
#    for _ in xrange(10):
#        gibbs_assign(dpmm, dataset.data(shuffle=True))
#
#    for typ, y in zip(likelihoods, Y[0]):
#        assert typ.Value == type(np.asscalar(y))
