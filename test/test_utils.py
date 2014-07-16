"""
routines to make writing test cases less painful
"""

import numpy as np
import numpy.ma as ma
import itertools as it
from microscopes.py.common.util import \
        KL_approx, KL_discrete, logsumexp
from microscopes.py.mixture.model import \
        fill as mixturemodel_fill
from microscopes.cxx.irm.model import \
        fill as irm_fill
from nose.tools import assert_almost_equals

class OurAssertionError(Exception):
    def __init__(self, ex):
        self._ex = ex

def our_assert_almost_equals(first, second, places=None, msg=None, delta=None):
    try:
        assert_almost_equals(first, second, places=places, msg=msg, delta=delta)
    except AssertionError as ex:
        raise OurAssertionError(ex)

def assert_1d_cont_dist_approx_sps(sample_fn,
                                   rv,
                                   support=None,
                                   ntries=5,
                                   nsamples=1000,
                                   nbins=1000,
                                   mean_places=3,
                                   var_places=3,
                                   kl_places=3):
    """
    Assert that the distributions of samples from sample_fn
    approaches the 1D continuous (real) distribution described by
    the (scipy.stats) rv object.

    Currently, three statistics are checked for convergence:
      (a) mean
      (b) variance
      (c) approximate KL-divergence
    """

    if support is None:
        support = rv.interval(1)
    if np.isinf(support[0]) or np.isinf(support[1]):
        raise ValueError("support is infinite: " + support)
    if support[1] <= support[0]:
        raise ValueError("support is empty")
    if ntries <= 0:
        raise ValueError("bad ntries: " + ntries)

    smoothing = 1e-5
    true_mean, true_var = rv.mean(), rv.var()
    raw_samples = []
    while 1:
        raw_samples.extend(sample_fn() for _ in xrange(nsamples))
        samples = np.array(raw_samples, dtype=np.float)
        try:
            # estimate mean
            est_mean = samples.mean()
            print 'true_mean', true_mean, 'est_mean', est_mean, 'diff', np.abs(true_mean - est_mean)
            our_assert_almost_equals(true_mean, est_mean, places=mean_places)

            # estimate variance
            est_var = samples.var(ddof=1) # used unbiased estimator
            print 'true_var', true_var, 'est_var', est_var, 'diff', np.abs(true_var - est_var)
            our_assert_almost_equals(true_var, est_var, places=var_places)

            # estimate empirical KL
            bins = np.linspace(support[0], support[1], nbins)

            est_hist, _ = np.histogram(samples, bins=bins, density=False)
            est_hist = np.array(est_hist, dtype=np.float)
            est_hist += smoothing
            est_hist /= est_hist.sum()

            points = (bins[1:]+bins[:-1])/2.
            actual_hist = rv.pdf(points)
            actual_hist /= actual_hist.sum()

            kldiv = KL_approx(actual_hist, est_hist, bins[1]-bins[0])
            print 'kldiv:', kldiv
            our_assert_almost_equals(kldiv, 0., places=kl_places)

            return # success
        except OurAssertionError as ex:
            print 'warning:', ex._ex.message
            ntries -= 1
            if not ntries:
                raise ex._ex

def assert_discrete_dist_approx(sample_fn,
                                dist,
                                ntries=5,
                                nsamples=1000,
                                kl_places=3):
    """
    Assert that the distributions of samples from sample_fn
    approaches the discrete distribution given by dist

    Currently, this is done by checking the KL-divergence
    (in both directions)
    """

    assert_almost_equals(dist.sum(), 1.0, places=4)

    if ntries <= 0:
        raise ValueError("bad ntries: " + ntries)

    smoothing = 1e-5
    est_hist = np.zeros(len(dist), dtype=np.int)
    while 1:
        for _ in xrange(nsamples):
            est_hist[sample_fn()] += 1
        try:
            hist = np.array(est_hist, dtype=np.float)
            hist += smoothing
            hist /= hist.sum()

            ab = KL_discrete(hist, dist)
            ba = KL_discrete(dist, hist)

            print 'KL_discrete(emp, act):', ab
            print 'KL_discrete(act, emp):', ba

            our_assert_almost_equals(ab, 0., places=kl_places)
            our_assert_almost_equals(ba, 0., places=kl_places)

            return # success
        except OurAssertionError as ex:
            print 'warning:', ex._ex.message
            ntries -= 1
            if not ntries:
                raise ex._ex

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

def mixturemodel_cluster(Y, assignments):
    """
    takes Y, a numpy struct (possibly masked) array, and
    generates a clustering which can be passed as an argument
    to fill()
    """
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

def irm_cluster(assignment):
    """
    takes an assignment and turns it into a clustering which
    can be passed as an argument to fill()
    """
    k = {}
    for eid, gid in enumerate(assignment):
        v = k.get(gid, [])
        v.append(eid)
        k[gid] = v
    return list(k.values())

def dist_on_all_clusterings(score_fn, N):
    """
    Enumerate all possible clusterings of N entities, calling
    score_fn with each assignment.

    The reslting enumeration is then turned into a valid
    discrete probability distribution
    """
    scores = np.array(map(score_fn, permutation_iter(N)))
    scores -= logsumexp(scores)
    scores = np.exp(scores)
    return scores

def mixturemodel_posterior(factory_fn, Y):
    """
    Invoking factory_fn should produce a new mixture model
    state object which is ready to have fill called on it
    """
    N = Y.shape[0]

    def score_fn(assignments):
        s = factory_fn()
        data = mixturemodel_cluster(Y, assignments)
        mixturemodel_fill(s, data)
        return s.score_joint()

    return dist_on_all_clusterings(score_fn, N)

def irm_single_domain_posterior(factory_fn, data, r):
    proto = factory_fn()
    N = proto.nentities(0)
    assert proto.ndomains() == 1

    def score_fn(assignments):
        s = factory_fn()
        irm_fill(s, [irm_cluster(assignments)], data, r)
        assign = s.score_assignment(0)
        likelihood = s.score_likelihood(r)
        return assign + likelihood

    return dist_on_all_clusterings(score_fn, N)
