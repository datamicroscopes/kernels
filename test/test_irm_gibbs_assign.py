from microscopes.cxx.irm.model import state, bind, fill, random_initialize
from microscopes.cxx.common.rng import rng
from microscopes.cxx.common.sparse_ndarray.dataview import numpy_dataview
from microscopes.cxx.models import bb
from microscopes.cxx.kernels.gibbs import assign

from microscopes.py.common.util import KL_discrete

import numpy as np
import numpy.ma as ma

import itertools as it
from scipy.misc import logsumexp

# XXX: dont duplicate code -- need a general model convergence test
# based on entity_state objects!

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

def test_simple():
    # 1 domain, 1 binary relation

    r = rng()

    domains = [3]
    relations = [((0,0), bb)]
    data = [numpy_dataview(
        ma.array(
            np.random.choice([False, True], size=(domains[0], domains[0])),
            mask=np.random.choice([False, True], size=(domains[0], domains[0]))))]

    def mk():
        s = state(domains, relations)
        s.set_domain_hp(0, {'alpha':2.0})
        s.set_relation_hp(0, {'alpha':1., 'beta':1.})
        return s

    idmap = { C : i for i, C in enumerate(permutation_iter(domains[0])) }
    def posterior(assignments):
        k = {}
        for eid, gid in enumerate(assignments):
            v = k.get(gid, [])
            v.append(eid)
            k[gid] = v
        brute = mk()
        fill(brute, [k.values()], data, r)
        return brute.score_assignment() + brute.score_likelihood(r);
    actual_scores = np.array(map(posterior, permutation_iter(domains[0])))
    actual_scores -= logsumexp(actual_scores)
    actual_scores = np.exp(actual_scores)

    s = mk()
    random_initialize(s, data, r)
    bound_s0 = bind(s, 0, data)

    burnin_niters = 20000
    nsamples = 2000
    skip = 10
    attempts = 5
    threshold = 0.01

    # burnin
    for _ in xrange(burnin_niters):
        assign(bound_s0, r)

    print 'finished burnin of', burnin_niters, 'iters'

    smoothing = 1e-5
    gibbs_scores = np.zeros(len(actual_scores)) + smoothing

    outer = attempts
    last_kl = None
    while outer > 0:
        # now grab nsamples samples, every skip iters
        for _ in xrange(nsamples):
            for _ in xrange(skip):
                assign(bound_s0, r)
            gibbs_scores[idmap[tuple(permutation_canonical(bound_s0.assignments()))]] += 1
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
