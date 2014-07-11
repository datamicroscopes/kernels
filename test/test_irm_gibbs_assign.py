from microscopes.cxx.irm.model import \
        state as irm_state, \
        bind as irm_bind , \
        fill as irm_fill, \
        random_initialize as irm_random_initialize
from microscopes.cxx.mixture.model import \
        state as mm_state, \
        bind as mm_bind
from microscopes.py.mixture.model import fill as mm_fill
from microscopes.cxx.common.rng import rng
from microscopes.cxx.common.sparse_ndarray.dataview import \
        numpy_dataview as spnd_numpy_dataview
from microscopes.cxx.common.recarray.dataview import \
        numpy_dataview as rec_numpy_dataview

from microscopes.cxx.models import bb
from microscopes.cxx.kernels.gibbs import assign

from microscopes.py.common.util import KL_discrete

import numpy as np
import numpy.ma as ma

import itertools as it
from scipy.misc import logsumexp

from nose.plugins.attrib import attr

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

def _assign_to_clustering(assignment):
    k = {}
    for eid, gid in enumerate(assignment):
        v = k.get(gid, [])
        v.append(eid)
        k[gid] = v
    return list(k.values())

@attr('wip')
def test_compare_to_mixture_model():
    r = rng()

    N, D = 4, 5

    Y = np.random.uniform(size=(N,D)) > 0.8
    Y_rec = np.array([tuple(y) for y in Y], dtype=[('',bool)]*D)

    mm_view = rec_numpy_dataview(Y_rec)

    irm_view = spnd_numpy_dataview(Y)

    mm_s = mm_state(N, [bb]*D)
    irm_s = irm_state([N, D], [((0,1),bb)])

    mm_s.set_cluster_hp({'alpha':2.})
    mm_s.set_feature_hp(0, {'alpha':1.,'beta':1.})
    irm_s.set_domain_hp(0, {'alpha':2.})
    irm_s.set_relation_hp(0, {'alpha':1.,'beta':1.})

    perms = list(permutation_iter(N))
    assignment = perms[np.random.randint(0, len(perms))]
    print assignment

    mm_fill(mm_s, cluster(Y_rec, assignment), r)
    irm_fill(irm_s, [_assign_to_clustering(assignment), [[i] for i in xrange(D)]], [irm_view], r)

    bound_mm_s = mm_bind(mm_s, mm_view)
    bound_irm_s = irm_bind(irm_s, 0, [irm_view])

    # doesn't really have to be true, just is true of impl
    assert not bound_mm_s.empty_groups()
    assert not bound_irm_s.empty_groups()

    bound_mm_s.create_group(r)
    bound_irm_s.create_group(r)

    gid_a = bound_mm_s.remove_value(0, r)
    gid_b = bound_irm_s.remove_value(0, r)

    assert gid_a == gid_b

    print bound_mm_s.score_value(0, r)
    print bound_irm_s.score_value(0, r)


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
        s = irm_state(domains, relations)
        s.set_domain_hp(0, {'alpha':2.0})
        s.set_relation_hp(0, {'alpha':1., 'beta':1.})
        return s

    idmap = { C : i for i, C in enumerate(permutation_iter(domains[0])) }
    def posterior(assignments):
        brute = mk()
        irm_fill(brute, [_assign_to_clustering(assignments)], data, r)
        return brute.score_assignment() + brute.score_likelihood(r);
    actual_scores = np.array(map(posterior, permutation_iter(domains[0])))
    actual_scores -= logsumexp(actual_scores)
    actual_scores = np.exp(actual_scores)

    s = mk()
    irm_random_initialize(s, data, r)
    bound_s0 = irm_bind(s, 0, data)

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
