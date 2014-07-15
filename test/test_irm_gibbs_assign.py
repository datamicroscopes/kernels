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

from microscopes.cxx.models import bb, bbnc
from microscopes.cxx.kernels.gibbs import assign, assign_resample
from microscopes.cxx.kernels.slice import theta

from microscopes.py.common.util import KL_discrete, logsumexp

import numpy as np
import numpy.ma as ma

import itertools as it
import time

from nose.tools import assert_almost_equals
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
    for i in xrange(D):
        mm_s.set_feature_hp(i, {'alpha':1.,'beta':1.})
    irm_s.set_domain_hp(0, {'alpha':2.})
    irm_s.set_relation_hp(0, {'alpha':1.,'beta':1.})

    perms = list(permutation_iter(N))
    assignment = perms[np.random.randint(0, len(perms))]

    mm_fill(mm_s, cluster(Y_rec, assignment), r)
    irm_fill(irm_s, [_assign_to_clustering(assignment), [[i] for i in xrange(D)]], [irm_view], r)

    def assert_suff_stats_equal():
        assert set(mm_s.groups()) == set(irm_s.groups(0))
        assert irm_s.groups(1) == range(D)
        groups = mm_s.groups()
        for g in groups:
            for i in xrange(D):
                a = mm_s.get_suffstats(g, i)
                b = irm_s.get_suffstats(0, [g, i])
                if b is None:
                    b = {'heads':0L, 'tails':0L}
                assert a['heads'] == b['heads'] and a['tails'] == b['tails']

    assert_suff_stats_equal()
    assert_almost_equals( mm_s.score_assignment(), irm_s.score_assignment(0), places=3 )

    bound_mm_s = mm_bind(mm_s, mm_view)
    bound_irm_s = irm_bind(irm_s, 0, [irm_view])

    # XXX: doesn't really have to be true, just is true of impl
    assert not bound_mm_s.empty_groups()
    assert not bound_irm_s.empty_groups()

    bound_mm_s.create_group(r)
    bound_irm_s.create_group(r)

    gid_a = bound_mm_s.remove_value(0, r)
    gid_b = bound_irm_s.remove_value(0, r)

    assert gid_a == gid_b
    assert_suff_stats_equal()

    x0, y0 = bound_mm_s.score_value(0, r)
    x1, y1 = bound_irm_s.score_value(0, r)
    assert x0 == x1 # XXX: not really a requirement

    # XXX: should really normalize and then check
    for a, b in zip(y0, y1):
        assert_almost_equals(a, b, places=2)

#@attr('wip')
#def test_compare_to_mixture_model_1():
#
#    r = rng()
#
#    N, D = 3, 5
#
#    Y = np.random.uniform(size=(N,D)) > 0.8
#    Y_rec = np.array([tuple(y) for y in Y], dtype=[('',bool)]*D)
#
#    mm_view = rec_numpy_dataview(Y_rec)
#
#    irm_view = spnd_numpy_dataview(Y)
#
#    mm_s = mm_state(N, [bbnc]*D)
#    irm_s = irm_state([N, D], [((0,1),bbnc)])
#
#    def init_mm(s):
#        s.set_cluster_hp({'alpha':2.})
#        for i in xrange(D):
#            s.set_feature_hp(i, {'alpha':1.,'beta':1.})
#        return s
#
#    def init_irm(s):
#        s.set_domain_hp(0, {'alpha':2.})
#        s.set_relation_hp(0, {'alpha':1.,'beta':1.})
#        return s
#
#    init_mm(mm_s)
#    init_irm(irm_s)
#
#    def posterior_mm(assignments):
#        brute = init_mm(mm_state(N, [bb]*D))
#        mm_fill(brute, cluster(Y_rec, assignments), r)
#        assign = brute.score_assignment()
#        likelihood = brute.score_data(None, None, r)
#        return assign + likelihood
#
#    def posterior_irm(assignments):
#        brute = init_irm(irm_state([N, D], [((0,1),bb)]))
#        irm_fill(brute, [_assign_to_clustering(assignments), [[i] for i in xrange(D)]], [irm_view], r)
#        assign = brute.score_assignment(0)
#        likelihood = brute.score_likelihood(r)
#        return assign + likelihood
#
#    mm_actual_scores = np.array(map(posterior_mm, permutation_iter(N)))
#    mm_actual_scores -= logsumexp(mm_actual_scores)
#    mm_actual_scores = np.exp(mm_actual_scores)
#
#    irm_actual_scores = np.array(map(posterior_irm, permutation_iter(N)))
#    irm_actual_scores -= logsumexp(irm_actual_scores)
#    irm_actual_scores = np.exp(irm_actual_scores)
#
#    assert_almost_equals(np.abs(mm_actual_scores - irm_actual_scores).max(), 0, places=5)
#    print 'actual_scores', mm_actual_scores
#
#    perms = list(permutation_iter(N))
#    assignment = perms[np.random.randint(0, len(perms))]
#    idmap = { C : i for i, C in enumerate(permutation_iter(N)) }
#
#    print 'number of initial groups:', len(set(assignment))
#
#    mm_fill(mm_s, cluster(Y_rec, assignment), r)
#    irm_fill(irm_s, [_assign_to_clustering(assignment), [[i] for i in xrange(D)]], [irm_view], r)
#
#    bound_mm_s = mm_bind(mm_s, mm_view)
#    bound_irm_s_d0 = irm_bind(irm_s, 0, [irm_view])
#
#    def mkparam():
#        return {'p':0.05}
#    params = { 0 : mkparam() }
#    def kernel(s, r):
#        assign_resample(s, 10, r)
#        #theta(s, params, r)
#
#    smoothing = 1e-5
#    mm_scores = np.zeros(len(perms)) + smoothing
#    irm_scores = np.zeros(len(perms)) + smoothing
#
#    nsamples = 1000
#    skip = 10
#
#    #nsamples, skip = 1, 1
#
#    print 'invocations:', bbnc_create_feature_group_invocations()
#
#    for _ in xrange(nsamples):
#        for _ in xrange(skip):
#            kernel(bound_mm_s, r)
#        mm_scores[idmap[tuple(permutation_canonical(bound_mm_s.assignments()))]] += 1
#
#    print 'invocations:', bbnc_create_feature_group_invocations()
#
#    for _ in xrange(nsamples):
#        for _ in xrange(skip):
#            kernel(bound_irm_s_d0, r)
#        irm_scores[idmap[tuple(permutation_canonical(bound_irm_s_d0.assignments()))]] += 1
#
#    print 'invocations:', bbnc_create_feature_group_invocations()
#
#    mm_scores /= mm_scores.sum()
#    irm_scores /= irm_scores.sum()
#    kldiv = KL_discrete(mm_scores, irm_scores)
#    print 'mm:', mm_scores
#    print 'irm:', irm_scores
#    print 'kl(mm, irm):', kldiv
#    print 'kl(mm, actual):', KL_discrete(mm_scores, mm_actual_scores)
#    print 'kl(irm, actual):', KL_discrete(irm_scores, mm_actual_scores)


def _test_convergence(domains, data, reg_relations, brute_relations, kernel,
                      burnin_niters=10000,
                      nsamples=1000,
                      skip=10,
                      attempts=5,
                      threshold=0.01):
    r = rng()
    assert len(domains) == 1

    def mk(relations):
        s = irm_state(domains, relations)
        for d in xrange(len(domains)):
            s.set_domain_hp(d, {'alpha':2.0})
        for r in xrange(len(relations)):
            s.set_relation_hp(r, {'alpha':1., 'beta':1.})
        return s

    idmap = { C : i for i, C in enumerate(permutation_iter(domains[0])) }
    def posterior(assignments):
        brute = mk(brute_relations)
        irm_fill(brute, [_assign_to_clustering(assignments)], data, r)
        assign = brute.score_assignment(0)
        likelihood = brute.score_likelihood(r)
        return assign + likelihood
    actual_scores = np.array(map(posterior, permutation_iter(domains[0])))
    actual_scores -= logsumexp(actual_scores)
    actual_scores = np.exp(actual_scores)

    s = mk(reg_relations)
    irm_random_initialize(s, data, r)
    bound_s0 = irm_bind(s, 0, data)

    # burnin
    start = time.time()
    for i in xrange(burnin_niters):
        kernel(bound_s0, r)
        if not ((i+1) % 1000):
            print 'burning finished iteration', (i+1), 'in', (time.time() - start), 'seconds'
            start = time.time()

    print 'finished burnin of', burnin_niters, 'iters'

    smoothing = 1e-5
    gibbs_scores = np.zeros(len(actual_scores)) + smoothing

    outer = attempts
    last_kl = None
    while outer > 0:
        # now grab nsamples samples, every skip iters
        for _ in xrange(nsamples):
            for _ in xrange(skip):
                kernel(bound_s0, r)
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

def test_one_binary():
    # 1 domain, 1 binary relation
    domains = [4]
    def mk_relations(model): return [((0,0), model)]
    data = [spnd_numpy_dataview(
        ma.array(
            np.random.choice([False, True], size=(domains[0], domains[0])),
            mask=np.random.choice([False, True], size=(domains[0], domains[0]))))]
    _test_convergence(domains, data, mk_relations(bb), mk_relations(bb), assign)

def test_one_binary_nonconj_kernel():
    # 1 domain, 1 binary relation
    domains = [4]
    def mk_relations(model): return [((0,0), model)]
    data = [spnd_numpy_dataview(
        ma.array(
            np.random.choice([False, True], size=(domains[0], domains[0])),
            mask=np.random.choice([False, True], size=(domains[0], domains[0]))))]
    kernel = lambda s, r: assign_resample(s, 10, r)
    _test_convergence(domains, data, mk_relations(bb), mk_relations(bb), kernel)

def test_two_binary():
    # 1 domain, 2 binary relations
    domains = [4]
    def mk_relations(model): return [((0,0), model), ((0,0), model)]
    data = [
        spnd_numpy_dataview(
            ma.array(
                np.random.choice([False, True], size=(domains[0], domains[0])),
                mask=np.random.choice([False, True], size=(domains[0], domains[0])))),
        spnd_numpy_dataview(
            ma.array(
                np.random.choice([False, True], size=(domains[0], domains[0])),
                mask=np.random.choice([False, True], size=(domains[0], domains[0])))),
    ]
    _test_convergence(domains, data, mk_relations(bb), mk_relations(bb), assign)

def test_one_binary_one_ternary():
    # 1 domain, 1 binary, 1 ternary
    domains = [4]
    def mk_relations(model): return [((0,0), model), ((0,0,0), model)]
    data = [
        spnd_numpy_dataview(
            ma.array(
                np.random.choice([False, True], size=(domains[0], domains[0])),
                mask=np.random.choice([False, True], size=(domains[0], domains[0])))),
        spnd_numpy_dataview(
            ma.array(
                np.random.choice([False, True], size=(domains[0], domains[0], domains[0])),
                mask=np.random.choice([False, True], size=(domains[0], domains[0], domains[0])))),
    ]
    _test_convergence(domains, data, mk_relations(bb), mk_relations(bb), assign)

def test_one_binary_nonconj():
    # 1 domain, 1 binary relation, nonconj
    domains = [3]
    def mk_relations(model): return [((0,0), model)]
    data = [spnd_numpy_dataview(
        ma.array(
            np.random.choice([False, True], size=(domains[0], domains[0])),
            mask=np.random.random(size=(domains[0], domains[0]))>0.8))]
    def mkparam():
        return {'p':0.05}
    params = { 0 : mkparam() }
    def kernel(s, r):
        assign_resample(s, 10, r)
        theta(s, params, r)
    _test_convergence(domains, data, mk_relations(bbnc), mk_relations(bb), kernel)
