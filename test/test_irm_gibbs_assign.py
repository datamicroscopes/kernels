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
from test_utils import \
        assert_discrete_dist_approx, \
        irm_single_domain_posterior, \
        permutation_iter, \
        permutation_canonical, \
        mixturemodel_cluster, \
        irm_cluster

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

    mm_fill(mm_s, mixturemodel_cluster(Y_rec, assignment), r)
    irm_fill(irm_s, [irm_cluster(assignment), [[i] for i in xrange(D)]], [irm_view], r)

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

def _test_convergence(domain_size,
                      data,
                      reg_relations,
                      brute_relations,
                      kernel,
                      burnin_niters=10000,
                      nsamples=1000,
                      skip=10,
                      attempts=5,
                      places=2):
    """
    one domain, beta-bernoulli relations only
    """
    r = rng()

    def mk(relations):
        s = irm_state([domain_size], relations)
        s.set_domain_hp(0, {'alpha':2.0})
        for r in xrange(len(relations)):
            s.set_relation_hp(r, {'alpha':1., 'beta':1.})
        return s

    posterior = irm_single_domain_posterior(lambda: mk(brute_relations), data, r)

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

    idmap = { C : i for i, C in enumerate(permutation_iter(domain_size)) }
    def sample_fn():
        for _ in xrange(skip):
            kernel(bound_s0, r)
        return idmap[tuple(permutation_canonical(bound_s0.assignments()))]

    assert_discrete_dist_approx(
            sample_fn, posterior,
            ntries=attempts, nsamples=nsamples,
            kl_places=places)

def test_one_binary():
    # 1 domain, 1 binary relation
    domains = [4]
    def mk_relations(model): return [((0,0), model)]
    data = [spnd_numpy_dataview(
        ma.array(
            np.random.choice([False, True], size=(domains[0], domains[0])),
            mask=np.random.choice([False, True], size=(domains[0], domains[0]))))]
    _test_convergence(domains[0], data, mk_relations(bb), mk_relations(bb), assign)

def test_one_binary_nonconj_kernel():
    # 1 domain, 1 binary relation
    domains = [4]
    def mk_relations(model): return [((0,0), model)]
    data = [spnd_numpy_dataview(
        ma.array(
            np.random.choice([False, True], size=(domains[0], domains[0])),
            mask=np.random.choice([False, True], size=(domains[0], domains[0]))))]
    kernel = lambda s, r: assign_resample(s, 10, r)
    _test_convergence(domains[0], data, mk_relations(bb), mk_relations(bb), kernel)

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
    _test_convergence(domains[0], data, mk_relations(bb), mk_relations(bb), assign)

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
    _test_convergence(domains[0], data, mk_relations(bb), mk_relations(bb), assign)

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
    _test_convergence(domains[0], data, mk_relations(bbnc), mk_relations(bb), kernel, attempts=10)
