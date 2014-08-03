from microscopes.cxx.kernels.slice import theta
from microscopes.cxx.mixture.model import \
        initialize as mm_initialize, bind as mm_bind
from microscopes.cxx.irm.model import \
        initialize as irm_initialize, bind as irm_bind
from microscopes.cxx.common.recarray.dataview import \
        numpy_dataview as mm_numpy_dataview
from microscopes.cxx.common.relation.dataview import \
        numpy_dataview as irm_numpy_dataview
from microscopes.cxx.common.rng import rng
from microscopes.models import bbnc
from microscopes.mixture.definition import \
        model_definition as mm_model_definition
from microscopes.irm.definition import \
        model_definition as irm_model_definition

from microscopes.py.common.util import KL_approx
from test_utils import assert_1d_cont_dist_approx_sps
from scipy.stats import beta

import numpy as np

from nose.plugins.attrib import attr

def test_slice_theta_mm():
    N = 100
    data = np.array(
        [(np.random.random() < 0.8,) for _ in xrange(N)],
        dtype=[('',bool)])
    defn = mm_model_definition(N, [bbnc])
    r = rng()
    prior = {'alpha':1.0, 'beta':9.0}
    view = mm_numpy_dataview(data)
    s = mm_initialize(
        defn,
        view,
        cluster_hp={'alpha':1.,'beta':9.},
        feature_hps=[prior],
        r=r,
        assignment=[0]*N)

    heads = len([1 for y in data if y[0]])
    tails = N - heads

    alpha1 = prior['alpha'] + heads
    beta1 = prior['beta'] + tails

    bs = mm_bind(s, view)
    params = {0:{'p':0.05}}
    def sample_fn():
        theta(bs, r, tparams=params)
        return s.get_suffstats(0, 0)['p']

    rv = beta(alpha1, beta1)
    assert_1d_cont_dist_approx_sps(sample_fn, rv, nsamples=50000)

def test_slice_theta_irm():
    N = 10
    defn = irm_model_definition([N], [((0,0), bbnc)])
    data = np.random.random(size=(N,N)) < 0.8
    view = irm_numpy_dataview(data)
    r = rng()
    prior = {'alpha':1.0, 'beta':9.0}

    s = irm_initialize(
            defn,
            [view],
            r=r,
            cluster_hps=[{'alpha':2.0}],
            relation_hps=[prior],
            domain_assignments=[[0]*N])

    bs = irm_bind(s, 0, [view])

    params = {0:{'p':0.05}}

    heads = len([1 for y in data.flatten() if y])
    tails = len([1 for y in data.flatten() if not y])

    alpha1 = prior['alpha'] + heads
    beta1 = prior['beta'] + tails

    def sample_fn():
        theta(bs, r, tparams=params)
        return s.get_suffstats(0, [0,0])['p']

    rv = beta(alpha1, beta1)
    assert_1d_cont_dist_approx_sps(sample_fn, rv, nsamples=50000)
