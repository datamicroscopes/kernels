from distributions.dbg.models import \
        bb as py_bb, gp as py_gp, nich as py_nich
from microscopes.py.mixture.model import \
        state as py_state, sample, fill, bind as py_bind
from microscopes.py.common.recarray.dataview import \
        numpy_dataview as py_numpy_dataview
from microscopes.py.common.util import random_orthonormal_matrix
from microscopes.py.kernels.gibbs import \
        assign as py_gibbs_assign, \
        assign_resample as py_gibbs_assign_nonconj
from microscopes.py.kernels.slice import theta as py_slice_theta
from microscopes.py.kernels.bootstrap import \
        likelihood as py_bootstrap_likelihood
from microscopes.py.models import bbnc as py_bbnc, niw as py_niw
from microscopes.cxx.mixture.model import state as cxx_state, bind as cxx_bind
from microscopes.cxx.models import bb as cxx_bb, \
        gp as cxx_gp, \
        nich as cxx_nich, \
        bbnc as cxx_bbnc, \
        niw as cxx_niw
from microscopes.cxx.common.recarray.dataview import \
        numpy_dataview as cxx_numpy_dataview
from microscopes.cxx.common.rng import rng
from microscopes.cxx.kernels.bootstrap import likelihood as cxx_likelihood
from microscopes.cxx.kernels.gibbs import \
        assign as cxx_gibbs_assign, \
        assign_resample as cxx_gibbs_assign_nonconj
from microscopes.cxx.kernels.slice import theta as cxx_slice_theta
from microscopes.cxx.kernels.bootstrap import likelihood as cxx_bootstrap_likelihood

from microscopes.py.common.util import KL_discrete, logsumexp
from microscopes.py.models import bbnc

from test_utils import \
        assert_discrete_dist_approx, \
        mixturemodel_posterior, \
        permutation_iter, \
        permutation_canonical

import itertools as it
import math
import numpy as np
import numpy.ma as ma

from nose.plugins.attrib import attr

def make_dp(ctor, n, alpha, models):
    mm = ctor(n, [m[0] for m in models])
    mm.set_cluster_hp({'alpha':alpha})
    for i, hp in enumerate(models):
        mm.set_feature_hp(i, hp[1])
    return mm

def sample_data(n, alpha, models):
    s = make_dp(py_state, n, alpha, models)
    Y_clustered, _ = sample(n, s)
    Y = np.hstack(Y_clustered)
    assert Y.shape[0] == n
    return Y

def data_with_posterior(N, D, alpha, models, preprocess_data_fn):
    factory_fn = lambda: make_dp(py_state, N, alpha, models)
    Y = sample_data(N, alpha, models)
    if preprocess_data_fn:
        Y = preprocess_data_fn(Y)
    posterior = mixturemodel_posterior(factory_fn, Y)
    return Y, posterior

def _test_convergence(bs,
                      posterior,
                      kernel,
                      burnin_niters,
                      skip,
                      ntries,
                      nsamples,
                      kl_places):
    N = bs.nentities()
    for _ in xrange(burnin_niters):
        kernel(bs)
    idmap = { C : i for i, C in enumerate(permutation_iter(N)) }
    def sample_fn():
        for _ in xrange(skip):
            kernel(bs)
        return idmap[tuple(permutation_canonical(bs.assignments()))]
    assert_discrete_dist_approx(
            sample_fn, posterior,
            ntries=ntries, nsamples=nsamples, kl_places=kl_places)

def _test_convergence_bb_py(N,
                            D,
                            kernel,
                            preprocess_data_fn=None,
                            nonconj=False,
                            burnin_niters=10000,
                            skip=10,
                            ntries=5,
                            nsamples=1000,
                            kl_places=2):
    alpha = 2.0
    models = [(py_bb, {'alpha':1.0,'beta':1.0})]*D
    nonconj_models = [(py_bbnc, {'alpha':1.0,'beta':1.0})]*D
    Y, posterior = data_with_posterior(N, D, alpha, models, preprocess_data_fn)
    s = make_dp(py_state, N, alpha, nonconj_models if nonconj else models)
    view = py_numpy_dataview(Y)
    py_bootstrap_likelihood(s, view.view(False))
    bs = py_bind(s, view)
    _test_convergence(
        bs,
        posterior,
        kernel,
        burnin_niters,
        skip,
        ntries,
        nsamples,
        kl_places)

def _test_convergence_bb_cxx(N,
                             D,
                             kernel,
                             preprocess_data_fn=None,
                             nonconj=False,
                             burnin_niters=10000,
                             skip=10,
                             ntries=5,
                             nsamples=1000,
                             kl_places=2):
    r = rng()
    alpha = 2.0
    py_models = [(py_bb, {'alpha':1.0,'beta':1.0})]*D
    cxx_models = [(cxx_bb, {'alpha':1.0,'beta':1.0})]*D
    cxx_nonconj_models = [(cxx_bbnc, {'alpha':1.0,'beta':1.0})]*D
    Y, posterior = data_with_posterior(N, D, alpha, py_models, preprocess_data_fn)
    s = make_dp(cxx_state, N, alpha, cxx_nonconj_models if nonconj else cxx_models)
    view = cxx_numpy_dataview(Y)
    cxx_bootstrap_likelihood(s, view.view(False, r), r)
    bs = cxx_bind(s, view)
    wrapped_kernel = lambda s: kernel(s, r)
    _test_convergence(
        bs,
        posterior,
        wrapped_kernel,
        burnin_niters,
        skip,
        ntries,
        nsamples,
        kl_places)

def test_convergence_bb_py():
    N, D = 4, 5
    _test_convergence_bb_py(N, D, py_gibbs_assign)

def test_convergence_bb_py_missing():
    N, D = 4, 5
    def preprocess_fn(Y):
        masks = [tuple(j == (i % len(Y)) for j in xrange(D)) for i in xrange(len(Y))]
        return ma.array(Y, mask=masks)
    _test_convergence_bb_py(N, D, py_gibbs_assign, preprocess_fn)

def test_convergence_bb_cxx():
    N, D = 4, 5
    _test_convergence_bb_cxx(N, D, cxx_gibbs_assign)

def test_convergence_bb_cxx_missing():
    N, D = 4, 5
    def preprocess_fn(Y):
        masks = [tuple(j == (i % len(Y)) for j in xrange(D)) for i in xrange(len(Y))]
        return ma.array(Y, mask=masks)
    _test_convergence_bb_cxx(N, D, cxx_gibbs_assign, preprocess_fn)

@attr('slow')
def test_convergence_bb_nonconj_py():
    N, D = 3, 5
    thetaparams = {fi:{'p':0.1} for fi in xrange(D)}
    def kernel(s):
        py_gibbs_assign_nonconj(s, 10)
        py_slice_theta(s, thetaparams)
    _test_convergence_bb_py(N, D, kernel, preprocess_data_fn=None, nonconj=True)

def test_convergence_bb_nonconj_cxx():
    N, D = 3, 5
    thetaparams = {fi:{'p':0.1} for fi in xrange(D)}
    def kernel(s, r):
        cxx_gibbs_assign_nonconj(s, 10, r)
        cxx_slice_theta(s, thetaparams, r)
    _test_convergence_bb_cxx(N, D, kernel, preprocess_data_fn=None, nonconj=True)

def _test_multivariate_models(ctor, bbmodel, niwmodel, dataview, bootstrap, bind, gibbs_assign, R):
    # XXX: this test only checks that the operations don't crash
    mu0 = np.ones(3)
    lambda_ = 0.3
    Q = random_orthonormal_matrix(3)
    psi = np.dot(Q, np.dot(np.diag([1.0, 0.5, 0.2]), Q.T))
    nu = 6

    N = 10
    s = ctor(N, [bbmodel, niwmodel])
    s.set_cluster_hp({'alpha':2.})
    s.set_feature_hp(0, {'alpha':2.,'beta':2.})
    s.set_feature_hp(1, {'mu0':mu0, 'lambda':lambda_, 'psi':psi, 'nu': nu})

    def genrow():
        return tuple([np.random.choice([False,True]), [np.random.uniform(-3.0, 3.0) for _ in xrange(3)]])

    X = np.array([genrow() for _ in xrange(N)], dtype=[('',bool),('',float,(3,))])
    view = dataview(X)
    bootstrap(s, view.view(False, R), R)
    bound_s = bind(s, view)

    for _ in xrange(10):
        gibbs_assign(bound_s, R)

def test_multivariate_models_py():
    _test_multivariate_models(
        py_state,
        py_bb,
        py_niw,
        py_numpy_dataview,
        py_bootstrap_likelihood,
        py_bind,
        py_gibbs_assign,
        None)

def test_multivariate_models_cxx():
    _test_multivariate_models(
        cxx_state,
        cxx_bb,
        cxx_niw,
        cxx_numpy_dataview,
        cxx_bootstrap_likelihood,
        cxx_bind,
        cxx_gibbs_assign,
        rng())

def _test_nonconj_inference(ctor,
                            bbncmodel,
                            dataview,
                            bind,
                            assign_nonconj_fn,
                            slice_theta_fn,
                            R,
                            ntries,
                            nsamples,
                            tol):
    N = 1000
    D = 5
    dpmm = make_dp(ctor, N, 0.2, [(bbncmodel, {'alpha':1.0, 'beta':1.0})]*D)
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
    bound_dpmm = bind(dpmm, view)

    def mkparam():
        return {'p':0.1}
    thetaparams = { fi : mkparam() for fi in xrange(D) }
    def kernel():
        assign_nonconj_fn(bound_dpmm, 10, R)
        slice_theta_fn(bound_dpmm, thetaparams, R)

    def inference(niters):
        for _ in xrange(niters):
            kernel()
            groups = dpmm.groups()
            inferred_dominant = groups[np.argmax([dpmm.groupsize(gid) for gid in groups])]
            inferred = np.array([dpmm.get_suffstats(inferred_dominant, d)['p'] for d in xrange(D)])
            yield inferred

    posterior = []
    while ntries:
        samples = list(inference(nsamples))
        posterior.extend(samples)
        inferred = sum(posterior) / len(posterior)
        diff = np.linalg.norm(truth-inferred)
        print 'inferred:', inferred
        print 'diff:', diff
        if diff <= tol:
            return
        ntries -= 1
        print 'tries left:', ntries

    assert False, 'did not converge'

@attr('slow')
def test_nonconj_inference_py():
    _test_nonconj_inference(
            py_state, py_bbnc, py_numpy_dataview, py_bind,
            py_gibbs_assign_nonconj, py_slice_theta, None, ntries=5, nsamples=100, tol=0.2)

@attr('slow')
def test_nonconj_inference_cxx():
    _test_nonconj_inference(
            cxx_state, cxx_bbnc, cxx_numpy_dataview, cxx_bind,
            cxx_gibbs_assign_nonconj, cxx_slice_theta, rng(), ntries=5, nsamples=100, tol=0.2)
