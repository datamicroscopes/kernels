from microscopes.py.mixture.model import initialize as py_initialize, bind as py_bind
from microscopes.py.kernels.gibbs import hp as py_gibbs_hp
from microscopes.py.kernels.slice import hp as py_slice_hp

from microscopes.cxx.mixture.model import initialize as cxx_initialize, bind as cxx_bind
from microscopes.cxx.kernels.gibbs import hp as cxx_gibbs_hp
from microscopes.cxx.kernels.slice import hp as cxx_slice_hp

from microscopes.cxx.common.rng import rng
from microscopes.cxx.common.scalar_functions import \
    log_exponential, log_noninformative_beta_prior

from microscopes.py.common.recarray.dataview import numpy_dataview as py_numpy_dataview
from microscopes.cxx.common.recarray.dataview import numpy_dataview as cxx_numpy_dataview

from microscopes.models import bb
from microscopes.mixture.definition import model_definition

from microscopes.py.common.util import almost_eq
#from microscopes.py.kernels.mh import mh_hp

import numpy as np
import math
import scipy as sp
import scipy.stats

try:
    import matplotlib.pylab as plt
    has_plt = True
except ImportError:
    has_plt = False

import itertools as it

from test_utils import OurAssertionError, our_assert_almost_equals, assert_1d_cont_dist_approx_emp
from nose.plugins.attrib import attr

def _bb_hyperprior_pdf(hp):
    alpha, beta = hp['alpha'], hp['beta']
    if alpha > 0.0 and beta > 0.0:
        # http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-9.html
        return -2.5 * np.log(alpha + beta)
    return -np.inf

def data_with_assignment(Y_clusters):
    assignments = it.chain.from_iterable(
        [i]*len(cluster) for i, cluster in enumerate(Y_clusters))
    return np.hstack(Y_clusters), list(assignments)

def _make_one_feature_bb_mm(initialize_fn, dataview, Nk, K, alpha, beta, r):
    # XXX: the rng parameter passed does not get threaded through the
    # random *data* generation
    # use the py_bb for sampling
    py_bb = bb.py_desc()._model_module
    shared = py_bb.Shared()
    shared.load({'alpha':alpha,'beta':beta})
    def init_sampler():
        samp = py_bb.Sampler()
        samp.init(shared)
        return samp
    samplers = [init_sampler() for _ in xrange(K)]
    def gen_cluster(samp):
        data = [(samp.eval(shared),) for _ in xrange(Nk)]
        return np.array(data, dtype=[('', bool)])
    Y_clustered = tuple(map(gen_cluster, samplers))
    Y, assignment = data_with_assignment(Y_clustered)
    view = dataview(Y)
    s = initialize_fn(model_definition(Y.shape[0], [bb]),
                      view,
                      cluster_hp={'alpha':2.},
                      feature_hps=[{'alpha':alpha,'beta':beta}],
                      r=r,
                      assignment=assignment)
    return s, view

def _grid_actual(s, prior_fn, lo, hi, nelems, r):
    x = np.linspace(lo, hi, nelems)
    y = x.copy()
    xv, yv = np.meshgrid(x, y)
    z = np.zeros(xv.shape)
    for i in xrange(nelems):
        for j in xrange(nelems):
            alpha = xv[i, j]
            beta = yv[i, j]
            raw = {'alpha':alpha, 'beta':beta}
            s.set_feature_hp(0, raw)
            z[i, j] = prior_fn(raw) + s.score_data(0, None, r)
    return xv, yv, z

def _add_to_grid(xv, yv, z, value):
    xmin, xmax = xv.min(axis=1).min(), xv.max(axis=1).max()
    ymin, ymax = yv.min(axis=1).min(), yv.max(axis=1).max()
    if value[0] < xmin or value[0] > xmax or value[1] < ymin or value[1] > ymax:
        # do not add
        return False
    xrep = xv[0,:]
    yrep = yv[:,0]
    xidx = min(np.searchsorted(xrep, value[0]), len(xrep)-1)
    yidx = min(np.searchsorted(yrep, value[1]), len(yrep)-1)
    z[yidx, xidx] += 1
    return True

def _test_hp_inference(initialize_fn,
                       prior_fn,
                       grid_min,
                       grid_max,
                       grid_n,
                       dataview,
                       bind_fn,
                       init_inf_kernel_state_fn,
                       inf_kernel_fn,
                       map_actual_postprocess_fn,
                       grid_filename,
                       prng,
                       burnin=1000,
                       nsamples=1000,
                       skip=10,
                       trials=5,
                       tol=0.1):

    print '_test_hp_inference: burnin', burnin, 'nsamples', nsamples, 'skip', skip, 'trials', trials, 'tol', tol

    Nk = 1000
    K = 100
    s, view = _make_one_feature_bb_mm(
        initialize_fn, dataview, Nk, K, 0.8, 1.2, prng)
    bound_s = bind_fn(s, view)

    xgrid, ygrid, z_actual = _grid_actual(s, prior_fn, grid_min, grid_max, grid_n, prng)

    i_actual, j_actual = np.unravel_index(np.argmax(z_actual), z_actual.shape)
    assert almost_eq(z_actual[i_actual, j_actual], z_actual.max())
    alpha_map_actual, beta_map_actual = \
        xgrid[i_actual, j_actual], ygrid[i_actual, j_actual]
    map_actual = np.array([alpha_map_actual, beta_map_actual])
    map_actual_postproc = map_actual_postprocess_fn(map_actual)
    print 'MAP actual:', map_actual
    print 'MAP actual postproc:', map_actual_postproc

    th_draw = lambda: np.random.uniform(grid_min, grid_max)
    alpha0, beta0 = th_draw(), th_draw()
    s.set_feature_hp(0, {'alpha':alpha0,'beta':beta0})
    print 'start values:', alpha0, beta0

    z_sample = np.zeros(xgrid.shape)
    opaque = init_inf_kernel_state_fn(s)
    for _ in xrange(burnin):
        inf_kernel_fn(bound_s, opaque, prng)
    print 'finished burnin of', burnin, 'iterations'

    def trial():
        def posterior(k, skip):
            for _ in xrange(k):
                for _ in xrange(skip-1):
                    inf_kernel_fn(bound_s, opaque, prng)
                inf_kernel_fn(bound_s, opaque, prng)
                hp = s.get_feature_hp(0)
                yield np.array([hp['alpha'], hp['beta']])
        for samp in posterior(nsamples, skip):
            #print 'gridding:', samp
            _add_to_grid(xgrid, ygrid, z_sample, samp)

    def draw_grid_plot():
        if not has_plt:
            return
        plt.imshow(z_sample, cmap=plt.cm.binary, origin='lower',
            interpolation='nearest',
            extent=(grid_min, grid_max, grid_min, grid_max))
        plt.hold(True) # XXX: restore plt state
        plt.contour(np.linspace(grid_min, grid_max, grid_n),
                    np.linspace(grid_min, grid_max, grid_n),
                    z_actual)
        plt.savefig(grid_filename)
        plt.close()

    while trials:
        trial()
        i_sample, j_sample = np.unravel_index(np.argmax(z_sample), z_sample.shape)
        alpha_map_sample, beta_map_sample = \
            xgrid[i_sample, j_sample], ygrid[i_sample, j_sample]
        map_sample = np.array([alpha_map_sample, beta_map_sample])
        diff = np.linalg.norm(map_actual_postproc - map_sample)
        print 'map_sample:', map_sample, 'diff:', diff, 'trials left:', (trials-1)
        if diff <= tol:
            # draw plot and bail
            draw_grid_plot()
            return
        trials -= 1

    draw_grid_plot() # useful for debugging
    assert False, 'MAP value did not converge to desired tolerance'

def _test_kernel_gibbs_hp(initialize_fn, dataview, bind_fn, gibbs_hp_fn, fname, prng):
    grid_min, grid_max, grid_n = 0.01, 5.0, 10
    grid = tuple({'alpha':alpha,'beta':beta} \
        for alpha, beta in it.product(np.linspace(grid_min, grid_max, grid_n), repeat=2))

    def init_inf_kernel_state_fn(dpmm):
        hparams = {0:{'hpdf':_bb_hyperprior_pdf,'hgrid':grid}}
        return hparams

    def map_actual_postprocess_fn(map_actual):
        # find closest grid point to actual point
        dists = np.array([np.linalg.norm(np.array([g['alpha'], g['beta']]) - map_actual) for g in grid])
        closest = grid[np.argmin(dists)]
        closest = np.array([closest['alpha'], closest['beta']])
        return closest

    _test_hp_inference(
        initialize_fn,
        _bb_hyperprior_pdf,
        grid_min,
        grid_max,
        grid_n,
        dataview,
        bind_fn,
        init_inf_kernel_state_fn,
        gibbs_hp_fn,
        map_actual_postprocess_fn,
        grid_filename=fname,
        prng=prng,
        burnin=100,
        trials=10,
        nsamples=100)

@attr('slow')
def test_kernel_gibbs_hp_py():
    _test_kernel_gibbs_hp(py_initialize,
                          py_numpy_dataview,
                          py_bind,
                          py_gibbs_hp,
                          'grid_gibbs_hp_samples_py.pdf',
                          prng=None)

def test_kernel_gibbs_hp_cxx():
    _test_kernel_gibbs_hp(cxx_initialize,
                          cxx_numpy_dataview,
                          cxx_bind,
                          cxx_gibbs_hp,
                          'grid_gibbs_hp_samples_cxx.pdf',
                          rng())

def _test_kernel_slice_hp(initialize_fn,
                          init_inf_kernel_state_fn,
                          prior_fn,
                          dataview,
                          bind_fn,
                          slice_hp_fn,
                          fname,
                          prng):
    grid_min, grid_max, grid_n = 0.01, 5.0, 200
    _test_hp_inference(
        initialize_fn,
        prior_fn,
        grid_min,
        grid_max,
        grid_n,
        dataview,
        bind_fn,
        init_inf_kernel_state_fn,
        slice_hp_fn,
        map_actual_postprocess_fn=lambda x: x,
        grid_filename=fname,
        prng=prng,
        burnin=100,
        trials=100,
        nsamples=100)

@attr('slow')
def test_kernel_slice_hp_py():
    indiv_prior_fn = log_exponential(1.2)
    def init_inf_kernel_state_fn(s):
        hparams = {
            0 : {
                'alpha' : (indiv_prior_fn, 1.5),
                'beta'  : (indiv_prior_fn, 1.5),
                }
            }
        return hparams
    def prior_fn(raw):
        return indiv_prior_fn(raw['alpha']) + indiv_prior_fn(raw['beta'])
    kernel_fn = lambda s, arg, rng: py_slice_hp(s, rng, hparams=arg)
    _test_kernel_slice_hp(py_initialize,
                          init_inf_kernel_state_fn,
                          prior_fn,
                          py_numpy_dataview,
                          py_bind,
                          kernel_fn,
                          'grid_slice_hp_py_samples.pdf',
                          prng=None)

def test_kernel_slice_hp_cxx():
    indiv_prior_fn = log_exponential(1.2)
    def init_inf_kernel_state_fn(s):
        hparams = {
            0 : {
                'alpha' : (indiv_prior_fn, 1.5),
                'beta'  : (indiv_prior_fn, 1.5),
                }
            }
        return hparams
    def prior_fn(raw):
        return indiv_prior_fn(raw['alpha']) + indiv_prior_fn(raw['beta'])
    kernel_fn = lambda s, arg, rng: cxx_slice_hp(s, rng, hparams=arg)
    _test_kernel_slice_hp(cxx_initialize,
                          init_inf_kernel_state_fn,
                          prior_fn,
                          cxx_numpy_dataview,
                          cxx_bind,
                          kernel_fn,
                          'grid_slice_hp_cxx_samples.pdf',
                          rng())

@attr('slow')
def test_kernel_slice_hp_noninform_py():
    def init_inf_kernel_state_fn(s):
        hparams = {
            0 : {
                  ('alpha', 'beta') : (log_noninformative_beta_prior, 1.0),
                }
            }
        return hparams
    def prior_fn(raw):
        return log_noninformative_beta_prior(raw['alpha'], raw['beta'])
    kernel_fn = lambda s, arg, rng: py_slice_hp(s, rng, hparams=arg)
    _test_kernel_slice_hp(py_initialize,
                          init_inf_kernel_state_fn,
                          prior_fn,
                          py_numpy_dataview,
                          py_bind,
                          kernel_fn,
                          'grid_slice_hp_noninform_py_samples.pdf',
                          prng=None)

def test_kernel_slice_hp_noninform_cxx():
    def init_inf_kernel_state_fn(s):
        hparams = {
            0 : {
                  ('alpha', 'beta') : (log_noninformative_beta_prior, 1.0),
                }
            }
        return hparams
    def prior_fn(raw):
        return log_noninformative_beta_prior(raw['alpha'], raw['beta'])
    kernel_fn = lambda s, arg, rng: cxx_slice_hp(s, rng, hparams=arg)
    _test_kernel_slice_hp(cxx_initialize,
                          init_inf_kernel_state_fn,
                          prior_fn,
                          cxx_numpy_dataview,
                          cxx_bind,
                          kernel_fn,
                          'grid_slice_hp_noninform_cxx_samples.pdf',
                          rng())

def _test_cluster_hp_inference(initialize_fn,
                               prior_fn,
                               grid_min,
                               grid_max,
                               grid_n,
                               dataview,
                               bind_fn,
                               init_inf_kernel_state_fn,
                               inf_kernel_fn,
                               map_actual_postprocess_fn,
                               prng,
                               burnin=1000,
                               nsamples=1000,
                               skip=10,
                               trials=100,
                               places=2):
    print '_test_cluster_hp_inference: burnin', burnin, 'nsamples', nsamples, \
            'skip', skip, 'trials', trials, 'places', places

    N = 1000
    D = 5

    # create random binary data, doesn't really matter what the values are
    Y = np.random.random(size=(N, D)) < 0.5
    Y = np.array([tuple(y) for y in Y], dtype=[('', np.bool)]*D)
    view = dataview(Y)

    defn = model_definition(N, [bb]*D)
    latent = initialize_fn(defn, view, r=prng)
    model = bind_fn(latent, view)

    def score_alpha(alpha):
        prev_alpha = latent.get_cluster_hp()['alpha']
        latent.set_cluster_hp({'alpha':alpha})
        score = prior_fn(alpha) + latent.score_assignment()
        latent.set_cluster_hp({'alpha':prev_alpha})
        return score

    def sample_fn():
        for _ in xrange(skip-1):
            inf_kernel_fn(model, opaque, prng)
        inf_kernel_fn(model, opaque, prng)
        return latent.get_cluster_hp()['alpha']

    alpha0 = np.random.uniform(grid_min, grid_max)
    print 'start alpha:', alpha0
    latent.set_cluster_hp({'alpha':alpha0})

    opaque = init_inf_kernel_state_fn(latent)
    for _ in xrange(burnin):
        inf_kernel_fn(model, opaque, prng)
    print 'finished burnin of', burnin, 'iterations'

    print 'grid_min', grid_min, 'grid_max', grid_max
    assert_1d_cont_dist_approx_emp(sample_fn,
                                   score_alpha,
                                   grid_min,
                                   grid_max,
                                   grid_n,
                                   trials,
                                   nsamples,
                                   places)

    # MAP estimation over a large range doesn't really work
    #alpha_grid = np.linspace(grid_min, grid_max, grid_n)
    #alpha_scores = np.array(map(score_alpha, alpha_grid))
    #alpha_grid_map_idx = np.argmax(alpha_scores)
    #alpha_grid_map = alpha_grid[alpha_grid_map_idx]
    #alpha_grid_map_postproc = map_actual_postprocess_fn(alpha_grid_map)
    #print 'alpha MAP:', alpha_grid_map, \
    #      'alpha MAP postproc:', alpha_grid_map_postproc

    #alpha0 = np.random.uniform(grid_min, grid_max)
    #print 'start alpha:', alpha0
    #latent.set_cluster_hp({'alpha':alpha0})

    #opaque = init_inf_kernel_state_fn(latent)
    #for _ in xrange(burnin):
    #    inf_kernel_fn(model, opaque, prng)
    #print 'finished burnin of', burnin, 'iterations'

    #def posterior(k, skip):
    #    for _ in xrange(k):
    #        for _ in xrange(skip-1):
    #            inf_kernel_fn(model, opaque, prng)
    #        inf_kernel_fn(model, opaque, prng)
    #        yield latent.get_cluster_hp()['alpha']

    #bins = np.zeros(grid_n, dtype=np.int)
    #while 1:
    #    for sample in posterior(nsamples, skip):
    #        idx = min(np.searchsorted(alpha_grid, sample), grid_n-1)
    #        bins[idx] += 1
    #    est_map = alpha_grid[np.argmax(bins)]
    #    try:
    #        our_assert_almost_equals(est_map, alpha_grid_map, places=places)
    #        return # success
    #    except OurAssertionError as ex:
    #        print 'warning:', ex._ex.message
    #        trials -= 1
    #        if not trials:
    #            raise ex._ex

@attr('slow')
def test_kernel_slice_cluster_hp_py():
    prior_fn = log_exponential(1.5)
    def init_inf_kernel_state_fn(s):
        cparam = {'alpha':(prior_fn, 1.)}
        return cparam
    kernel_fn = lambda s, arg, rng: py_slice_hp(s, rng, cparam=arg)
    grid_min, grid_max, grid_n = 0.0, 50., 100
    _test_cluster_hp_inference(py_initialize,
                               prior_fn,
                               grid_min,
                               grid_max,
                               grid_n,
                               py_numpy_dataview,
                               py_bind,
                               init_inf_kernel_state_fn,
                               kernel_fn,
                               map_actual_postprocess_fn=lambda x: x,
                               prng=None)

def test_kernel_slice_cluster_hp_cxx():
    prior_fn = log_exponential(1.5)
    def init_inf_kernel_state_fn(s):
        cparam = {'alpha':(prior_fn, 1.)}
        return cparam
    kernel_fn = lambda s, arg, rng: cxx_slice_hp(s, rng, cparam=arg)
    grid_min, grid_max, grid_n = 0.0, 50., 100
    _test_cluster_hp_inference(cxx_initialize,
                               prior_fn,
                               grid_min,
                               grid_max,
                               grid_n,
                               cxx_numpy_dataview,
                               cxx_bind,
                               init_inf_kernel_state_fn,
                               kernel_fn,
                               map_actual_postprocess_fn=lambda x: x,
                               prng=rng())

#@attr('slow')
#def test_kernel_mh_hp():
#    # use a gaussian proposal
#    sigma2 = 0.2
#    Qfn = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
#    Q = Qfn(math.pi/8.)
#    cov = np.dot(np.dot(Q, np.diag([sigma2, 1e-3])), Q.T)
#
#    def gauss_prop_pdf(a, b):
#        a = np.array([a['alpha'], a['beta']])
#        b = np.array([b['alpha'], b['beta']])
#        # log Q(b|a)
#        return sp.stats.multivariate_normal.logpdf(b, mean=a, cov=cov)
#
#    def gauss_prop_sampler(a):
#        a = np.array([a['alpha'], a['beta']])
#        # rejection sample only in the first quadrant
#        while True:
#            alpha, beta = np.random.multivariate_normal(mean=a, cov=cov)
#            if alpha > 0.0 and beta > 0.0:
#                break
#        return {'alpha':alpha,'beta':beta}
#
#    def init_inf_kernel_state_fn(dpmm):
#        hparams = {0:{'hpdf':_bb_hyperprior_pdf,'hcondpdf':gauss_prop_pdf,'hsamp':gauss_prop_sampler}}
#        return hparams
#
#    _test_hp_inference(
#        init_inf_kernel_state_fn,
#        mh_hp,
#        lambda x: x,
#        'grid_mh_hp_samples.pdf')
