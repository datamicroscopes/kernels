from microscopes.py.mixture.dp import state as py_state, fill
from microscopes.py.kernels.gibbs import gibbs_hp as py_gibbs_hp
from distributions.dbg.models import bb as py_bb

from microscopes.cxx.mixture.model import state as cxx_state
from microscopes.cxx.kernels.gibbs import hp as cxx_gibbs_hp
from microscopes.cxx.models import bb as cxx_bb
from microscopes.cxx.common.rng import rng

from microscopes.py.common.util import almost_eq
from microscopes.py.kernels.mh import mh_hp
from microscopes.py.kernels.slice import slice_hp

import numpy as np
import math
import scipy as sp
import scipy.stats
import matplotlib.pylab as plt
import itertools as it

from nose.plugins.attrib import attr

def _bb_hyperprior_pdf(hp):
    alpha, beta = hp['alpha'], hp['beta']
    if alpha > 0.0 and beta > 0.0:
        # http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-9.html
        return -2.5 * np.log(alpha + beta)
    return -np.inf

def _make_one_feature_bb_mm(ctor, bbmodel, Nk, K, alpha, beta, r):
    # XXX: the rng parameter passed does not get threaded through the
    # random *data* generation
    s = ctor(K*Nk, [bbmodel])
    s.set_cluster_hp({'alpha':2.0})
    s.set_feature_hp(0, {'alpha':alpha,'beta':beta})
    # use the py_bb for sampling
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
    fill(s, Y_clustered, r)
    return s, Y_clustered

def _grid_actual(s, lo, hi, nelems, r):
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
            z[i, j] = _bb_hyperprior_pdf(raw) + s.score_data(0, r)
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

def _test_hp_inference(
    ctor,
    bbmodel,
    grid_min, grid_max, grid_n,
    init_inf_kernel_state_fn,
    inf_kernel_fn,
    map_actual_postprocess_fn,
    grid_filename,
    prng,
    burnin=1000, nsamples=1000, skip=10, trials=5, tol=0.1):

    print '_test_hp_inference: burnin', burnin, 'nsamples', nsamples, 'skip', skip, 'trials', trials, 'tol', tol

    Nk = 1000
    K = 100
    s, Y_clustered = _make_one_feature_bb_mm(ctor, bbmodel, Nk, K, 0.8, 1.2, prng)

    xgrid, ygrid, z_actual = _grid_actual(s, grid_min, grid_max, grid_n, prng)

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
        inf_kernel_fn(s, opaque, prng)
    print 'finished burnin of', burnin, 'iterations'

    def trial():
        def posterior(k, skip):
            for _ in xrange(k):
                for _ in xrange(skip-1):
                    inf_kernel_fn(s, opaque, prng)
                inf_kernel_fn(s, opaque, prng)
                hp = s.get_feature_hp(0)
                yield np.array([hp['alpha'], hp['beta']])
        for samp in posterior(nsamples, skip):
            _add_to_grid(xgrid, ygrid, z_sample, samp)

    def draw_grid_plot():
        plt.imshow(z_sample, cmap=plt.cm.binary, origin='lower',
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

def _test_kernel_gibbs_hp(ctor, bbmodel, gibbs_hp_fn, fname, prng):
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
        ctor,
        bbmodel,
        grid_min, grid_max, grid_n,
        init_inf_kernel_state_fn,
        gibbs_hp_fn,
        map_actual_postprocess_fn,
        grid_filename=fname,
        prng=prng,
        burnin=0,
        trials=5,
        nsamples=100)

@attr('slow')
def test_kernel_gibbs_hp_py():
    _test_kernel_gibbs_hp(py_state, py_bb, py_gibbs_hp, 'grid_gibbs_hp_samples_py.pdf', None)

@attr('slow')
def test_kernel_gibbs_hp_cxx():
    _test_kernel_gibbs_hp(cxx_state, cxx_bb, cxx_gibbs_hp, 'grid_gibbs_hp_samples_cxx.pdf', rng(3954335))

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


#@attr('slow')
#def test_kernel_slice_hp():
#    def init_inf_kernel_state_fn(dpmm):
#        hparams = {0:{'hpdf':_bb_hyperprior_pdf,'hw':{'alpha':0.5,'beta':0.5}}}
#        return hparams
#
#    def inf_kernel_fn(dpmm, hparams):
#        slice_hp(dpmm, hparams)
#
#    _test_hp_inference(
#        init_inf_kernel_state_fn,
#        slice_hp,
#        lambda x: x,
#        'grid_slice_hp_samples.pdf',
#        burnin=1000,
#        nsamples=1000,
#        tol=0.15)
