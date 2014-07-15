from microscopes.cxx.kernels.slice import theta
from microscopes.cxx.mixture.model import state, bind
from microscopes.cxx.common.recarray.dataview import numpy_dataview
from microscopes.cxx.common.rng import rng
from microscopes.cxx.models import bbnc

from microscopes.py.common.util import KL_approx
from scipy.stats import beta

import numpy as np

from nose.tools import assert_almost_equals

def test_slice_theta_simple():
    N = 100
    s = state(N, [bbnc])
    s.set_cluster_hp({'alpha':2.0})

    prior = {'alpha':1.0, 'beta':9.0}
    s.set_feature_hp(0, prior)

    data = np.array(
        [(np.random.random() < 0.8,) for _ in xrange(N)],
        dtype=[('',bool)])

    heads = len([1 for y in data if y[0]])
    tails = N - heads

    alpha1 = prior['alpha'] + heads
    beta1 = prior['beta'] + tails

    r = rng()
    view = numpy_dataview(data)

    bs = bind(s, view)
    bs.create_group(r)
    for i in xrange(N):
        bs.add_value(0, i, r)

    params = {0:{'p':0.05}}

    nsamples, ntries = 50000, 5
    raw_samples = []

    while 1:
        try:
            for _ in xrange(nsamples):
                theta(bs, params, r)
                raw_samples.append(s.get_suffstats(0, 0)['p'])
            samples = np.array(raw_samples)

            # check the mean
            true_mean = alpha1/(alpha1+beta1)
            est_mean = samples.mean()

            print 'true_mean', true_mean, 'est_mean', est_mean
            print np.abs(true_mean - est_mean)
            assert_almost_equals(true_mean, est_mean, places=3)

            # check variance
            true_var = alpha1*beta1/(((alpha1+beta1)**2)*(alpha1+beta1+1.))
            est_var = samples.var(ddof=1)

            print 'true_var', true_var, 'est_var', est_var
            print np.abs(true_var - est_var)
            assert_almost_equals(true_var, est_var, places=3)

            # check empirical KL
            bins = np.linspace(0., 1., 1000)
            smoothing = 1e-5
            est_hist, _ = np.histogram(samples, bins=bins, density=False)
            est_hist = np.array(est_hist, dtype=np.float)
            est_hist += smoothing
            est_hist /= est_hist.sum()

            points = (bins[1:]+bins[:-1])/2.
            actual_hist = beta.pdf(points, alpha1, beta1)
            actual_hist /= actual_hist.sum()

            print 'actual_hist', actual_hist[:5]
            print 'est_hist', est_hist[:5]

            kldiv = KL_approx(actual_hist, est_hist, bins[1]-bins[0])
            assert_almost_equals(kldiv, 0., places=3)

            #import matplotlib.pylab as plt
            #plt.plot(points, actual_hist, 'ro')
            #plt.plot(points, est_hist, 'gx')
            #plt.show()
            break
        except AssertionError:
            ntries -= 1
            if not ntries:
                raise
