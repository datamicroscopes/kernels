"""
routines to make writing test cases less painful
"""

import numpy as np
from microscopes.py.common.util import KL_approx
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
            ntries -= 1
            if not ntries:
                raise ex._ex
