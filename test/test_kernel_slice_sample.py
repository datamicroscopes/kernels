from microscopes.py.kernels.slice import slice_sample as py_slice_sample
from microscopes.cxx.kernels.slice import sample as cxx_slice_sample
from microscopes.cxx.common.scalar_functions import log_normal
from microscopes.cxx.common.rng import rng
from microscopes.py.common.util import KL_approx

import numpy as np

def hist(data, bins):
    H, _ = np.histogram(data, bins=bins, density=False)
    return H

def _test_gauss(slice_sample_fn, prng):
    # sample from N(0,1)
    pdf = log_normal(0., 1.)
    def sampler(x0, niters):
        x = x0
        for _ in xrange(niters):
            x = slice_sample_fn(pdf, x, 2.0, prng)
            yield x

    bins = np.linspace(-3, 3, 1000)
    smoothing = 1e-5

    actual_samples = np.random.normal(size=10000)
    actual_hist = hist(actual_samples, bins) + smoothing
    actual_hist /= actual_hist.sum()

    slice_samples = np.array(list(sampler(1.0, 10000)))
    slice_hist = hist(slice_samples, bins) + smoothing
    slice_hist /= slice_hist.sum()

    # KL-divergence
    kldiff = KL_approx(actual_hist, slice_hist, bins[1]-bins[0])
    print 'KL:', kldiff

    # statistical distance
    maxdiff = np.abs(actual_hist - slice_hist).max()
    print 'maxdiff:', maxdiff

    assert kldiff  <= 0.005
    assert maxdiff <= 0.005

def test_gauss_py():
    _test_gauss(py_slice_sample, None)

def test_gauss_cxx():
    import time
    _test_gauss(cxx_slice_sample, rng(int(time.time())))
