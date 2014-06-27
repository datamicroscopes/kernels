from scipy.stats import norm
from microscopes.py.kernels.slice import slice_sample
from microscopes.py.common.util import KL_approx

import numpy as np

def hist(data, bins):
    H, _ = np.histogram(data, bins=bins, density=False)
    return H

def test_gauss():
    # sample from N(0,1)

    pdf = lambda x: norm.logpdf(x)
    def sampler(x0, niters):
        x = x0
        for _ in xrange(niters):
            x = slice_sample(x, pdf, 0.1)
            yield x

    bins = np.linspace(-3, 3, 1000)
    smoothing = 1e-5

    actual_samples = np.random.normal(size=1000)
    actual_hist = hist(actual_samples, bins) + smoothing
    actual_hist /= actual_hist.sum()

    slice_samples = np.array(list(sampler(1.0, 1000)))
    slice_hist = hist(slice_samples, bins) + smoothing
    slice_hist /= slice_hist.sum()

    assert KL_approx(actual_hist, slice_hist, bins[1]-bins[0]) <= 0.1
