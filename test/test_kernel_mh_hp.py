from microscopes.kernels.mh import mh_hp
from common import make_one_feature_bb_mm, bb_hyperprior_pdf

import itertools as it
import numpy as np
import math
import scipy as sp
import scipy.stats

from nose.plugins.attrib import attr

@attr('slow')
def test_kernel_mh_hp():
    Nk = 1000
    K = 100
    dpmm = make_one_feature_bb_mm(Nk, K, 1.0, 1.0)
    dpmm.set_feature_hp(0, {'alpha':1.5,'beta':1.5}) # don't start w/ the right answer

    # use a gaussian proposal
    sigma2 = 0.2
    Qfn = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = Qfn(math.pi/8.)
    cov = np.dot(np.dot(Q, np.diag([sigma2, 1e-3])), Q.T)

    def gauss_prop_pdf(a, b):
        a = np.array([a['alpha'], a['beta']])
        b = np.array([b['alpha'], b['beta']])
        # log Q(b|a)
        return sp.stats.multivariate_normal.logpdf(b, mean=a, cov=cov)

    def gauss_prop_sampler(a):
        a = np.array([a['alpha'], a['beta']])
        while True:
            alpha, beta = np.random.multivariate_normal(mean=a, cov=cov)
            if alpha > 0.0 and beta > 0.0:
                break
        return {'alpha':alpha,'beta':beta}

    hpdfs = (bb_hyperprior_pdf,)
    hcondpdfs = (gauss_prop_pdf,)
    hsamplers = (gauss_prop_sampler,)
    def posterior(k, skip):
        for _ in xrange(k):
            for _ in xrange(skip-1):
                mh_hp(dpmm, hpdfs, hcondpdfs, hsamplers)
            mh_hp(dpmm, hpdfs, hcondpdfs, hsamplers)
            hp = dpmm.get_feature_hp(0)
            yield np.array([hp['alpha'], hp['beta']])
    values = list(posterior(2000, 100))
    avg = sum(values) / len(values)

    #print avg
    #print values
    assert np.linalg.norm( np.array([1., 1.]) - avg ) <= 0.15