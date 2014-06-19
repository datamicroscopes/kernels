"""
Slice sampler based on:
    http://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    https://github.com/ericmjonas/netmotifs/blob/master/irm/slicesample.cc
"""

import numpy as np

import math
import logging
logger = logging.getLogger(__name__)

def interval(pdf, x0, y, w, m):
    """
    Fig. 3 of http://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    """
    U = np.random.random()
    L = x0 - w*U
    R = L + w
    V = np.random.random()
    J = int(math.floor(m*V))
    K = m-1-J

    while J > 0 and y < pdf(L):
        L -= w
        J -= 1

    while K > 0 and y < pdf(R):
        R += w
        K -= 1

    if J == 0 or K == 0:
        logging.warn('interval hit maximum expansions')
    return L, R

def shrink(pdf, x0, y, L, R):
    """
    Fig. 5 of http://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    """
    ntries = 100
    while ntries:
        U = np.random.random()
        x1 = L + U*(R-L)
        if y < pdf(x1):
            return x1
        if x1 < x0:
            L = x1
        else:
            R = x1
        ntries -= 1

    logging.warn('shrink exceeded maximum iterations (%d)' % (ntries))
    return x1

def slice_sample(x0, pdf, w):
    y = np.log(np.random.random()) + pdf(x0)
    L, R = interval(pdf, x0, y, w, 1000)
    return shrink(pdf, x0, y, L, R)

def slice_hp(m, hpdfs, hws):
    # XXX: this can be done in parallel
    for fi, (hpdf, hw) in enumerate(zip(hpdfs, hws)):
        for key, w in hw:
            hp = m.get_feature_hp(fi)
            hp0 = dict(hp)
            def pdf(x):
                hp0[key] = x
                m.set_feature_hp(fi, hp0)
                return hpdf(hp0) + m.score_data(fi)
            hp0[key] = slice_sample(pdf, hp[key], hw[key])
            m.set_feature_hp(fi, hp0)
