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

def shrink(pdf, x0, y, L, R, ntries):
    """
    Fig. 5 of http://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    """
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

def slice_sample(pdf, x0, w, r=None):
    y = np.log(np.random.random()) + pdf(x0)
    L, R = interval(pdf, x0, y, w, m=10000)
    return shrink(pdf, x0, y, L, R, ntries=100)

class scalar_param(object):
    def __init__(self, prior, w):
        self._prior = prior
        self._w = w
    def set(self, hp, key, value):
        hp[key] = value
    def get(self, hp, key):
        return hp[key]
    def index(self):
        return None

class vector_param(object):
    def __init__(self, idx, prior, w):
        self._idx = idx
        self._prior = prior
        self._w = w
    def set(self, hp, key, value):
        hp[key][self._idx] = value
    def get(self, hp, key):
        return hp[key][self._idx]
    def index(self):
        return self._idx

def slice_hp(m, cparams, hparams, r=None):
    # XXX: this can be done in parallel
    for fi, hparam in hparams.iteritems():
        hp = m.get_feature_hp(fi)
        items = list(hparam.iteritems())
        for i in np.random.permutation(np.arange(len(items))):
            key, objs = items[i]
            if not hasattr(objs, '__iter__'):
                objs = [objs]
            for param in objs:
                def pdf(x):
                    param.set(hp, key, x)
                    m.set_feature_hp(fi, hp)
                    return param._prior(x) + m.score_data(fi, None)
                param.set(hp, key, slice_sample(pdf, param.get(hp, key), param._w))
        m.set_feature_hp(fi, hp)
    hp = m.get_cluster_hp()
    for key, objs in cparams.iteritems():
        if not hasattr(objs, '__iter__'):
           objs = [objs]
        for param in objs:
            def pdf(x):
                param.set(hp, key, x)
                m.set_cluster_hp(hp)
                return param._prior(x) + m.score_assignment()
            param.set(hp, key, slice_sample(pdf, param.get(hp, key), param._w))
    m.set_cluster_hp(hp)

def slice_theta(m, tparams, r=None):
    groups = np.array(m.groups(), dtype=np.int)
    for fi, params in tparams.iteritems():
        for k, w in params.iteritems():
            for gi in groups[np.random.permutation(len(groups))]:
                theta = m.get_suff_stats(gi, fi)
                def pdf(x):
                    theta[k] = x
                    m.set_suff_stats(gi, fi, theta)
                    return m.score_data(fi, gi, r)
                theta[k] = slice_sample(pdf, theta[k], w)
