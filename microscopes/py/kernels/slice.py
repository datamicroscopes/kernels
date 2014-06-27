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

def slice_hp(m, hparams):
    """
    hparams: dict mapping feature id to the following dict:
        {
            'hpdf' : <TODO: fill me in>,
            'hgrid': <TODO: fill me in>
        }
    """
    # XXX: this can be done in parallel
    for fi, hparam in hparams.iteritems():
        hpdf, hw = hparam['hpdf'], hparam['hw']
        items = list(hw.iteritems())
        for i in np.random.permutation(np.arange(len(items))):
            key, w = items[i]
            hp = m.get_feature_hp(fi)
            #print 'slicing on', key, 'with current value', hp
            def pdf(x):
                hp[key] = x
                #print '  * slice:', hp
                m.set_feature_hp(fi, hp)
                return hpdf(hp) + m.score_data(fi)
            hp[key] = slice_sample(hp[key], pdf, hw[key])
            #print 'sample result:', hp
            #print
            m.set_feature_hp(fi, hp)

def slice_theta(m, thetaparams):
    """
    XXX: doc
    """
    # XXX: this can be done in parallel
    for fi, thetaparam in thetaparams.iteritems():
        thetaw = thetaparam['thetaw']
        items = list(thetaw.iteritems())
        shared = m.get_feature_hp_shared(fi)
        for _, g in m.get_suff_stats_for_feature(fi):
            for i in np.random.permutation(np.arange(len(items))):
                key, w = items[i]
                theta = g.dump()
                def pdf(x):
                    theta[key] = x
                    g.load(theta)
                    return g.score_data(shared)
                theta[key] = slice_sample(theta[key], pdf, thetaw[key])
                g.load(theta)
